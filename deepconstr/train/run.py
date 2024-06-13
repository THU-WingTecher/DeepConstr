import copy
import functools
import glob
import json
import random
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple, Type
import yaml
import hydra
from nnsmith.backends.factory import BackendFactory
from nnsmith.materialize import Model
from omegaconf import DictConfig, ListConfig
from deepconstr.gen.record import Record, load_yaml, process_record, save_record
from deepconstr.gen.dtype import gen_dtype_info, add_default_rules
from deepconstr.train.constr import Constraint, convert_constr_to_executable, convert_dtypes_to_z3s
from deepconstr.train.errmsg import ErrorMessage, is_similar, map_error_messages_to_clusters_dynamic
from deepconstr.train.executor import Executor, is_normal_error
from deepconstr.train.inferencer import Inferencer
from deepconstr.train.prepare import check_trainable
from deepconstr.train.prompter import Prompter
from deepconstr.train.synthesizer import Synthesizer, segment_constr, parse_from_raw_txt
from deepconstr.utils import formatted_dict
from deepconstr.logger import TRAIN_LOG

# Status.csv
# Each line represents the status of one generation.
# META: seed,stat,tgen,tsmt,tsave,trun,elapsed_s
# stat can be one of:
#  - "ok": no error.
#  - "fail": invalid testcase.
#  - "bug": bug found.
SPLITTER = "\n"
class Scores(Dict): pass

def gen_type_info(api_name, path, library, inferencer) -> List :
    TRAIN_LOG.info(f"No files found for {api_name} ==> Try to generate ...")
    save_dir = os.path.dirname(path)
    file = gen_dtype_info(save_dir, api_name, library, inferencer)
    return file

class TrainingLoop:
    def __init__(
        self,
        cfg: DictConfig,
    ):
        
        self.cfg = cfg
        self.inferencer = Inferencer(cfg['llm']['settings'])
        cmpwith = cfg["cmp"]["with"]
        if cfg["backend"]["type"] == "tflite" and (
            cmpwith is None or cmpwith["target"] != "cuda"
        ):
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        if (
            cfg["train"]["crash_safe"]
            and cfg["backend"]["type"] == "xla"
            and cfg["backend"]["target"] == "cuda"
        ) or (
            cmpwith is not None
            and cmpwith["type"] == "xla"
            and cmpwith["target"] == "cuda"
        ):
            raise ValueError(
                "Please set `fuzz.crash_safe=false` for XLA on CUDA. "
                "Also see https://github.com/ise-uiuc/nnsmith/blob/main/doc/known-issues.md"
            )

        self.n_infer = int(cfg["train"]["n_infer_per_round"])

        model_cfg = self.cfg["model"]
        self.constr_target_map = {}
        self.ModelType = Model.init(
            model_cfg["type"], backend_target=cfg["backend"]["target"]
        )

        self.executor : Executor = Executor(self.ModelType, parallel = cfg["train"]["parallel"])
        self.train_list = self.traverse_train_list(target = cfg["train"]["target"])
        TRAIN_LOG.info(
            f"{len(self.train_list)} opsets wait for inferring"
        )

    def traverse_train_list(self, target) -> List[str]:
        """
        get operator yaml file path
        """
        
        def sort_key(file_name):
            base_name, index_with_extension = file_name.split('-')[0], file_name.split('-')[1]
            index = index_with_extension.replace('.yaml', '')
            return base_name, index

        train_list = []
        record_paths = []
        wait_for_gen = []
        if isinstance(target, (list, ListConfig)) : 
            train_list = target 
        elif isinstance(target, str) :
            if target.endswith("json") :
                with open(target, "r") as f:
                    train_list = json.load(f)
            else :
                train_list = [target]
        else :
            raise ValueError(f"Invalid train list(list, or str(api name or api name list)) : {type(target)}")
        
        root_path = self.cfg["train"]["record_path"] 
        for api_name in train_list :
            TRAIN_LOG.info(f"Finding Type information with {api_name}")
            all_files = []
            name_to_path = os.path.join(os.path.dirname(root_path), api_name.replace(".", "/"))
            search_pattern = name_to_path + "-*"
            files_found = glob.glob(search_pattern)
            if files_found:
                # If files are found, append them to the all_files list
                all_files.extend(files_found)
                TRAIN_LOG.info(f"Files found for {api_name}: {files_found}")
            else:
                func = functools.partial(gen_type_info, api_name, self.cfg["train"]["record_path"], self.cfg["model"]["type"])
                wait_for_gen.append(func)
                # If no files are found for this api, log the information
            record_paths.extend(all_files)

        record_paths = sorted(record_paths, key=sort_key)
        return record_paths + wait_for_gen
    
    def get_pass_rate_and_err_msgs(self, record, ntimes) -> Tuple[float, List[ErrorMessage]]:
        success_count = 0
        clusters = []
        raw_err_msgs = []
        copied_record = copy.deepcopy(record)
        executable_constr = convert_constr_to_executable(copied_record) # unactivated
        results = self.executor.execute(record = copied_record, 
                                        constraints = executable_constr,
                                        ntimes = ntimes, 
                                        noise = self.cfg["train"]["noise"],
                                        allow_zero_length_rate = self.cfg["train"]["allow_zero_length_rate"],
                                        allow_zero_rate = self.cfg["train"]["allow_zero_rate"],
                                        num_of_try = self.cfg["train"]["num_of_try"]
                                        )
        instance_mapping = {}
        for result in results:
            if not is_normal_error(result) :
                continue
            success, error_instance = result
            if success:
                success_count += 1
            else:
                msg_key = error_instance.get_core_msg()
                if instance_mapping.get(msg_key, False) :
                    pass
                else :
                    instance_mapping[msg_key] = error_instance
                raw_err_msgs.append(msg_key)

        if raw_err_msgs :
            dynamic_cluster_mapping = map_error_messages_to_clusters_dynamic(raw_err_msgs, self.cfg["train"]["str_sim_threshold"])
            for _, value in dynamic_cluster_mapping.items():
                instances = []
                for v in value :
                    if instance_mapping.get(v, None) is not None :
                        instances.append(instance_mapping[v])
                clusters.append(instances)
            clusters = list(sorted(clusters, key=lambda item: len(item), reverse=True))
            success_rate = success_count / (success_count + len(raw_err_msgs))
        else :
            success_rate = 1 

        if clusters:
            most_frequent_errs = clusters[0]
            distributions = {messages[0] : len(messages) for messages in clusters}
            TRAIN_LOG.debug(f"Current error distribution :\n{formatted_dict(distributions)}")
            # if self.is_special_err_msg(most_frequent_errs[0]):
            #     raise Exception("Special error message encountered. Cannot proceed.")
        else:
            TRAIN_LOG.info("No error messages encountered.")

        # Return success rate and the sorted list of error message instances
        return success_rate*100, clusters
    
    def select_train_op(self) :
        if self.train_list :
            return self.train_list.pop(0)
        else :
            return None 

    def parse_and_generate_rules(self, raw_infered : str, target : ErrorMessage, arg_names : List[str]) -> List[Constraint]:
        """
        Parse LLM response and generate rules.
        Filter ill-formed rules and return the valid ones.
        """
        generated = []
        segmented = []
        rules = []
        if raw_infered is None :
            return rules
        infered, cot = parse_from_raw_txt(raw_infered)
        dtypes = target.get_dtypes(arg_names)
        for rule_txt in infered.split(';'):
            generated.append(rule_txt.strip())
            segmented.extend(segment_constr(rule_txt.strip()))
        
        for i, rule_txts in enumerate([generated, segmented]):
            for rule_txt in rule_txts:
                if rule_txt :
                    if i == 1 : # segmented
                        cot = "divided"
                    rule = Constraint(rule_txt, cot, target, arg_names, dtypes)
                    if not rule.is_error() and rule.check() :
                        rules.append(rule)
        TRAIN_LOG.debug(f"Generated rules : {[c.txt for c in rules]}")
        return rules

    def update_tolerance_and_history(self, result : Tuple[Scores, Constraint], tolerance, highest_prev, prev_answer):
        """
        Update the training session's state based on the latest rule evaluation results.
        
        :param results: The results from the latest round of rule evaluations.
        :param tolerance: The current tolerance level.
        :param highest_prev: The highest score achieved so far.
        :param prev_answer: The current best rule based on previous evaluations.
        :return: A tuple containing updated tolerance, highest_prev, and prev_answer.
        """
        if result is None or highest_prev["overall_score"] >= result[0]["overall_score"] :
            # No new rules to evaluate or no improvement found, increase tolerance
            tolerance += 1
        else:
            # Extract the best result from the current evaluation
            best_current_score, best_constr = result
            # Found a better rule, reset tolerance and update history
            TRAIN_LOG.info(f'New highest score: from {highest_prev["overall_score"]} to {best_current_score["overall_score"]} with rule {best_constr.txt}')
            highest_prev = best_current_score
            prev_answer = best_constr
            tolerance = 0  # Reset tolerance since we found an improvement
        
        # Check if an optimal rule has been found
        solved = highest_prev["overall_score"] == 100
        return solved, tolerance, highest_prev, prev_answer

    def finalize_training_session(self, 
                                  target : ErrorMessage, 
                                  highest_prev : Dict[str, float], 
                                  record : Dict[str, Any], 
                                  constr_list : List[Tuple[Constraint, Scores]],
                                  synthesizer : Synthesizer, 
                                  prev_answer : Constraint = None) -> bool:
        """
        Finalize the training session, handling the selection of the most optimal rule.
        
        :param highest_prev: The highest score achieved during training.
        :param synthesizer: The synthesizer instance used during training.
        :param prev_answer: The rule corresponding to the highest score.
        :return: Boolean indicating whether a satisfactory rule was found.
        """
        # Log the tried rules and their outcomes
        TRAIN_LOG.debug(f"Error-Solved: {target.get_core_msg()}Tried rules:\n{SPLITTER.join(map(str, synthesizer.tried))}")
        
        if highest_prev["overall_score"] == 0:
            # No improvement found
            TRAIN_LOG.info("No improvement found.")
            return False
        elif highest_prev["overall_score"] == 100:
            # A rule with perfect score found
            TRAIN_LOG.info(f"Perfect rule found: {prev_answer.txt}")
            self.handle_solved_rule(prev_answer, highest_prev, record, constr_list)
            return True
        else:
            # Select the best rule based on the training outcome
            if synthesizer.non_FP:
                # Select the first non-false positive rule if available
                best_rule = synthesizer.non_FP[0][1]
            else:
                # Fallback to the best seed rule if the precision threshold is met
                if len(synthesizer.seeds) < 1 or synthesizer.seeds[0][0]["precision"] < self.cfg["train"]['precision_threshold']:
                    TRAIN_LOG.info(f"Best rule({synthesizer.seeds[0][0]['precision']}) does not meet the precision threshold({self.cfg['train']['precision_threshold']}).")
                    return False
                best_rule = synthesizer.seeds[0][1]
            
            TRAIN_LOG.info(f"Applying best found rule: {best_rule.txt}")
            self.handle_solved_rule(best_rule, highest_prev, record, constr_list) ## save rule to record
            return True   
        
    def get_only_acc_save_path(self, record_path) :
        ## change constraints -> only_acc
        new_path = record_path.replace("records", "only_acc")
        return new_path
    
    def get_retrain_list(self, record, constr_list : List[Tuple[Constraint, Scores]]) -> List[Tuple[Constraint, Scores]] :
        # pass_rate, sorted_err_instances = self.get_pass_rate_and_err_msgs(op_record)
        data = constr_list[:]
        for constr, scores in self.get_constrs_from_rules(record) :
            if constr.txt in [c.txt for c, _ in data] :
                continue 
            data.append((constr, scores))
        TRAIN_LOG.info(f"start retraining with {len(data)} constraints")
        return data
    
    @staticmethod
    def get_constrs_from_rules(record : Record) -> List[Any] :
        if record.get("rules", None) is None :
            record["rules"] = []
        return [(Constraint.load(rule_data), scores) for rule_data, scores in record["rules"]]
    
    def pop_constr_from_record(self, txt : str, record) :
        for i, (rule_data, _) in enumerate(record["rules"]) :
            if rule_data["txt"] == txt :
                popped = record["rules"].pop(i)
                TRAIN_LOG.info(f"popped : {popped[0]}")
                return (Constraint.load(popped[0]), popped[1])
        raise ValueError(f"no such constraint in record : {txt}")

    def is_trained(self, record : Record) :
        return record.get("pass_rate", 0) >= self.cfg["train"]["pass_rate"] #or record.get("trained", False)
    
    def is_trainable(self, record : Record) :
        return (
            record is not None and \
            record.get("error", None) is None and \
            self.is_trained(record) is False \
        )
    
    def inspect_trainable(self, record_path) :
        
        TRAIN_LOG.info(f"Validate the properties of {record_path}")
        with open(os.path.join(record_path), "r") as f:
            record = yaml.safe_load(f)
        if record.get("error", None) is not None :
            TRAIN_LOG.info(f"{record_path} is errored record")
            return False
        elif not self.cfg["train"]["retrain"] and any(rule.get('cot') != "default" for rule, score in record.get("rules", [])) > 0 :
            # TRAIN_LOG.info(f"{record_path} is already trained")
            return False
        TRAIN_LOG.info(f"Check whether {record_path} is valid")
        is_trainable, record = check_trainable(record, self.executor)
        if is_trainable :
            record = add_default_rules(record)
            save_record(record, record_path)
            return True
        
    def runs(self) :
        n_func = 0
        trained_func_index = self.cfg["temp"]["start"] if self.cfg["temp"]["start"] is not None else 0
        end_index = self.cfg["temp"]["end"] if self.cfg["temp"]["end"] is not None else len(self.train_list)
        TRAIN_LOG.info(f"Start training from {trained_func_index} to {end_index}")
        # while True :
        while n_func < end_index :
            n_func+=1
            record_path_or_func = self.select_train_op()
            if not isinstance(record_path_or_func, str) :
                record_path_or_func = record_path_or_func(self.inferencer)
            else :
                record_path_or_func = [record_path_or_func]
            for record_path in record_path_or_func :
                if not self.inspect_trainable(record_path) :
                    continue
                if n_func >= trained_func_index :
                    if record_path is None :
                        break
                    op_record = process_record(record_path, filter={})
                    TRAIN_LOG.info(f"Start infering {op_record['name']}({n_func})")
                    # self.get_err_msgs(op_record)
                    if self.is_trainable(op_record) :
                        try :
                            self.run(op_record, record_path)
                            self.finalize_infering(op_record, record_path)
                        except :
                            TRAIN_LOG.error(f"{traceback.format_exc()}")
                    else :
                        TRAIN_LOG.warning(f"""Record don't need-train/trainable : {formatted_dict(op_record, sep=":", split=SPLITTER)}""")

    def finalize_infering(self, record, record_path) :
        pass_rate, unsolved = self.get_pass_rate_and_err_msgs(record, self.cfg["train"]["num_eval"])
        unsolved_msgs = [e[0].dump() for e in unsolved if e] if unsolved else []
        self.update_pass_rate(record, pass_rate)
        save_record(record, record_path)
        constr_tar_dict = {r[0]['target']['msg']:(r[0]['txt'],('f1',r[1]['f1_score']),('prec',r[1]['precision'])) for r in record['rules']}
        TRAIN_LOG.info(
f"""{record['name']} infered finished\n
constrs : {formatted_dict(constr_tar_dict)}\n
final_pass_rate : {record['pass_rate']}\n
unsolved_err_msgs : {unsolved_msgs}
            """
        )  
        self.inferencer.finalize()
    
    def update_pass_rate(self, record, pass_rate = None) :
        if pass_rate is None :
            pass_rate, unsolved = self.get_pass_rate_and_err_msgs(record, self.cfg["train"]["num_eval"])
        TRAIN_LOG.info(f"pass_rate : {pass_rate}")
        record["pass_rate"] = pass_rate

    def save_only_acc(self, record, record_path) :
        save_path = self.get_only_acc_save_path(record_path)
        if os.path.exists(save_path) :
            data = load_yaml(save_path)
            if data["pass_rate"] > self.cfg["train"]["precision_threshold"] :
                return 
        save_record(record, save_path)
    
    def is_triaged_err_msg(self, errmsg : ErrorMessage, constr_list) -> bool:
        for constr, _ in constr_list :
            if is_similar(errmsg.get_core_msg(), constr.target.get_core_msg(), self.cfg["train"]["str_sim_threshold"]) :
                return True
        return False

    def is_retrainable(self, record : Record) :
        return record is not None and record.get("pass_rate", 0) > 0
    def extra_exit_check(self, pass_rate, sorted_err_instances, constr_list) :
        # if pass_rate >= self.cfg["train"]["pass_rate"] :
        #     TRAIN_LOG.info(f"pass_rate is over {self.cfg['train']['precision_threshold']}")
        #     return True
        if not sorted_err_instances :
            TRAIN_LOG.info(f"No error messages encountered")
            return True 
        if all([self.is_triaged_err_msg(e[0], constr_list) for e in sorted_err_instances]) :
            TRAIN_LOG.info(f"All err messages[{SPLITTER.join([e[0].get_core_msg() for e in sorted_err_instances])} are alredy triaged")
            return True 
        return False 
    
    def get_err_msgs(self, op_record): 

        file_name = f'{self.cfg["model"]["type"]}_err_msgs.json'
        if not os.path.exists(file_name) :
            data = {"solved" : {}, "unsolved" : {}}
        else : 
            with open(file_name, "r") as f :
                data = json.load(f)

        solved_msgs = []
        unsolved_msgs = []
        constrs = self.get_constrs_from_rules(op_record)
        if op_record["name"] in data["solved"] : 
            solved_msgs.extend(data["solved"][op_record["name"]]) 
            unsolved_msgs.extend(data["unsolved"][op_record["name"]]) 
        else :
            data["solved"][op_record["name"]] = []
            data["unsolved"][op_record["name"]] = []

        for constr, _ in constrs : 
            if any(is_similar(constr.target.msg, msg, threshold=self.cfg["train"]["str_sim_threshold"]) for msg in solved_msgs) :
                continue
            solved_msgs.append(constr.target.msg)

        #get other error messages
        pass_rate, sorted_err_instances = self.get_pass_rate_and_err_msgs(op_record, self.cfg["train"]["num_eval"])
        for err in sorted_err_instances : 
            if self.is_triaged_err_msg(err[0], constrs) :
                continue 
            elif any(is_similar(err[0].msg, msg, threshold=self.cfg["train"]["str_sim_threshold"]) for msg in unsolved_msgs) :
                continue
            else :
                unsolved_msgs.append(err[0].msg)

        # save as txt file 
        data["solved"][op_record["name"]] = solved_msgs 
        data["unsolved"][op_record["name"]] = unsolved_msgs 
        
        with open(file_name, "w") as f :
            json.dump(data, f, indent=4)

    def run(self, op_record, record_path):

        n_try = 0
        pass_rate = 0
        constr_list : List[Constraint] = []
        TRAIN_LOG.info(f"""Start training {op_record['name']}{SPLITTER}{formatted_dict(op_record, sep=":", split=SPLITTER)}""")
        constr_list.extend(self.get_constrs_from_rules(op_record))
        self.inferencer.init()
        while n_try < self.cfg["train"]["n_try"] :
            pass_rate, sorted_err_instances = self.get_pass_rate_and_err_msgs(op_record, self.cfg["train"]["num_eval"])
            self.update_pass_rate(op_record, pass_rate)
            if self.extra_exit_check(pass_rate, sorted_err_instances, constr_list) :
                break
            while sorted_err_instances :
                if all(self.is_triaged_err_msg(err[0], constr_list) for err in sorted_err_instances) :
                    TRAIN_LOG.warning(f"ALL err messages has been triaged.")
                    break
                most_frequents = sorted_err_instances.pop(0)
                succeed = self.train(op_record, most_frequents, constr_list, mode="acc")
                if succeed :
                    save_record(op_record, record_path)
                else :
                    TRAIN_LOG.error(f"Failed to train {most_frequents[0].get_core_msg()}")
                    return
                pass_rate, sorted_err_instances = self.get_pass_rate_and_err_msgs(op_record, self.cfg["train"]["num_eval"])
                self.update_pass_rate(op_record, pass_rate)
        
        self.save_only_acc(op_record, record_path)
        # if not self.is_retrainable(op_record) :
        #     TRAIN_LOG.warning(f"Record is not retrainable : {formatted_dict(op_record, sep=':', split=', ')}")
        #     return
        queue = self.get_retrain_list(op_record, constr_list)
        while queue :
            constr, scores = queue.pop()
            self.pop_constr_from_record(constr.txt, op_record)
            targets = self.reproduce_errs(op_record, constr.target)
            if targets :
                TRAIN_LOG.info(f"Reproduced {constr.target.get_core_msg()} after popped {constr.txt}")
                succeed = self.train(op_record, targets, [], mode="f1", seeds = [(scores, constr)])
                if succeed :
                    save_record(op_record, record_path)
                else :
                    break
            else :
                # unreproducable means the popped constraint are not solving the error message, it may solved by others, unretrainable
                self.handle_solved_rule(constr, scores, op_record, []) 
                constr_tar_dict = {r[0]['target']['msg']:r[0]['txt'] for r in op_record['rules']}
                TRAIN_LOG.warning(f"Failed to reproduce {constr.target.get_core_msg()} after popped {constr.txt}{SPLITTER}constrs : {formatted_dict(constr_tar_dict)}")

    def reproduce_errs(self, record, target : ErrorMessage) :
        targets = []
        targets = self.get_sim_errors(target, record, num_of_check = self.cfg["train"]["num_eval"])
        if targets : 
            return targets
        else :# no similar error messages
            return False
        
    def fix_record_attrs(self, record, target : ErrorMessage) :
        """
        fix dtypes and rules
        multiple dtypes -> single dtype
        txt rule -> executable rule(unactivated)
        """
        copied = copy.deepcopy(record)
        copied["rules"] = convert_constr_to_executable(copied)
        copied["args"]["dtype_obj"] = [[d] for d in target.get_dtypes(copied["args"]["name"])]
        return copied
    
    def get_same_dtype_errmsg(self, msgs : List[ErrorMessage]) :
        clusters = []
        for errmsg in msgs :
            if clusters :
                found = False
                for cluster in clusters :
                    if errmsg.get_dtypes_map() == cluster[0].get_dtypes_map() :
                        cluster.append(errmsg)
                        found = True
                        break
                if not found :
                    clusters.append([errmsg])
            else :
                clusters.append([errmsg])
        
        return sorted(clusters, key=lambda x : len(x), reverse=True)
    
    def get_sim_errors(self, target, record, num_of_check : int =10) : 

        instance_mapping = {}
        target_cluster = None
        raw_err_msgs = [] # [target.get_core_msg()]
        instance_mapping = {} # {target.get_core_msg() : target}
        executable_constr = convert_constr_to_executable(record) # unactivated
        results = self.executor.execute(record=record, constraints=executable_constr, ntimes=num_of_check)
        for result in results :
            if result is None :
                pass
            else:
                _, error_instance = result
                msg_key = error_instance.get_core_msg()
                if instance_mapping.get(msg_key, False) :
                    pass
                else :
                    instance_mapping[msg_key] = error_instance
                raw_err_msgs.append(msg_key)
        if len(raw_err_msgs) == 1 : # not added any result(= all result is None)
            return False
        dynamic_cluster_mapping = map_error_messages_to_clusters_dynamic(raw_err_msgs, self.cfg["train"]["str_sim_threshold"])
        for key, values in dynamic_cluster_mapping.items() :
            for value in values :
                if is_similar(value, target.get_core_msg(), threshold=self.cfg["train"]["str_sim_threshold"]) :
                    target_cluster = key 
                    break
            if target_cluster is not None : break
        
        return [instance_mapping[msg] for msg in dynamic_cluster_mapping.get(target_cluster, []) if instance_mapping.get(msg, None) is not None]
    
    def train(self, 
              orig_record, 
              targets : List[ErrorMessage],
              constr_list : List[Tuple[Constraint, Dict[str, Any]]],  
              mode : Literal["acc", "f1"], 
              seeds = []):
        # Initialize training session
        self.inferencer.change_to_gpt3()
        targets = self.get_same_dtype_errmsg(targets)[0]
        solved = False
        infer_times = 0
        prev_answer = None
        highest_prev : Dict[str, float] = {"overall_score" : 0, "precision" : 0, "recall" : 0, "f1_score" : 0}
        tolerance = 0
        new_rules : List[Constraint] = []
        record = self.fix_record_attrs(orig_record, targets[0]) # deep copy and fix dtypes and rules
        prompter = Prompter(record)
        synthesizer = Synthesizer(targets[0], self.executor, record, self.cfg["train"])
        # Synthesizer responsible for filtering, evaluating, and finding the best rule
        synthesizer.set_mode(mode)
        if seeds : 
            TRAIN_LOG.info(f"Start training with : {seeds[0][1]}")
            if seeds[0][0]["f1_score"] == 100 :
                self.update_record_constr(seeds[0][1], orig_record, seeds[0][0])
                return True
            synthesizer.save_state(seeds)
        while not solved:
            if tolerance >= self.cfg["train"]['tolerance']:
                if self.inferencer.is_gpt4() :
                    break
                else : 
                    self.inferencer.change_to_gpt4()
                    tolerance = 2 # for 4, give self.cfg["train"]['tolerance']-2 chance
            
            context, prompts = prompter.gen(targets, 
                                            func_name=record["name"], 
                                            synthesizer=synthesizer,
                                            prev_answer=prev_answer)
            
            raw_infered = self.inferencer.inference(prompts, context) 
            tolerance += 1

            new_rules = self.parse_and_generate_rules(raw_infered, 
                                                      targets[0] if targets else prompter.Q_history[-1], 
                                                      record["args"]["name"]
                                                      )
            if new_rules and not all(synthesizer.is_tried(r) for r in new_rules) :
                result = synthesizer.run(new_rules)
            else :
                result = None
                
            solved, tolerance, highest_prev, prev_answer = self.update_tolerance_and_history(result, tolerance, highest_prev, prev_answer)
            
            if synthesizer.get_errs_of_FP_cases() :
                targets.insert(0, synthesizer.get_errs_of_FP_cases()[0])
        return self.finalize_training_session(targets[0] if targets else prompter.Q_history[-1],
                                              highest_prev, orig_record, constr_list, synthesizer, prev_answer
                                              )
    
    def update_record_constr(self, constr : Constraint, record, scores) :
        record["rules"].append(
            [constr.dump(), scores]
        )
    def handle_solved_rule(self, 
                           constr : Constraint, 
                           scores : Dict[str, Any], 
                           record : Dict[str, Any],
                           constr_list : List[Tuple[Constraint, Dict[str, Any]]],
                           ) : 
        constr_list.append((constr, scores))
        self.update_record_constr(constr, record, scores)

    def is_special_err_msg(self, errmsg : ErrorMessage) -> bool:
        return errmsg.error_type in [TypeError]


@hydra.main(version_base=None, config_path="../../nnsmith/config", config_name="main")
def main(cfg: DictConfig):
    try :
        trainer = TrainingLoop(cfg)
        trainer.runs()
    except KeyboardInterrupt :
        raise ValueError(f"User requested to stop")
    except :
        TRAIN_LOG.error(f"{traceback.format_exc()}")


if __name__ == "__main__":
    main()
    # pass
