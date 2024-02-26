import copy
import random
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple, Type

import yaml
from neuri.constrinf import process_record
from neuri.constrinf.constr import Constraint, convert_constr_to_executable, convert_dtypes_to_z3s
from neuri.constrinf.errmsg import ErrorMessage
from neuri.constrinf.executor import Executor
from neuri.constrinf.inferencer import Inferencer
import hydra
from omegaconf import DictConfig

from neuri.backends.factory import BackendFactory
from neuri.constrinf.parser import segment_constr, parse_from_raw_txt
from neuri.constrinf.prompter import Prompter
from neuri.constrinf.synthesizer import Synthesizer
from neuri.error import WrongInferenceError
from neuri.gir import GraphIR
from neuri.logger import TRAIN_LOG, AUTOINF_LOG, TRAIN_LOG
from neuri.macro import NNSMITH_BUG_PATTERN_TOKEN
from neuri.materialize import BugReport, Model, TestCase
from neuri.util import mkdir, parse_timestr, set_seed

# Status.csv
# Each line represents the status of one generation.
# META: seed,stat,tgen,tsmt,tsave,trun,elapsed_s
# stat can be one of:
#  - "ok": no error.
#  - "fail": invalid testcase.
#  - "bug": bug found.
SPLITTER = "\n"
def transform_record_for_saving(record: dict) -> dict:
    """
    Transform the record dictionary to the original format expected for saving.

    Args:
        record (dict): The modified record dictionary.

    Returns:
        dict: The transformed record dictionary suitable for saving.
    """
    transformed = {}
    for key, value in record.items():
        if key == 'name':
            transformed['title'] = value
        elif key == 'args':
            pass
        else:
            transformed[key] = value
    return transformed

def save_record(record: dict, path: str) -> None:
    """
    Save a given record dictionary to a YAML file, ensuring that the dictionary is
    transformed back to its original expected format before saving.

    Args:
        record (dict): The record dictionary to be saved.
        path (str): The file path where the record should be saved.

    Raises:
        FileNotFoundError: If the directory specified in the path does not exist.
        Exception: For any unexpected errors during the save operation.
    """
    # Transform the record back to the expected format
    record = copy.deepcopy(record)
    record = transform_record_for_saving(record)

    # Ensure the directory exists
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        # raise FileNotFoundError(f"The directory {directory} does not exist.")
    
    try:
        with open(path, 'w') as file:
            yaml.dump(record, file)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"The directory {directory} does not exist.") from e
    except Exception as e:
        raise Exception(f"An error occurred while saving the record to {path}.") from e

    print(f"Record saved successfully to {path}.")

class StatusCollect:
    def __init__(self, root, resume=False, op_usage=False):
        self.root = Path(root)
        stat_fname = self.root / "status.csv"
        self.n_infer = 0
        self.n_synthesized = 0 
        self.cur_high_score = 0
        self.score_history = []
        self.n_bugs = 0
        self.n_fail_make_test = 0
        self.n_testcases = 0
        self._elapsed_s = 0
        self.infer_history = {}
        if (
            os.path.exists(self.root) and resume
        ):  # parse n_bugs, n_fail_make_test, n_testcases.
            with open(stat_fname, "r") as f:
                lines = f.readlines()
            for line in lines[1:]:
                if line == "":
                    continue
                tokens = line.split(",")
                stat = tokens[1]
                self._elapsed_s = float(tokens[-1])
                if stat == "bug":
                    self.n_bugs += 1
                elif stat == "fail":
                    self.n_fail_make_test += 1
                elif stat != "ok":
                    raise ValueError(f"Unknown stat: {stat}")
                self.n_testcases += 1
            TRAIN_LOG.info(f"Resuming from {self.n_testcases} testcases.")
        else:
            mkdir(self.root)
            with open(stat_fname, "w") as f:
                f.write("seed,stat,tgen(ms),tsmt(ms),tsave(ms),trun,elapsed(s)\n")

        self.stat_file = open(
            self.root / "status.csv", "a", buffering=1024
        )  # 1KB buffer
        self.tik = None

        self.op_usage = op_usage
        if self.op_usage:
            self.op_used_file = open(self.root / "op_used.txt", "a", buffering=128)
            self.op_rej_file = open(self.root / "op_rej.txt", "a", buffering=128)

    def ok_inst(self, ir: GraphIR):
        if self.op_usage:
            insts = ir.leaf_inst()
            if insts and insts[0].iexpr.op.__class__.__name__ == "AutoInfOpBase":
                self.op_used_file.write(f"{insts[0].iexpr.op.inst.name_index}\n")

    def rej_inst(self, ir: GraphIR):
        if self.op_usage:
            insts = ir.leaf_inst()
            if insts and insts[0].iexpr.op.__class__.__name__ == "AutoInfOpBase":
                self.op_rej_file.write(f"{insts[0].iexpr.op.inst.name_index}\n")
                self.op_rej_file.flush()  # flush every time to avoid losing data.

    @property
    def elapsed_s(self) -> float:
        if self.tik is None:
            self.tik = time.time()
        else:
            tok = time.time()
            self._elapsed_s += tok - self.tik
            self.tik = tok
        return self._elapsed_s

    def record(self, seed, stat, tgen, tsmt=0, trun=0, tsave=0):
        self.n_testcases += 1
        if stat == "bug":
            self.n_bugs += 1
        elif stat == "fail":
            self.n_fail_make_test += 1
        elif stat != "ok":
            raise ValueError(f"Unknown stat: {stat}")
        self.stat_file.write(
            f"{seed},{stat},{tgen},{tsmt},{tsave},{trun},{self.elapsed_s:.3f}\n"
        )
        self.stat_file.flush()
        TRAIN_LOG.info(
            f"tgen={tgen:.1f}ms, tsmt={tsmt:.1f}ms, trun={trun:.1f}ms, tsave={tsave:.1f}ms"
        )

    def get_next_bug_path(self):
        return self.root / f"bug-{NNSMITH_BUG_PATTERN_TOKEN}-{self.n_bugs}"

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

        self.factory = BackendFactory.init(
            cfg["backend"]["type"],
            target=cfg["backend"]["target"],
            optmax=cfg["backend"]["optmax"],
        )

        model_cfg = self.cfg["model"]
        self.constr_target_map = {}
        self.ModelType = Model.init(
            model_cfg["type"], backend_target=cfg["backend"]["target"]
        )
        # set_seed(self.cfg["train"]["seed"])
        # self.ModelType.add_seed_setter()
        self.executor : Executor = Executor(self.ModelType, cfg["train"]["parallel"])
        self.train_list = self.get_train_list()
        TRAIN_LOG.info(
            f"{len(self.train_list)} opsets wait for inferring"
        )
        # self.crash_safe = bool(cfg["train"]["crash_safe"])
        # self.save_test = cfg["train"]["save_test"]
        # resume = cfg["train"]["resume"]
        # self.status = StatusCollect(
        #     cfg["train"]["root"],
        #     resume=resume,
        #     op_usage=(
        #         cfg["mgen"]["max_nodes"] == 1 and cfg["mgen"]["method"] == "neuri-i"
        #     ),
        # )
        # if isinstance(self.save_test, str) and not (
        #     os.path.exists(self.save_test) and resume
        # ):  # path of root dir.
        #     TRAIN_LOG.info(f"Saving all intermediate testcases to {self.save_test}")
        #     mkdir(self.save_test)

    def get_train_list(self, module=None) -> List[str]:
        """
        get operator yaml file path
        """
        li = []
        root_path = self.cfg["train"]["record_path"] 
        if module is not None :
            raise NotImplementedError("module is not implemented")
        
        for root, _, files in os.walk(root_path):
            for file in files:
                if file.endswith(".yaml"):
                    li.append(os.path.join(root, file))
        return li
    
    def get_pass_rate_and_err_msgs(self, record, ntimes) -> Tuple[float, ErrorMessage]:
        # Assuming `run` is replaced with `self.execute` as per the revised requirement
        success_count = 0
        error_messages = {}
        copied_record = copy.deepcopy(record)
        executable_constr = convert_constr_to_executable(copied_record) # unactivated
        # wrapped = wrap(executable_constr)
        # copied_record["args"]["dtype"] = convert_dtypes_to_z3s(copied_record["args"]["dtype"])
        results = self.executor.execute(record = copied_record, 
                                        constraints = executable_constr,
                                        ntimes = ntimes, 
                                        noise = self.cfg["train"]["noise"],
                                        allow_zero_length_rate = self.cfg["train"]["allow_zero_length_rate"],
                                        allow_zero_rate = self.cfg["train"]["allow_zero_rate"],
                                        num_of_try = self.cfg["train"]["num_of_try"]
                                        )
        for result in results:
            success, error_instance = result
            if success:
                success_count += 1
            else:
                errmsg = error_instance.get_core_msg()
                if errmsg not in error_messages:
                    error_messages[errmsg] = {'instance': error_instance, 'count': 0}
                error_messages[errmsg]['count'] += 1

        # Calculate success rate
        success_rate = success_count / ntimes if ntimes else 0

        # Process error messages and their frequencies
        errmsg_counts = {errmsg: details['count'] for errmsg, details in error_messages.items()}

        # Sort error messages by frequency
        sorted_errmsgs = sorted(errmsg_counts.items(), key=lambda item: item[1], reverse=True)

        sorted_error_instances = [error_messages[errmsg]['instance'] for errmsg, _ in sorted_errmsgs]

        # Log the most frequent error message
        if sorted_errmsgs:
            most_frequent_errmsg, count = sorted_errmsgs[0]
            TRAIN_LOG.info(f"Most frequent error message: {most_frequent_errmsg} with {count}")
            if self.is_special_err_msg(most_frequent_errmsg):
                raise Exception("Special error message encountered. Cannot proceed.")
        else:
            TRAIN_LOG.info("No error messages encountered.")

        # Return success rate and the sorted list of error message instances
        return success_rate, sorted_error_instances
    
    def select_train_op(self) :
        if self.train_list :
            return self.train_list.pop()
        else :
            return None 

    def parse_and_generate_rules(self, raw_infered : str, target : ErrorMessage, arg_names : List[str]) -> List[Constraint]:
        """
        Parse LLM response and generate rules.
        Filter ill-formed rules and return the valid ones.
        """
        generated = []
        rules = []
        infered, cot = parse_from_raw_txt(raw_infered)
        dtypes = target.get_dtypes()
        for rule_txt in infered.split(';'):
            generated.append(rule_txt)
            generated.extend(segment_constr(rule_txt))
        
        for rule_txt in generated:
            if rule_txt :
                rule = Constraint(rule_txt, cot, target, arg_names, dtypes)
                if not rule.is_error() and rule.check() :
                    rules.append(rule)
        return rules

    def update_tolerance_and_history(self, result : Tuple[Dict[str, float], Constraint], tolerance, highest_prev, prev_answer):
        """
        Update the training session's state based on the latest rule evaluation results.
        
        :param results: The results from the latest round of rule evaluations.
        :param tolerance: The current tolerance level.
        :param highest_prev: The highest score achieved so far.
        :param prev_answer: The current best rule based on previous evaluations.
        :return: A tuple containing updated tolerance, highest_prev, and prev_answer.
        """
        if highest_prev["overall_score"] >= result[0]["overall_score"] :
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
                                  highest_prev : Dict[str, float], 
                                  record : Dict[str, Any], 
                                  constr_target_map : Dict[ErrorMessage, Tuple[Dict[str, float],Constraint]], 
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
        TRAIN_LOG.info(f"Tried rules:\n{SPLITTER.join(map(str, synthesizer.tried))}")
        
        if highest_prev["overall_score"] == 0:
            # No improvement found
            TRAIN_LOG.info("No improvement found.")
            return False
        elif highest_prev["overall_score"] == 100:
            # A rule with perfect score found
            TRAIN_LOG.info(f"Perfect rule found: {prev_answer.txt}")
            self.handle_solved_rule(prev_answer, highest_prev, record, constr_target_map)
            return True
        else:
            # Select the best rule based on the training outcome
            if synthesizer.non_FP:
                # Select the first non-false positive rule if available
                best_rule = synthesizer.non_FP[0][1]
            else:
                # Fallback to the best seed rule if the precision threshold is met
                if synthesizer.seeds[0][0]["precision"] < self.cfg["train"]['precision_threshold']:
                    TRAIN_LOG.info("Best rule does not meet the precision threshold.")
                    return False
                best_rule = synthesizer.seeds[0][1]
            
            TRAIN_LOG.info(f"Applying best found rule: {best_rule.txt}")
            self.handle_solved_rule(best_rule, highest_prev) ## save rule to record
            return True   
    def get_only_acc_save_path(self, record_path) :
        ## change constraints -> only_acc
        new_path = record_path.replace("constraints", "only_acc")
        if os.path.exists(new_path) :
            return new_path
        else :
            os.makedirs(new_path, exist_ok=True)
            return new_path
    
    def get_retrain_list(self, record, constr_target_map) -> List[Tuple[ErrorMessage, Constraint]] :
        # pass_rate, sorted_err_instances = self.get_pass_rate_and_err_msgs(op_record)
        return [(target, constr) for target, constr in constr_target_map.items()]
    
    @staticmethod
    def get_constrs_from_rules(record) :
        return record["rules"]
    
    def pop_constr_from_record(self, txt : str, record) :
        constrs = self.get_constrs_from_rules(record)
        for const_instance in constrs :
            if const_instance["txt"] == txt :
                constrs.remove(const_instance)
                return True
        raise ValueError(f"no such constraint in record : {txt}")

    def runs(self) :
        while True :
            record_path = self.select_train_op()
            if record_path is None :
                break
            op_record = process_record(record_path)
            if op_record is None :
                continue
            self.run(op_record, record_path)

    def run(self, op_record, record_path):
        n_try = 0
        pass_rate = 0
        record_path =record_path.replace("constraints", "debug") # for debugging
        op_record['rules'] = [] # for debugging
        constr_target_map : Dict[ErrorMessage, Tuple[Dict[str, float],Constraint]] = {}
        while pass_rate < 100 and n_try < self.cfg["train"]["n_try"] :
            pass_rate, sorted_err_instances = self.get_pass_rate_and_err_msgs(op_record, self.cfg["train"]["eval_asset"])
            if sorted_err_instances :
                succeed = self.train(op_record, sorted_err_instances[0], constr_target_map, mode="acc")
                if succeed :
                    save_record(op_record, record_path)
                else :
                    break
            else :
                break
        
        save_record(op_record, self.get_only_acc_save_path(record_path))
        queue = self.get_retrain_list(op_record, constr_target_map)
        while queue :
            target_err, (scores, constr) = queue.pop()
            self.pop_constr_from_record(constr.txt, op_record)
            succeed = self.train(op_record, target_err, constr_target_map, mode="f1", seeds = [(scores, constr)])
            if succeed :
                save_record(op_record, record_path)
            else :
                break
    
    def fix_record_attrs(self, record, target : ErrorMessage) :
        """
        fix dtypes and rules
        multiple dtypes -> single dtype
        txt rule -> executable rule(unactivated)
        """
        copied = copy.deepcopy(record)
        copied["rules"] = convert_constr_to_executable(copied)
        copied["args"]["dtype"] = target.get_dtypes()
        return copied
    
    def train(self, 
              orig_record, 
              target : ErrorMessage,
              constr_target_map : Dict[ErrorMessage, Tuple[Dict[str, float],Constraint]],  
              mode : Literal["acc", "f1"], 
              seeds = []):
        # Initialize training session

        solved = False
        infer_times = 0
        prev_answer = None
        highest_prev : Dict[str, float] = {"overall_score" : 0, "precision" : 0, "recall" : 0, "f1_score" : 0}
        tolerance = 0
        new_rules : List[Constraint] = []
        record = self.fix_record_attrs(orig_record, target) # deep copy and fix dtypes and rules
        prompter = Prompter(record)
        synthesizer = Synthesizer(target, self.executor, record, self.cfg["train"])
        # Synthesizer responsible for filtering, evaluating, and finding the best rule
        synthesizer.set_mode(mode)
        if seeds : 
            synthesizer.save_state(seeds)
        while self.cfg["train"]['infer_asset_per_epoch'] >= infer_times and not solved:
            if tolerance >= self.cfg["train"]['tolerance']:
                break
            
            # Generate prompts for inference
            context, prompts = prompter.gen([target.get_core_msg()], 
                                    args_values=target.get_values_map(),
                                    num_of_ex=random.randint(2, 3),
                                    prev_answer=prev_answer)
            
            # raw_infered = self.inferencer.inference(prompts, context) # for debugging
            raw_infered = """Error is triggered because the input tensor A must have at least 2 dimensions, but the provided value is a 1-dimensional tensor. To prevent this error from occurring again, we can generate constraints that ensure the input tensor has at least 2 dimensions. We can define the constraint as follows:
```len(input.shape) >= 2```"""
            infer_times += 1
            
            # Parse LLM response and generate rules
            new_rules = self.parse_and_generate_rules(raw_infered, 
                                                      target, 
                                                      record["args"]["name"]
                                                      )
            # Evaluate and optimize generated rules
            result = synthesizer.run(new_rules)
            # Update tolerance, history, and select the best rule based on results
            solved, tolerance, highest_prev, prev_answer = self.update_tolerance_and_history(result, tolerance, highest_prev, prev_answer)
        
        # Finalize training session and handle rule selection
        return self.finalize_training_session(highest_prev, orig_record, constr_target_map, synthesizer, prev_answer)
    
    def update_record_constr(self, constr, record, scores) :
        self.get_constrs_from_rules(record).append({
            "txt" : constr.txt,
            "cot" : constr.cot,
            "target" : constr.target.get_core_msg(),
            "scores" : scores
        })
    def handle_solved_rule(self, 
                           constr : Constraint, 
                           scores : Dict[str, Any], 
                           record : Dict[str, Any],
                           constr_target_map : Dict[ErrorMessage, Tuple[Dict[str, float],Constraint]]
                           ) : 
        constr_target_map.update({constr.target : (scores, constr)})
        self.update_record_constr(constr, record, scores)

    def is_special_err_msg(self, errmsg) :
        lowered = errmsg.lower()
        if "typeerror" in lowered :
            return True


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):
    TrainingLoop(cfg).runs()


if __name__ == "__main__":
    main()
