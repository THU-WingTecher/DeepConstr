import functools
import logging
from typing import Callable, Union, List, Tuple, Dict, Any, Literal
from logger import TRAIN_LOG
import copy
from deepconstr.train.constr import Constraint
from deepconstr.train.errmsg import ErrorMessage, is_similar, map_error_messages_to_clusters_dynamic
from deepconstr.train.executor import NOERR_MSG, Executor, is_normal_error
from deepconstr.utils import formatted_dict

class Evaluator() :
    def __init__(self, 
                 target : ErrorMessage,
                 constr : Constraint,
                 record : Dict[str, Any],
                 executor : Executor,
                 cfg : Dict[str, Any] = None,
                 FP_container = []
                 ) :
        
        self.target = target 
        self.constr = constr
        self.record = copy.deepcopy(record)
        self.record["args"]["dtype_obj"] = [[d] for d in self.target.get_dtypes(self.record["args"]["name"])]
        self.cfg = cfg
        self.execute = functools.partial(executor.execute, 
                                        noise = self.cfg["noise"],
                                        allow_zero_length_rate = self.cfg["allow_zero_length_rate"],
                                        allow_zero_rate = self.cfg["allow_zero_rate"],
                                        num_of_try = self.cfg["num_of_try"])
        self.latest_args_values = {}
        self.args_values = {}
        self.tries = 0
        self.latest_errmsgs = ''
        self.container = {}
        self.errs_of_FP_cases = FP_container
        self.precision = 0
        self.recall = 0
        self.f1_score = 0
        self.gen_interfered : bool = False
    def is_same(self, msg1, msg2) :
        # target args is same? 
        # target args's type is same ?
        return is_similar(msg1, msg2, threshold=self.cfg['str_sim_threshold'])

    def save_err_cases(self, container, msgs) :
        for msg in msgs :
            if msg :
                if msg.get_core_msg() in container.keys() : 
                    container[msg.get_core_msg()].append(msg)
                else :
                    container[msg.get_core_msg()] = [msg]
    
    def assess_TP(self, num_of_check=20) :
        # self.prompter.collect_control(FP=True, FN=False)
        res = self.evaluate(self.constr, num_of_check=num_of_check)
        if res is False : 
            return self.handle_unable_to_gen()
        else :
            solved, unsolved = res
        self.save_err_cases(self.errs_of_FP_cases, unsolved)
        TP_ratio = len(solved)/(len(solved) + len(unsolved))
        return TP_ratio
    def handle_unable_to_gen(self) :
        TRAIN_LOG.info(f"Unable to generate")
        self.gen_interfered = True
        return 0
    def assess_FN(self,num_of_check=20) :
        FN_ratio = 0
        opposed = self.constr.gen_opposed()
        # self.prompter.collect_control(FN=True, FP=False)
        res = self.evaluate(opposed, num_of_check=num_of_check)
        if res is False : 
            return self.handle_unable_to_gen()
        solved, unsolved = res
        FN_ratio = len(solved)/(len(solved) + len(unsolved))
        return FN_ratio
    def cal_recall(self, FN, TP) :
        if TP == 0 : return 0 
        return (TP / (TP+FN)) * 100
    def cal_precision(self, FN, TP) : 
        if TP == 0 : 
            self.precision = 0
        else :
            self.precision = TP * 100
        return self.precision 
    def get_precision(self, num_of_check) : 
        fn = 0
        tp = self.assess_TP(num_of_check)
        precision = self.cal_precision(fn, tp)
        return precision
    def get_f1(self, num_of_check, ratio) :
        tp = self.assess_TP(num_of_check)
        if self.gen_interfered : return 0
        fn = self.assess_FN(int(num_of_check*ratio))
        if self.gen_interfered : return 0
        self.precision = self.cal_precision(fn, tp)
        self.recall = self.cal_recall(fn, tp)
        # self.save_err_cases()
        if self.precision == 0 and self.recall == 0 : 
            self.f1_score = 0
        else :
            self.f1_score = 2 * self.precision*self.recall/(self.precision+self.recall)

        return self.f1_score


    def evaluate(self, constr : Constraint, num_of_check : int =10) : 
        solved_instances = []
        unsolved_instances = []
        ungen = 0
        target_cluster = None
        raw_err_msgs = [self.target.get_core_msg()]
        error_messages = {self.target.get_core_msg() : self.target}
        temp_constrs = copy.deepcopy(self.record["rules"])
        temp_constrs.append(constr.get_executable())
        results = self.execute(record=self.record, constraints=temp_constrs, ntimes=num_of_check)
        
        for result in results :
            if not is_normal_error(result) :
                ungen+=1 
                continue
            success, error_instance = result
            msg_key = error_instance.get_core_msg()
            assert msg_key is not None
            error_messages[msg_key] = error_instance
            raw_err_msgs.append(msg_key)
        if len(raw_err_msgs) == 1 : # not added any result(= all result is None)
            return False
        dynamic_cluster_mapping = map_error_messages_to_clusters_dynamic(raw_err_msgs, self.cfg["str_sim_threshold"])
        for key, values in dynamic_cluster_mapping.items() :
            for value in values :
                if value == self.target.get_core_msg() :
                    target_cluster = key 
                    break
            if target_cluster is not None : break
        raw_err_msgs.remove(self.target.get_core_msg())
        for msg in raw_err_msgs : 
            if (msg in dynamic_cluster_mapping[target_cluster]) and (msg != NOERR_MSG) :
                unsolved_instances.append(error_messages[msg])
            else :
                solved_instances.append(error_messages[msg])

        if TRAIN_LOG.getEffectiveLevel()  <= logging.DEBUG:
            sorted_cluster_mapping = dict(sorted(dynamic_cluster_mapping.items(), key=lambda item: len(item[1]), reverse=True))
            distributions = {messages[0] : len(messages) for _, messages in sorted_cluster_mapping.items()}
            TRAIN_LOG.debug(f"[{constr.txt}] Current error distribution :\n{formatted_dict(distributions)}")

        TRAIN_LOG.info(f"[{constr.txt}] Solved: {len(solved_instances)}, Unsolved: {len(unsolved_instances)}, Unable to generate: {ungen}")
        return solved_instances, unsolved_instances