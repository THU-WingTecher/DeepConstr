import functools
from typing import Callable, Union, List, Tuple, Dict, Any, Literal
from logger import TRAIN_LOG
import copy
from neuri.constrinf.constr import Constraint, convert_dtypes_to_z3s
from neuri.constrinf.errmsg import is_similar
from neuri.constrinf.executor import Executor

class Evaluator() :
    def __init__(self, 
                 target : str,
                 constr : Constraint,
                 record : Dict[str, Any],
                 executor : Executor,
                 cfg : Dict[str, Any] = None
                 ) :
        
        self.target = target 
        self.constr = constr
        self.record = copy.deepcopy(record)
        self.record["args"]["dtype"] = [[dtype] for dtype in self.record["args"]["dtype"]]
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
        self.errmsgs_of_FP_cases = []
        self.precision = 0
        self.recall = 0
        self.f1_score = 0
        self.gen_interfered : bool = False
        
    def is_same(self, msg1, msg2) :
        # target args is same? 
        # target args's type is same ?
        return is_similar(msg1, msg2, threshold=self.cfg['str_sim_threshold'])

    def assess_TP(self, num_of_check=20) :
        TRAIN_LOG.info(f"###Checking TP")
        # self.prompter.collect_control(FP=True, FN=False)
        res = self.evaluate(self.constr, num_of_check=num_of_check)
        if res is False : 
            return self.handle_unable_to_gen()
        else :
            solved, unsolved = res
        if self.gen_interfered : 
            return 0
        TP_ratio = solved/num_of_check
        return TP_ratio
    def handle_unable_to_gen(self) :
        TRAIN_LOG.info(f"###Unable to generate")
        self.gen_interfered = True
        return 0
    def assess_FN(self,num_of_check=20) :
        TRAIN_LOG.info(f"###Checking FN")
        FN_ratio = 0
        opposed = self.constr.gen_opposed()
        # self.prompter.collect_control(FN=True, FP=False)
        res = self.evaluate(opposed, num_of_check=num_of_check)
        if res is False : 
            return self.handle_unable_to_gen()
        solved, unsolved = res
        FN_ratio = solved/num_of_check
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
    def get_f1(self, num_of_check) :
        tp = self.assess_TP(num_of_check)
        if self.gen_interfered : return 0
        fn = self.assess_FN(num_of_check)
        if self.gen_interfered : return 0
        self.precision = self.cal_precision(fn, tp)
        self.recall = self.cal_recall(fn, tp)
        # self.save_err_cases()
        if self.precision == 0 and self.recall == 0 : 
            self.f1_score = 0
        else :
            self.f1_score = 2 * self.precision*self.recall/(self.precision+self.recall)
        TRAIN_LOG.info(f"Score of Rule : {self.constr.txt}")
        TRAIN_LOG.info(f"### precision : {round(self.precision,2)}%, recall : {round(self.recall,2)}%, F1 : {round(self.f1_score,2)}%")
        return self.f1_score
    def evaluate(self, constr : Constraint, num_of_check : int =10) : 
        solved = 0
        unsolved = 0
        ungen = 0
        temp_constrs = copy.deepcopy(self.record["rules"])
        temp_constrs.append(constr.get_executable())
        results = self.execute(record=self.record, constraints=temp_constrs, ntimes=num_of_check)
        if results is None :
            TRAIN_LOG.info(f"This Rule interfere generation : {constr.txt}")
            return False 

        for res in results :
            if res is None :
                ungen+=1
            else :
                success, errmsg = res
                if success :
                    ##FN_case
                    TRAIN_LOG.debug(f'###Changed : No error occured with {errmsg.get_values_map()}')
                    solved+=1
                elif self.is_same(errmsg.get_core_msg(), self.target) == False : 
                    TRAIN_LOG.debug(f'###Changed : Diff error with {errmsg.get_values_map()}')
                    TRAIN_LOG.info(f'Changed {self.target.get_core_msg()} -> {errmsg.get_core_msg()}')
                    solved+=1
                else :
                    ##FP_case
                    self.latest_errmsg = errmsg
                    TRAIN_LOG.debug(f'!!!!UNChanged : Same error with {errmsg.get_values_map()}')
                    TRAIN_LOG.info(f'UNChanged {self.target} -> {errmsg.get_core_msg()}')
                    unsolved+=1
        
        TRAIN_LOG.info(f"Solved : {solved}, Unsolved : {unsolved}, Unable to generate : {ungen}")
        return solved, unsolved