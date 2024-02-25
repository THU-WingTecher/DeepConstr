import functools
from typing import Callable, Union, List, Tuple, Dict, Any, Literal
from logger import LOGGER
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
        self.gen_interfered : int = 0
        
    def is_same(self, msg1, msg2) :
        # target args is same? 
        # target args's type is same ?
        return is_similar(msg1, msg2, threshold=self.cfg['str_sim_threshold'])

    def score_TP(self, num_of_check=20) :
        LOGGER.info(f"###Checking TP")
        # self.prompter.collect_control(FP=True, FN=False)
        solved, unsolved = self.evaluate(self.constr, num_of_check=num_of_check)
        if self.gen_interfered : return 0
        TP_ratio = solved/num_of_check
        
        return TP_ratio 
    def score_FN(self,num_of_check=20) :
        LOGGER.info(f"###Checking FN")
        FN_ratio = 0
        opposed = self.constr.gen_opposed()
        # self.prompter.collect_control(FN=True, FP=False)
        solved, unsolved = self.evaluate(opposed, num_of_check=num_of_check)
        FN_ratio = solved/num_of_check
        return FN_ratio
    def score_recall(self, FN, TP) :
        if TP == 0 : return 0 
        return (TP / (TP+FN))
    def score_precision(self, FN, TP) : return TP 
    def score_F1(self, num_of_check) :
        tp = self.score_TP(num_of_check)
        if tp == 0 : return 0
        fn = self.score_FN(num_of_check)
        if self.unable_to_gen : return 0
        self.precision = self.score_precision(fn, tp) * 100
        self.recall = self.score_recall(fn, tp) * 100
        # self.save_err_cases()
        if self.precision == 0 and self.recall == 0 : 
            self.f1_score = 0
        else :
            self.f1_score = 2 * self.precision*self.recall/(self.precision+self.recall)
        LOGGER.info(f"Score of Rule : {self.constr.txt}")
        LOGGER.info(f"### precision : {round(self.precision,2)}%, recall : {round(self.recall,2)}%, F1 : {round(self.f1_score,2)}%")
        return self.f1_score
    
    def unable_to_gen(self, record, constrs) : 
        n_unable = 0
        times = 5
        results = self.execute(record=record, constraints=constrs, ntimes=times)
        for res in results : 
            if res is None :
                n_unable+=1
        return n_unable == times 


    def evaluate(self, constr : Constraint, num_of_check : int =10) : 
        solved = 0
        unsolved = 0
        ungen = 0
        temp_constrs = copy.deepcopy(self.record["constraints"])
        temp_constrs.append(constr.get_executable())
        if self.unable_to_gen(self.record, temp_constrs) :
            LOGGER.info(f"This Rule interfere generation : {constr.txt}")
            return False 
        
        results = self.execute(record=self.record, constraints=temp_constrs, ntimes=num_of_check)

        for res in results :
            if res is None :
                ungen+=1
            else :
                success, errmsg = res
                if success :
                    ##FN_case
                    LOGGER.debug(f'###Changed : No error occured with {errmsg.get_values_map()}')
                    solved+=1
                elif self.is_same(errmsg.get_core_msg(), self.target) == False : 
                    LOGGER.debug(f'###Changed : Diff error with {errmsg.get_values_map()}')
                    LOGGER.info(f'Changed {self.target.get_core_msg()} -> {errmsg.get_core_msg()}')
                    solved+=1
                else :
                    ##FP_case
                    self.latest_errmsg = errmsg
                    LOGGER.debug(f'!!!!UNChanged : Same error with {errmsg.get_values_map()}')
                    LOGGER.info(f'UNChanged {self.target} -> {errmsg.get_core_msg()}')
                    unsolved+=1
        
        LOGGER.info(f"Solved : {solved}, Unsolved : {unsolved}, Unable to generate : {ungen}")
        return solved, unsolved, ungen