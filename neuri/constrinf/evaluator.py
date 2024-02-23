from multiprocessing import Pool
import re
from tqdm import trange
import traceback 
from typing import Callable, Union, List, Tuple, Dict, Any, Literal
import tqdm
from logger import LOGGER
import copy
from schema.rule import Rule 
from generators import ArgumentGenerator
from train.semantic import is_similar
import gc


class Evaluator() :
    def __init__(self, 
                 target : str,
                 rule : Rule,
                 record : Dict[str, Any],
                 executor : Executor,
                 cfg : Dict[str, Any] = None
                 ) :
        
        self.target = target 
        self.rule = rule
        self.record = record
        self.executor = executor
        self.cfg = cfg
        self.latest_args_values = {}
        self.args_values = {}
        self.tries = 0
        self.latest_errmsgs = ''
        self.container = {}
        self.errmsgs_of_FP_cases = []
        self.precision = 0
        self.recall = 0
        self.f1_score = 0
        self.unable_to_gen : int = 0
        
    def is_same(self, msg1, msg2) :
        # target args is same? 
        # target args's type is same ?
        return is_similar(msg1, msg2, threshold=self.cfg['str_sim_threshold'])

    def score_TP(self, num_of_check=20) :
        LOGGER.info(f"###Checking TP")
        # self.prompter.collect_control(FP=True, FN=False)
        solved, unsolved = self.evaluate(self.rule, num_of_check=num_of_check)
        if self.unable_to_gen : return 0
        TP_ratio = solved/num_of_check
        
        return TP_ratio 
    def score_FN(self, generator, num_of_check=20) :
        LOGGER.info(f"###Checking FN")
        FN_ratio = 0
        opposed = self.rule.gen_opposed()
        # self.prompter.collect_control(FN=True, FP=False)
        solved, unsolved = self.evaluate(opposed, num_of_check=num_of_check)
        FN_ratio = solved/num_of_check
        return FN_ratio
    def score_recall(self, FN, TP) :
        if TP == 0 : return 0 
        return (TP / (TP+FN))
    def score_precision(self, FN, TP) : return TP 
    def score_F1(self, generator : ArgumentGenerator, num_of_check) :
        tp = self.score_TP(generator, num_of_check)
        if tp == 0 : return 0
        fn = self.score_FN(generator, num_of_check)
        if self.unable_to_gen : return 0
        self.precision = self.score_precision(fn, tp) * 100
        self.recall = self.score_recall(fn, tp) * 100
        self.save_err_cases()
        if self.precision == 0 and self.recall == 0 : 
            self.f1_score = 0
        else :
            self.f1_score = 2 * self.precision*self.recall/(self.precision+self.recall)
        LOGGER.info(f"Score of Rule : {self.rule.txt}")
        LOGGER.info(f"### precision : {round(self.precision,2)}%, recall : {round(self.recall,2)}%, F1 : {round(self.f1_score,2)}%")
        return self.f1_score
    
    def unable_to_gen(self, record) : 
        unable = 0
        times = 5
        results = self.executor.execute(self.record, 4)
        for res in results : 
            if res is None :
                unable+=1
        return unable == times 

    def rm_rule_from_record(self, rule, record) :
        record['rules'].remove(rule)
        return record
    
    def evaluate(self, constr : Constraint, num_of_check : int =10) : 
        solved = 0
        unsolved = 0
        ungen = 0
        temp_constrs = copy.deepcopy(self.recorder["constraint"])
        temp_constrs.append(constr.get_executable())
        if self.unable_to_gen(self.record) :
            LOGGER.info(f"This Rule interfere generation : {constr.txt}")
            return False 
        
        results = self.executor.execute(self.record, temp_constrs, num_of_check)
        for res in results :
            if res is None :
                ungen+=1
            else :
                success, errmsg = res
                if success :
                    ##FN_case
                    LOGGER.debug(f'###Changed : No error occured with {generator.args_values}')
                    solved+=1
                elif self.is_same(errmsg.get_core_msg(), self.target) == False : 
                    LOGGER.debug(f'###Changed : Diff error with {generator.args_values}')
                    LOGGER.info(f'Changed {self.target} -> {cur_errmsg}')
                    solved+=1
                else :
                    ##FP_case
                    self.latest_errmsgs = cur_errmsg
                    LOGGER.debug(f'!!!!UNChanged : Same error with {generator.args_values}')
                    LOGGER.info(f'UNChanged {self.target} -> {cur_errmsg}')
                    unsolved+=1
        
        LOGGER.info(f"Solved : {solved}, Unsolved : {unsolved}, Unable to generate : {ungen}")
        return solved, unsolved, ungen