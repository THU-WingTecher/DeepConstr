from logger import TRAIN_LOG
from typing import Callable, Any, List, Tuple, Dict, Optional, Union, Literal
from deepconstr.train.constr import Constraint, conn_rule_txt
from deepconstr.train.evaluator import Evaluator
from deepconstr.grammar.base import is_same_constr, is_implies, has_same_rule
from deepconstr.utils import SPLITER
from typing import Tuple

OR = " or "
AND = " and "
SYNTHESIZED_COT = "synthesized"
class Synthesizer:
    def __init__(self, 
                 target,
                 executor,
                 record,
                 cfg):
        self.target = target
        self.executor = executor
        self.record = record
        self.cfg = cfg
        self.saved_state : List[Tuple[Dict[str, float], Constraint]] = []
        self.seeds : List[Tuple[Dict[str, float], Constraint]] = []
        self.non_FP : List[Tuple[Dict[str, float], Constraint]] = []  # For storing the best constraints found so far
        self.non_FN : List[Tuple[Dict[str, float], Constraint]] = []  # For storing the best constraints found so far
        self.tried : List["z3.ExprRef"] = []
        self.mode : Literal["acc", "f1"] = 'f1'
        self.errs_of_FP_cases = {}
        self.FN_err_instances = []
    
    def set_target(self, target) :
        self.target = target

    def set_mode(self, mode : Literal["acc", "f1"]) : 
        assert mode in ["acc", "f1"], f"mode should be either 'acc' or 'f1', but got {mode}"
        self.mode = mode 
    def goal_is_acc(self) : return self.mode == 'acc'
    def cal_overall_score(self, scores : Dict[str, Any], length) -> int :
        # Placeholder: Evaluate and return the score based on your criteria (e.g., F1 score / num_of_atomic_constraints)
        alpha = 0.5
        score = 0
        if self.goal_is_acc() : 
            beta = 3
            score = alpha * scores.get("precision", 0) + beta * (1 / length)
        else :
            beta = 12
            score = alpha * scores.get("f1_score", 0) + beta * (1 / length)
        scores["overall_score"] = score

    def save_state(self, seeds) :
        for seed in seeds :
            self.saved_state.append(seed)

    def get_varied_errs_from_evaluator(self, evaluator) :
        self.errs_of_FP_cases = evaluator.errs_of_FP_cases

    def get_errs_of_FP_cases(self) :
        clusters = list(self.errs_of_FP_cases.values())
        sorted(clusters, key=lambda x : len(x), reverse=True)
        return clusters[0] if clusters else []

    def evaluate(self, constraint : Constraint, skim : bool = False) -> Dict[str, Any]:
        # Placeholder: Calculate F1, precision, and recall for the given constraint
        scores = {}
        evaluator = Evaluator(self.target, constraint, self.record, self.executor, self.cfg, self.errs_of_FP_cases)
        TRAIN_LOG.info(f"Evaluating [[{constraint.txt}]] ::: Z3 [[{constraint.z3expr}]]")

        evaluator.get_f1(num_of_check=self.cfg['simple_num_eval'] if skim else self.cfg['num_eval'], ratio=self.cfg['eval_ratio'])
        self.get_varied_errs_from_evaluator(evaluator)
        scores["precision"] = evaluator.precision
        scores["recall"] = evaluator.recall
        scores["f1_score"] = evaluator.f1_score
        TRAIN_LOG.info(f"Score of Rule : {constraint.txt}")
        TRAIN_LOG.info(f"prec : {round(scores['precision'], 2)}%, recall : {round(scores['recall'], 2)}%, F1 : {round(scores['f1_score'], 2)}%")
        return scores
    
    def synthesize_with_semi_complete(self, constraints : List[Constraint]) -> Constraint:
        # Placeholder: Combine the given 'best' constraint with all other atomic constraints
        # and add them to the queue for further testing
        res = []
        changed = False
        for constr in constraints :
            synthesized = constr 
            if self.non_FP and self.need_synthesis(constr, self.non_FP[0][1]):
                changed = True
                synthesized = self.synthesize_constrs(constr, self.non_FP[0][1], OR)
                assert synthesized is not None
            if self.non_FN and self.need_synthesis(constr, self.non_FN[0][1]):
                changed = True
                synthesized = self.synthesize_constrs(constr, self.non_FN[0][1], AND)
                assert synthesized is not None
            if changed :
                res.append(synthesized)
        return res
    def update_seeds(self, new_seeds : List[Tuple[Dict[str,float], Constraint]]):
        for new_seed in new_seeds :
            if self.seeds and has_same_rule(new_seed[1].z3expr, [c.z3expr for _, c in self.seeds]) :
                pass 
            else :
                self.seeds.append(new_seed)
        self.seeds.sort(key=lambda pair: pair[0].get("overall_score", 0), reverse=True)
        self.seeds = self.seeds[:self.cfg['max_num_of_seeds']]
        TRAIN_LOG.debug(f"current seeds : {[(score, c.txt, c.length) for score, c in self.seeds]}")

    def need_synthesis(self, left : Constraint, right : Constraint) -> bool :
        return not is_implies(left.z3expr, right.z3expr) and not is_implies(right.z3expr, left.z3expr) and (left.length+right.length <= self.cfg["max_length"])
    def synthesize_constrs(self, left : Constraint, right : Constraint, method = ' or ') -> List[Constraint] :
        # Placeholder: Combine the given 'best' constraint with all other atomic constraints
        # and add them to the queue for further testing
        synthesized_txt = conn_rule_txt([left.txt, right.txt], method=method)
        synthesized_constr = Constraint(synthesized_txt, SYNTHESIZED_COT, left.target, left.arg_names, left.dtypes)
        synthesized_constr.length = left.length + right.length
        if synthesized_constr.check() :
            return synthesized_constr
        else :
            print(synthesized_constr.txt, "is not valid")
            return None 

    def get_all_synthesized(self, orig : List[Constraint], new : List[Constraint]) : #update_queue
        res : List[Constraint] = []
        for orig_constr in orig : 
            for new_constr in new : 
                if self.need_synthesis(orig_constr, new_constr) :
                    res.append(self.synthesize_constrs(orig_constr, new_constr, OR))
                    res.append(self.synthesize_constrs(orig_constr, new_constr, AND))
        return res
    def is_tried(self, const : Constraint) : 
        return has_same_rule(const.z3expr, self.tried) 
    def update_tried(self, constraint :  Constraint) :
        self.tried.append(constraint.z3expr)
    def rm_from_tried(self, constraint : Constraint) :
        self.tried = [c for c in self.tried if not is_same_constr(c, constraint.z3expr)]
    def rm_dupe(self, rules : List[Constraint]) -> List[Constraint] :
        return [r for r in rules if not self.is_tried(r)]
    def load_high_quality_constrs(self, top_k = None, return_score = False) -> Union[List[Tuple[Dict[str, Any], Constraint]], List[Constraint]] :
        if top_k is None :
            top_k = self.cfg['top_k']
        if return_score : 
            return [seed for seed in self.seeds][:top_k]
        else :
            return [seed[1] for seed in self.seeds][:top_k]
    
    def load_saved_state(self) :
        return [state[1] for state in self.saved_state]
    
    def run(self, new_rules : List[Constraint]) -> Tuple[float, Constraint] :

        queue : List[Constraint]= []
        high_quality = self.load_high_quality_constrs()
        if self.saved_state : 
            saved_rules = self.load_saved_state()
            queue.extend(saved_rules)
            self.saved_state = False
        filtered_new = self.rm_dupe(new_rules) 
        queue.extend(filtered_new)
        synthesized = self.get_all_synthesized(high_quality, filtered_new)
        queue.extend(synthesized)
        synthesized = self.synthesize_with_semi_complete(queue)
        queue.extend(synthesized)
        if len(queue) > self.cfg['top_k']*2 :
            filtered_res = self.find_optimal_constrs(queue, skim=True)
            queue = [c for _, c in filtered_res][:self.cfg['top_k']*2]

        results = self.find_optimal_constrs(queue)
        self.update_seeds(results)
        res = self.load_high_quality_constrs(top_k=1, return_score=True) 
        return res[0] if res else None
    
    def find_optimal_constrs(self,
            queue : List[Constraint], 
            skim : bool = False,
            ) -> List[Tuple[Dict[str, float],Constraint]]:
        
        res = []
        found_complete = False
        TRAIN_LOG.debug(f'finding optimal constr(skim={skim}){SPLITER}{SPLITER.join([c.txt for c in queue])}')
        while queue :
            constraint = queue.pop()
            self.update_tried(constraint)
            scores = self.evaluate(constraint, skim=skim)
            self.cal_overall_score(scores, constraint.length)
            if not skim :
                if scores.get("f1_score", 0) == 100:
                    found_complete = True
                    TRAIN_LOG.debug(f"Found optimal({self.mode}) constraint[{constraint.txt}]. Ending search.")
                    break
                if scores.get("precision", 0) == 100 :
                    if self.goal_is_acc() : 
                        found_complete = True
                        TRAIN_LOG.debug(f"Found optimal({self.mode}) constraint[{constraint.txt}]. Ending search.")
                        break
                    self.non_FP.append((scores, constraint))
                    TRAIN_LOG.debug(f"Found Non FP constraint[{constraint.txt}].")
                if scores.get("recall", 0) == 100:
                    self.non_FN.append((scores, constraint))
                    TRAIN_LOG.debug(f"Found NON FN constraint[{constraint.txt}]")
            res.append((scores, constraint))
        
        if not skim and found_complete :
            scores["overall_score"] = 100
            res = [(scores, constraint)]
        else :
            #return best f1 score from self.queue 
            res.sort(key=lambda x : x[0]["overall_score"], reverse=True)
            self.non_FP.sort(key=lambda x : x[0]["overall_score"], reverse=True)
            if self.non_FP :
                self.non_FP = [self.non_FP[0]]
            self.non_FN.sort(key=lambda x : x[0]["overall_score"], reverse=True)
            if self.non_FN :
                self.non_FN = [self.non_FN[0]]
            TRAIN_LOG.debug(f"current non_FPs : {', '.join([c[1].txt for c in self.non_FP])}")
            TRAIN_LOG.debug(f"current non_FNs : {', '.join([c[1].txt for c in self.non_FN])}")
        if skim :
            for _, c in res :
                self.rm_from_tried(c)
        return res


def parse_from_raw_txt(raw_infer) -> Tuple[str, str] :
    """
    raw_infer txt -> answer_output, cot str
    """
    ans_flag = '```'
    if ans_flag in raw_infer :
        cot, answers = raw_infer.split(ans_flag)[0], raw_infer.split(ans_flag)[1:]
        return ';'.join([ans.replace('python','') for ans in answers]), cot.strip()
    else : 
        return '', ''
    
def segment_constr(target : str) : 
    
    """
    txt : a or b -> a, b 
    txt : a and b -> a, b
    txt : a or (b or c) -> a, b or c
    txt : (a) or (b or c and (d and e)) -> a, b or c and (d and e)
    """
    parts = []
    split_hints = [' or ', ' and '] 
    split_mark = "##"
    if not any(hint in target for hint in split_hints) :
        return [] 
    for hint in split_hints :
        target = target.replace(hint, split_mark)
    
    for part in target.split(split_mark) :
        while part.count('(') != part.count(')') :
            if part.count('(') == 0 and part.count(')') == 0 : break
            if part.count('(') > part.count(')') :
                pos = part.index('(')
                part = part[:pos] + part[pos+1:]
                
            if part.count('(') < part.count(')') :
                pos = part.rindex(')')
                part = part[:pos] + part[pos+1:]
        parts.append(part)
    return parts