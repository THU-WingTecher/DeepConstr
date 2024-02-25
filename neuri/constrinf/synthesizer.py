from logger import TRAIN_LOG
from typing import Callable, Any, List, Tuple, Dict, Optional, Union, Literal
from neuri.constrinf.constr import Constraint, conn_rule_txt
from neuri.constrinf.evaluator import Evaluator
from neuri.constrinf.smt_funcs import is_same_constr, is_implies, has_same_rule


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
        self.has_save_point = False
        self.seeds : List[Tuple[Dict[str, float], Constraint]] = []
        self.non_FP : List[Tuple[Dict[str, float], Constraint]] = []  # For storing the best constraints found so far
        self.non_FN : List[Tuple[Dict[str, float], Constraint]] = []  # For storing the best constraints found so far
        self.tried : List[z3.ExprRef] = []
        self.mode : Literal["acc", "f1"] = 'f1'
    
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
            beta = 5
            score = alpha * scores.get("f1_score", 0) + beta * (1 / length)
        scores["overall_score"] = score

    def save_state(self, seeds) :
        self.update_seeds(({}, seed) for seed in seeds)
        self.has_save_point = True
    def evaluate(self, constraint : Constraint, skim : bool = False) -> Dict[str, Any]:
        # Placeholder: Calculate F1, precision, and recall for the given constraint
        scores = {}
        evaluator = Evaluator(self.target, constraint, self.record, self.executor, self.cfg)
        if self.goal_is_acc() :
            evaluator.get_precision(num_of_check=self.cfg['simple_eval_asset'] if skim else self.cfg['eval_asset'])
            scores["precision"] = evaluator.precision
        else :
            evaluator.get_f1(num_of_check=self.cfg['simple_eval_asset'] if skim else self.cfg['eval_asset'])
            scores["precision"] = evaluator.precision
            scores["recall"] = evaluator.recall
            scores["f1_score"] = evaluator.f1_score
        return scores
    
    def synthesize_with_semi_complete(self, constraints : List[Constraint]) -> Constraint:
        # Placeholder: Combine the given 'best' constraint with all other atomic constraints
        # and add them to the queue for further testing
        res = []
        changed = False
        for constr in constraints :
            synthesized = constr 
            if self.non_FP :
                changed = True
                synthesized = self.synthesize_constrs(constr, self.non_FP[0][1], OR)
                assert synthesized is not None
            if self.non_FN :
                changed = True
                synthesized = self.synthesize_constrs(constr, self.non_FN[0][1], AND)
                assert synthesized is not None
            if changed :
                res.append(synthesized)
        return res
    def update_seeds(self, new_seeds : List[Tuple[Dict[str,float], Constraint]]):
        for _, seed in new_seeds :
            self.seeds.append(seed)
        self.seeds = [(score, seed) for score, seed in sorted(self.seeds, \
                                                 key=lambda pair: pair[0]["overall_score"], reverse=True)][:self.cfg['max_num_of_seeds']]
        TRAIN_LOG.debug(f"current seeds : {[(score, c.txt, c.length) for score, c in self.seeds]}")

    def need_synthesis(self, left : Constraint, right : Constraint) -> bool :
        return not is_implies(left.z3expr, right.z3expr) and not is_implies(right.z3expr, left.z3expr)
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
    
    def load_save_point(self) :
        seeds = self.load_high_quality_constrs()
        self.has_save_point = False
        return seeds
    def run(self, new_rules : List[Constraint]) -> Tuple[float, Constraint] :

        queue : List[Constraint]= []
        high_quality = self.load_high_quality_constrs()
        if self.has_save_point : 
            saved_rules = self.load_save_point()
            queue.extend(saved_rules)
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
        return self.load_high_quality_constrs(top_k=1, return_score=True)[0]
    
    def find_optimal_constrs(self,
            queue : List[Constraint], 
            skim : bool = False,
            ) -> List[Tuple[Dict[str, float],Constraint]]:
        
        res = []
        found_complete = False
        TRAIN_LOG.debug(f'to_test(skim={skim}) : {", ".join([c.txt for c in queue])}')
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
            self.non_FN.sort(key=lambda x : x[0]["overall_score"], reverse=True)
            TRAIN_LOG.debug(f"current non_FPs : {', '.join([c[1].txt for c in self.non_FP])}")
            TRAIN_LOG.debug(f"current non_FNs : {', '.join([c[1].txt for c in self.non_FN])}")
        if skim :
            for _, c in res :
                self.rm_from_tried(c)
        return res