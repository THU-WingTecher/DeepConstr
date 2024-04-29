
from deepconstr.grammar.dtype import AbsDType
from deepconstr.grammar.ast2z3 import Ast2z3
from typing import Callable, Dict, Any, List, Literal, Optional, Tuple, Union
from deepconstr.train.errmsg import ErrorMessage
from deepconstr.logger import CONSTR_LOG
from z3 import Solver

_checker = Solver()

class Constraint:
    def __init__(self, txt, cot, target, arg_names, dtypes, length = 1) :
        self.txt = txt
        self.cot = cot
        self.length = length
        self.target : ErrorMessage = target
        self.arg_names : List[str] = arg_names
        self.dtypes : List[Any] = dtypes # chosen dtypes, should be able to activated
        assert not isinstance(self.dtypes[0], list)
        self.unactivated = self.z3(self.arg_names, self.dtypes) 
        if self.unactivated is not None :
            self.z3expr = Constraint.activate(self.unactivated, self.arg_names, self.dtypes)
        else :
            self.z3expr = None

    @staticmethod
    def load(data) :
        errmsg = ErrorMessage.load(data["target"])
        dtype_map = errmsg.get_dtypes_map()
        arg_names = list(dtype_map.keys())
        dtypes = list(dtype_map.values())
        return Constraint(
            data["txt"],
            data["cot"],
            errmsg,
            arg_names,
            dtypes,
            length = data.get("length", 1)
        )
    
    def dump(self) :
        return {
            "txt" : self.txt,
            "cot" : self.cot,
            "target" : self.target.dump(),
            "length" : self.length
        }
    
    def __repr__(self) -> str:
        return f"Constr({self.txt})"
    
    def is_error(self) -> bool :
        return self.z3expr is None 
    
    def get_executable(self) :
        return self.unactivated
    
    def get_z3_constr(self) :
        return self.z3expr
    
    def check(self, arg_names = None, dtypes : List[AbsDType]=None) :
        if arg_names is None :
            arg_names = self.arg_names
        if dtypes is None :
            dtypes = self.dtypes
        converter = Ast2z3(arg_names, [[dtype] for dtype in dtypes], self.txt)
        try :
            result = converter.convert(no_suff=True)
            if result is None or result == True :
                return False 
            _checker.add(result)
            _checker.reset()
            return True 
        except :
            return False
    
    def z3(self, arg_names=None, dtypes : List[AbsDType]=None) :
        if arg_names is None :
            arg_names = self.arg_names
        if dtypes is None :
            dtypes = self.dtypes
        converter = Ast2z3(arg_names, [[dtype] for dtype in dtypes], self.txt)
        unactivated_constr = converter.convert()
        return unactivated_constr
    
    @staticmethod
    def activate(constr, arg_names, dtypes) :
        
        return constr(
            z3objs = {
                name : abs.z3()(name)
                for name, abs in zip(arg_names, dtypes) \
                    if abs is not None
            }
        )
    
    def gen_opposed(self) :
        opposed_txt = opposed_rule_txt(self.txt)
        return Constraint(opposed_txt, self.cot, self.target, self.arg_names, self.dtypes)

    def __hash__(self) -> int:
        return hash(self.txt)

def conn_rule_txt(merged_txts : List[str], method = ' or ') :
    assert len(merged_txts) > 0, "merged_txts should not be empty"
    res = ""
    for txt in merged_txts[:-1] : 
        res += f"({txt}){method}"
    res += f"({merged_txts[-1]})"
    return res

def opposed_rule_txt(txt : str) :
    res = ""
    sign = "not"
    res += f"{sign} ({txt})"
    return res


def convert_constr_to_executable(record, rule_cnt = None) -> List[Callable] : 
    """
    convert to unactivated executable constr
    """
    exec_rules = []
    rules = record.get('rules', [])
    for rule, _ in rules :
        CONSTR_LOG.debug(f"rule : {rule['txt']}")
        constr = Constraint.load(rule)
        rule = constr.z3()
        if rule is None : 
            CONSTR_LOG.warning(f"rule generation Failed : {rule}")
            # raise ValueError(f"rule generation Failed : {rule}")
            continue
        
        exec_rules.append(rule)
    if rule_cnt is not None :
        rule_cnt["cnt"] += len(exec_rules)
        CONSTR_LOG.info(f"{len(exec_rules)} rules are generated")
    return exec_rules

def convert_dtypes_to_z3s(dtypes : List[Any]) -> List["z3.ExprRef"]:
    return [[dtype.z3() for dtype in dtypes] for dtypes in dtypes]