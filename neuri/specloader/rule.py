import ast
import copy
import re
import traceback
from logger import AUTOINF_LOG
from typing import Callable, Any, List, Tuple, Dict, Optional, Union, Set, Literal
from specloader.utils import parse_and_transform, ArgsSearcher
from specloader.ast2z3 import Ast2z3
from abstract.dtype import AbsDType, AbsLiteral, AbsIter, DType
from neuri.abstract.tensor import AbsTensor
from specloader import MAX_ARR_LEN

class Rule() :
    def __init__(self, 
                 target : str,
                 rule_txt : str, 
                 cot : str,
                 args_type_dict : Dict[str, Any]
                 ) :
        self.txt = rule_txt
        self.target = target
        self.cot = cot
        self.new_gen = False
        self.ast = None
        self.related_args = None
        self.args_type_dict = args_type_dict
        self.activated = False
        self.length = 1 # num of atomic rule combined
        self.ast2z3 : Ast2z3 = None
        self.parse_rule(self.txt)
        if self.related_args is None or len(self.related_args) == 0 :
            self.ast = None
        else :
            for arg_name in self.related_args :
                if arg_name not in self.args_type_dict.keys() :
                    self.ast=None 
                    break
            
        if self.ast != None :
            self.build_converter()
    def build_converter(self) : 
        try :
            self.ast2z3 = Ast2z3(
                        self.related_args,
                        self.txt,
                        self.ast,
                        self.args_type_dict
                    )
        except :
            AUTOINF_LOG.error(f"Rule({self.txt}) : Ast2z3 build failed.")
            AUTOINF_LOG.debug(f"{self.info()}\n{traceback.format_exc()}")
            self.set_err_flag()
        if self.ast2z3.error == False :
            self.check() 
        else : 
            pass 
    def check(self) :
        if not self.ast2z3.is_unsolverable() :
            pass
        else :
            if self.type_rule_check() == False : self.ast2z3.error = True  
    def type_rule_check(self) :
        for dtypes in self.ast2z3.types_map.values() :
            for dtype in dtypes : 
                if not any(isinstance(dtype[1], allowed_dtype) for allowed_dtype in [AbsDType, AbsLiteral, AbsIter, DType, AbsTensor]) :
                    return False
        if not all(arg in self.ast2z3.types_map.keys() for arg in self.related_args) :
            return False
        return True

    def info(self) : 
        rule_txt = f'Rule : {self.txt}'
        rule_ast = f'AST : {ast.dump(self.ast, indent=4)}'
        rule_args = f'Args : {self.related_args}'
        if len(self.ast2z3.types_map) != 0 :
            is_type_map = f'Type Map : {self.ast2z3.types_map}'
        else :
            is_type_map = f'Type Map : None'
        return '\n'.join([rule_txt, rule_ast, rule_args, is_type_map])
    def rewrap_txt(self, merged_txts : List[str], method = ' or ') :
        res = ""
        for txt in merged_txts : 
            res += f"({txt}){method}"
        res += f"({self.txt})"
        return res
    def re_init(self, new_ast) : 
        self.ast = new_ast
        self.build_converter()
    def serialize(self) : 
        return {
            'txt' : self.txt, 
            'target' : self.target, 
            'cot' : self.cot
        }

    def gen_z3(self) : 
        dtypes = {}
        args_types = {key : dtype[0] for key, dtype in self.args_type_dict.items()}
        self.ast2z3.set_args_types(args_types)
        equation = self.ast2z3.gen_constraints()
        return equation
    
    def export_opposed(self) : 
        from specloader.utils import set_location
        if isinstance(self.ast.body, ast.Tuple) : 
            new_body = ast.BoolOp(op=ast.And(), values=self.ast.body.elts)
        else :
            new_body = self.ast.body
        unary_op_node = ast.UnaryOp(op=ast.Not(), operand=new_body)
        expr_node = ast.Expression(body=unary_op_node)
        set_location(expr_node, 1, 0)
        return expr_node

    def export_merged(self, to_merge : List[ast.Expr], op : Literal['or','and'] = 'or') : 
        from specloader.utils import set_location
        values = [self.ast.body]+[rule.ast.body for rule in to_merge]
        if op=='or' : 
            op = ast.Or() 
        else : 
            op = ast.And()
        merged_ast = ast.Expression(
                    body=ast.BoolOp(
                        op=op,
                        values=values
                    )
                )
    
        # Fix missing locaions
        set_location(merged_ast, 1, 0)
        return merged_ast
    
    def init_solver(self) -> "z3.Solver" : 
        from specloader import DEFAULT_PREMISES
        from z3 import Solver
        solver = Solver()
        for premise in DEFAULT_PREMISES :
            solver.add(premise)
        return solver

    def type_rule_act(self, args_type_dict : Dict[str, Any]) -> Dict[str, Any]: 
        """
        str -> return converted types_def dict.
        ex) type('padding') == str, return def that has changed the value of 'padding'
        """
        pass 
        # from neuri.abstract.tensor import AbsTensor
        # from abstract.utils import is_same_abs
        # from specloader import NOT_SIGN
        # LOGGER.debug(f"type_rule({self.txt}) : From {args_type_dict}")

        # def get_tensor_arg(dtype_list) : 
        #     if any(isinstance(dtype, AbsTensor) for dtype in dtype_list) :
        #         idx = 0  
        #         while  idx < len(dtype_list) and not isinstance(dtype_list[idx], AbsTensor) :
        #             idx+=1
        #         if idx < len(dtype_list) :
        #             return (arg_name, idx)

        # to_rm = []
        # to_include = []
        # if len(self.ast2z3.types_map) == 0 : 
        #     return args_type_dict 
        # for arg_name in self.related_args :
        #     for dtype_def in self.ast2z3.types_map[arg_name] :
        #         given_types = dtype_def[1]
        #         if type(given_types) == str and given_types in args_type_dict.keys() :
        #             given_types = args_type_dict[given_types]
        #         if dtype_def[0] == NOT_SIGN :
        #             for arg_dtype in args_type_dict[arg_name] :
        #                 if is_same_abs(given_types, arg_dtype) : # if it is same dtype, return True
        #                     # check whether given_dtype is in arg_dtypes, at this, 
        #                     # DType -> look in to possible dtypes of abstensor
        #                     to_rm.append(given_types)
                    
        #         else :
        #             to_include.append(given_types)
            
        #     if len(to_include) != 0 :
        #         new_types = []
        #         arg_nm_idx = get_tensor_arg(args_type_dict[arg_name])

        #         for dtype in to_include :   
        #             if isinstance(dtype, AbsLiteral) : 
        #                 arg_nm_idx = None 
        #                 new_types.append(dtype)
        #             elif isinstance(dtype, AbsTensor) and arg_nm_idx is not None : 
        #                 new_types.extend(dtype.possible_dtypes)
                        
        #             new_types.append(dtype)           

        #         args_type_dict[arg_name] = new_types

        #     if len(to_rm) != 0 :
        #         for dtype in to_rm :
        #             for orig_dtype in args_type_dict[arg_name] :
        #                 if is_same_abs(dtype, orig_dtype) :
        #                     args_type_dict[arg_name].remove(orig_dtype)

        # LOGGER.debug(f"To {args_type_dict}")
        # return args_type_dict
    
    # def convert_tensor_to_iter(self, args_type_dict : Dict[str, Any]) -> Dict[str, Any] :
    #     for key, dtype in args_type_dict.items() :
    #         if isinstance(dtype, AbsTensor) :
    #             if all(dtype in DTYPE_INTS for dtype in dtype.possible_dtypes) : 
    #                 args_type_dict[key] = AbsDType.int.iter()

    def revise_rule(self, line : str) -> str :
        line = re.sub(r'^[-\d]+', '', line)
        line = line.replace('.', '')
        return line.strip()
    
    
    def _parse_rule(self, 
                   rule_txt : str) -> None :

        try :
            astree = parse_and_transform(rule_txt)
            if isinstance(astree, ast.Constant) : # Eval failed, it takes rule as a normal string.
                ## AST Revise Logic
                revised_rule = astree
                if revised_rule == rule_txt :
                    AUTOINF_LOG.debug(f'Eval ##FAILED## : {rule_txt}') 
                    return None
                else :
                    return self._parse_rule(revised_rule)
            else :
                self.ast, self.related_args = self.identify_related_args(astree)
                # LOGGER.debug(f'Eval ##SUCCEED## related args {self.related_args} : {rule_txt}\n{ast.dump(self.ast, indent=4)}')

        except :
            revised = self.revise_rule(rule_txt)
            if revised == rule_txt :
                # LOGGER.debug(f'Eval ##FAILED## : {rule_txt}') 
                return None
            # LOGGER.debug(f'{rule_txt} -> retry with {revised}')
            return self._parse_rule(revised)
        
    def parse_rule(self, 
                   rule_txt : str) -> None :
        astree = self._parse_rule(rule_txt.replace(',', ';'))
        if astree is None :
            astree = self._parse_rule(rule_txt)
        return astree 
    
    def identify_related_args(self,astree) : 
        searcher = ArgsSearcher(self.args_type_dict.keys())
        tree = searcher.visit(astree)
        return tree, searcher.get()
    
    def test(self) : 
        if self.ast2z3.is_unsolverable() :
            self.type_rule_act(copy.deepcopy(self.args_type_dict))
            return True
        else :
            return self.gen_z3()

            # if equation generate failed, error flag has been set automately.
    

def gen_rules(
              target : str,
              cot : str,
              rules : Union[List[str], str], 
              args_type_dict : Dict[str, Any]) -> List[Rule] :
    gen = []
    if type(rules) == list :
        for rule in rules :
            ruleobj = gen_rule(target, cot, rule.strip(), args_type_dict)
            if ruleobj is not None :
                gen.append(ruleobj)
    elif type(rules) == str :
        res = []
        for rule in rules.split('\\n') :
            for r in rule.split('\n') : 
                for rr in r.split(';') :
                    if len(rr)!=0 :
                        res.append(rr)
        for rule in res :
            ruleobj = gen_rule(target, cot, rule.strip(), args_type_dict)
            if ruleobj is not None :
                gen.append(ruleobj)
    else :
        pass
    return gen

def gen_rule(target : str,
             cot : str,
             rule_txt : str, 
             args_type_dict : Dict[str, Any]) -> Union[Rule,None] :
        
    if rule_txt is None or rule_txt == '' :
        return None 
    else :
        ruleobj = Rule(target, rule_txt, cot, args_type_dict)
        if ruleobj.ast is not None and ruleobj.ast2z3.error == False :
            # ruleobj.test()
            return ruleobj
        else :
            return None


def tensor_default_rule(arg_name) : 
    return f"all([{arg_name}.shape[i]>=0 and {arg_name}.shape[i]<={MAX_ARR_LEN} for i in range(len({arg_name}.shape))])"
def unsupport_dtypes_rule(arg_name, dtypes : List[str]) : 
    if dtypes :
        return f"{arg_name}.dtype not in {dtypes}"
    else : 
        return ""
