import ast
import random
import string
import traceback
import operator as op 
import z3
from typing import *
from neuri.abstract.dtype import AbsDType, DType
from neuri.constrinf import gen_inst_with_records
from neuri.specloader import z3_funcs

###### TENSOR DTYPE DEINITION ######

## Define : tensor, int, float, bool, complex, str
## Define : Array(tensor, int, float, bool, complex, str)
## Will be updated to support Dict, object

iter_specific_funcs = ['len', 'rank', 'sorted', 'min', 'max']
Z3DTYPE = z3.Datatype('DType')
for dtype in DType : 
    Z3DTYPE.declare(dtype.name)
Z3DTYPE = Z3DTYPE.create()

Complex = z3.Datatype('complex')
Complex.declare('complex_instance', ('real', z3.RealSort()), ('imag', z3.RealSort()))
Complex = Complex.create()

TensorZ3 = z3.Datatype('TensorZ3')
TensorZ3.declare('tensor_instance', 
                ('shape', z3.ArraySort(z3.IntSort(), z3.IntSort())),
                ('dtype', Z3DTYPE),
                ('rank', z3.IntSort())),
TensorZ3 = TensorZ3.create()

def DeclareArr(sort):
    Arr = z3.Datatype('Arr_of_%s' % sort.name())
    Arr.declare('arr_instance', 
                ('value', z3.ArraySort(z3.IntSort(), sort)),
                ('len', z3.IntSort())),
    Arr = Arr.create()
    return Arr

IntArr = DeclareArr(z3.IntSort())
FloatArr = DeclareArr(z3.RealSort())
StrArr = DeclareArr(z3.StringSort())
BoolArr = DeclareArr(z3.BoolSort())
ComplexArr = DeclareArr(Complex)
TensorArr = DeclareArr(TensorZ3)

class BaseWrapper():
    def __init__(self, const, datatype):
        self.const : z3.Datatype = const
        self.datatype : z3.Datatype = datatype
        self.info = {

        }
    def len(self): 
        if self.info.get("sliced") is not None : 
            return self.info["sliced"][1] - self.info["sliced"][0]
        else :
            return self.datatype.len(self.const)
    
    @property
    def name(self):
        return self.const.decl().name()
    
    @property
    def rank(self):
        return self.datatype.len(self.const)
    
    def update_info(self, *args, **kwargs):
        for key, value in kwargs.items() : 
            self.info[key] = value
    
    def get_wrapped_object(self):
        return self.const
    
    def __getitem__(self, idx):
        return self.datatype.value(self.const)[idx]

class TensorWrapper(BaseWrapper):
    def __init__(self, const, datatype):
        super().__init__(const, datatype)
    
    def __getitem__(self, idx):
        return self.datatype.shape(self.const)[idx]
    
    @property
    def rank(self):
        return self.datatype.rank(self.const)
    
    def __str__(self):
        return f"Tensor:{self.const.name()}"

    @property
    def shape(self):
        return self.datatype.shape(self.const)
    
    @property
    def dtype(self):
        return self.datatype.dtype(self.const)
    
class ComplexWrapper(BaseWrapper):
    def __init__(self, const, datatype):
        super().__init__(const, datatype)

    def __str__(self):
        return f"Complex:{self.const.name()}"
    
    @property
    def rank(self):
        raise NotImplementedError
    
    def __lt__(self, __value: object) -> bool:
        pass
    
class ArrWrapper(BaseWrapper):
    def __init__(self, const, datatype):
        super().__init__(const, datatype)
    
    def __str__(self):
        return f"Arr:{self.const.name()}"
    
    @property
    def value(self):
        return self.datatype.value(self.const)
    
    def range(self):
        assert "sliced" in self.info.keys(), "sliced info is not defined"
        return self.info["sliced"]
def load_z3_const(name, var_type, is_array=False):
    """
    Define a Z3 variable of a given type. 
    If it's an array, specify the element type.
    """
    if var_type == 'int':
        if is_array : 
            return ArrWrapper(z3.Const(name, IntArr), IntArr)
        else : 
            return z3.Const(name, z3.IntSort())
    elif var_type == 'float':
        if is_array : 
            return ArrWrapper(z3.Const(name, FloatArr), FloatArr)
        else :
            return z3.Const(name, z3.RealSort())
    elif var_type == 'str':
        if is_array : 
            return ArrWrapper(z3.Const(name, StrArr), StrArr)
        else :
            return z3.Const(name, z3.StringSort())
    elif var_type == 'complex':
        if is_array : 
            return ArrWrapper(z3.Const(name, ComplexArr), ComplexArr)
        else :
            return ComplexWrapper(z3.Const(name, Complex), Complex)
    elif var_type == 'bool':
        if is_array : 
            return ArrWrapper(z3.Const(name, BoolArr), BoolArr)
        else :
            return z3.Const(name, z3.BoolSort())
    elif var_type == 'tensor':
        if is_array : 
            return ArrWrapper(z3.Const(name, TensorArr), TensorArr)
        else :
            return TensorWrapper(z3.Const(name, TensorZ3), TensorZ3)
    else:
        raise ValueError("Unsupported variable type")


class z3funcs:
    """
    Class to hold custom Z3 functions.
    """
    # Class variable to hold names of all functions
    function_names = ['all', 'any', 'len', 'type', 'sorted', 'abs', 'min', 'max', 'in_', 'not_in', 'rank', 'range']
    _z3_dataarr = [IntArr, FloatArr, StrArr, ComplexArr, BoolArr, TensorArr, TensorZ3]
    iterables = []
    non_iterables= []
    def __init__(self) -> None:
        self.constrs = []

    @staticmethod
    def sorted(vs, length = None) : 
        pass 

    @staticmethod
    def abs(vs, length = None) : 
        pass 

    @staticmethod
    def in_(a,b) : 
        pass 

    @staticmethod
    def in_(a,b) : 
        pass 

    @staticmethod
    def not_in(a,b) : 
        pass 

    @classmethod 
    def is_iterable(cls, v) :
        return cls._find_datatype_by_id(v.sort().get_id()) is not None 
    @staticmethod
    def type(v) : 
        return v.sort().name()
    
    @classmethod
    def clear(cls) :
        cls.constrs.clear()
    
    @classmethod
    def _find_datatype_by_id(cls, id):
        """
        Helper function to get a Z3 datatype by its id.
        """
        return next(((dv, dv.name()) for dv in cls._z3_dataarr if dv.get_id() == id), None)
    @staticmethod
    def all(input_array):
        """
        Custom 'all' function for Z3 expressions.
        Returns z3.And of all elements in the input array.
        """
        assert all(isinstance(expr, z3.ExprRef) for expr in input_array), "All elements must be Z3 expressions"
        return z3.And(input_array)

    @staticmethod
    def any(input_array):
        """
        Custom 'any' function for Z3 expressions.
        Returns z3.Or of any element in the input array.
        """
        assert all(isinstance(expr, z3.ExprRef) for expr in input_array), "All elements must be Z3 expressions"
        return z3.Or(input_array)
    
    @staticmethod
    def range(start, end=None, step=None):
        """
        """
        lower_bound = 0
        upper_bound = None 
        if step is None : step = 1
        if end is None : upper_bound = start
        else :
            lower_bound = start
            upper_bound = end
        
        if step is not None and step < 0 :
            lower_bound, upper_bound = upper_bound, lower_bound
            step = -step

        return (lower_bound, upper_bound, step)
    
    @classmethod
    def _load_attrs(cls, v):
        """
        """
        datatype = None
        datatype = cls._find_datatype_by_id(v.sort().get_id())
        if datatype is None :
            raise TypeError("Undefined in z3func object")
        else : 
            obj, name = datatype
            if name == TensorZ3.name() : 
                return obj.rank(v), obj.shape(v), obj.dtype(v)
            else : 
                return obj.len(v), obj.value(v)

    @classmethod
    def len(cls, v):
        """
        Custom 'len' function for Z3 arrays and tensors.
        Returns the length of an array or the rank of a tensor.
        """
        if hasattr(v, 'len') :
            return v.len()
        else :
            attrs = cls._load_attrs(v)
            return attrs[0]
    
    def min(self, vs, lev_var=None):
        if not hasattr(vs, '__len__') :
            assert self.is_iterable(vs), f"{vs.sort().name()} is not iterable variable"
            if lev_var is None :
                attrs = self._load_attrs(vs)
                lev_var = attrs[0]
                values = attrs[1]
            
            min_idx = z3.Int('min_idx')
            assert values[min_idx].sort().name() not in ['complex', 'tensor'], f"{values[min_idx].sort().name()} don't support min func"
            i = z3.Int('i')
            min_const = z3.ForAll(
                [i], 
                z3.Implies(z3.And(i >= 0, i < lev_var), 
                        values[min_idx] <= values[i]))
            exists_in_array = z3.And(min_idx >= 0, min_idx < lev_var)
            self.constrs.append(z3.And(exists_in_array, min_const))
            return values[min_idx] 
        else :
            assert len(vs) > 0, f"len of list should be positive, cur : {len(vs)}"
            if len(vs) == 1 :
                return vs[0]
            m = vs[0]
            for v in vs[1:]:
                m = z3.If(v < m, v, m)
            return m

# z3_constrs = []
# ia = load_z3_const('a', 'int', is_array=True)
# ib = load_z3_const('b', 'complex', is_array=True)
# ic = load_z3_const('c', 'tensor', is_array=False)
# z3funcs_const = z3funcs()
# z3_constrs.extend([
#     z3funcs_const.len(ib) <3,
#     z3funcs_const.min(ia) <3,
#     # z3funcs.len(ic) <3
#                   ])
# z3_constrs = [Z3ARRINT.value(arr_int_var)[0] > 1, Z3ARRINT.len(arr_int_var) <3]
# print(z3funcs.all(z3_constrs))
# arr_int_var = load_z3_const('a', 'float', is_array=True)
# arr_int_var = load_z3_const('a', 'str', is_array=True)
# arr_int_var = load_z3_const('a', 'complex', is_array=True)
# arr_int_var = load_z3_const('a', 'tensor', is_array=True)
# print(arr_int_var)

def is_same_ast_name(left_ast : str, right_ast) : 
    return left_ast == right_ast.__name__

def get_operator(astop):
    if is_same_ast_name(astop, ast.Eq):
        return lambda a,b : op.eq(a, b)
    elif is_same_ast_name(astop, ast.NotEq):
        return lambda a,b : op.ne(a, b)
    elif is_same_ast_name(astop, ast.Lt):
        return lambda a,b : op.lt(a, b)
    elif is_same_ast_name(astop, ast.LtE):
        return lambda a,b : op.le(a, b)
    elif is_same_ast_name(astop, ast.Gt):
        return lambda a,b : op.gt(a, b)
    elif is_same_ast_name(astop, ast.USub):
        return lambda a: -a
    elif is_same_ast_name(astop, ast.GtE):
        return lambda a,b : op.ge(a, b)
    elif is_same_ast_name(astop, ast.Is):
        return lambda a,b : op.eq(a, b)
    elif is_same_ast_name(astop, ast.IsNot):
        return lambda a,b : op.ne(a, b)
    elif is_same_ast_name(astop, ast.In):
        return lambda a,b : z3_funcs.in_(a,b)
    elif is_same_ast_name(astop, ast.NotIn):
        return lambda a,b : z3_funcs.not_in(a,b)
    elif is_same_ast_name(astop, ast.Add):
        return lambda a,b : op.add(a, b)
    elif is_same_ast_name(astop, ast.Sub):
        return lambda a,b : op.sub(a, b)
    elif is_same_ast_name(astop, ast.Mult):
        return lambda a,b : op.mul(a, b)
    elif is_same_ast_name(astop, ast.MatMult):
        return lambda a,b : op.matmul(a, b)
    elif is_same_ast_name(astop, ast.Div):
        return lambda a,b : op.truediv(a, b)
    elif is_same_ast_name(astop, ast.FloorDiv):
        return lambda a,b : op.truediv(a, b)
    elif is_same_ast_name(astop, ast.Mod):
        return lambda a,b : op.mod(a, b)
    elif is_same_ast_name(astop, ast.Pow):
        return lambda a,b : op.pow(a, b)
    elif is_same_ast_name(astop, ast.Or):
        return z3.Or
    elif is_same_ast_name(astop, ast.Not):
        return z3.Not
    elif is_same_ast_name(astop, ast.And):
        return z3.And
    else:
        raise ValueError(f"Unknown operator: {ast.dump(astop)}")

def gen_z3_obj(arg, arg_map, idx=None, ret_wrapper=False, no_const=False) : 
    if ast2z3.is_sym(arg) : 
        z3obj = arg
    elif arg in arg_map.keys() : 
        z3obj = arg_map[arg]
    else :
        if no_const : return z3.Int(arg) 
        return arg # constant
    if idx is not None : 
        if type(idx) == int :
            return z3obj[z3obj.rank + idx] if idx<0 else z3obj[idx]
        else : # when index is 'i' for generator
            if idx in arg_map.keys() : 
                return z3obj[arg_map[idx]]
            elif type(idx) == str :
                return z3obj[z3.Int(idx)]
            else : 
                return z3obj[idx]
    else : 
        if hasattr(z3obj, 'get_wrapped_object') and not ret_wrapper :
            return z3obj.get_wrapped_object()
        else :
            return z3obj


def random_gen_name() : 
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(5))

def change_val_from_expr(expr, target, new_target):
    return z3.substitute(expr, (target, new_target))

    # res = gen_z3_obj(obj)[{start}:{end}]

def gen_exp_constr(generator, arg_map):
    # Assuming generator is a dictionary with keys "element" and "generators"
    constraints = []
    for comp in generator["generators"]:
        # Assuming comp is a dictionary with keys "target", "iter", "ifs"
        target = gen_z3_obj(comp["target"], arg_map, no_const=True)
        iter = gen_z3_obj(comp["iter"], arg_map)
        ifs = [gen_z3_obj(if_clause, arg_map) for if_clause in comp["ifs"]]
        # Check if iteration is over a list/array or a range
        if isinstance(iter, ArrWrapper) :
            assert hasattr(iter, 'range'), f"iter should have range attribute, cur : {iter}"
            range = iter.range()
            lower_bound = range[0]
            upper_bound = range[1]
            step = 1
            if ifs :
                ifs = change_val_from_expr(ifs, target, iter[target])
            generator["element"] = change_val_from_expr(generator["element"], target, iter[target])
            # target = iter[target]
        else:
            # Case: Iteration over a range
            lower_bound, upper_bound, step = iter
        assert abs(step) < 2, f"step should be less than 2, cur : {step}"
        ### list comprehension
        is_in = z3.And(target >= lower_bound, target < upper_bound)
        # Use z3.And to combine all conditions in 'ifs'
        combined_ifs = z3.And(*ifs) if ifs else z3.BoolVal(True)

        # Element of the generator expression
        element = gen_z3_obj(generator["element"], arg_map)

        # Construct the Z3 constraint using z3.Imply
        constraint = z3.ForAll([target], z3.Implies(z3.And(is_in, combined_ifs), element))
        constraints.append(constraint)

    # Combine all constraints
    return constraints

def merge_constr(constrs) : 
    return z3.And(constrs)

class ast2z3(z3funcs) : 
    def __init__(self, arg_names, dtypes, txt, func_name) -> None : 
        super().__init__()
        self.func_name = func_name
        self.txt = txt
        self.arg_map = {
            name : dtype for name, dtype in zip(arg_names, dtypes)
        }
        self.constr_flags : Dict[str, Dict[str, bool]] = {
            name : {
                'must_iter' : False,
                'must_int' : False,
                'must_not_iter' : False,
                'must_str' : False
            } for name in arg_names
        }
    def parse(self, txt) : 
        try :
            ast_tree = ast.parse(txt, mode='eval')
            return ast_tree.body
        except :
            raise ValueError(f"Unsupported AST node {ast.dump(txt)})")
    def set_flag(self, name, **kwargs) :
        #must_iter=None, must_int=None, must_not_iter=None, must_str= 
        pass
    def convert(self) : 
        ast = self.parse(self.txt)
        return self._convert(ast, self.arg_map)
    
    def gen_func_obj(self, func_name, *args) :
        func_exec = getattr(self, func_name)
        res = func_exec(*args)
        return res

    def set_ele_type_flag(self, name, comparator) : 
        if type(comparator) == bool : self.set_flag(name, must_bool=True)
        if type(comparator) == int : self.set_flag(name, must_num=True)
        if type(comparator) == str : self.set_flag(name, must_str=True)   

    def set_flag_by_func_name(self, func_name, args) :
        if func_name in iter_specific_funcs :
            for arg in args : 
                if self.is_sym(arg) :
                    self.set_flag(self.get_name(arg), must_iter=True)
        
    @staticmethod
    def is_sym(a):
        # Check if 'a' is a Z3 expression or sort
        is_z3_obj = isinstance(a, (z3.ExprRef, z3.SortRef))
        is_wrapper = hasattr(a, 'get_wrapped_object')
        return is_z3_obj or is_wrapper

    @staticmethod
    def get_name(a) : 
        if hasattr(a, 'get_wrapped_object') :
            return a.name
        if a.num_args() == 0 : 
            return a.decl().name()
        else :
            return ast2z3.get_name(a.arg(0))
        
    def gen_basic_constr(self, op, *args) : 
        if len(args) > 1 : 
            name = ast2z3.get_name(args[0]) if self.is_sym(args[0]) else args[0]
            self.set_ele_type_flag(name, args[1])
            if all(
                not self.is_sym(arg) for arg in args
            ) : 
                args = list(args)
                args[0] = gen_z3_obj(args[0], self.arg_map, no_const=True)  
        # set flag( len(args)> 2, not in, and not_in, make operation comparable )
    
        res = get_operator(op)(*args)
        return res
    
    def gen_sliced_obj(self, obj, arg_map, start = None, end = None) :
        z3obj = gen_z3_obj(obj, arg_map, ret_wrapper=True)
        if start is None : start = 0 
        if end is None : end = z3obj.rank
        idx_range = (start, end)
        z3obj.update_info(sliced=idx_range)
        return z3obj
    def _convert(self, node, arg_map):
        if isinstance(node, ast.BoolOp):
            op = type(node.op).__name__
            values = [self._convert(value, arg_map) for value in node.values]
            return self.gen_basic_constr(op, *values)
        elif isinstance(node, ast.UnaryOp):
            op = type(node.op).__name__
            operand = self._convert(node.operand, arg_map)
            return self.gen_basic_constr(op, operand)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                attribute_value = self._convert(node.func.value, arg_map)
                return self.gen_func_obj(node.func.attr, attribute_value)
            else:
                func_name = node.func.id
                args = [self._convert(arg, arg_map) for arg in node.args]
                assert func_name in z3funcs.function_names, f"Unsupported function {func_name}"
                self.set_flag_by_func_name(func_name, args)
                return self.gen_func_obj(func_name, *args)
            
        elif isinstance(node, ast.Attribute):
            value = self._convert(node.value, arg_map)
            return self.gen_func_obj(node.attr, value)
        elif isinstance(node, ast.BinOp):
            op_type = type(node.op).__name__
            left = self._convert(node.left, arg_map)
            right =self._convert(node.right, arg_map)
            return self.gen_basic_constr(op_type, left, right)
        elif isinstance(node, ast.Compare):
            results = []
            left = self._convert(node.left, arg_map)
            for op, right_node in zip(node.ops, node.comparators):
                right = self._convert(right_node, arg_map)
                op_type = type(op).__name__
                results.append(self.gen_basic_constr(op_type, left, right))
            return merge_constr(results)
        elif isinstance(node, ast.Subscript):
            # Handle negative indices and slicing
            if isinstance(node.slice, ast.Index):
                return gen_z3_obj(self._convert(node.value, arg_map),
                                    arg_map,
                                    self._convert(node.slice.value, arg_map))
            elif isinstance(node.slice, ast.Slice):
                # Slicing, e.g., a[1:] or a[:-1]
                start = self._convert(node.slice.lower, arg_map) if node.slice.lower else None
                end = self._convert(node.slice.upper, arg_map) if node.slice.upper else None
                array = self._convert(node.value, arg_map)
                if start : 
                    if start < 0 :
                        start = self.gen_func_obj('len', array) - start
                if end : 
                    if end < 0 :
                        end = self.gen_func_obj('len', array) - end
                
                ### typically sliced obj is used in list comprehension 
                return self.gen_sliced_obj(array, arg_map, start, end)

        elif isinstance(node, (ast.GeneratorExp, ast.ListComp)):
            # Handle generator expressions
            elt = self._convert(node.elt, arg_map)
            generators = [self._convert(gen, arg_map) for gen in node.generators]
            generator_exp = {"type": "GeneratorExp", "element": elt, "generators": generators}
            return gen_exp_constr(generator_exp, arg_map)
        elif isinstance(node, ast.comprehension):
            # Handle comprehension part of generator expression
            target = self._convert(node.target, arg_map)
            iter = self._convert(node.iter, arg_map)
            ifs = [self._convert(if_clause, arg_map) for if_clause in node.ifs]
            comprehension = {"type": "comprehension", "target": target, "iter": iter, "ifs": ifs}
            return comprehension
        elif isinstance(node, ast.IfExp):
            # Handle IfExp (Ternary Conditional Expression)
            test = self._convert(node.test, arg_map)
            body = self._convert(node.body, arg_map)
            orelse = self._convert(node.orelse, arg_map)
            return f"({body} if {test} else {orelse})"
        elif isinstance(node, (ast.List, ast.Tuple)):
            return merge_constr([self._convert(elem, arg_map) for elem in node.elts])
        elif isinstance(node, ast.Name):
            return gen_z3_obj(
                node.id,
                arg_map,
                ret_wrapper = True
                )
        elif isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Num):
            return node.n
        else:
            raise ValueError(f"Unsupported AST node {ast.dump(node)})")
    # Process and print each constraint


# dir_path = "/artifact/data/constraints"
# records = gen_inst_with_records(dir_path)
# for record in records : 
#     arg_names = record['args']['name']
#     rule_txts = [r['txt'] for r in record['rules']]
#     ast_to_mimic_z3(arg_names, rule_txts, record['name'])
# Test cases

arg_names = ['a', 'b']
dtypes = [load_z3_const('a','int', is_array=True), load_z3_const('b','int', is_array=True)]
test_constraints = [
    "all((a[i]>1 and a[i]<4) for i in range(len(a[2:])))",
    "((i>1 and i<4) for i in a[2:])",
    'a[-1] > b[-2]', 'a[:-1] == b[1:]'
]
all_results = []
func_name = 'test'
for constr in test_constraints:
    result = "error"
    try :
        result = ast2z3(arg_names, dtypes, constr, func_name).convert()
        # print(result)

    except :
        print(traceback.format_exc())
        print(constr, func_name)
    print(f"Constraint: {constr}\nMimic Z3: {result}\n")
# ast_to_mimic_z3(arg_names, test_constraints, func_name="test")
    



import ast
from typing import List, Union, Dict, Any, Callable, Tuple, Optional, get_args
import traceback
import operator as op
from specloader.materalize import materalize_dtypes, materalize_dtype
from logger import AUTOINF_LOG
from neuri.abstract.tensor import AbsTensor
from abstract.dtype import AbsDType, AbsIter, AbsLiteral, DType
import z3
from specloader import RANK_ATTR, TYPE_FUNCS, \
        NOT_SIGN, POS_OPS, NEG_OPS, LEN_ATTR, \
        TYPE_ATTRS, TYPE_ATTR, z3_funcs, gen_len_obj
from neuri.abstract.op import __MAX_RANK__
from specloader.irs import Select, IRcompare, IRexpr, symbolize_idx
from specloader.smt import gen_sufficient_condition
class Ast2z3 : 
    
    def __init__(self, arg_names, txt, _ast, args_type_dict) -> None:
        
        self.arg_names = arg_names
        self.txt = txt
        self.is_sliced= {arg_name : [] for arg_name in arg_names}
        self.must_not_None = {arg_name : False for arg_name in arg_names}
        self.min_len = {arg_name : 0 for arg_name in arg_names}
        self.must_be_seq = {arg_name : False for arg_name in arg_names}
        self.is_subscriptable = {arg_name : False for arg_name in arg_names}
        self.args_types : Dict[str, Union[AbsDType,AbsIter,AbsLiteral,AbsTensor]] = {arg_name : None for arg_name in arg_names}
        self.args_length = {arg_name : None for arg_name in arg_names}
        self.is_rank_rules = {arg_name : False for arg_name in arg_names}
        self.is_suff_cond_need = {arg_name : True for arg_name in arg_names}
        self.args_type_dict = args_type_dict
        self.unsolverable = False
        self.ast = _ast
        self.error = False
        self.type_constraints = None
        self.types_map : Dict[str, List[str, Union[AbsDType, AbsLiteral, AbsTensor]]] = dict()
        self.irs : List[IRcompare] = []
        self.constraints = None
        self.inspecting = False

        self.inspect_ast_tree()     
        self.set_err_flag()

    def set_rank_rule_flag(self, arg_name) : 
        self.is_rank_rules[arg_name] = True
    def is_rank_rule(self, arg_name) :
        return self.is_rank_rules[arg_name] 
    def info(self) -> str :
        info = ""
        info += f"txt : {self.txt}\n"
        info += f"arg_names : {self.arg_names}\n"
        info += f"is_sliced : {self.is_sliced}\n"
        info += f"min_len : {self.min_len}\n"
        info += f"is_subscriptable : {self.is_subscriptable}\n"
        info += f"args_types : {self.args_types}\n"
        info += f"args_length : {self.args_length}\n"
        info += f"unsolverable : {self.unsolverable}\n"
        info += f"args_type_dict : {self.args_type_dict}\n"
        info += f"ast : {ast.dump(self.ast)}\n"
        info += f"error : {self.error}\n"
        info += f"types_map : {self.types_map}\n"
        return info
    def set_args_types(self, args_types : Dict[str, Any]) -> None :
        for arg_name in self.arg_names :
            self.args_types[arg_name] = args_types[arg_name]
    def set_args_length(self, args_length : Dict[str, Any]) -> None :
        for arg_name in self.arg_names :
            if args_length[arg_name] is not None : 
                self.args_length[arg_name] = args_length[arg_name]
    def get_mode(self) : 
        if self.is_converting() :
            return '[convert-mode]'
        else :
            return '[inspect-mode]'
    def set_model_unsolverable_flag(self, switch : bool = True) : 
        AUTOINF_LOG.debug(f"{self.get_mode()} would not solved by z3. types_map : {self.types_map}")
        self.unsolverable = switch
    def set_types_map(self, left : str, 
                      right : Any = None , 
                      sign : Optional[str] = None) : 
        if left not in self.types_map.keys() : 
            self.types_map[left] = []
        if right is not None :
            self.types_map[left].append((sign, right))
    def inspect_ast_tree(self) -> None :
        self.args_types = {arg_name : None for arg_name in self.arg_names}
        self.inspecting = True  
        self.convert(self.ast)
        self.inspecting = False
        if self.is_in_types_map('undefined') and len(self.arg_names) == 1 :
            item = self.types_map['undefined']
            self.set_types_map(self.arg_names[0], item[1], item[0])
            del self.types_map['undefined']
    def _is_suff_cond_need(self) -> None :
        """ 
        we are building suff_cond by the index of array.
        however, for generatorexpr which are a[0]>1 ... a[MAX_LEN]>1, this way of building would be wrong.
        Therefore, for generatorexpr, we disable suff_cond building.
        """
    def gen_constraints(self) -> None : 
        try :
            constraints = []
            if self.is_unsolverable() : 
                return True
            ir = self.convert(self.ast)
            if isinstance(ir, bool) : 
                self.set_model_unsolverable_flag()
                return ir
            if hasattr(ir, 'concrete') :
                z3_equations = ir.concrete(self.args_types)
            else : 
                if hasattr(ir, '__len__') : 
                    for i in range(len(ir)) :
                        ir[i] = ir[i].concrete(self.args_types)[0] if hasattr(ir[i], 'concrete') else z3.And(ir[i])
                z3_equations = z3.And(ir)

            z3_equations = ir.concrete(self.args_types) if hasattr(ir, 'concrete') else z3.And(ir)
            if not hasattr(z3_equations, '__iter__') :
                z3_equations = [z3_equations]
            for z3_equation in z3_equations :
                sufficient_conds = gen_sufficient_condition(z3_equation, self.is_suff_cond_need)
                if sufficient_conds :
                    z3_equation = z3.Implies(z3.And(sufficient_conds), z3_equation)
                # AUTOINF_LOG.debug(f"z3_equation : {z3_equation}, \nsuff_conds : {sufficient_conds}")
                constraints.append(z3_equation)
            # return z3.And(constraints)
            return z3.simplify(z3.And(constraints))
        except :
            raise ValueError(f"{self.info()}\n{traceback.format_exc()}")

    def type_rule_behavior(self,
                           left : str, 
                           comparators : List[Union[str, AbsLiteral]],
                           op : ast.Expr,
                           ) -> None :

        ## if rights == Literal , we assume that if dtype is kindof AbsLiteral, it cannot be exist with other dtype
        if any(isinstance(op, neg_op) for neg_op in NEG_OPS) : 
            sign = NOT_SIGN
        elif any(isinstance(op, pos_op) for pos_op in POS_OPS) :
            sign = None
        
        infered_types=[]
        for comparator in comparators :
            if type(comparator) == str :
                materalized = materalize_dtypes(comparator, merge_tensor=False) 
                if materalized is not None :
                    infered_types.extend(materalized)
            else :
                infered_types.append(comparator)
        for dtype in infered_types : 
            self.set_types_map(left, dtype, sign)

    def is_types_map_inited(self) : 
        return len(self.types_map) != 0
    def is_converting(self) : return self.inspecting == False 
    def set_iter_rule_flag(self, arg_name) : 
        self.must_be_seq[arg_name] = True
        self.is_subscriptable[arg_name] = True
    def clean_types_map(self) :
        to_rm=[]
        for key in self.types_map.keys() :
            if len(self.types_map[key]) == 0 :
                to_rm.append(key)
        for key in to_rm :
            del self.types_map[key]
    def is_unsolverable_dtype(self, orig_type, materalized) :
        ## materalized dtype is one of literal or iter(Tuple, list) -> z3 unsolverable ex) type(a) == list[int], or type(a) == Literal['a', 'b']
        ## if orig_type is not tensor, materalized dtype is absdtype -> z3 unsolverable ex) type(a) == int (if tensor -> to_tensor_dtype)
        return any(isinstance(materalized, dtype) for dtype in \
                   [AbsLiteral,
                    AbsIter]) or \
                    (not any(isinstance(orig_dtype, AbsTensor) for orig_dtype in self.args_type_dict[orig_type]) and \
                     any(isinstance(materalized, dtype) for dtype in \
                    [AbsDType]))
    def set_idx_constraints(self, arg_name, sliced) : 
        self.is_sliced[arg_name] = [sliced]
        self.min_len[arg_name] = max(self.min_len[arg_name], sliced+1 if sliced >= 0 else -sliced)
    def is_unsolverable(self) :
        return self.unsolverable
    def is_in_types_map(self, left) :
        return left in self.types_map.keys()
    def is_tensor(self, arg_name : str) :
        # Tensor dtype arg_name would not be changed dtype to int, or list[int], or bool.
        # dtype would be retrieved with args_type_dict that has been given when generating the rule.
        # Therefore, the type_actvated new dtype would not be shown here.
        arg_name = self.get_arg_name(arg_name)
        return isinstance(self.args_types[arg_name], AbsTensor)
        
    def get_arg_name(self, obj) :
        return str(obj)
    def is_generator_exp(self, left) :
        if isinstance(left, Select) :
            return left.has_symbol()
        elif isinstance(left, str) :
            return left in ['i', 'j']
        else :
            return False 
    def is_z3_obj(self, arg_name) :
        if isinstance(arg_name, str) and arg_name in self.arg_names :
            return False 
        else :
            return True 
    def check_tensor_dtype(self, arg_name) :
        if isinstance(self.args_types[arg_name], AbsTensor) :
            self._is_tensor[arg_name] = True
        else :
            self._is_tensor[arg_name] = False
    def materalize_dtype(self, dtype) : 
        if dtype not in self.arg_names : 
            return materalize_dtype(dtype)
        else :
            return dtype 
    def convert(self,
               expr : Union[ast.Expression,List[ast.Expression]]
               ) -> Union[IRcompare, IRexpr, bool] :
        """
        This function do two things : 
            1.collect arg info from given ast (first time executed) -> return bool
            2.gen z3 obj according to given ast -> return z3.ExprRef
        it return "bool" only when collecting arg_info, which only happens at the first time.
        Other times, it should return z3.ExprRef
        
        """
        if isinstance(expr, ast.Expression):
            return self.convert(expr.body)
        if isinstance(expr, ast.Expr):
            return self.convert(expr.value) ## if it is not body?
        elif isinstance(expr, ast.Module):
            return self.convert(expr.body[0])
        elif isinstance(expr, ast.UnaryOp):
            # Handle Boolean operations (e.g., And, Or)
            operand_to_oppo = self.convert(expr.operand)
            if isinstance(operand_to_oppo, str) : 
                return True 
            else :
                return IRexpr(expr.op, [operand_to_oppo])
        elif isinstance(expr, ast.BoolOp):
            # Handle Boolean operations (e.g., And, Or)
            return IRexpr(expr.op, [self.convert(value) for value in expr.values])
        elif isinstance(expr, ast.Tuple) :
            return [self.convert(value) for value in expr.elts]
        elif isinstance(expr, ast.Compare) : ## return left op comparators ex) 'output_size'<0 or 'input_size'
            res=[]
            left=expr.left 
            ops=expr.ops
            astop = ops[0]
            comparators=expr.comparators
            left = self.convert_vars(left) # -> filter out type/len constraints
            rights = self.convert_comparators_to_z3(comparators)
            op = convert_ops(ops)
            if self.is_in_types_map(left) and not self.is_converting() : # type()
                dtype_instances = [] 
                for right in rights :
                    if isinstance(right, str) :
                        right = self.materalize_dtype(right)
                    dtype_instances.append(right)
                for dtype in dtype_instances :
                    if self.is_unsolverable_dtype(left, dtype) :
                        self.type_rule_behavior(left, rights, astop)
                        self.set_model_unsolverable_flag()
                
            if self.is_unsolverable() or not self.is_converting()  : 
                return True 
            if self.is_generator_exp(left) : ## interpret ir in Generator part 
                ir = IRcompare(left, op, rights)
                return ir
            else : ## interpret ir 
                if isinstance(left, Select) and left.attr in TYPE_ATTRS : # comparator dtype interpreting if left attr == dtype 
                    left = left.concrete(self.args_types)
                    z3dtypes = set()
                    for _dtype in rights :
                        # string, or Select obj
                        ## currently only allowed btw tensor dtypes.
                        if isinstance(_dtype, Select) : 
                            z3dtypes.add(_dtype.concrete(self.args_types))
                            continue
                        dtype = self.materalize_dtype(_dtype)
                        if not isinstance(dtype, DType) :
                            ## currently only allowed btw tensor dtypes.
                            if hasattr(dtype, 'to_tensor_dtype') :
                                if isinstance(astop, ast.Eq) and len(dtype.to_tensor_dtype())>1: 
                                    astop = ast.In()
                                    op = convert_ops([astop])
                                temp = dtype.to_tensor_dtype()
                                z3s = [d.z3() for d in temp]
                                z3dtypes.update(z3s)
                            elif hasattr(dtype, 'z3') : 
                                z3dtypes.add(dtype.z3())
                            else :
                                z3dtypes.add(dtype)
                        else :
                            z3dtypes.add(dtype.z3())
                    z3dtypes = list(z3dtypes)
                    if len(z3dtypes) == 1 : z3dtypes = z3dtypes[0]
                    if is_compatiable(astop, left, z3dtypes) :
                        res.append(op(left, z3dtypes))
                    else :
                        if any(isinstance(astop, eq_noteq) for eq_noteq in [ast.Eq, ast.NotEq]) : 
                            for z3dtype in z3dtypes :
                                res.append(convert_ops([astop])(left, z3dtype))
                        else :
                            AUTOINF_LOG.debug(f"{self.get_mode()} Uncompatiable - left : {left}, z3dtypes : {z3dtypes}")
                            return True
                else : # left.attr in [RANK_ATTR, LEN_ATTR] or other cases
                    concreted = []
                    left = left.concrete(self.args_types) if hasattr(left, 'concrete') else left
                    for right in rights :
                        if isinstance(right, Select) : 
                            concreted.append(right.concrete(self.args_types))
                        else :
                            concreted.append(right)
                    if is_compatiable(astop, left, concreted) :
                        res.append(op(left, concreted))
                    elif hasattr(left, '__len__') :
                        if is_compatiable(astop, left[0], concreted) :
                            res.append([op(left[i], concreted) for i in range(len(left))])
                    else :
                        if hasattr(left, 'sort') and left.sort().kind() in [6, 5] : # TensorZ3).sort()
                            if len(concreted) == 1 and hasattr(concreted[0], 'sort') and concreted[0].sort().kind() in [6, 5] :
                                res.append(op(left, concreted[0]))
                                res.append(gen_len_obj(left)==gen_len_obj(concreted[0]))
                            else :
                                len_obj = gen_len_obj(left)
                                res.append(len_obj==len(concreted))
                                r = [op(left[i], concreted[i]) for i in range(len(concreted))]
                                res.extend(r)
                        else :
                            for con in concreted : 
                                if is_compatiable(astop, left, con) :
                                    res.append(op(left, con))
            
            return IRexpr(ast.And(), [res])
        elif isinstance(expr, ast.Call):
            # Handle function calls
            converted = self.convert_vars(expr)
            if isinstance(converted, str) : return True
            else : return converted
        elif isinstance(expr, ast.comprehension):
            target = self.convert_vars(expr.target)
            iters = self.convert(expr.iter)
            # if expr.iter.func.id == 'range' :
            if len(expr.ifs) > 0 : raise(NotImplementedError)
            return (target, iters) 

        else:
            return self.convert_vars(expr)
    

    def convert_vars(self, expr : ast.Expression) -> Union[str, z3.ExprRef, bool] : 
        """
        return (arg_name, func_name, slices )
        expr : ast.Expression which should contain the information of expr 
        types : Dict[str, Any] which contains the information of type of each variable(var name should be same with left arg_name)
        """

        if isinstance(expr, ast.Name) :
            if self.is_converting() :
                if expr.id in self.arg_names :
                    return Select(expr.id)
            return expr.id
        elif isinstance(expr, ast.Constant) :
            # Handle constant values
            if self.is_converting() :
                if expr.value in self.arg_names :
                    return Select(expr.value)
            return expr.value 
        elif isinstance(expr, ast.UnaryOp) :
            if isinstance(expr.op, ast.USub) :
                left = self.convert_vars(expr.operand)
                if self.is_converting() :
                    if isinstance(left, Select) :
                        return - left.concrete(self.args_types)
                    elif isinstance(left, str) : # 'i' symbol
                        return (ast.Mult(), left, -1)
                    else :
                        return -left 
                else :
                    return left
        elif isinstance(expr, ast.Tuple) :
            return self.convert(expr)
        elif isinstance(expr, ast.Subscript) :
            arg_name = self.convert_vars(expr.value)
            sliced = self.convert_vars(expr.slice)
            if self.is_converting() :
                if isinstance(sliced, int) :
                    arg_name.set_idx(sliced)
                elif isinstance(sliced, tuple) : 
                    astop, left, right = sliced 
                    arg_name.set_idx(left)
                    arg_name.set_binops((get_operator(astop), right))
                else : # str (plain 'i') or Select
                    arg_name.set_idx(sliced)
                return arg_name
            else : # inspecting 
                if isinstance(arg_name, str) and arg_name.lower() == 'literal' :
                    self.check_types_map(create=True)
                    if type(sliced) == str : # 'same' -> ['same']
                        sliced = [sliced]
                    type_name = AbsLiteral(sliced)
                    return type_name
                elif isinstance(arg_name, str) and arg_name.lower() == 'list' :
                    self.check_types_map(create=True)
                    return arg_name.lower() + '[' + str(sliced) + ']'
                else :
                    if arg_name in self.arg_names : #just for inspecting.
                        return arg_name
                    else :
                        AUTOINF_LOG.info(f"{self.get_mode()} Unsupported subscript of {arg_name}")
                        return arg_name      
        elif isinstance(expr, ast.Attribute) :

            arg_name = self.convert_vars(expr.value)
            if self.is_converting() :
                if hasattr(arg_name, 'set_attr') :
                    arg_name.set_attr(expr.attr)
                    return arg_name
                else :
                    return arg_name + '.' + expr.attr
                    # raise Exception(f"{self.get_mode()} wrong rule generated, {arg_name} not in arg_names")
            else :
                if expr.attr in TYPE_ATTRS :
                    self.set_types_map(arg_name)
                return arg_name   
        elif isinstance(expr, ast.Call) :
            if isinstance(expr.func, ast.Attribute) :
                return self.convert_vars(expr.func)
            call = expr 
            if call.func.id in TYPE_FUNCS :
                arg_names = [self.convert_vars(arg) for arg in expr.args]
                assert len(arg_names) == 1 ## we restrict to only accept type() of dtype rule
                arg_name = arg_names[0]  
                if self.is_converting() :
                    ## arg_name should be the instance of Select 
                    arg_name.set_attr(TYPE_ATTR)
                    return arg_name
                else :
                    ## some checking if needed 
                    self.set_types_map(arg_name)
                    return arg_name  
            elif call.func.id == 'range' : # -> (inspecting)arg_name, (converting)Int
                # -> strongly interleave with generaterexp,
                # it should return list in conventional, for simplicity, we only return int here.
                range_args = [] 
                for arg in expr.args : # will return args [start,end,step]
                    range_args.append(self.convert_vars(arg))
 
                if self.is_converting() :
                    return range_args
                else :
                    return self.get_arg_name(range_args)
            elif hasattr(z3_funcs, call.func.id) :
                args = []
                for arg in expr.args : 
                    args.append(self.convert_vars(arg))
                
                if self.is_converting() :
                    if len(args) == 1 : args = args[0]
                    func = getattr(z3_funcs, call.func.id)
                    if isinstance(args, Select) :
                        args.set_func(func)
                        return args
                    else : 
                        for i in range(len(args)) :
                            if isinstance(args[i], Select) :
                                args[i] = args[i].concrete(self.args_types)
                        return func(args)
                else : # inspecting 
                    return self.get_arg_name(args)

            elif call.func.id in ['all', 'any'] :
                if self.is_converting() : 
                    constraints = self.convert_vars(expr.args[0])
                    if call.func.id == 'all' :
                        constraints = IRexpr(ast.And(), constraints)
                    elif call.func.id == 'any' :
                        constraints = IRexpr(ast.Or(), constraints)
                    return constraints
                else : # inspecting
                    return True 
            elif call.func.id in [LEN_ATTR, RANK_ATTR] :
                    arg_name = self.convert_vars(expr.args[0]) #FIXME : it should be 
                    if self.is_converting() : 
                        arg_name.set_attr(LEN_ATTR)
                        return arg_name
                    else : 
                        self.set_iter_rule_flag(arg_name)
                        return arg_name
            else :
                raise ValueError(f"{self.get_mode()} Unsupported function {call.func.id}")

        elif isinstance(expr, ast.GeneratorExp) or isinstance(expr, ast.ListComp) :
            ## interpret generators -> idx = symbol , name cannot change 
            if not self.is_converting() : return True 
            syms = []
            idx_constraints = []
            start, end, step = 0, __MAX_RANK__, 1 
            irexpr = self.convert(expr.elt)
            if len(expr.generators) > 1 : raise NotImplementedError(f"{self.get_mode()} Not implemented yet, multiple generators")
            idx_nm, iters = self.convert(expr.generators[0])
            irexpr.find_sym(idx_nm)
            sym_idx = symbolize_idx(idx_nm)
            syms.append(sym_idx)
            if isinstance(iters, Select) : # all(i>0 for i in input.shape)
                conc = iters.concrete(self.args_types)
                length = iters.export_len_var(self.args_types)
                new_sym_idx = symbolize_idx(iters.name+'_i')
                syms.append(new_sym_idx)
                idx_constraints.extend([new_sym_idx>=0, new_sym_idx<length] )
                idx_constraints.append(sym_idx == conc[new_sym_idx])
            else :
                if len(iters) == 1 : #only have end / all(input.shape[i]>0 for i in range(len(input.shape)))
                    if isinstance(iters[0], Select) : # if list -> ?
                        if iters[0].attr in [LEN_ATTR, RANK_ATTR] :
                            end = iters[0].concrete(self.args_types)
                    else : # int
                        end = iters[0]

                elif len(iters) == 2 : # range -> start, end 
                    start = iters[0]
                    end = iters[1]

                else :  # range -> start, end, step 
                    step = iters[2]
                    assert(isinstance(step, int) and abs(step) == 1)
                    if step < 0 : #range(a,b,-1) -> range(b-1, a-1, 1)
                        start = iters[1] + 1
                        end = iters[0] + 1
                    else :
                        start = iters[0]
                        end = iters[1]

                idx_constraints.append(sym_idx<end)
                idx_constraints.append(sym_idx>=start)
                
            self.mark_suff_cond_needed(self.locate_symbol(irexpr))
            irexpr.assign(sym_idx)
            generator_expr = irexpr.concrete(self.args_types, sym_idx) 
            combined = [z3.ForAll(syms, z3.Implies(z3.And(idx_constraints), z3.And(generator_expr)))]
            return combined
        
        elif isinstance(expr, ast.BinOp) :
            if self.is_converting() :
                left = self.convert_vars(expr.left)
                binop = get_operator(expr.op)
                rights = self.convert_comparators_to_z3([expr.right])
                right = rights[0]
                if isinstance(left, str) : # 'i + 1' 
                    return (expr.op, left, right)
                if hasattr(left, 'concrete') : left = left.concrete(self.args_types)
                if hasattr(right, 'concrete') : right = right.concrete(self.args_types)
                return binop(left, right)
            else : # inspecting 
                return True 
        else : 
            raise ValueError(f"{self.get_mode()} Unsupported generator expression {expr}")
    def get_arg_name(self, args : List[Any]) :
        for arg_name in args :
            if arg_name in self.arg_names :
                return arg_name            
    def mark_suff_cond_needed(self, sym_pos : List[Tuple[Select, str]]) : 
        for pos in sym_pos : 
            if pos[1] == 'idx' : 
                self.is_suff_cond_need[pos[0].name] = False
            else :
                pass 

    def locate_symbol(self, irexprs) :
        pos = []
        if isinstance(irexprs, IRcompare) :
            pos.extend(irexprs.whereis_symbols)
        elif isinstance(irexprs, IRexpr) :
            for irexpr in irexprs.values :
                pos.extend(self.locate_symbol(irexpr))
        return pos

    def check_types_map(self, create : bool = False) :
        res = True  
        if not self.is_types_map_inited() :
            AUTOINF_LOG.warning(f"{self.get_mode()} unconsidered behavior, generated by obvious unsolverable rule.")
            res = False 
        if create : 
            arg_name = self.arg_names[0]
            self.set_types_map(arg_name)
        return res 
    def set_idx(self, idx, iters) : 
        if type(iters) == int :
            return range(iters) 
        else : 
            return iters 
    def set_err_flag(self) : 
        self.error = True
    def convert_comparators_to_z3(self, comparators : List[ast.Expression]) -> List[ast.Expression] : 
  
        converted_comparators=[]
        if len(comparators) > 1 :
            for comparator in comparators :
                converted_comparators.extend(self.convert_comparators_to_z3([comparator]))
            return converted_comparators
        elif len(comparators) < 1 :
            return []
        else : 
            comparators = comparators[0]
            if isinstance(comparators, ast.Tuple) or isinstance(comparators, ast.List) :
                converted_comparators.extend(self.convert_comparators_to_z3(comparators.elts))
                return converted_comparators
            else : return [self.convert_vars(comparators)]

    def gen_z3_obj(self, arg_name) -> z3.ExprRef :
        
        return gen_z3_obj(arg_name,
                          self.args_types[arg_name],
                            )


def gen_z3_obj(arg_name,
               arg_type : Union[AbsDType, AbsLiteral, AbsTensor, AbsLiteral],                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
) -> Optional[z3.ExprRef] :
    z3_obj = None
    if arg_type in [AbsDType.none, AbsDType.str] :
        pass
    elif hasattr(arg_type, 'z3') : 
        z3_obj = arg_type.z3()(arg_name)
    elif isinstance(arg_type, AbsTensor) : 
        z3_obj = arg_type.z3()(arg_name)
    else :
        raise NotImplementedError(f"Unsupported {arg_name} type {arg_type}")
    return z3_obj

def is_compatiable(op : ast, a, b) -> bool :
    if not hasattr(a, '__iter__') and (not isinstance(b, str) and hasattr(b, '__iter__')) :
        return any(isinstance(op, allowed) for allowed in [ast.In, ast.NotIn])
    elif not hasattr(a, '__iter__') and (isinstance(b, str) or not hasattr(b, '__iter__')) :
        return any(isinstance(op, allowed) for allowed in [
                                                        ast.Eq,
                                                        ast.NotEq,
                                                        ast.Lt,
                                                        ast.LtE,
                                                        ast.Gt,
                                                        ast.GtE,
                                                        ast.Is,
                                                        ])
  
    else :
        return False 
def convert_ops(ops: List[ast.Expression]) -> Callable : # operator of ast 

    if len(ops) > 1 :
        # for op in ops :
        #     converted_ops.extend(op)
        # return converted_ops
        raise NotImplementedError(f"Unsupported multiple ops {' '.join([type(op) for op in ops])}")
    elif len(ops) < 1 :
        raise NotImplementedError(f"Unsupported empty ops")
    else :
        ops = ops[0]
        if isinstance(ops, ast.Eq):
            return lambda a,b : op.eq(a, b)
        elif isinstance(ops, ast.NotEq):
            return lambda a,b : op.ne(a, b)
        elif isinstance(ops, ast.Lt):
            return lambda a,b : op.lt(a, b)
        elif isinstance(ops, ast.LtE):
            return lambda a,b : op.le(a, b)
        elif isinstance(ops, ast.Gt):
            return lambda a,b : op.gt(a, b)
        elif isinstance(ops, ast.GtE):
            return lambda a,b : op.ge(a, b)
        elif isinstance(ops, ast.Is):
            return lambda a,b : op.eq(a, b)
        elif isinstance(ops, ast.IsNot):
            return lambda a,b : op.ne(a, b)
        elif isinstance(ops, ast.In):
            return lambda a,b : z3_funcs.in_(a,b)
        elif isinstance(ops, ast.NotIn):
            return lambda a,b : z3_funcs.not_in(a,b)
        else:
            raise ValueError(f"Unknown comparison operation: {op}")

def get_operator(operator):
    if isinstance(operator, ast.Add):
        return lambda a,b : op.add(a, b)
    elif isinstance(operator, ast.Sub):
        return lambda a,b : op.sub(a, b)
    elif isinstance(operator, ast.Mult):
        return lambda a,b : op.mul(a, b)
    elif isinstance(operator, ast.MatMult):
        return lambda a,b : op.matmul(a, b)
    elif isinstance(operator, ast.Div):
        return lambda a,b : op.truediv(a, b)
    elif isinstance(operator, ast.FloorDiv):
        return lambda a,b : op.truediv(a, b)
    elif isinstance(operator, ast.Mod):
        return lambda a,b : op.mod(a, b)
    elif isinstance(operator, ast.Pow):
        return lambda a,b : op.pow(a, b)
    # elif isinstance(operator, ast.LShift):
    #     return op.lshift
    # elif isinstance(operator, ast.RShift):
    #     return op.rshift
    # elif isinstance(operator, ast.BitOr):
    #     return op.or_
    # elif isinstance(operator, ast.BitXor):
    #     return op.xor
    # elif isinstance(operator, ast.BitAnd):
    #     return op.and_
    # elif isinstance(operator, ast.FloorDiv): ## FIXME : z3 does not support floor div
    #     return op.truediv 
    else:
        raise ValueError(f"Unknown operator: {operator}")

def convert_boolop_to_z3(op : ast.BoolOp) -> z3.ExprRef :
    if isinstance(op, ast.Or):
        return z3.Or
    elif isinstance(op, ast.Not):
        return z3.Not
    elif isinstance(op, ast.And):
        return z3.And
    else :
        raise NotImplementedError(f"Unsupported boolop type {type(op)}")   

def flatten_list(nested_list : List[Any]) -> List[Any] :
    flattened_list = []
    [flattened_list.extend(sublist) for sublist in nested_list]
    return flattened_list
