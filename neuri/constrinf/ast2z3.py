import ast
import random
import string
import traceback
import operator as op 
import z3
from typing import *
from neuri.abstract.dtype import AbsDType, DType
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

def gen_z3_obj(arg, arg_map, ret_wrapper=False, no_const=False) : 
    if ast2z3.is_sym(arg) : 
        z3obj = arg
    elif arg in arg_map.keys() : 
        z3obj = arg_map[arg]
    else :
        if isinstance(arg, str) and no_const : # index 'i' generator
            return z3.Int(arg) 
        return arg # constant
    if hasattr(z3obj, 'get_wrapped_object') and not ret_wrapper :
        return z3obj.get_wrapped_object()
    else :
        return z3obj
    # if idx is not None : 
    #     if type(idx) == int :
    #         return z3obj[z3obj.rank + idx] if idx<0 else z3obj[idx]
    #     else : # when index is 'i' for generator
    #         if idx in arg_map.keys() : 
    #             return z3obj[arg_map[idx]]
    #         elif type(idx) == str :
    #             return z3obj[z3.Int(idx)]
    #         else : 
    #             return z3obj[idx]
    # else : 
    #     if hasattr(z3obj, 'get_wrapped_object') and not ret_wrapper :
    #         return z3obj.get_wrapped_object()
    #     else :
    #         return z3obj


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
        iter = gen_z3_obj(comp["iter"], arg_map, ret_wrapper=True)
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
    def gen_sym(a):
        # Check if 'a' is a Z3 expression or sort
        return z3.Int(a)

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
                slice_value = self._convert(node.slice.value, arg_map)
                slice_value = gen_z3_obj(slice_value, arg_map, no_const=True)
                val = gen_z3_obj(self._convert(node.value, arg_map), arg_map, ret_wrapper=True)
                return val[slice_value]
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
                ret_wrapper=True,
                )
        elif isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Num):
            return node.n
        else:
            raise ValueError(f"Unsupported AST node {ast.dump(node)})")
    # Process and print each constraint
