import ast
import itertools
import random
import string
import traceback
import operator as op 
import z3
from typing import *
from neuri.abstract.dtype import AbsDType, DType
from neuri.specloader import z3_funcs

###### TENSOR DTYPE DEINITION ######

## Define : tensor, int, float, bool, complex, str``
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
    def __init__(self, const, datatype, value_func, len_func):
        self.const : z3.Datatype = const
        self.datatype : z3.Datatype = datatype
        self.value_func = value_func
        self.len_func = len_func
        self.info = {

        }
    
    def corrected_idx(self, idx, datatype = None, len_func_name = "len"):
        """ 
        return corrected idx for possible negative index
        """ 
        if datatype is None : datatype = self.datatype
        if len_func is None : len_func = self.len_func
        else : len_func = getattr(datatype, len_func_name)
        return z3.If(idx >= 0, idx, self.len_func(self.const) + idx)  
    
    def len(self): 
        if self.info.get("sliced") is not None : 
            return self.info["sliced"][1] - self.info["sliced"][0]
        else :
            return self.len_func(self.const)

    @property
    def value(self):
        return self.value_func(self.const)
     
    def __getitem__(self, idx):
        return self.corrected_idx(idx, self.datatype, "len")

    @property
    def name(self):
        return self.const.decl().name()
    
    @property
    def rank(self):
        return self.len_func(self.const)
    
    def update_info(self, *args, **kwargs):
        for key, value in kwargs.items() : 
            self.info[key] = value
    
    def get_wrapped_object(self):
        return self.const
    
    def range(self):
        # for generator conversion # range of idx
        assert "sliced" in self.info.keys(), "sliced info is not defined"
        return self.info["sliced"]
class TensorWrapper(BaseWrapper):
    def __init__(self, const, datatype, value_func, len_func):
        super().__init__(const, datatype, value_func, len_func)
        self.dtype_func = datatype.dtype
    def __getitem__(self, idx):
        return self.corrected_idx(idx)
     
    def __str__(self):
        return f"Tensor:{self.const.name()}"
    
    @property
    def shape(self):
        return self.value_func(self.const)
    
    @property
    def dtype(self):
        return self.dtype_func(self.const)
    
class ComplexWrapper(BaseWrapper):
    def __init__(self, const, datatype):
        super().__init__(const, datatype)

    def range(self):
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError
    
    def __getitem__(self, idx):
        raise NotImplementedError
       
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
    
def load_z3_const(name, var_type, is_array=False):
    """
    Define a Z3 variable of a given type. 
    If it's an array, specify the element type.
    """
    if var_type == 'int':
        if is_array : 
            return ArrWrapper(z3.Const(name, IntArr), IntArr, IntArr.value, IntArr.len)
        else : 
            return z3.Const(name, z3.IntSort())
    elif var_type == 'float':
        if is_array : 
            return ArrWrapper(z3.Const(name, FloatArr), FloatArr, FloatArr.value, FloatArr.len)
        else :
            return z3.Const(name, z3.RealSort())
    elif var_type == 'str':
        if is_array : 
            return ArrWrapper(z3.Const(name, StrArr), StrArr, StrArr.value, StrArr.len)
        else :
            return z3.Const(name, z3.StringSort())
    elif var_type == 'complex':
        if is_array : 
            return ArrWrapper(z3.Const(name, ComplexArr), ComplexArr, ComplexArr.value, ComplexArr.len)
        else :
            return ComplexWrapper(z3.Const(name, Complex), Complex)
    elif var_type == 'bool':
        if is_array : 
            return ArrWrapper(z3.Const(name, BoolArr), BoolArr, BoolArr.value, BoolArr.len)
        else :
            return z3.Const(name, z3.BoolSort())
    elif var_type == 'tensor':
        if is_array : 
            return ArrWrapper(z3.Const(name, TensorArr), 
                              TensorWrapper(z3.Const(name, TensorZ3), TensorZ3),
                              TensorArr.value, TensorArr.len)
        else :
            return TensorWrapper(z3.Const(name, TensorZ3), TensorZ3, TensorZ3.shape, TensorZ3.rank)
    else:
        raise ValueError("Unsupported variable type")

def is_wrapper(obj) : 
    return hasattr(obj, 'get_wrapped_object')

class z3funcs:
    """
    Class to hold custom Z3 functions.
    """
    # Class variable to hold names of all functions
    function_names = ['all', 'any', 'len', 'type', 'sorted', 'abs', 'min', 'max', 'in_', 'not_in', 'rank', 'range']
    _z3_dataarr = [IntArr, FloatArr, StrArr, ComplexArr, BoolArr, TensorArr, TensorZ3, TensorZ3.shape]
    iterables = []
    non_iterables= []
    rank_idx = 0
    value_idx = 1
    dtype_idx = 2
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
    def not_in(a,b) : 
        pass 

    @classmethod 
    def is_iterable(cls, v) :
        return cls._find_datatype_by_id(v.sort().get_id()) is not None 
    
    @classmethod
    def type(cls, v) : 
        if hasattr(v, 'dtype') :
            return v.dtype
        else :
            attrs = cls._load_attrs(v)
            assert len(attrs) > cls.dtype_idx, f"{v} seems not be Tensor object"
            return attrs[cls.dtype_idx]
    @staticmethod
    def sort(v) : 
        if is_wrapper(v) : 
            return v.datatype.name()
        else : 
            return v.sort().name()
    
    def clear(self) :
        self.constrs.clear()
    
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
             # Tensor.shape - dirty fix
            datatype = cls._find_datatype_by_id(v.decl().get_id())

        if datatype is None :
            return None
        else : 
            obj, name = datatype
            if name == TensorZ3.name() : 
                return obj.rank(v), obj.shape(v), obj.dtype(v)
            elif name == "shape" : #shape(t) -> load name "t" -> TensorZ3.rank(t)
                return TensorZ3.rank(v.arg(0)), v, TensorZ3.dtype(v.arg(0))
            else : 
                return obj.len(v), obj.value(v)
    @classmethod
    def is_tensor(cls, v):
        return cls.sort(v) == TensorZ3.name()
    
    @classmethod
    def shape(cls, v):
        """
        Custom 'len' function for Z3 arrays and tensors.
        Returns the length of an array or the rank of a tensor.
        """
        if hasattr(v, 'shape') :
            return v.shape
        else :
            attrs = cls._load_attrs(v)
            assert len(attrs) > cls.value_idx, f"{v} seems not be Tensor object"
            return attrs[cls.value_idx]
        
    @classmethod
    def dtype(cls, v):
        if hasattr(v, 'dtype') :
            return v.dtype
        else :
            attrs = cls._load_attrs(v)
            assert len(attrs) > cls.dtype_idx, f"{v} seems not be Tensor object"
            return attrs[cls.dtype_idx]
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
            assert len(attrs) > cls.rank_idx, f"{v} seems not be Tensor object"
            return attrs[cls.rank_idx]
    @classmethod
    def dim(cls, v) :
        assert cls.is_tensor(v), f"{v} is not tensor object"
        return cls.len(v)
    @classmethod
    def ndim(cls, v) :
        return cls.dim(v)
    @staticmethod
    def in_(a,b) : 
        from specloader.irs import SYM_LEN
        if isinstance(b[0], z3.ArrayRef) : 
            b_len = SYM_LEN[b[0].arg(0).decl().name()] if b[0].num_args() else SYM_LEN[b[0].decl().name()]
            i = z3.Int('i')
            exists = z3.Exists([i], z3.Implies(z3.And(i >= 0, i < b_len), b[0][i] == a))
            return exists
        elif hasattr(b, '__len__') :
            return z3.Or([a == v for v in b])
        else :
            return z3.Or([a == b[i] for i in range(MAX_ARR_LEN)])
    
    @staticmethod
    def not_in(a,b) : 
        if isinstance(b[0], z3.ArrayRef) : 
            from specloader.irs import SYM_LEN
            b_len = SYM_LEN[b[0].arg(0).decl().name()] if b[0].num_args() else SYM_LEN[b[0].decl().name()]
            i = z3.Int('i')
            exists = z3.ForAll([i], z3.Implies(z3.And(i >= 0, i < b_len), b[0][i] != a))
            return exists
        elif hasattr(b, '__len__') :
            return z3.And([a != v for v in b])
        else :
            return z3.And([a != b[i] for i in range(MAX_ARR_LEN)])
    @classmethod
    def get_corrected_idx(cls, idx, v) :
        rank = cls.len(v)
        corrected_idx = z3.If(idx >= 0, idx, rank + idx)
        return corrected_idx
    def min(self, *vs):
        if len(vs) > 1 :
            assert len(vs) > 0, f"len of list should be positive, cur : {len(vs)}"
            if len(vs) == 1 :
                return vs[0]
            m = vs[0]
            for v in vs[1:]:
                m = z3.If(v < m, v, m)
            return m
        else : 
            vs = vs[0]
            assert self.is_iterable(vs), f"{vs.sort().name()} is not iterable variable"
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

def is_same_ast_name(left_ast : str, right_ast) : 
    return left_ast == right_ast.__name__

def get_bool_operator(astop : str): 
    if is_same_ast_name(astop, ast.And):
        return z3.And
    elif is_same_ast_name(astop, ast.Or):
        return z3.Or
    elif is_same_ast_name(astop, ast.Not):
        return z3.Not
    else:
        raise ValueError(f"Unknown operator: {astop}")
    
def get_operator(astop : str):
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
        return lambda a,b : z3funcs.in_(a,b)
    elif is_same_ast_name(astop, ast.NotIn):
        return lambda a,b : z3funcs.not_in(a,b)
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
    else:
        raise ValueError(f"Unknown operator: {astop}")

def random_gen_name() : 
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(5))

def change_val_from_expr(expr, target, new_target):
    return z3.substitute(expr, (target, new_target))

    # res = gen_z3_obj(obj)[{start}:{end}]


def is_dtype_constant(arg) :
    from neuri.specloader.materalize import STR_TO_ABS
    return arg in STR_TO_ABS.keys()

def get_dtype_z3_obj(arg) : 
    from neuri.specloader.materalize import STR_TO_ABS
    dtypeobj = STR_TO_ABS[arg] 
    if hasattr(dtypeobj, "to_tensor_dtype") : 
        if dtypeobj.to_tensor_dtype() is not None :
            return [dtype.z3() for dtype in dtypeobj.to_tensor_dtype()]
    return dtypeobj.z3()

def gen_z3_obj(arg, arg_map, ret_wrapper=True, no_const=False) : 
    """
    in most of the case, 
    we need to return wrapper, 
    wrapper has correction for negative index
    """
    
    if ast2z3.is_sym(arg) : 
        z3obj = arg
    elif arg in arg_map.keys() : 
        z3obj = arg_map[arg]
    elif is_dtype_constant(arg) : # TODO : very inefficient(every conversion need to check)
        return get_dtype_z3_obj(arg)
    else :
        if isinstance(arg, str) and no_const : # index 'i' generator
            return z3.Int(arg) 
        return arg # constant
    if hasattr(z3obj, 'get_wrapped_object') and not ret_wrapper :
        # array but need to its const, why?
        return z3obj.get_wrapped_object()
    else :
        return z3obj

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

def dict_combinations(input_dict):
    # Create a list of tuples where each tuple is (key, option)
    keys, values = zip(*[(key, value) for key, values in input_dict.items() for value in values])
    
    # Generate all combinations
    all_combinations = itertools.product(*[set(vals) for vals in input_dict.values()])

    # Create a list of dictionaries for each combination
    result = []
    for combination in all_combinations:
        result.append(dict(zip(keys, combination)))

    return result
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
    def dtype_hijack(self, dtype) : 
        ## if dtype-related rule exist, then hijack dtype to be the same as the rule
        pass 
    def gen_dtype_obj(self) :
        return dict_combinations(self.arg_map) 
    def dtype_constant_to_z3(self, dtype) :
        if dtype in DType : 
            return DType[dtype].z3()
        else :
            return False 
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
        for dtype_map in self.gen_dtype_obj() : 
            # dtype_hijack else : 
            z3_type_objs = {
                    name : dtype.z3()(name) for name, dtype in dtype_map.items()
                }
            return self._convert(ast, z3_type_objs)
    
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
        return is_z3_obj or is_wrapper(a)

    @staticmethod
    def get_name(a) : 
        if is_wrapper(a) :
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
        idx_range = (z3obj.corrected_idx(start), z3obj.corrected_idx(start))
        z3obj.update_info(sliced=idx_range)
        return z3obj
    
    def gen_bool_constr(self, op, *args) : 
        res = get_bool_operator(op)(*args)
        return res

    def replace_op_accord_constant(self, op, arg) : 
        """
        some dtype is float -> float16, float32, float64
        eq -> in 
        not_eq -> not in 
        """
        if hasattr(arg, "__len__") : # suppose that arg is replaced
            if is_same_ast_name(op, ast.Eq) : 
                return ast.In.__name__
            elif is_same_ast_name(op, ast.NotEq) :
                return ast.NotIn.__name__
        return op
    def _convert(self, node, arg_map):
        if isinstance(node, ast.BoolOp):
            op = type(node.op).__name__
            values = [self._convert(value, arg_map) for value in node.values]
            return self.gen_bool_constr(op, *values)
        elif isinstance(node, ast.UnaryOp):
            op = type(node.op).__name__
            operand = self._convert(node.operand, arg_map)
            if is_same_ast_name(op, ast.Not) :
                return self.gen_bool_constr(op, operand)
            else : 
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
                op_type = self.replace_op_accord_constant(op_type, right)
                results.append(self.gen_basic_constr(op_type, left, right))
            return merge_constr(results)
        elif isinstance(node, ast.Subscript):
            # Handle negative indices and slicing
            if isinstance(node.slice, ast.Index):
                slice_value = self._convert(node.slice.value, arg_map)
                slice_value = gen_z3_obj(slice_value, arg_map, no_const=True)
                val = gen_z3_obj(self._convert(node.value, arg_map), arg_map, ret_wrapper=True)
                if not is_wrapper(val) : 
                    slice_value = self.get_corrected_idx(slice_value, val)
                return val[slice_value]
            elif isinstance(node.slice, ast.Slice):
                # Slicing, e.g., a[1:] or a[:-1]
                start = self._convert(node.slice.lower, arg_map) if node.slice.lower else None
                end = self._convert(node.slice.upper, arg_map) if node.slice.upper else None
                array = self._convert(node.value, arg_map)
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
            if is_dtype_constant(node.value) : # TODO : very inefficient(every conversion need to check)
                node.value = get_dtype_z3_obj(node.value)
            return node.value
        elif isinstance(node, ast.Num):
            return node.n
        else:
            raise ValueError(f"Unsupported AST node {ast.dump(node)})")
    # Process and print each constraint
