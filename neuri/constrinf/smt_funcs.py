import z3
from typing import *
from neuri.error import IncompatiableConstrError, IncorrectConstrError
###### TENSOR DTYPE DEINITION ######

## Define : tensor, int, float, bool, complex, str``
## Define : Array(tensor, int, float, bool, complex, str)
## Will be updated to support Dict, object

### add is_iter attribute True 

SUPPORTED_DTYPES = [
    "float32",
    "int32",
    "int64",
    "bool",
    "complex32",
    "complex64",
    "float16",
    "float64",
    "int8",
    "int16",
    "complex128",
    "quint8",
    "qint8",
    "qint16",
    "qint32",
    "bfloat16",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    ]

Z3DTYPE = z3.Datatype("DType")
for dtype in SUPPORTED_DTYPES : 
    Z3DTYPE.declare(dtype)
Z3DTYPE = Z3DTYPE.create()

iter_specific_funcs = ["len", "rank", "sorted", "min", "max", "dim", "ndim"]
tensor_dtype_check_funcs = ["dtype"]

Complex = z3.Datatype("complex")
Complex.declare("complex_instance", ("real", z3.RealSort()), ("imag", z3.RealSort()))
Complex = Complex.create()

TensorZ3 = z3.Datatype("TensorZ3")
TensorZ3.declare("tensor_instance", 
                ("shape", z3.ArraySort(z3.IntSort(), z3.IntSort())),
                ("dtype", Z3DTYPE),
                ("rank", z3.IntSort())),
TensorZ3 = TensorZ3.create()

def DeclareArr(sort):
    Arr = z3.Datatype("Arr_of_%s" % sort.name())
    Arr.declare("arr_instance", 
                ("value", z3.ArraySort(z3.IntSort(), sort)),
                ("len", z3.IntSort())),
    Arr = Arr.create()
    return Arr

IntArr = DeclareArr(z3.IntSort())
FloatArr = DeclareArr(z3.RealSort())
StrArr = DeclareArr(z3.StringSort())
BoolArr = DeclareArr(z3.BoolSort())
ComplexArr = DeclareArr(Complex)
TensorArr = DeclareArr(TensorZ3)
ARRTYPES = [IntArr, FloatArr, StrArr, BoolArr, ComplexArr, TensorArr]

def is_same_constr(A : z3.ExprRef, B : z3.ExprRef) -> bool :
    s = z3.Solver()
    s.add(z3.Not(z3.Implies(A, B)))
    if s.check() == z3.unsat:
        s.reset()
        s.add(z3.Not(z3.Implies(B, A)))
        if s.check() == z3.unsat:
            return True
    return False

def has_same_rule(B : z3.ExprRef, A : List[z3.ExprRef]) -> bool :
    return any(is_same_constr(a, B) for a in A)

def is_implies(A : z3.ExprRef, B : z3.ExprRef) -> bool :
    s = z3.Solver()
    s.push()
    s.add(z3.And(A,z3.Not(B)))
    if s.check() == z3.unsat: return True 
    s.pop()
    s.add(z3.And(B,z3.Not(A)))
    if s.check() == z3.unsat: return True 
    return False 

def gen_type_constr(z3instance, flag):
    if flag == "must_iter":
        # Check if the z3instance is of any array type
        return z3.Or([z3instance.sort() == arr_type for arr_type in ARRTYPES])
    elif flag == "must_int":
        # Check if the z3instance is of integer type
        return z3instance.sort() == z3.IntSort()
    elif flag == "must_not_iter":
        # Check if the z3instance is not of any array type
        return z3.Not(z3.Or([z3instance.sort() == arr_type for arr_type in ARRTYPES]))
    elif flag == "must_str":
        # Check if the z3instance is of string type
        return z3instance.sort() == z3.StringSort()
    else:
        raise ValueError("Invalid type constraint flag")

class BaseWrapper():
    def __init__(self, const, datatype, value_func=None, len_func=None):
        self.const : z3.Datatype = const
        self.datatype : z3.Datatype = datatype
        self.value_func = value_func
        self.len_func = len_func
        self.info = {

        }
    
    def evaluate(self, model, instance, attr : str = "len", idx : int = None) : 
        if attr in ["len", "rank"] : 
            return model.evaluate(self.len_func(instance))
        elif attr in ["value", "shape"] : 
            return model.evaluate(self.value_func(instance)[idx])
        elif attr in ["dtype"] : 
            return model.evaluate(self.dtype_func(instance))
            return self.dtype_func(model_value)
        
    def corrected_idx(self, idx, datatype = None, len_func_name = "len"):
        """ 
        return corrected idx for possible negative index
        """ 
        if datatype is None : datatype = self.datatype
        if len_func_name is None : len_func = self.len_func
        else : len_func = getattr(datatype, len_func_name)
        return z3.If(idx >= 0, idx, len_func(self.const) + idx)  
    
    def len(self): 
        if self.info.get("sliced", None) is None :
            return self.len_func(self.const)
        else :
            return self.info["sliced"][1] - self.info["sliced"][0]
        
    def __repr__(self):
        return self.__str__()
    
    @property
    def value(self):
        return self.value_func(self.const)
     
    def __getitem__(self, idx):
        return self.value[idx]

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
        if self.info.get("sliced", None) is None : 
            return (0, self.rank)
        return self.info["sliced"]
    
class TensorWrapper(BaseWrapper):
    def __init__(self, const, datatype, value_func, len_func):
        super().__init__(const, datatype, value_func, len_func)
        self.dtype_func = datatype.dtype
     
    def __str__(self):
        return f"Tensor:{self.const.decl()}"
    
    @property
    def shape(self):
        return self.value_func(self.const)
    
    @property
    def dtype(self):
        return self.dtype_func(self.const)
class ArrWrapper(BaseWrapper):
    def __init__(self, const, datatype, value_func, len_func):
        super().__init__(const, datatype, value_func, len_func)
    
    def __str__(self):
        return f"Arr:{self.const.decl()}"
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
        return f"Complex:{self.const.decl()}"
    
    @property
    def rank(self):
        raise NotImplementedError
    
    def __lt__(self, __value: object) -> bool:
        pass
    
def load_z3_const(name, var_type, is_array=False):
    """
    Define a Z3 variable of a given type. 
    If it"s an array, specify the element type.
    """
    if var_type == "int":
        if is_array : 
            return ArrWrapper(z3.Const(name, IntArr), IntArr, IntArr.value, IntArr.len)
        else : 
            return z3.Const(name, z3.IntSort())
    elif var_type == "float":
        if is_array : 
            return ArrWrapper(z3.Const(name, FloatArr), FloatArr, FloatArr.value, FloatArr.len)
        else :
            return z3.Const(name, z3.RealSort())
    elif var_type == "str":
        if is_array : 
            return ArrWrapper(z3.Const(name, StrArr), StrArr, StrArr.value, StrArr.len)
        else :
            return z3.Const(name, z3.StringSort())
    elif var_type == "complex":
        if is_array : 
            return ArrWrapper(z3.Const(name, ComplexArr), ComplexArr, ComplexArr.value, ComplexArr.len)
        else :
            return ComplexWrapper(z3.Const(name, Complex), Complex)
    elif var_type == "bool":
        if is_array : 
            return ArrWrapper(z3.Const(name, BoolArr), BoolArr, BoolArr.value, BoolArr.len)
        else :
            return z3.Const(name, z3.BoolSort())
    elif var_type == "tensor":
        if is_array : 
            return ArrWrapper(z3.Const(name, TensorArr), 
                              TensorWrapper(z3.Const(name, TensorZ3), TensorZ3, TensorZ3.shape, TensorZ3.rank),
                              TensorArr.value, TensorArr.len)
        else :
            return TensorWrapper(z3.Const(name, TensorZ3), TensorZ3, TensorZ3.shape, TensorZ3.rank)
    else:
        raise ValueError("Unsupported variable type")

def is_wrapper(obj) : 
    return hasattr(obj, "get_wrapped_object")

def change_val_from_expr(expr, target, new_target):
    return z3.substitute(expr, (target, new_target))

class SMTFuncs:
    """
    Class to hold custom Z3 functions.
    """
    # Class variable to hold names of all functions
    function_names = ["all", "any", "len", "type", "sorted", "abs", "min", "max", "in_", "not_in", "rank", "range", "isinstance", "T"]
    _z3_dataarr = [(IntArr, (IntArr, IntArr.value, IntArr.len)),
                   (FloatArr, (FloatArr, FloatArr.value, FloatArr.len)),
                   (StrArr, (StrArr, StrArr.value, StrArr.len)),
                   (ComplexArr, (ComplexArr, ComplexArr.value, ComplexArr.len)),
                   (BoolArr, (BoolArr, BoolArr.value, BoolArr.len)),
                   (TensorArr, (TensorArr, TensorArr.value, TensorArr.len)),
                   (TensorZ3, (TensorZ3, TensorZ3.shape, TensorZ3.dtype, TensorZ3.rank))
                   ]
    iterables = []
    non_iterables= []
    rank_idx = 0
    value_idx = 1
    dtype_idx = 2
    def __init__(self) -> None:
        self.constrs = []

    @staticmethod
    def check_suff_conds(expr) :
        return expr == False
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
    @staticmethod
    def gen_sym(a):
        # Check if 'a' is a Z3 expression or sort
        return z3.Int(a)    
    @staticmethod 
    def is_func_applied(v) : 
        if hasattr(v, "num_args") :
            return v.num_args() > 0
        else : 
            return False
    @classmethod 
    def is_iterable(cls, v) :
        return cls._find_sort_by_id(
            v.decl().get_id(), pool = [
                                        IntArr.value,
                                        FloatArr.value,
                                        StrArr.value,
                                        ComplexArr.value,
                                        BoolArr.value,
                                        TensorArr.value,
                                        TensorZ3.shape,]
                                        ) is not None 
    
    @classmethod 
    def is_int(cls, v) :
        return cls._find_sort_by_id(v.sort().get_id(), pool = [z3.IntSort(), z3.RealSort()]) is not None 
    
    @classmethod 
    def is_str(cls, v) :
        return cls._find_sort_by_id(v.sort().get_id(), pool = [z3.StringSort()]) is not None 
    
    @classmethod
    def type(cls, v) : 
        raise IncompatiableConstrError(f"type function not supported for now.")
    
    @classmethod
    def isinstance(cls, v, *args) : 
        raise IncompatiableConstrError(f"isinstance function not supported for now.")
    
    @staticmethod
    def sort(v) : 
        if is_wrapper(v) : 
            return v.datatype.name()
        else : 
            return v.sort().name()
    
    def clear(self) :
        self.constrs.clear()

    @classmethod
    def _find_datatype_by_decl_id(cls, id, pool = None):
        """
        Helper function to get a Z3 datatype by its id.
        """
        if pool is None : pool = cls._z3_dataarr
        return next(((dv, dv.name()) for dv, attrs in pool if any(attr.get_id() == id for attr in attrs)), None)
    
    @classmethod
    def _find_sort_by_id(cls, id, pool = None):
        """
        Helper function to get a Z3 datatype by its id.
        """
        assert pool is not None, "pool must be specified"
        return next(((dv, dv.name()) for dv in pool if dv.get_id() == id), None)
    
    @staticmethod
    def all(input_array):
        """
        Custom "all" function for Z3 expressions.
        Returns z3.And of all elements in the input array.
        """
        assert all(isinstance(expr, z3.ExprRef) for expr in input_array), "All elements must be Z3 expressions"
        return z3.And(input_array)

    @staticmethod
    def get_syms_from_wrappers(*args) :
        args = list(args)
        for i in range(len(args)) : 
            if is_wrapper(args[i]) :
                args[i] = args[i].get_wrapped_object() 
        
        return args
    @staticmethod
    def get_syms_val_from_wrappers(*args) :
        args = list(args)
        for i in range(len(args)) : 
            if is_wrapper(args[i]) :
                args[i] = args[i].value
        
        return args

    @staticmethod
    def any(input_array):
        """
        Custom "any" function for Z3 expressions.
        Returns z3.Or of any element in the input array.
        """
        assert all(isinstance(expr, z3.ExprRef) for expr in input_array), "All elements must be Z3 expressions"
        return z3.Or(input_array)


    @staticmethod
    def is_sym(a):
        # Check if "a" is a Z3 expression or sort
        is_z3_obj = isinstance(a, (z3.ExprRef, z3.SortRef))
        return is_z3_obj

    @staticmethod
    def get_name(a) : 
        if is_wrapper(a) :
            return a.name
        if a.num_args() == 0 : 
            return a.decl().name()
        else :
            return SMTFuncs.get_name(a.arg(0))

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
        if isinstance(v, str) : 
            raise IncorrectConstrError(f"the name \"{v}\" may not in the related args?")
        datatype = None
        is_attr = True
        datatype = cls._find_datatype_by_decl_id(v.decl().get_id())
        if datatype is None :
             # Tensor.shape - dirty fix
            is_attr = False
            datatype = cls._find_datatype_by_decl_id(v.decl().range().get_id())

        if datatype is None :
            raise IncompatiableConstrError(f"Unsupported datatype : {v.sort()} - {v.decl().name()}")
        else : 
            obj, name = datatype
            v = v.arg(0) if is_attr else v
            if name == TensorZ3.name() : 
                return obj.rank(v), obj.shape(v), obj.dtype(v)
            # elif v.decl().name() == "shape" : #shape(t) -> load name "t" -> TensorZ3.rank(t)
            #     return TensorZ3.rank(v), v, TensorZ3.dtype(v)
            else : 
                return obj.len(v), obj.value(v)
    @classmethod
    def is_tensor(cls, v):
        return cls.sort(v) == TensorZ3.name()
   
    @classmethod 
    def T(cls, v) : 
        raise IncompatiableConstrError(f"T function not supported for now.")
        assert isinstance(v, z3.ArrayRef) 
        len_var = cls.len(v) 

    @classmethod
    def shape(cls, v):
        """
        Custom "len" function for Z3 arrays and tensors.
        Returns the length of an array or the rank of a tensor.
        """
        if hasattr(v, "shape") :
            return v.shape
        else :
            attrs = cls._load_attrs(v)
            assert len(attrs) > cls.value_idx, f"{v} seems not be Tensor object"
            return attrs[cls.value_idx]
    
    @classmethod
    def size(cls, v, idx = None):
        res = cls.shape(v) 
        if idx is not None :
            res = res[cls.get_corrected_idx(idx, v)]
        return res
            
    @classmethod
    def dtype(cls, v):
        if hasattr(v, "dtype") :
            return v.dtype
        else :
            attrs = cls._load_attrs(v)
            assert len(attrs) > cls.dtype_idx, f"{v} seems not be Tensor object"
            return attrs[cls.dtype_idx]
    @classmethod
    def rank(cls, v) : 
        return cls.len(v)
    
    @classmethod
    def len(cls, v):
        """
        Custom "len" function for Z3 arrays and tensors.
        Returns the length of an array or the rank of a tensor.
        """
        if hasattr(v, "len") :
            return v.len()
        else :
            attrs = cls._load_attrs(v)
            assert len(attrs) > cls.rank_idx, f"{v} seems not be Tensor object"
            return attrs[cls.rank_idx]
    @classmethod
    def dim(cls, v, idx=None) :
        return cls.len(v)
    @classmethod
    def ndims(cls, v) :
        return cls.dim(v)
    @classmethod
    def ndim(cls, v) :
        return cls.dim(v)
    @staticmethod
    def in_(a,b) : 
        from specloader.irs import SYM_LEN
        if isinstance(b[0], z3.ArrayRef) : 
            b_len = SYM_LEN[b[0].arg(0).decl().name()] if b[0].num_args() else SYM_LEN[b[0].decl().name()]
            i = z3.Int("i")
            exists = z3.Exists([i], z3.Implies(z3.And(i >= 0, i < b_len), b[0][i] == a))
            return exists
        elif hasattr(b, "__len__") :
            return z3.Or([a == v for v in b])
        else :
            return z3.Or([a == b[i] for i in range(MAX_ARR_LEN)])
    
    @staticmethod
    def not_in(a,b) : 
        if isinstance(b[0], z3.ArrayRef) : 
            from specloader.irs import SYM_LEN
            b_len = SYM_LEN[b[0].arg(0).decl().name()] if b[0].num_args() else SYM_LEN[b[0].decl().name()]
            i = z3.Int("i")
            exists = z3.ForAll([i], z3.Implies(z3.And(i >= 0, i < b_len), b[0][i] != a))
            return exists
        elif hasattr(b, "__len__") :
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
            # if len(vs) == 1 :
            #     if self.is_sym(vs[0]) and self.is_iterable(vs[0]) :
            #         return self.min(vs[0])
            #     return vs[0]
            m = vs[0]
            for v in vs[1:]:
                m = z3.If(v < m, v, m)
            return m
        else : 
            vs = self.get_syms_val_from_wrappers(*vs)
            vs = vs[0]
            attrs = self._load_attrs(vs)
            lev_var = attrs[0]
            values = attrs[1]
            
            min_idx = z3.Int("min_idx")
            assert values[min_idx].sort().name() not in ["complex", "tensor"], f"{values[min_idx].sort().name()} dont support min func"
            i = z3.Int("i")
            min_const = z3.ForAll(
                [i], 
                z3.Implies(z3.And(i >= 0, i < lev_var), 
                        values[min_idx] <= values[i]))
            exists_in_array = z3.And(min_idx >= 0, min_idx < lev_var)
            self.constrs.append(z3.And(exists_in_array, min_const))
            return values[min_idx]

    def max(self, *vs):
        if len(vs) > 1 :
            if len(vs) == 1 :
                return vs[0]
            m = vs[0]
            for v in vs[1:]:
                m = z3.If(v > m, v, m)
            return m
        else : 
            vs = self.get_syms_val_from_wrappers(*vs)
            vs = vs[0]
            attrs = self._load_attrs(vs)
            lev_var = attrs[0]
            values = attrs[1]
            
            max_idx = z3.Int("max_idx")
            assert values[max_idx].sort().name() not in ["complex", "tensor"], f"{values[max_idx].sort().name()} dont support max func"
            i = z3.Int("i")
            max_const = z3.ForAll(
                [i], 
                z3.Implies(z3.And(i >= 0, i < lev_var), 
                        values[max_idx] >= values[i]))
            exists_in_array = z3.And(max_idx >= 0, max_idx < lev_var)
            self.constrs.append(z3.And(exists_in_array, max_const))
            return values[max_idx]

