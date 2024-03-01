import collections
from enum import Enum, unique
from functools import partial, reduce
import random
from typing import Any, Callable, Dict, List, Literal, Tuple, Union, get_args, get_origin
import numpy as np
from z3 import Const, BoolSort, IntSort, RealSort, StringSort, Array, Datatype, Const
from neuri.abstract.arith import ConstraintCheck, SanityCheck, nnsmith_eq, nnsmith_ge, nnsmith_gt, nnsmith_mul, z3
from neuri.error import ConstraintCheck, SanityCheck
from neuri.logger import LOGGER


@unique
class DType(Enum):
    qint8 = "qint8"
    qint16 = "qint16"
    qint32 = "qint32"
    quint8 = "quint8"
    bfloat16 = "bfloat16"
    float16 = "float16"
    float32 = "float32"
    float64 = "float64"
    uint8 = "uint8"  # Support quantized models.
    uint16 = "uint16"
    uint32 = "uint32"
    uint64 = "uint64"
    int8 = "int8"
    int16 = "int16"
    int32 = "int32"
    int64 = "int64"
    bool = "bool"
    complex32 = "complex32"
    complex64 = "complex64"
    complex128 = "complex128"
    __all__ = [
        "quint8",
        "qint8",
        "qint16",
        "qint32",
        "bfloat16",
        "float16",
        "float32",
        "float64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "int8",
        "int16",
        "int32",
        "int64",
        "bool",
        "complex32",
        "complex64",
        "complex128",
    ]
    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        s = super().__str__()
        assert s.startswith("DType."), s
        return s[len("DType.") :]

    def short(self) -> str:
        return {
            DType.quint8: "qu8",
            DType.qint8: "q8",
            DType.qint16: "q16",
            DType.qint32: "q32",
            DType.bfloat16: "bf16",
            DType.float16: "f16",
            DType.float32: "f32",
            DType.float64: "f64",
            DType.uint8: "u8",
            DType.uint16: "u16",
            DType.uint32: "u32",
            DType.uint64: "u64",
            DType.int8: "i8",
            DType.int16: "i16",
            DType.int32: "i32",
            DType.int64: "i64",
            DType.complex32: "c32",
            DType.complex64: "c64",
            DType.complex128: "c128",
            DType.bool: "b",
        }[self]

    @staticmethod
    def is_float(dtype):  # Don't use string. Make it well-formed.
        assert isinstance(dtype, DType)
        return dtype in [DType.float32, DType.float64]

    @staticmethod
    def from_str(s):
        return {
            "qu8": DType.quint8,
            "q8": DType.qint8,
            "q16": DType.qint16,
            "q32": DType.qint32,
            "bf16": DType.bfloat16,
            "f16": DType.float16,
            "f32": DType.float32,
            "f64": DType.float64,
            "u8": DType.uint8,
            "i8": DType.int8,
            "i32": DType.int32,
            "i64": DType.int64,
            "c32": DType.complex32,
            "c64": DType.complex64,
            "c128": DType.complex128,
            "quint8": DType.quint8,
            "qint8": DType.qint8,
            "qint16": DType.qint16,
            "qint32": DType.qint32,
            "bfloat16": DType.bfloat16,
            "float16": DType.float16,
            "float32": DType.float32,
            "float64": DType.float64,
            "uint8": DType.uint8,
            "uint16": DType.uint16,
            "uint32": DType.uint32,
            "uint64": DType.uint64,
            "int8": DType.int8,
            "int16": DType.int16,
            "int32": DType.int32,
            "int64": DType.int64,
            "complex32": DType.complex32,
            "complex64": DType.complex64,
            "complex128": DType.complex128,
            "bool": DType.bool,
        }[s]
    def z3_const(self) -> str:
        from neuri.constrinf.smt_funcs import Z3DTYPE
        return {
            DType.quint8: Z3DTYPE.quint8,
            DType.float16: Z3DTYPE.float16,
            DType.float32: Z3DTYPE.float32,
            DType.float64: Z3DTYPE.float64,
            DType.uint8: Z3DTYPE.uint8,
            DType.uint16: Z3DTYPE.uint16,
            DType.uint32: Z3DTYPE.uint32,
            DType.uint64: Z3DTYPE.uint64,
            DType.qint8: Z3DTYPE.qint8,
            DType.qint16: Z3DTYPE.qint16,
            DType.qint32: Z3DTYPE.qint32,
            DType.bfloat16: Z3DTYPE.bfloat16,
            DType.int8: Z3DTYPE.int8,
            DType.int16: Z3DTYPE.int16,
            DType.int32: Z3DTYPE.int32,
            DType.int64: Z3DTYPE.int64,
            DType.complex32: Z3DTYPE.complex32,
            DType.complex64: Z3DTYPE.complex64,
            DType.complex128: Z3DTYPE.complex128,
            DType.bool: Z3DTYPE.bool,
        }[self]
    def numpy(self):
        return {
            # DType.qint8: "q8",
            # DType.qint16: "q16",
            # DType.qint32: "q32",
            # DType.bfloat16: np.bfloat16,
            # DType.complex32: "c32",
            DType.float16: np.float16,
            DType.float32: np.float32,
            DType.float64: np.float64,
            DType.uint8: np.uint8,
            DType.uint8: np.uint8,
            DType.uint16: np.uint16,
            DType.uint32: np.uint32,
            DType.uint64: np.uint64,
            DType.int8: np.int8,
            DType.int16: np.int16,
            DType.int32: np.int32,
            DType.int64: np.int64,
            DType.complex64: np.complex64,
            DType.complex128: np.complex128,
            DType.bool: np.bool_,

        }[self]

    # TODO(@ganler): put "torchization" in a separate file.
    def torch(self) -> "torch.dtype":
        import torch

        return {
            DType.float16: torch.float16,
            DType.float32: torch.float32,
            DType.float64: torch.float64,
            DType.uint8: torch.uint8,
            # PyTorch does not support other unsigned int types: https://github.com/pytorch/pytorch/issues/58734
            DType.int8: torch.int8,
            DType.int16: torch.int16,
            DType.int32: torch.int32,
            DType.int64: torch.int64,
            DType.bfloat16: torch.bfloat16,
            DType.complex32: torch.complex32,
            DType.complex64: torch.complex64,
            DType.complex128: torch.complex128,
            DType.bool: torch.bool,
        }[self]

    @staticmethod
    def from_torch(dtype) -> "DType":
        import torch

        return {
            torch.float16: DType.float16,
            torch.float32: DType.float32,
            torch.float64: DType.float64,
            torch.bfloat16: DType.bfloat16,
            torch.uint8: DType.uint8,
            torch.int8: DType.int8,
            torch.int16: DType.int16,
            torch.int32: DType.int32,
            torch.int64: DType.int64,
            torch.complex32: DType.complex32,
            torch.complex64: DType.complex64,
            torch.complex128: DType.complex128,
            torch.bool: DType.bool,
        }[dtype]


    def tensorflow(self) :
        import tensorflow as tf

        return {
            DType.quint8: tf.quint8,
            DType.bfloat16: tf.bfloat16,
            DType.float16: tf.float16,
            DType.float32: tf.float32,
            DType.float64: tf.float64,
            DType.qint8: tf.qint8,
            DType.qint16: tf.qint16,
            DType.qint32: tf.qint32,
            DType.uint8: tf.uint8,
            DType.uint16: tf.uint16,
            DType.uint32: tf.uint32,
            DType.uint64: tf.uint64,
            DType.int8: tf.int8,
            DType.int16: tf.int16,
            DType.int32: tf.int32,
            DType.int64: tf.int64,
            DType.complex64: tf.complex64,
            DType.complex128: tf.complex128,
            DType.bool: tf.bool,
        }[self]

    @staticmethod
    def from_tensorflow(dtype) -> "DType":
        import tensorflow as tf

        return {
            tf.quint8: DType.quint8,
            tf.bfloat16: DType.bfloat16,
            tf.float16: DType.float16,
            tf.float32: DType.float32,
            tf.float64: DType.float64,
            tf.qint8: DType.qint8,
            tf.qint16: DType.qint16,
            tf.qint32: DType.qint32,
            tf.uint8: DType.uint8,
            tf.uint16: DType.uint16,
            tf.uint32: DType.uint32,
            tf.uint64: DType.uint64,
            tf.int8: DType.int8,
            tf.int16: DType.int16,
            tf.int32: DType.int32,
            tf.int64: DType.int64,
            tf.complex64: DType.complex64,
            tf.complex128: DType.complex128,
            tf.bool: DType.bool,
        }[dtype]

    def sizeof(self) -> int:
        return {
            DType.bfloat16: 2,
            DType.quint8: 1,
            DType.qint8: 1,
            DType.qint16: 2,
            DType.qint32: 4,
            DType.float16: 2,
            DType.float32: 4,
            DType.float64: 8,
            DType.uint8: 1,
            DType.uint16: 2,
            DType.uint32: 4,
            DType.uint64: 8,
            DType.int8: 1,
            DType.int16: 2,
            DType.int32: 4,
            DType.int64: 8,
            DType.complex64: 8,
            DType.complex128: 16,
            DType.bool: 1,  # Follow C/C++ convention.
        }[self]
@unique
class AbsDType(Enum):
    bool = "bool"
    int = "int"
    float = "float"
    str = "str"
    complex = "complex"
    none = "None"
    def __repr__(self) -> str:
        return "AbsDType:" + str(self.name)
    def init(self) -> None :
        pass 
    def is_iter(self) -> bool :
        return False
    def to_str(self) -> Any :
        return {
            AbsDType.bool: 'bool',
            AbsDType.int: 'int',
            AbsDType.float: 'float',
            AbsDType.str: 'str',
            AbsDType.complex: 'complex',
            AbsDType.none: 'None',
        }[self]
    def _z3(self) -> "z3.Dtype" :
        return {
            AbsDType.bool: BoolSort(),
            AbsDType.int: IntSort(),
            AbsDType.float: RealSort(),
            AbsDType.complex: RealSort(),
            AbsDType.str: StringSort(),
            AbsDType.none: None,
        }[self]
    def z3(self) -> "z3.Dtype" :
        from neuri.constrinf.ast2z3 import load_z3_const
        z3_load_func = partial(load_z3_const, var_type=self.to_str(), is_array=False)
        return z3_load_func
    def to_iter(self) -> "AbsIter" :
        return AbsIter([self])
    def get_arg_dtype(self) : 
        return self
    def z3_const(self) -> List[DType]:
        return {
            AbsDType.bool: [dtype.z3_const() for dtype in [DType.bool]],
            AbsDType.int: [dtype.z3_const() for dtype in [DType.int32, DType.int64, DType.int8, DType.int16]],
            AbsDType.float: [dtype.z3_const() for dtype in [DType.float16,DType.float32,DType.float64]],
            AbsDType.complex: [dtype.z3_const() for dtype in [DType.complex64, DType.complex128]],
            AbsDType.none: [None],
        }[self]
    def get_tensor_dtypes(self) -> List[DType]:
        return {
            AbsDType.bool: [DType.bool],
            AbsDType.int: [DType.int32, DType.int64, DType.int8, DType.int16],
            AbsDType.float: [DType.float16,DType.float32,DType.float64],
            AbsDType.complex: [DType.complex64, DType.complex128],
            AbsDType.none: [None],
        }[self]
    
class AbsIter():
    def __init__(self, values : List[AbsDType]):
        self.values = values
        self.length = len(values)
        self.arg_type = values[0]() if isinstance(values[0], Callable) else values[0]
        self._values = values
        self._length = len(values)
        self._arg_type = values[0]
    def init(self) : 
        self.values = self._values
        self.length = self._length
        self.arg_type =self._arg_type

    def set_length(self, length : int) :
        self.length = length
        self.values = [self.arg_type for i in range(self.length)]
    def is_iter(self) -> bool :
        return True
    def get_arg_dtype(self) : 
        return self.arg_type
    def __repr__(self) -> str:
        return f"AbsIter:{self.arg_type}:{self.length}"
    def to_str(self) -> str:
        return f"list[{self.arg_type.to_str()}]"
    def z3(self) -> "z3.Dtype" :
        from neuri.constrinf.ast2z3 import load_z3_const
        z3_load_func = partial(load_z3_const, 
                               var_type=self.arg_type.to_str(), 
                               is_array=True)
        return z3_load_func
    def z3_const(self) -> "z3.Dtype" :
        """
        Not IMPLEMENTED YET
        """
        from neuri.constrinf.ast2z3 import load_z3_const
        z3_load_func = partial(load_z3_const, 
                               var_type=self.arg_type.to_str(), 
                               is_array=True)
        return z3_load_func
    
class AbsLiteral() :
    def __init__ (self, choices : List[str]):
        self.choices = choices 
    def init(self) : pass 
    def is_iter(self) -> bool :
        return False 
    def __repr__(self) -> str:
        return f"Literal:{self.choices}"
    def to_str(self) -> str:
        return f"Literal{list(self.choices)}"
    def get_arg_dtype(self) :
        return self.choices
    def z3(self) -> "z3.Dtype" :

        # def literal(arg_name) :
        #     DType = Datatype('literal')
        #     for choice in self.choices :
        #         DType.declare(choice)
        #     DType = DType.create()
        #     return [Const(arg_name, DType), DType]
        # return literal
        return AbsDType.str.z3()

# "DTYPE_GEN*" means data types used for symbolic generation.
# "DTYPE_GEN_ALL" is surely a subset of all types but it is
# used to conservatively to avoid unsupported data types while
# applying nnsmith to various frameworks.
DTYPE_INCOMMON = [DType.uint16, DType.uint32, DType.uint64]
DTYPE_GEN_ALL = [e for e in DType if e not in DTYPE_INCOMMON]
DTYPE_GEN_NON_BOOL = [dtype for dtype in DTYPE_GEN_ALL if dtype != DType.bool]
DTYPE_GEN_FLOATS = [DType.float16, DType.float32, DType.float64]
DTYPE_GEN_INTS = [
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
    DType.uint8,
]

DTYPE_NOT_SUPPORTED : Dict[str, List[DType]] = { 
    "numpy" : [
        DType.bfloat16,
        DType.complex32,
        DType.qint8,
        DType.qint16,
        DType.qint32,
    ],
    "torch" : [
    DType.qint8,
    DType.quint8,
    DType.qint16,
    DType.qint32,
    DType.uint8,
    DType.uint16,
    DType.uint32,
    DType.uint64,
    ],
    "tensorflow":[
        DType.complex32
    ]
}

DTYPE_ALL = {
    "torch" : [
            DType.float16,
            DType.float32,
            DType.float64,
            DType.bfloat16,
            DType.int8,
            DType.int16,
            DType.int32,
            DType.int64,
            DType.complex32,
            DType.complex64,
            DType.complex128,
            DType.bool,
    ],
    "tensorflow" :
    [
            DType.bfloat16,
            DType.float16,
            DType.float32,
            DType.float64,
            DType.qint8,
            DType.qint16,
            DType.qint32,
            DType.uint8,
            DType.uint16,
            DType.uint32,
            DType.uint64,
            DType.int8,
            DType.int16,
            DType.int32,
            DType.int64,
            DType.complex64,
            DType.complex128,
            DType.bool,
    ]
}

TYPE_TO_ABS = {
    ## __origin__ of typing object
    int : AbsDType.int,
    float : AbsDType.float,
    bool : AbsDType.bool,
    str : AbsDType.str,
    list : AbsIter([AbsDType.int]),
    tuple : AbsIter([AbsDType.int]),
    type(None) : AbsDType.none,
}


class AbsTensor:
    def __init__(self,
                 shape: List[Union[int, z3.ExprRef]] = [],
                 dtype: DType = None,
                 possible_dtypes : List[DType] = [],
                 **kwargs):
        assert isinstance(
            shape, (list, tuple)
        ), f"Shape must be a list/tuple, but got {shape}"
        self.shape = list(shape)
        self.rank = len(self.shape)
        # assert possible_dtypes or dtype, "Must provide dtype or possible_dtypes"
        self.possible_dtypes : List[DType] = possible_dtypes
        self.dtype = dtype

    @staticmethod
    def from_numpy(x: "np.ndarray") -> "AbsTensor":
        return AbsTensor(list(x.shape), str(x.dtype))
    @staticmethod
    def to_str() -> Any :
        return 'tensor'
    def downcast_rank(self):
        return AbsTensor(shape=[None] * self.ndims, dtype=self.dtype)

    def concrete_shape(self, symb_2_value: Dict[str, Any]) -> List[int]:
        return [symb_2_value[s] for s in self.shape]

    def set_shape(self, shape: List[Union[int, z3.ExprRef]]):
        self.shape = shape
        if len(self.shape) != self.rank:
            self.rank = len(self.shape)

    def concretize(
        self,
        symb_2_value: Dict[str, Any],
        tensor_from_numpy: Callable = lambda x: x,
        only_shape : bool = False,
        *args,
        **kwargs,
    ):
        if only_shape :
            return AbsTensor(shape=self.concrete_shape(symb_2_value), dtype=self.dtype)
        from neuri.autoinf.instrument.utils import (
            numpy_random,
        )
        shape = [symb_2_value[s] for s in self.shape]
        return tensor_from_numpy(numpy_random(shape, str(self.dtype)))

    def concretize_with_concrete_values(
        self,
        tensor_from_numpy: Callable = lambda x: x,
    ):
        from neuri.autoinf.instrument.utils import (
            numpy_random,
        )
        return tensor_from_numpy(numpy_random(self.shape, str(self.dtype)))

    def __hash__(self) -> int:
        return hash((tuple(self.shape), self.dtype))

    def __repr__(self) -> str:

        if self.dtype is None :
            return f"AbsTensor<null[{str(len(self.possible_dtypes))}]>{str(self.shape)}"
        else :
            return f"AbsTensor<{self.dtype.short()}>{str(self.shape)}"

    def pretty(self) -> str:
        return f"{self.dtype.short()}{self.shape}"

    def weak_compare(self, other: "AbsTensor") -> bool:
        if self.dtype != other.dtype or self.ndims != other.ndims:
            return False
        for l, r in zip(self.shape, other.shape):
            if isinstance(l, z3.ExprRef) or isinstance(r, z3.ExprRef):
                continue
            if l != r:
                return False
        return True

    def strong_compare(self, other: "AbsTensor") -> bool:
        return self.shape == other.shape and self.dtype == other.dtype

    def __eq__(self, other: "AbsTensor") -> bool:
        return isinstance(other, AbsTensor) and self.strong_compare(other)

    def ge_zero(self):
        ret = []
        for s in self.shape:
            if isinstance(s, z3.ExprRef):
                ret.append(nnsmith_ge(s, 0))
            else:
                ConstraintCheck.ge(s, 0)
        return ret

    def sym_gt_conc_ge_zero(self):
        ret = []
        for s in self.shape:
            if isinstance(s, z3.ExprRef):
                ret.append(nnsmith_gt(s, 0))
            else:
                ConstraintCheck.ge(s, 0)
        return ret

    def gt_zero(self):
        ret = []
        for s in self.shape:
            if isinstance(s, z3.ExprRef):
                ret.append(nnsmith_gt(s, 0))
            else:
                ConstraintCheck.gt(s, 0)
        return ret

    def eq(self, other):
        SanityCheck.eq(self.ndims, other.ndims)
        ret = []
        for i in range(self.ndims):
            if isinstance(self.shape[i], z3.ExprRef) or isinstance(
                other.shape[i], z3.ExprRef
            ):
                ret.append(nnsmith_eq(self.shape[i], other.shape[i]))
            else:
                ConstraintCheck.eq(self.shape[i], other.shape[i])
        return ret

    def torch(self):
        import torch

        return torch.Size(self.shape)
    def consistent_constr(self, other : str) -> List[z3.BoolRef] :
        """ 
        generate constraints that ensure the shape, rank, and dtype as consistent with
        the name of $other tensor
        """
        ## gen z3 var 
        other_obj = self.z3()(other)
        ## rank consistent constr 
        rank_cons = [other_obj.rank == self.ndims]
        ## shape consistent constr 
        shape_cons = [
            other_obj.shape[i] == self.shape[i] for i in range(self.ndims)
        ]
        ## dtype consistent constr
        dtype_cons = [other_obj.dtype == self.dtype.z3_const()]

        return rank_cons + shape_cons + dtype_cons

    @classmethod
    def z3(cls) -> "z3.Dtype" :
        from neuri.constrinf.ast2z3 import load_z3_const
        z3_load_func = partial(load_z3_const,
                        var_type=cls.to_str(),
                        is_array=False)

        return z3_load_func

    def constains_symbol(self) -> bool:
        return any(isinstance(s, z3.ExprRef) for s in self.shape)

    def nelement(self):
        if len(self.shape) == 0:  # Scalar
            return 1
        return reduce(lambda x, y: nnsmith_mul(x, y), self.shape, 1)

    def nbytes(self) -> int:
        return self.nelement() * self.dtype.sizeof()

    def deepcopy(self):
        return AbsTensor(shape=list(self.shape), dtype=self.dtype)
    @staticmethod
    def to_iter() :
        from neuri.abstract.dtype import AbsIter
        return AbsIter([AbsTensor])
    @property
    def ndims(self):
        return len(self.shape)

    def is_concrete(self) -> bool:
        return all(isinstance(s, int) for s in self.shape)

    def htype(self):  # High-level type
        return (self.dtype, self.ndims)
STR_TO_ABS = {
    # AbsDType
    'float': AbsDType.float,
    'real': AbsDType.float,
    'complex': AbsDType.complex,
    'floats': AbsDType.float,
    'floating': AbsDType.float,
    'int': AbsDType.int,
    'ints': AbsDType.int,
    'integer': AbsDType.int,
    'numeric': AbsDType.int,
    'number': AbsDType.int,
    'bool': AbsDType.bool,
    'boolean': AbsDType.bool,
    'str': AbsDType.str,
    'strings': AbsDType.str,
    'string': AbsDType.str,
    'tf.str': AbsDType.str,
    'torch.str': AbsDType.str,
    'list[int]': AbsDType.int.to_iter(),
    'list[bool]': AbsDType.bool.to_iter(),
    'list[float]': AbsDType.float.to_iter(),
    'list[complex]': AbsDType.complex.to_iter(),
    'list[str]': AbsDType.str.to_iter(),
    'list[none]': AbsDType.none.to_iter(),
    'list[tensor]': AbsTensor().to_iter(),
    'list': AbsDType.int.to_iter(),
    'lists': AbsDType.int.to_iter(),
    'array': AbsDType.int.to_iter(),
    'arrays': AbsDType.int.to_iter(),
    'vector': AbsDType.int.to_iter(),
    'vectors': AbsDType.int.to_iter(),
    'tuple': AbsDType.int.to_iter(),
    'array_like': AbsDType.int.to_iter(),
    'sequence[tensor]': AbsTensor().to_iter(),
    'sequence[int]': AbsDType.int.to_iter(),
    'sequences[int]': AbsDType.int.to_iter(),
    'sequence': AbsDType.int.to_iter(),
    'sequences': AbsDType.int.to_iter(),
    'optional[number]': [AbsDType.int, AbsDType.none],
    # DType
    #qint

    'dt_uint8': DType.uint8,
    'dt_int16': DType.int16,
    'dt_int8': DType.int8,
    'dt_complex64': DType.complex64,
    'dt_bool': DType.bool,
    'dt_qint8': DType.qint8,
    'dt_quint8': DType.qint8,
    'dt_qint32': DType.qint32,
    'dt_uint16': DType.uint16,
    'dt_complex128': DType.complex128,
    'dt_uint32': DType.uint32,
    'dt_uint64': DType.uint64,
    'dt_float8_e5m2': DType.float32,
    'dt_float8_e4m3fn': DType.float32,
    'dt_int4': DType.uint8,
    'dt_uint4': DType.uint8,
    'dt_uint8': DType.uint8,
    'dt_int16': DType.int16,
    'dt_int8': DType.int8,
    'dt_complex64': DType.complex64,
    'complexdouble': DType.complex128,
    'dt_bool': DType.bool,
    'dt_qint8': DType.qint8,
    'dt_quint8': DType.qint8,
    'dt_qint32': DType.qint32,
    'dt_uint16': DType.uint16,
    'dt_complex128': DType.complex128,
    'dt_uint32': DType.uint32,
    'dt_uint64': DType.uint64,
    'dt_float8_e5m2': DType.float32,
    'dt_float8_e4m3fn': DType.float32,
    'dt_int4': DType.uint8,
    'dt_uint4': DType.uint8,
    'dt_uint8': DType.uint8,
    'dt_int16': DType.int16,
    'dt_int8': DType.int8,
    'dt_complex64': DType.complex64,
    'dt_bool': DType.bool,
    'dt_qint8': DType.qint8,
    'dt_quint8': DType.qint8,
    'dt_qint32': DType.qint32,
    'dt_uint16': DType.uint16,
    'dt_complex128': DType.complex128,
    'dt_uint32': DType.uint32,
    'dt_uint64': DType.uint64,
    'dt_float8_e5m2': DType.float32,
    'dt_float8_e4m3fn': DType.float32,
    'dt_int4': DType.int8,
    'dt_uint4': DType.uint8,
    'tf.qint8': DType.qint8,
    'torch.qint8': DType.qint8,
    'qint8': DType.qint8,
    'tf.qint16': DType.qint16,
    'torch.qint16': DType.qint16,
    'qint16': DType.qint16,
    'tf.qint32': DType.qint32,
    'torch.qint32': DType.qint32,
    'qint32': DType.qint32,
    #float,bfloat
    'dt_float': DType.float32,
    'dt_double': DType.float64,
    'dt_bfloat16': DType.bfloat16,
    'dt_half' : DType.float16,
    'tf.float': DType.float32,
    'torch.float': DType.float32,
    'tf.float16': DType.float16,
    'torch.float16': DType.float16,
    'float16': DType.float16,
    'half': DType.float16,
    'double': DType.float64,
    'tf.float32': DType.float32,
    'torch.float32': DType.float32,
    'float32': DType.float32,
    'tf.float64': DType.float64,
    'torch.float64': DType.float64,
    'float64': DType.float64,
    'bfloat': DType.bfloat16,
    'bfloat16': DType.bfloat16,
    'tf.bfloat': DType.bfloat16,
    'torch.bfloat': DType.bfloat16,
    'tf.bfloat16': DType.bfloat16,
    'torch.bfloat16': DType.bfloat16,
    #uint
    'uint8': DType.uint8,
    'tf.uint8': DType.uint8,
    'torch.uint8': DType.uint8,
    'uint16': DType.uint16,
    'tf.uint16': DType.uint16,
    'torch.uint16': DType.uint16,
    'uint32': DType.uint32,
    'tf.uint32': DType.uint32,
    'torch.uint32': DType.uint32,
    'uint64': DType.uint64,
    'tf.uint64': DType.uint64,
    'torch.uint64': DType.uint64,
    #int
    "quint8": DType.quint8,
    "dt_quint8": DType.quint8,
    'tf.int': DType.int32,
    'torch.int': DType.int32,
    'dt_int32': DType.int32,
    'dt_int64': DType.int64,
    'int8': DType.int8,
    'tf.int8': DType.int8,
    'torch.int8': DType.int8,
    'int16': DType.int16,
    'tf.int16': DType.int16,
    'torch.int16': DType.int16,
    'int32': DType.int32,
    'tf.int32': DType.int32,
    'torch.int32': DType.int32,
    'int64': DType.int64,
    'tf.int64': DType.int64,
    'torch.int64': DType.int64,
    #bool&complex
    'tf.boolean': DType.bool,
    'torch.boolean': DType.bool,
    'tf.bool': DType.bool,
    'torch.bool': DType.bool,
    'tf.complex': DType.complex64,
    'tf.complex32': DType.complex32,
    'tf.complex64': DType.complex64,
    'tf.complex128': DType.complex128,
    'torch.complex': DType.complex64,
    'torch.complex64': DType.complex64,
    'complex64': DType.complex64,
    'complex32': DType.complex32,
    'torch.complex128': DType.complex128,
    'complex128': DType.complex128,

    # AbsTensor and List[AbsTensor]
    'tf.tensor': AbsTensor(),
    'torch.tensor': AbsTensor(),
    'tensor': AbsTensor(),
    'tf.tensors': AbsTensor(),
    'torch.tensors': AbsTensor(),
    'longtensor': AbsTensor(),
    'sequence of Tensors':  AbsTensor.to_iter(),
    'list[tensor]':  AbsTensor.to_iter(),
    'sequence[tensor]':  AbsTensor.to_iter(),
    'sequences[tensor]':  AbsTensor.to_iter(),

    # torch 
    'complexfloat' : AbsDType.complex,
    'cfloat' : DType.complex64,
    'cdouble' : DType.complex128,
    'short' : DType.int16,
    'long' : DType.int64,
}


def typing_to_abs(dtype : Any) -> Any :
    origin = get_origin(dtype)
    if origin is Union :
        abs_types = [typing_to_abs(dtype_arg) for dtype_arg in get_args(dtype)]
        return abs_types
    elif origin is Tuple :
        tuple_types = get_args(dtype)
        if (...) in tuple_types :
            length = random.randint(MIN_LENGTH, MAX_LENGTH)
        else :
            length = len(tuple_types)

        tuple_types = tuple_types[0]
        return AbsIter(values = [typing_to_abs(tuple_types) for _ in length])
    elif origin in [List, collections.abc.Sequence, list] :
        list_type = get_args(dtype)[0]
        return AbsIter(values = [typing_to_abs(list_type)])

    elif origin is Literal :
        args = get_args(dtype)
        return AbsLiteral(args)
    elif origin is None :
        if dtype in TYPE_TO_ABS.keys() :
            return TYPE_TO_ABS[dtype]
        else :
            raise NotImplementedError(f"Unsupported type {dtype}")
    else :
        return None


def materalize_typing_obj(target_str : str) :
    try :
        obj = eval(target_str)
        return typing_to_abs(obj)
    except :
        return None


def materalize_dtype(target_str : str) -> Any :

    target_str = target_str.replace("null", "None")\
                            .replace('`','')\
                            .replace(' ','')\
                            .replace('\'','')\
                            .replace('\"','')

    lowered = target_str.lower()
    if lowered in STR_TO_ABS.keys() :
        materalized = STR_TO_ABS[lowered]
        return materalized
    else :
        LOGGER.debug(f"Unsupported type {target_str}, may be literal arg")
        return target_str


def materalize_dtypes(dtypes : str, merge_tensor : bool = True) -> List[Any] :
    targets : List[str] = []
    res : List[Any] = []
    typing_obj = materalize_typing_obj(dtypes)
    if typing_obj is not None :
        return typing_obj if isinstance(typing_obj, list) else [typing_obj]
    dtypes = str(dtypes)
    # if "tensor" in dtypes.lower() : 
    #     dtypes = "tensor"
    dtypes = dtypes.replace(':',',').replace('->',',').replace('-',',')
    if dtypes.startswith('[') and dtypes.endswith(']') or \
        dtypes.startswith('(') and dtypes.endswith(')') or \
        dtypes.startswith('{') and dtypes.endswith('}') :
        dtypes = dtypes[1:-1]
    if ' or ' in dtypes :
        for splited in dtypes.split(' or ') :
            targets.append(splited)
    elif ',' in dtypes :
        for splited in dtypes.split(',') :
            targets.append(splited)
    else :
        targets.append(dtypes)

    for target_str in targets :
        dtype = materalize_dtype(target_str)
        if dtype is not None :
            if isinstance(dtype, list) :
                res.extend(dtype)
            else :
                res.append(dtype)

    LOGGER.debug(f"form {dtypes} -> to {res}")
    if not merge_tensor :
        return res

    to_merge = []
    final = []
    for abs in res :
        if isinstance(abs, DType) :
            to_merge.append(abs)
        elif isinstance(abs, AbsDType) :
            if any([isinstance(ele, DType) for ele in res]) :
                to_merge.extend(abs.get_tensor_dtypes())
            else :
                final.append(abs)
        elif isinstance(abs, AbsTensor) :
            if any([isinstance(ele, DType) for ele in res]) :
                pass
            else :
                final.append(abs)
        elif isinstance(abs, (AbsIter, AbsLiteral)) :
            final.append(abs)
        else :
            pass
    if len(to_merge) > 0 :
        final.append(AbsTensor(possible_dtypes=to_merge))
    if len(final) == 0 :
        return None
    else :
        LOGGER.debug(f"[merged] from {dtypes} To {final}")
        return final