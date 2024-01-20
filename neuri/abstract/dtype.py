from enum import Enum, unique
from typing import Any, Dict, List

import numpy as np
from z3 import Const, BoolSort, IntSort, RealSort, StringSort, Array, Datatype, Const



@unique
class DType(Enum):
    qint8 = "qint8"
    qint16 = "qint16"
    qint32 = "qint32"
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

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        s = super().__str__()
        assert s.startswith("DType."), s
        return s[len("DType.") :]

    def short(self) -> str:
        return {
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
    def z3(self) -> str:
        from specloader import Z3DTYPE
        return {
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
            # DType.bfloat16: "bf16",
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


    def tensorflow(self) -> "tf.Dtype":
        import tensorflow as tf

        return {
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
        def Scalar(arg_name):
            return Const(arg_name, self._z3())
        return Scalar
    def to_iter(self) -> "AbsIter" :
        return AbsIter([self])
    def get_arg_dtype(self) : 
        return self
    def to_tensor_dtype(self) -> List[DType]:
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
        self.arg_type = values[0]
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
        def Vector(arg_name):
            return Array(arg_name, IntSort(), self.arg_type._z3())
        return Vector
    
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