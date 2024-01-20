import random
from typing import Callable, Literal, Any, List, Tuple, Dict, Optional, Union, Set, get_args, get_origin
import collections
import copy
from logger import LOGGER
from abstract.dtype import AbsDType, AbsIter, AbsLiteral, DType, DTYPE_ALL
from neuri.abstract.tensor import AbsTensor

NOT_SUPPORTED_DTYPES ={
    'torch' : [
    'qint8',
    'qint16',
    'qint32',
    'uint8',
    'uint16',
    'uint32',
    'uint64',
    ],
    'tf' : [
        'complex32'
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
    None : AbsDType.none,
}
STR_TO_ABS = {
    # AbsDType
    'scalar': 'scalar_float',
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
    'list': AbsDType.int.to_iter(),
    'lists': AbsDType.int.to_iter(),
    'array': AbsDType.int.to_iter(),
    'arrays': AbsDType.int.to_iter(),
    'vector': AbsDType.int.to_iter(),
    'vectors': AbsDType.int.to_iter(),
    'tuple': AbsDType.int.to_iter(),
    'array_like': AbsDType.int.to_iter(),
    'sequence[int]': AbsDType.int.to_iter(),
    'sequence[int]': AbsDType.int.to_iter(),
    'sequences[int]': AbsDType.int.to_iter(),
    'sequence': AbsDType.int.to_iter(),
    'sequences': AbsDType.int.to_iter(),

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
    'torch.complex': DType.complex64,
    'tf.complex64': DType.complex64,
    'torch.complex64': DType.complex64,
    'complex64': DType.complex64,
    'tf.complex128': DType.complex128,
    'torch.complex128': DType.complex128,
    'complex128': DType.complex128,

    # AbsTensor and List[AbsTensor]
    'tf.tensor': AbsTensor(possible_dtypes=DTYPE_ALL),
    'torch.tensor': AbsTensor(possible_dtypes=DTYPE_ALL),
    'tensor': AbsTensor(possible_dtypes=DTYPE_ALL),
    'tf.tensors': AbsTensor(possible_dtypes=DTYPE_ALL),
    'torch.tensors': AbsTensor(possible_dtypes=DTYPE_ALL),
    'longtensor': AbsTensor(possible_dtypes=DTYPE_ALL),
    'sequence of Tensors': List[AbsTensor],

    # torch 
    'complexfloat' : AbsDType.complex,
    'cfloat' : DType.complex64,
    'cdouble' : DType.complex128,
    'short' : DType.int16,
    'long' : DType.int64,
}

def demateralize(dtype : Union[List, AbsDType, AbsIter, AbsTensor, AbsLiteral]) -> str :
    
    if isinstance(dtype, AbsTensor) :
        return ",".join([ele.to_str() for ele in dtype.possible_dtypes])
    elif type(dtype) == list : 
        return ",".join([demateralize(ele) for ele in dtype])
    return dtype.to_str()

def typing_to_abs(dtype : Any) -> Any :
    origin = get_origin(dtype)
    if origin is Union :
        abs_types = [typing_to_abs(dtype_arg) for dtype_arg in get_args(dtype)]
        return abs_types
    elif origin is Tuple :
        tuple_types = get_args(dtype)
        if (...) in tuple_types :    
            length = random_define_length(dtype)
        else :
            length = len(tuple_types)
        
        tuple_types = tuple_types[0]
        return AbsIter(values = [typing_to_abs(tuple_types) for _ in length])
    elif origin in [List, collections.abc.Sequence] :
        length = random_define_length(dtype)
        list_type = get_args(dtype)[0]
        return AbsIter(values = [typing_to_abs(list_type) for _ in length])

    elif origin is Literal :
        args = get_args(dtype)
        return AbsLiteral(args)
    elif origin is None : return dtype
    else :
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
        # abs_from_dtype = exec_typing(target_str)
        # if abs_from_dtype is not None :
def materalize_dtypes(dtypes : str, merge_tensor : bool = True) -> List[Any] :
    targets : List[str] = []
    res : List[Any] = []
    dtypes = str(dtypes)
    if dtypes.startswith("Literal") :
        literal_obj = eval(dtypes)
        choices = literal_obj.__args__
        res.append(AbsLiteral(choices))
    else : 
        if "tensor" in dtypes.lower() : 
            dtypes = "tensor"
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
                to_merge.extend(abs.to_tensor_dtype())
            else :
                final.append(abs) 
        elif isinstance(abs, AbsTensor) :
            if any([isinstance(ele, DType) for ele in res]) :
                pass 
            else :
                final.append(abs) 
        elif isinstance(abs, AbsIter) or isinstance(abs, AbsLiteral) :
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

def exec_typing(dtype : Any) -> Any :
    if type(dtype) == str :
        try :
            typing_obj = eval(dtype)
        except :
            return None
    else :
        typing_obj = dtype
    return typing_to_abs(typing_obj)
    

def materialize_tensor(abs : AbsTensor) :
    
    assert abs.shape is not None and abs.dtype is not None
    if abs.package == 'torch' :
        from torch import from_numpy
        abs.materalized = abs.concretize(from_numpy)
    elif abs.package == 'tf' :
        from tensorflow import convert_to_tensor
        abs.materalized = abs.concretize(convert_to_tensor)

    return abs.materalized

def custom_copy(inputs : Dict[str, Any]) : 
    import torch
    if isinstance(inputs, torch.Tensor) :
        return inputs.clone()
    for key, item in inputs.items() :
        if isinstance(item, torch.Tensor) :
            inputs[key] = item.clone() 
        else :
            inputs[key] = copy.copy(item)
    return inputs

def materalize_func(func_name : str, package : Literal['torch', 'tf'] = 'torch') -> Union[Callable, None] :
    """
    Generate function with given name
    """
    if package == 'torch' :
        import torch 
    elif package == 'tf' :
        import tensorflow as tf 
    else :
        pass
    function_str = func_name
    function_parts = function_str.split('.')
    if len(function_parts) < 2:
        return eval(function_str)
    module_name = '.'.join(function_parts[:-1])
    function_name = function_parts[-1]

    if hasattr(eval(module_name), function_name):
        return getattr(eval(module_name), function_name)
        
    else:
        raise AttributeError(f"Function '{function_str}' does not exist.")
    
def random_define_length(dtype : Any) -> int :
    return RandomGenerator.random_choice_int()

class RandomGenerator:
    MAX_SEQUENCE_LENGTH = 6
    MAX_TUPLE_LENGTH = 6
    TUPLE_LENGTH_RANGE = (0,MAX_TUPLE_LENGTH)
    SEQUENCE_LENGTH_RANGE = (0,MAX_SEQUENCE_LENGTH)
    INT_RANGE = (0,8)
    FLOAT_RANGE = (-2,10)
    STRING_OPTIONS = ['not defined']

    @classmethod
    def random_choice_int(cls, drange : Optional[int] = None) -> int:
        if drange is None : drange = cls.INT_RANGE
        return random.randint(*drange)

    @classmethod
    def random_choice_float(cls) -> float:
        return random.uniform(*cls.FLOAT_RANGE)

    @classmethod
    def random_choice_bool(cls) -> bool:
        return random.choice([True, False])

    @classmethod
    def random_choice_str(cls, string_options=None) -> str:
        return cls.STRING_OPTIONS[0]

    @classmethod
    def random_choice_list(cls, type=int, length=None) -> List:
        if length is None :
            length = random.randint(*cls.SEQUENCE_LENGTH_RANGE)
        return [cls.random_choice_arg(type) for _ in range(length)]

    @classmethod
    def random_choice_tuple(cls, types, length=None) -> Tuple:
        if ... in types : # Tuple[...]
            types = types[0]
            return tuple([cls.random_choice_arg(types) for _ in range(length)])
        else :
            return tuple([cls.random_choice_arg(type) for type in types])

    @classmethod
    def random_choice_union(cls, type, length=None) -> Any:
        choice = cls.random_choice_arg(random.choice(type), length=None)
        return choice
    
    @classmethod
    def choice(cls, choices : List[Any]) -> Any : 
        return random.choice(choices)
    
    @classmethod
    def random_generate_input_rank(cls) -> int:
        return random.randint(1, 5)

    @classmethod
    def gen_None(cls) -> None:
        return None
    @classmethod
    def materalrize_abs(cls, 
                        abs : Union[AbsTensor, AbsDType, AbsIter],
                        length : int = None,
                        package : Literal['tf', 'torch'] = 'torch') -> Any : 
        if isinstance(abs, AbsTensor) :
            if abs.shape is None :
                if length is None :
                    length = cls.random_choice_int(cls.MAX_SEQUENCE_LENGTH)
                abs.set_rank(length)
                new_shape = [cls.random_choice_int() \
                                for shape in abs.shape \
                                if shape is None]
                abs.set_shape(new_shape)
            if abs.dtype is None :
                abs.dtype = cls.choice(abs.possible_dtypes) 
            return abs
        elif type(abs) ==list :
            return [cls.materalrize_abs(item) for item in abs]
        elif isinstance(abs, AbsDType) :
            if abs == AbsDType.none :
                return None
            elif abs == AbsDType.bool :
                return cls.random_choice_bool()
            elif abs == AbsDType.int :
                return cls.random_choice_int()
            elif abs == AbsDType.float :
                return cls.random_choice_float()
            elif abs == AbsDType.str :
                return cls.random_choice_str()
            else :
                raise NotImplementedError(f"Unsupported type {abs}")
        elif isinstance(abs, AbsLiteral) :
            return cls.choice(abs.choices)
        elif isinstance(abs, AbsIter) :
            if abs.length is None :
                if length is None :
                    length = cls.random_choice_int(cls.MAX_SEQUENCE_LENGTH)
                abs.set_length(length)
            return [cls.materalrize_abs(abs.arg_type) for _ in range(abs.length)] 
        else :
            raise NotImplementedError(f"Unsupported type {abs}")
    @classmethod
    def random_choice_arg(cls, dtype: Union[type, AbsTensor], length=None, string_options=None) -> Any:

        if hasattr(dtype, '__origin__'):
            if dtype.__origin__ is Union:
                subtypes = dtype.__args__
                return cls.random_choice_union(subtypes, length=length)
            elif dtype.__origin__ is tuple:
                tuple_types = dtype.__args__
                return cls.random_choice_tuple(tuple_types, length=length)
            elif dtype.__origin__ is list:
                list_type = dtype.__args__[0]
                return cls.random_choice_list(list_type, length=length)
            elif dtype.__origin__ is collections.abc.Sequence:
                list_type = dtype.__args__[0]
                return cls.random_choice_list(list_type, length=length)
            elif dtype.__origin__ is Literal:
                literal_values = dtype.__args__
                return random.choice(literal_values)
        elif dtype is int:
            return cls.random_choice_int()
        elif dtype is str:
            return cls.random_choice_str(string_options=string_options)
        elif dtype is bool:
            return cls.random_choice_bool()
        elif dtype is float:
            return cls.random_choice_float()
        elif dtype is None:
            return None
        elif dtype is type(None):
            return None
        else :
            raise NotImplementedError(f"Unsupported type {dtype}")
        
    @classmethod
    def sequence_ele_check(cls, dtype) :
        return dtype in (int, float)

    @classmethod
    def check(cls, dtype, length=None, string_options=None) -> Any:
        try : 
            cls.materalrize_abs(dtype) 
            if hasattr(dtype, 'init') :
                dtype.init()
            return True
        except :
            return False 

# mapping_torch = {
#     'Number': int,
#     'number': int,
#     'Tensor': torch.Tensor,
#     'LongTensor': torch.Tensor,
#     'int': int, 
#     'ints': int, 
#     'integers': int, 
#     'int64': int, 
#     'int32': int, 
#     'int16': int, 
#     'int8': int, 
#     'torch.int': int, 
#     'torch.int64': int, 
#     'torch.int32': int, 
#     'torch.int16': int, 
#     'torch.int8': int, 
#     'sequence of Tensors': List[torch.Tensor],
#     'float': float, 
#     'floating': float, 
#     'float64': float, 
#     'float32': float, 
#     'float16': float, 
#     'torch.float': float, 
#     'torch.float64': float, 
#     'torch.float32': float, 
#     'torch.float16': float, 

#     'bool': bool, 
#     'boolean': bool, 
#     'torch.bool': bool, 

#     'str': str, 
#     'string': str, 
#     'strings': str, 
#     'torch.char': str, 

#     'list': List[int], 
#     'lists': List[int], 
#     'array': List[int], 
#     'arrays': List[int], 
#     'vector': List[int], 
#     'vectors': List[int], 

#     'tuple': Tuple[int, ...], 

#     'dict': Dict, 
#     'dictionary': Dict, 

#     'tensor': torch.Tensor, 
#     'tensors': torch.Tensor, 
#     'torch.tensor': torch.Tensor,

#     'torch.floattensor': torch.Tensor, 
#     'torch.doubletensor': torch.Tensor, 
#     'torch.bytetensor': torch.Tensor, 
#     'torch.shorttensor': torch.Tensor, 
#     'torch.inttensor': torch.Tensor, 
#     'torch.longtensor': torch.Tensor, 
#     'torch.booltensor': torch.Tensor, 
#     'torch.halftensor': torch.Tensor,

#     'torch.short': int, 
#     'torch.long': int, 
#     'torch.double': int, 
#     'torch.half': int, 
#     'torch.uint8': int, 
#     'torch.qint8': int, 
#     'torch.qint16': int, 
#     'torch.qint32': int, 
#     'torch.quint8': int, 
#     'torch.quint16': int, 
#     'torch.quint32': int, 

#     'torch.chartensor': str, 

#     'torch.dtype': Any, 

#     'uint8': int, 
#     'short': int, 
#     'long': int, 
#     'half': int, 
#     'double': int, 
#     'numeric': int, 

#     'iterable': List[int], 
#     'sequence': List[int], 
#     'ndarray': List[int], 
#     'array_like': List[int], 

#     'sparsetensor': torch.sparse.FloatTensor
# }
# mapping_all = {}
# mapping_all.update(mapping_torch)
# mapping_all.update(mapping_tf)