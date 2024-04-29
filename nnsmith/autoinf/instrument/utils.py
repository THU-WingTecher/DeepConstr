import hashlib
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np


def hash_list_str(x: List[str]) -> str:
    return hashlib.sha256("".join(x).encode()).hexdigest()


def is_int_not_bool(x: Any) -> bool:
    return any(isinstance(x, t) for t in [int, np.integer]) and not any(
        isinstance(x, t) for t in [bool, np.bool_]
    )


def is_numpy_container(x):
    return (
        (isinstance(x, list) or isinstance(x, tuple))
        and len(x) > 0
        and all(isinstance(x_i, np.ndarray) for x_i in x)
    )


def get_ret_list(ret):
    if isinstance(ret, tuple):
        ret_list = list(ret)
    elif not isinstance(ret, list):
        ret_list = [ret]
    else:
        ret_list = ret
    return ret_list

def tensor_to_abs(tensor, abs_from_dtype : Callable) : 
    from nnsmith.abstract.dtype import AbsTensor
    return AbsTensor(list(tensor.shape), abs_from_dtype(tensor.dtype))

def data_type_str(x: Any, keep_int_value: bool = True, dtype_class: Any = None) -> str:
    """To distinguish operator instances in one API.
    1. Tensor: only consider rank
    2. int not bool:
        keep_int_value is True, int values should be the same in one op
        keep_int_value is False, int values can be symbolized
    3. bool/str: values should be the same in one op
    4. list: recursive
    5. others: ignore them; their values can vary in one op
    """
    if isinstance(x, np.ndarray):
        return f"Tensor<{x.ndim}>"
    elif is_int_not_bool(x):
        if keep_int_value:
            return f"int({x})"
        else:
            return "int"
    elif isinstance(x, bool):
        return str(x)
    elif isinstance(x, str):
        return f'"{x}"'
    elif isinstance(x, float):
        return "float"
    elif isinstance(x, complex):
        return "complex"
    elif isinstance(x, list):
        return f"[{', '.join([data_type_str(x_i, keep_int_value) for x_i in x])}]"
    elif dtype_class and isinstance(x, dtype_class):
        return str(x)
    else:
        """Ignore these types since they are not likely to affect the output shapes.
        memory_format, device, ...
        """
        return "ignored"
        # isinstance(eval(type), str)
    # else:
    #     print(f"Unknown input type for data: {x = }", flush=True)
    #     embed()


def numpy_random(shape: List[int], str_dtype: str) -> Union[np.ndarray, Tuple[List[int], str]]:
    """ 
    generate random numpy array.
    If numpy doesn't support the dtype, return the shape and dtype.(Tuple)
    """
    if np.prod(shape) > 2 * 1024**3 / 16:
        raise ValueError(f"Too large tensor shape: {shape = }")
    # print(f"Generating random tensor: {shape = }, {str_dtype = }", flush=True)
    rand_float = lambda size: np.random.uniform(-1000_000, 1000_000, size)
    # rand_float = lambda size: np.random.uniform(0, 1, size)
    ret: np.ndarray = None
    if "float" in str_dtype:
        if "bfloat" in str_dtype:
            ret = (shape, str_dtype)
        else :
            ret = np.array(rand_float(shape)).astype(str_dtype)
    elif "complex" in str_dtype:
        complex_2_float = {
            "complex64": "float32",
            "complex128": "float64",
            "complex256": "float128",
        }
        float_dtype = complex_2_float[str_dtype]
        ret = np.array(
            np.array(rand_float(shape)).astype(float_dtype)
            + 1j * np.array(rand_float(shape)).astype(float_dtype)
        )
    elif "int" in str_dtype:
        if "qint" in str_dtype or "uint" in str_dtype:
            ret = (shape, str_dtype)
        else:
            ret = np.array(np.random.randint(-1000_000, 1000_000, shape)).astype(str_dtype)
    elif "bool" in str_dtype:
        ret = np.array(np.random.randint(0, 2, shape)).astype(str_dtype)
    else:
        print(f"Unknown dtype: {str_dtype = }", flush=True)
        raise NotImplementedError(str_dtype)
    return ret


def numpy_random_like(x):
    input_type = type(x)
    if not (isinstance(x, list) or isinstance(x, tuple)):
        xs = [x]
    else:
        xs = x
    ret = []
    for x_i in xs:
        ret.append(numpy_random(x_i.shape, str(x_i.dtype)))
    if isinstance(x, list):
        return ret
    elif isinstance(x, tuple):
        return tuple(ret)
    else:
        return ret[0]


def tensors_from_numpys(x, tensor_from_numpy: Callable):
    if not (isinstance(x, list) or isinstance(x, tuple)):
        xs = [x]
    else:
        xs = x
    ret = []
    for x_i in xs:
        ret.append(tensor_from_numpy(x_i))
    if isinstance(x, list):
        return ret
    elif isinstance(x, tuple):
        return tuple(ret)
    else:
        return ret[0]


def inputs_from_record(
    record: Dict[str, Any],
    tensor_from_numpy: Callable,
    random_value: bool = False,
    replace: Dict[str, Any] = {},
):
    args = []
    kwargs = {}
    record_args = record["args"]
    for i_name, name in enumerate(record_args["name"]):
        if name in replace:
            value = replace[name]
        else:
            value = record_args["value"][i_name]
        if isinstance(value, np.ndarray) or is_numpy_container(value):
            if random_value:
                value = numpy_random_like(value)
            value = tensor_from_numpy(value)
        if record_args["is_pos"][i_name]:
            args.append(value)
        else:
            kwargs[name] = value
    for replace_name, replace_value in replace.items():
        if replace_name not in record_args["name"]:
            if isinstance(replace_value, np.ndarray) or is_numpy_container(
                replace_value
            ):
                if random_value:
                    replace_value = numpy_random_like(replace_value)
                replace_value = tensor_from_numpy(replace_value)
            kwargs[replace_name] = replace_value
    return args, kwargs
