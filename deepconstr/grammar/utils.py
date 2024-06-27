import hashlib
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np

class AbsValue:
    def __init__(self, value):
        self.value = value

    def concretize(self, *args, **kwargs):
        return self.value

    def __str__(self) -> str:
        return f"AbsValue({str(self.value)})"

    def __repr__(self) -> str:
        return f"AbsValue({str(self.value)})"

    def concrete_str(self, *args, **kwargs) -> str:
        return self.__str__()


class AbsInt(AbsValue):
    def concretize(self, symb_2_value: Dict[str, Any], *args, **kwargs):
        return symb_2_value[self.value]

    def concrete_str(self, symb_2_value: Dict[str, Any]) -> str:
        # s0=1
        if isinstance(self.value, str):
            return f"{self.value}={symb_2_value[self.value]}"
        else:
            return str(self.value)

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
    from deepconstr.grammar.op import AbsVector
    return AbsVector(list(tensor.shape), abs_from_dtype(tensor.dtype))

def data_type_str(x: Any, keep_int_value: bool = True, dtype_class: Any = None) -> str:
    """To distinguish operator instances in one API.
    1. AbsVector: only consider rank
    2. int not bool:
        keep_int_value is True, int values should be the same in one op
        keep_int_value is False, int values can be symbolized
    3. bool/str: values should be the same in one op
    4. list: recursive
    5. others: ignore them; their values can vary in one op
    """
    if isinstance(x, np.ndarray):
        return f"AbsVector<{x.ndim}>"
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