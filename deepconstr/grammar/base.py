from functools import partial
import z3
from typing import *
from deepconstr.gen.noise import should_generate_noise
from deepconstr.grammar import ARRTYPES, MAX_ARR_LEN, MAX_SHAPE_SUM, MAX_VALUE, MIN_VALUE, SMTFuncs
from deepconstr.grammar.dtype import DTYPE_NOT_SUPPORTED, AbsDType, AbsIter, AbsVector


def gen_dtype_constraints(arg_name : str, not_supported_dtypes : List["DType"]) -> z3.ExprRef :
    
    assert not_supported_dtypes, "DTYPE_NOT_SUPPORTED not defined"
    constr = SMTFuncs.not_in(
        AbsVector.z3()(arg_name).dtype, 
        [
        dtype.z3_const() for dtype in not_supported_dtypes 
    ]
    )
    return constr

DEFAULT_DTYPE_CONSTR : Dict[str, z3.ExprRef] = {
    "torch" : partial(gen_dtype_constraints, not_supported_dtypes=DTYPE_NOT_SUPPORTED.get("torch")),
    "tensorflow" : partial(gen_dtype_constraints, not_supported_dtypes=DTYPE_NOT_SUPPORTED.get("tensorflow")),
    "numpy" : partial(gen_dtype_constraints, not_supported_dtypes=DTYPE_NOT_SUPPORTED.get("numpy")),
}


def tensor_default_constr(
        tensor_shape,
        length,
        include_zero
    ) :
    return pos_max_constraints(tensor_shape, length, include_zero)

def gen_default_constr(
        args_types : Dict[str, Union[AbsDType, AbsVector]],
        args_lengths : Dict[str, Optional[int]],
        allow_zero_rate : float = 0.5,
                        ) -> List[z3.ExprRef] :
    rules = []
    if should_generate_noise(allow_zero_rate) : 
        include_zero = True
    else :
        include_zero = False
    for arg_name in args_types.keys() :
        if isinstance(args_types[arg_name], AbsVector) :
            shape_var = args_types[arg_name].z3()(arg_name)
            length_var = args_lengths[arg_name]
            rules.append(
                pos_max_constraints(shape_var,
                                length_var,
                                include_zero)
                )
        elif isinstance(args_types[arg_name], AbsIter) :
            if isinstance(args_types[arg_name].get_arg_dtype(), AbsVector) :
                # arr_wrapper = args_types[arg_name].z3()(arg_name)
                # for idx in range(len(args_lengths[arg_name])) :
                #     rules.append(pos_max_constraints(arr_wrapper.get_arg_attr(idx, "shape"), args_lengths[arg_name][idx], include_zero))
                pass
            elif args_types[arg_name].get_arg_dtype() in [AbsDType.int, AbsDType.float] :
                arr_wrapper = args_types[arg_name].z3()(arg_name)
                rules.append(min_max_constraints(arr_wrapper.value, args_lengths[arg_name]))
            else :
                raise NotImplementedError(f"Unsupported type {args_types[arg_name].get_arg_dtype()}")
        else :
            pass
    return rules

def length_constr(length, max_len = None, min_len = None) : 
    if max_len is None : max_len = MAX_ARR_LEN
    if min_len is None : min_len = 0
    return z3.And(length >= min_len, length <= max_len)

def pos_max_constraints(z3obj, len_var, include_zero) : 
    if isinstance(len_var, int) :
        length = len_var
    else :
        length = MAX_ARR_LEN
    if include_zero :
        return z3.And([
            z3.And(z3obj[i]<=MAX_VALUE, z3obj[i] >= 0) for i in range(length)
        ])
        # return z3.ForAll([i], z3.Implies(z3.And(i>=0, i<=len_var), z3.And(z3obj[i]<=MAX_VALUE, z3obj[i] >= 0)))
    else :
        return z3.And([
            z3.And(z3obj[i]<=MAX_VALUE, z3obj[i] > 0) for i in range(length)
        ])
        # return z3.ForAll([i], z3.Implies(z3.And(i>=0, i<=len_var), z3.And(z3obj[i]<=MAX_VALUE, z3obj[i] > 0)))
    
def min_max_constraints(z3obj, len_var) : 
    if isinstance(len_var, int) :
        length = len_var
    else :
        length = MAX_ARR_LEN
    return z3.And([
        z3.And(z3obj[i]<=MAX_VALUE, z3obj[i] >= MIN_VALUE) for i in range(length)
    ])
    # i = z3.Int('i')
    # return z3.ForAll([i], z3.Implies(z3.And(i>=0, i<=len_var), z3.And(z3obj[i]<=MAX_VALUE, z3obj[i] >= MIN_VALUE)))
    
def check_numel_constr(shape, len_var=None) :
    # Create a Z3 array of integers

    # Recursive function to calculate the product of elements in the array
    def array_product(arr, length):
        if length == 0:
            return 1
        else:
            return arr[length - 1] * array_product(arr, length - 1)

    if len_var is None : len_var = MAX_ARR_LEN
    product_constraint = array_product(shape, len_var) < MAX_SHAPE_SUM
    return product_constraint

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