import random 
import string
import z3
from deepconstr.grammar.dtype import AbsDType, AbsIter
from deepconstr.grammar.dtype import AbsVector, DType
from deepconstr.grammar import TensorZ3, BOOL_POOLS, MAX_VALUE, \
    MIN_VALUE, OP_POOLS
import operator

def add_noises(args_types, args_lengths, noise_prob, not_gen_names = []):
    noises = []
    noise_dist = {arg_name: noise_prob if isinstance(noise_prob, float) else noise_prob.get(arg_name, 0) for arg_name in args_types}

    for arg_name, arg_noise in noise_dist.items():
        if not arg_noise or arg_name in not_gen_names:
            continue
        noises.extend(
            gen_noise(arg_name, args_types[arg_name], args_lengths.get(arg_name), noise_dist[arg_name])
        )
    return noises

def should_generate_noise(noise_prob):
    return random.random() < noise_prob

CHARACTER_POOL = string.ascii_letters + string.digits

def gen_random_string_constr(z3obj, op_pools=None):
    """Generates a "noisy" string based on the input string by applying a random operation."""
    if op_pools is None:
        op_pools = BOOL_POOLS
    return random.choice(op_pools)(z3obj, random.choice(CHARACTER_POOL))

def gen_random_int_constr(z3obj, min = None, max = None, op_pools = None):
    if min is None : min = MIN_VALUE
    if max is None : max = MAX_VALUE
    if op_pools is None : op_pools = OP_POOLS
    return random.choice(op_pools)(z3obj, random.randint(min, max))

def gen_random_list_choice_constr(z3obj, pools, op_pools = None):
    if op_pools is None : 
        op_pools = BOOL_POOLS
    return random.choice(op_pools)(z3obj, random.choice(pools))

def gen_noise(arg_name, arg_type, args_length, noise_prob):
    noises = []
    if arg_type is AbsDType.none :
        return noises
    
    if isinstance(arg_type, AbsVector):
        if isinstance(arg_name, z3.ExprRef): # AbsVectorARR
            shape_var = TensorZ3.shape(arg_name)
            dtype_var = TensorZ3.dtype(arg_name)
        else :
            shape_var = arg_type.z3()(arg_name).shape
            dtype_var = arg_type.z3()(arg_name).dtype
            # args_length = len(args_length)
        for idx in range(args_length) :
            if should_generate_noise(noise_prob):
                # Generate noise for each dimension
                noise = gen_random_int_constr(shape_var[idx], 0, MAX_VALUE)
                noises.append(noise)

        if should_generate_noise(noise_prob):
            # Generate dtype noise
            dtype_noise = gen_random_list_choice_constr(dtype_var, [dtype.z3_const() for dtype in DType], [operator.ne])
            noises.append(dtype_noise)

    elif isinstance(arg_type, AbsIter):
        arr_wrapper = arg_type.z3()(arg_name)
        for idx in range(len(args_length) if isinstance(args_length, list) else args_length):
            if should_generate_noise(noise_prob):
                if arg_type.get_arg_dtype() in [AbsDType.complex] :
                    raise NotImplementedError
                elif isinstance(arg_type.get_arg_dtype(), AbsVector) :
                    noise = z3.And(gen_noise(arr_wrapper[idx], arg_type.get_arg_dtype(), args_length[idx], noise_prob))
                elif arg_type.get_arg_dtype() in [AbsDType.str] :
                    noise = None # gen_random_string_constr(arr_wrapper[idx])
                elif arg_type.get_arg_dtype() in [AbsDType.bool] :
                    noise = gen_random_list_choice_constr(arr_wrapper[idx], [True, False])
                else :
                    # Generate noise for each dimension
                    noise = gen_random_int_constr(arr_wrapper[idx], MIN_VALUE, MAX_VALUE)
                if noise is not None :
                    noises.append(noise)

    elif isinstance(arg_type, AbsDType) : 
        if should_generate_noise(noise_prob) : 
            if arg_type in [AbsDType.bool] :
                noise = gen_random_list_choice_constr(arg_type.z3()(arg_name), [True, False])
            elif arg_type in [AbsDType.int, AbsDType.float, AbsDType.complex] :
                noise = gen_random_int_constr(arg_type.z3()(arg_name), MIN_VALUE, MAX_VALUE)
            else : # AbsDType.str 
                noise = None #gen_random_string_constr(arg_type.z3()(arg_name))
            if noise is not None :
                noises.append(noise)
    else : 
        pass 
    return noises