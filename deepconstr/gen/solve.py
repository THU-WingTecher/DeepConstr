import logging
import z3
import random 
import string
import z3
from deepconstr.gen.noise import add_noises, gen_random_int_constr, should_generate_noise
from deepconstr.grammar.base import gen_default_constr, length_constr, min_max_constraints
from deepconstr.grammar.dtype import DTYPE_GEN_ALL, AbsDType, AbsIter, AbsVector, DType 
from deepconstr.grammar import MAX_TENSOR_LIST_RANK, SMTFuncs, TensorZ3, MAX_ARR_LEN, MAX_VALUE, MIN_VALUE, TensorArrWrapper
from deepconstr.logger import SMT_LOG
from typing import Callable, Dict, Any, List, Literal, Optional, Tuple, Union

from nnsmith.abstract.op import __MAX_RANK__
assert MAX_ARR_LEN > __MAX_RANK__, f"MAX_ARR_LEN should be greater than __MAX_RANK__"

def handle_int_sort(model, z3_instance, is_length=False):
    if is_length :
        return clip_unbound_val(z3_instance.as_long(), max = MAX_ARR_LEN, min = 0)
    return clip_unbound_val(z3_instance.as_long())

def handle_real_sort(model, z3_instance):
    return clip_unbound_val(float(z3_instance.as_decimal(prec=7).rstrip('?')))

def handle_bool_sort(model, z3_instance):
    return bool(z3_instance)

def handle_string_sort(model, z3_instance):
    return str(z3_instance).replace('"', '').replace("'", "")

def gen_abs_tensor(dtype, shape):
    return AbsVector(dtype=DType.from_str(str(dtype)), shape=shape)

def handle_abs_tensor(model, z3_instance, arg_name, args_types, args_values, args_lengths, ret_len=False):
    wrapper = args_types[arg_name].z3()(arg_name)
    dtype = wrapper.evaluate(model, z3_instance, attr="dtype")
    rank = wrapper.evaluate(model, z3_instance, attr="rank")
    rank = random_gen_arr_len() if rank is None else handle_int_sort(model, rank, is_length=True)
    if ret_len : 
        args_lengths[arg_name] = rank
        return 
    shapes = [wrapper.evaluate(model, z3_instance, attr="shape", idx=i) for i in range(rank)]
    for i, shape in enumerate(shapes):
        if shape is None:
            shapes[i] = random_gen(AbsDType.int)  # Assuming random_gen and AbsDType are defined
        else:
            shapes[i] = handle_int_sort(model, shape)
    
    # Assuming AbsVector and DType are defined and have a suitable constructor
    return gen_abs_tensor(dtype, shapes)

def handle_abs_iter(model, z3_instance, arg_name, args_types, args_values, args_lengths, ret_len=False):
    """
    Refactored to call sym_to_conc for each element value.
    """
    wrapper = args_types[arg_name].z3()(arg_name)
    length = wrapper.evaluate(model, z3_instance, attr="len")
    length = random_gen_arr_len() if length is None else handle_int_sort(model, length)
    if isinstance(wrapper, TensorArrWrapper) :
        length = [random_gen_arr_len() for _ in range(length)]
    if ret_len : 
        args_lengths[arg_name] = length
        return 
    values = [wrapper.evaluate(model, z3_instance, attr="value", idx=i) for i in range(len(length) if isinstance(length, list) else length)]
    for i, value in enumerate(values):
        if value is None:
            values[i] = random_gen(args_types[arg_name].get_arg_dtype())
        else:
            # Recursively handle nested abstract types
            if hasattr(value, "sort") and value.sort() == TensorZ3 :
                rank = handle_int_sort(model, wrapper.evaluate(model, value, arg_attr="rank"))
                shapes = [None] * rank
                for j in range(rank):
                    shapes[j] = handle_int_sort(model, wrapper.evaluate(model, value, arg_attr="shape", arg_idx=j))
                dtype = wrapper.evaluate(model, value, arg_attr="dtype")
                values[i] = gen_abs_tensor(dtype, shapes)
            else:
                values[i] = to_conc_py_val(model, value)
    
    if args_types[arg_name].get_arg_dtype() == AbsDType.str:
        return "".join(map(str, values)).replace('""', '')
    else:
        return values

def sym_to_conc(model, z3_arg, args_types, args_values, args_lengths, ret_len) : 
    """
    fill the args_values and args_lengths based on model of solver
    """
    values = [] 
    arg_name = z3_arg.name()
    z3_instance = model[z3_arg]
    datatype = z3_arg.range()

    if arg_name not in args_types:
        return None
    if datatype == z3.IntSort():
        args_values[arg_name] = handle_int_sort(model, z3_instance)
    elif datatype == z3.RealSort():
        args_values[arg_name] = handle_real_sort(model, z3_instance)
    elif datatype == z3.BoolSort():
        args_values[arg_name] = handle_bool_sort(model, z3_instance)
    elif datatype == z3.StringSort():
        args_values[arg_name] = handle_string_sort(model, z3_instance)
    else : 
        wrapper = args_types[arg_name].z3()(arg_name)
        if isinstance(args_types[arg_name], AbsVector) :
            args_values[arg_name] = handle_abs_tensor(model, z3_instance, arg_name, args_types, args_values, args_lengths, ret_len=ret_len)
        elif isinstance(args_types[arg_name], AbsIter):
            if args_types[arg_name].get_arg_dtype() in [AbsDType.complex] : 
                raise NotImplementedError
            args_values[arg_name] = handle_abs_iter(model, z3_instance, arg_name, args_types, args_values, args_lengths, ret_len=ret_len)
            # args_lengths[arg_name] = wrapper.evaluate(model, z3_instance, attr="len")
            # if args_lengths[arg_name] is None :
            #     args_lengths[arg_name] = random_gen_arr_len()
            # else :
            #     args_lengths[arg_name] = clip_unbound_val(args_lengths[arg_name].as_long(), max = MAX_ARR_LEN)
            # args_values[arg_name] = [wrapper.evaluate(model, z3_instance, attr = "value", idx = i) for i in range(args_lengths[arg_name])]
            # if args_types[arg_name].get_arg_dtype() in [AbsDType.str] :
            #     args_values[arg_name] = "".join(map(str, args_values[arg_name])).replace('""', '')
            # else :
            #     for i in range(len(args_values[arg_name])) :
            #         if args_values[arg_name][i] is None :
            #             args_values[arg_name][i] = random_gen(args_types[arg_name].get_arg_dtype())
            #         elif args_types[arg_name].get_arg_dtype() in [AbsVector] :
            #             args_values[arg_name][i] = sym_to_conc(model, args_values[arg_name][i], args_types, args_values, args_lengths)
            #         else :
            #             args_values[arg_name][i] = clip_unbound_val(to_conc_py_val(args_values[arg_name][i]), max = MAX_VALUE)
            # return args_values[arg_name]  # Returning only the values, not the length
        else :
            raise NotImplementedError(f"Unsupported type {z3_arg.range()}")
def process_len(
        args_types : Dict[str, Union[AbsDType, AbsVector]],
        args_lengths : Dict[str, Optional[int]],
        solver : z3.Solver,
        names : List[str],
        noise : float = 0.0,
        allow_zero_length_rate : float = 0.5,
        ) -> List[z3.ExprRef] :
    
    constrs = []
    noises = []
    for arg_name, arg_type in args_types.items() : 
        if isinstance(arg_type, (AbsVector, AbsIter)) :
            is_tensor_list = isinstance(arg_type, AbsIter) and isinstance(arg_type.get_arg_dtype(), AbsVector)
            args_lengths[arg_name] = None
            min_val = 1 if should_generate_noise(allow_zero_length_rate) else 0
            max_val = MAX_TENSOR_LIST_RANK if is_tensor_list else None
            ## gen default constr 
            len_sym = arg_type.z3()(arg_name).rank

            len_constr = length_constr(len_sym, max_val, min_val)
            constrs.append(len_constr)
            ## gen noise 
            if arg_name not in names and should_generate_noise(noise) :
                len_noise = gen_random_int_constr(len_sym, 0, MAX_ARR_LEN)
                noises.append(len_noise)
    
    solver.add(constrs)
    solver.add(noises)
    # solver = push_and_pop_noise(noises, solver)
    # Solving 
    if not is_solver_solvable(solver) :
        if SMT_LOG.getEffectiveLevel() <= logging.DEBUG:
            SMT_LOG.debug(f"Solver is not solvable with {solver.assertions()}")
        return None 
    
    model = solver.model()
    constrs.clear()
    for z3_arg in model.decls() :
        sym_to_conc(model, z3_arg, args_types, {}, args_lengths, ret_len=True)
    
    for arg_name in args_lengths.keys() :
        if args_lengths[arg_name] is None :
            args_lengths[arg_name] = random.randint(0, MAX_ARR_LEN)
    return args_lengths
    # for arg_name, len_val in args_lengths.items() :
    #     if isinstance(len_val, list) :
    #         for i, val in enumerate(len_val) :
    #             constrs.append(
    #                 args_types[arg_name].z3()(arg_name).get_arg_attr(i, "rank") == val
    #             )
    #             assert isinstance(val, int) and val <= MAX_ARR_LEN, f"Invalid length {val}"
    #         len_val = len(len_val)
    #     constrs.append(args_types[arg_name].z3()(arg_name).rank == len_val)
    #     # if isinstance(args_types[arg_name], AbsVector) :
    #     #     assert isinstance(len_val, int) and len_val <= MAX_ARR_LEN, f"Invalid length {len_val}"
    #         # constrs.append(check_numel_constr(args_types[arg_name].z3()(arg_name).shape, len_val))
    
    # return constrs

def push_and_pop_noise(noises : List[z3.ExprRef], solver : z3.Solver) -> None :
    n=0
    solverable = False
    for noise in noises :
        solver.push() 
        solver.add(noise)
        n+=1
    while n>0 and not solverable :
        solver.pop()
        n-=1
        solverable = is_solver_solvable(solver)
    return solver

def process_model(solver, noises, args_types : Dict[str, Any]) : 

    args_lengths = {}
    args_values = {}
    solver.add(noises)
    # solver = push_and_pop_noise(noises, solver)
    if not is_solver_solvable(solver):
        if SMT_LOG.getEffectiveLevel() <= logging.DEBUG:
            SMT_LOG.debug(f"Solver is not solvable with {solver.assertions()}")
        return None
    model = solver.model()
    if SMT_LOG.getEffectiveLevel() <= logging.DEBUG:
        SMT_LOG.debug(f"MODEL ---> {model}")
    model_decls = sorted(model.decls(), key=lambda d: d.name())
    for z3_arg in model_decls :
        sym_to_conc(model, z3_arg, args_types, args_values, args_lengths, ret_len=False)

    for key, value in args_values.items() : 
        if isinstance(value, dict) and "tensor" in value.keys() :
            args_values[key] = AbsVector(**value["tensor"])
    return args_values, args_lengths

def gen_val(num_of_try, *args, **kwargs) -> Optional[Dict[str, Any]] : 
    values = None 
    tries = 0
    num_of_try = int(num_of_try * kwargs["noise_prob"])
    while values is None and tries <= num_of_try :
        values = _gen_val(*args, **kwargs)
        tries += 1
    return values

def extract_names_from_constrs(constrs : List[z3.ExprRef]) -> List[str] :
    names = set()
    for constr in constrs :
        try :
            name = SMTFuncs.get_name(constr)
        except :
            name = None
        if name is not None :
            names.add(name)
    return names

def _gen_val(
          args_types : Dict[str, Union[AbsDType, AbsVector]],
          constrs : List[Callable] = [], 
          noise_prob : float = 0.0,
          allow_zero_length_rate : float = 0.5,
          allow_zero_rate : float = 0.5,
          constraints : List[z3.ExprRef] = [],
          dtype_constrs : List[z3.ExprRef] = [],
          api_name : str = "",
          ) -> Optional[Dict[str, Any]] : 
    """ 
    Gen failed -> return None 
    """
    args_lengths = {}
    ## activate constrs
    constrs = [constr(z3objs={
                        name : abs.z3()(name)
                        for name, abs in args_types.items() if abs is not None
                    }) for constr in constrs]
    names = extract_names_from_constrs(constraints)
    solver = z3.Solver()
    solver.add(constrs + constraints)
    len_res = process_len(args_types, args_lengths, solver, names, noise=noise_prob, allow_zero_length_rate=allow_zero_length_rate)
    if len_res is None :
        SMT_LOG.debug(f"rank related rules cannot be satisfied")
        return None 
    solver.reset() 
    solver.add(constrs + constraints + dtype_constrs)
    help_rules = gen_default_constr(args_types, args_lengths, allow_zero_rate=allow_zero_rate)
    solver.add(help_rules)
    # solve all rank and len varaibles_first 
    noises = add_noises(args_types, args_lengths, noise_prob, names)
    if SMT_LOG.getEffectiveLevel() <= logging.DEBUG:
        SMT_LOG.debug(f"---> Trying to solve: {api_name} ~ {solver.assertions()}")
    results = process_model(solver, noises, args_types)
    if results is not None :
        args_values, args_lengths = results
        for arg_name in args_types.keys() : 
            if args_values.get(arg_name, None) is None :
                args_values[arg_name] = random_gen(args_types[arg_name], args_lengths.get(arg_name))
    else : 
        args_values = None
    SMT_LOG.debug(f"args_values: {args_values}")
    return args_values

def is_solver_solvable(solver: z3.Solver) -> bool:
    res = False
    # if solver.check() == z3.unknown :
    #     solver.set("timeout", 100000)
    res = solver.check() == z3.sat
    return res 

def random_gen_arr_len() -> int :
    return random.randint(0, MAX_ARR_LEN)

def _random_gen(datatype):
    """
    Randomly generate a value based on the given datatype.
    """
    value = None
    if datatype == AbsDType.bool:
        value = random.choice([True, False])
    elif datatype == AbsDType.int:
        value = random.randint(0, 100)  # Adjust range as needed
    elif datatype == AbsDType.float:
        value = random.uniform(0, 1)
    elif datatype == AbsDType.str:
        value = ''.join(random.choices(string.ascii_letters, k=4))
    else:
        raise NotImplementedError(f"Unsupported datatype {datatype}")
    return value

def random_gen(abs, length=None):
    """
    Materialize an abstract data type into a concrete value.
    """
    if abs in [AbsDType.none, None] :
        return None
    elif isinstance(abs, AbsVector) :
        if length is None :
            length = random_gen_arr_len()
        shape = [random_gen_arr_len() for _ in range(length)]
        dtype = random.choice(DTYPE_GEN_ALL)
        return AbsVector(dtype=dtype, shape=shape)
    elif isinstance(abs, AbsIter) : 
        if length is None :
            length = random_gen_arr_len()
        arg_type = abs.get_arg_dtype()
        return [random_gen(arg_type) for _ in range(length)]
    elif isinstance(abs, AbsDType):
        return _random_gen(abs)
    else:
        raise NotImplementedError(f"Unsupported type {abs}")

def clip_unbound_val(value, max = None, min = None) :
    if max is None : max = MAX_VALUE
    if min is None : min = MIN_VALUE 
    if value > max : 
        return max
    elif value < min : 
        return min
    else :
        return value 

def to_conc_py_val(model, z3_obj: z3.ExprRef) -> Any:
    if isinstance(z3_obj, z3.IntNumRef):
        return handle_int_sort(model, z3_obj)
    elif isinstance(z3_obj, z3.RatNumRef):
        return handle_real_sort(model, z3_obj)
    elif isinstance(z3_obj, z3.BoolRef):
        return handle_bool_sort(model, z3_obj)
    elif isinstance(z3_obj, z3.SeqRef):
        return handle_string_sort(model, z3_obj)
    else :
        raise NotImplementedError