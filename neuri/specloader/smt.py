from functools import partial
import logging
import random
import string
import z3
from abstract.dtype import DTYPE_GEN_ALL, DTYPE_NOT_SUPPORTED, AbsDType, AbsIter, AbsLiteral, DType
from neuri.constrinf.smt_funcs import SMTFuncs
from neuri.logger import SMT_LOG
# from specloader.materalize import RandomGenerator
from neuri.abstract.tensor import AbsTensor 
from typing import Callable, Dict, Any, List, Literal, Optional, Tuple, Union
from specloader import  BOOL_POOLS, MAX_ARR_LEN, MAX_SHAPE_SUM, MAX_VALUE, MIN_VALUE, OP_POOLS, Z3DTYPE, TensorZ3, check_numel_constr, \
    length_default_constraints, length_not_zero_constraints, pos_constraints
import operator


def gen_dtype_constraints(arg_name : str, not_supported_dtypes : List[DType]) -> z3.ExprRef :
    
    assert not_supported_dtypes, "DTYPE_NOT_SUPPORTED not defined"
    constr = SMTFuncs.not_in(
        AbsTensor.z3()(arg_name).dtype, 
        [
        dtype.z3_const() for dtype in not_supported_dtypes 
    ]
    )
    return constr

DEFAULT_DTYPE_CONSTR : Dict[str, z3.ExprRef] = {
    "torch" : partial(gen_dtype_constraints, not_supported_dtypes=DTYPE_NOT_SUPPORTED.get("torch")),
    "tensorflow" : partial(gen_dtype_constraints, not_supported_dtypes=DTYPE_NOT_SUPPORTED.get("tensorflow")),
}

def gen_default_constr(
        args_types : Dict[str, Union[AbsDType, AbsTensor, AbsLiteral]],
        args_lengths : Dict[str, Optional[int]],
        allow_zero_rate : float = 0.5,
                        ) -> List[z3.ExprRef] :
    rules = []
    for arg_name in args_types.keys() :
        if isinstance(args_types[arg_name], AbsTensor) :
            tensor_wrapper = args_types[arg_name].z3()(arg_name)
            if should_generate_noise(allow_zero_rate) : 
                include_zero = True
            else :
                include_zero = False
            rules.append(
                pos_constraints(tensor_wrapper.shape,
                                args_lengths[arg_name],
                                include_zero)
                )
        elif isinstance(args_types[arg_name], AbsLiteral) :
            rules.append(SMTFuncs.in_(args_types[arg_name].z3()(arg_name), args_types[arg_name].choices))
        else :
            continue 
    return rules

def add_noises(args_types, args_lengths, noise_prob):
    noises = []
    noise_dist = {arg_name: noise_prob if isinstance(noise_prob, float) else noise_prob.get(arg_name, 0) for arg_name in args_types}

    for arg_name, arg_noise in noise_dist.items():
        if not arg_noise:
            continue
        noises.extend(
            gen_noise(arg_name, args_types[arg_name], args_lengths.get(arg_name), noise_dist[arg_name])
        )
    return noises

def should_generate_noise(noise_prob):
    return random.random() < noise_prob

def gen_random_int_constr(z3obj, min = None, max = None, op_pools = None):
    if min is None : min = MIN_VALUE
    if max is None : max = MAX_VALUE
    if op_pools is None : op_pools = OP_POOLS
    return random.choice(op_pools)(z3obj, random.randint(min, max))

def gen_radnom_list_choice_constr(z3obj, pools, bool_pools = None):
    if bool_pools is None : 
        bool_pools = BOOL_POOLS
    return random.choice(bool_pools)(z3obj, random.choice(pools))

def gen_noise(arg_name, arg_type, args_length, noise_prob):
    noises = []

    if isinstance(arg_type, AbsLiteral) and should_generate_noise(noise_prob):
        literal_noise = gen_radnom_list_choice_constr(arg_type.z3()(arg_name), arg_type.choices)
        noises.append(literal_noise)

    elif isinstance(arg_type, AbsTensor):
        tensor_wrapper = arg_type.z3()(arg_name)
        for idx in range(args_length) :
            if should_generate_noise(noise_prob):
                # Generate noise for each dimension
                noise = gen_random_int_constr(tensor_wrapper.shape[idx], 0, MAX_VALUE)
                noises.append(noise)

        if should_generate_noise(noise_prob):
            # Generate dtype noise
            dtype_noise = gen_radnom_list_choice_constr(tensor_wrapper.dtype, [dtype.z3_const() for dtype in DType], [operator.ne])
            noises.append(dtype_noise)

    elif isinstance(arg_type, AbsIter):
        arr_wrapper = arg_type.z3()(arg_name)
        for idx in range(args_length) :
            if should_generate_noise(noise_prob):
                # Generate noise for each dimension
                noise = gen_random_int_constr(arr_wrapper[idx], MIN_VALUE, MAX_VALUE)
                noises.append(noise)

    elif isinstance(arg_type, AbsDType) : 
        if should_generate_noise(noise_prob) : 
            if arg_type in [AbsDType.bool] :

                dtype_noise = gen_radnom_list_choice_constr(arg_type.z3()(arg_name), [True, False])
            elif arg_type in [AbsDType.int, AbsDType.float, AbsDType.complex] :
                dtype_noise = gen_random_int_constr(arg_type.z3()(arg_name), MIN_VALUE, MAX_VALUE)
            else : # AbsDType.str 
                dtype_noise = []
                pass # dont generate noise for str 
            noises.append(dtype_noise)

    return noises

def sym_to_conc(model, z3_arg, args_types, args_values, args_lengths) : 
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
        return z3_instance.as_long()
    elif datatype == z3.RealSort():
        return float(z3_instance.as_decimal(prec=7).rstrip('?'))
    elif datatype == z3.BoolSort():
        return bool(z3_instance)
    elif datatype == z3.StringSort():
        return str(z3_instance).replace('"', '').replace("'", "")
    else : 
        wrapper = args_types[arg_name].z3()(arg_name)
        if isinstance(args_types[arg_name], AbsTensor) :
            dtype = wrapper.evaluate(model, z3_instance, attr = "dtype")
            args_lengths[arg_name] = wrapper.evaluate(model, z3_instance, attr = "rank")
            if args_lengths[arg_name] is None : 
                args_lengths[arg_name] = random_gen_arr_len()
            else :
                args_lengths[arg_name] = clip_unbound_val(args_lengths[arg_name].as_long(), max = MAX_ARR_LEN)
            values = [wrapper.evaluate(model, z3_instance, attr = "shape", idx = i) for i in range(args_lengths[arg_name])]
            for i in range(len(values)) :
                if values[i] is None :
                    values[i] = random_gen(AbsDType.int)
                else :
                    values[i] = clip_unbound_val(values[i].as_long(), max = MAX_VALUE)
            args_values[arg_name] = AbsTensor(dtype=DType.from_str(str(dtype)), shape=values)
            return args_values[arg_name]
        elif isinstance(args_types[arg_name], AbsIter):
            if args_types[arg_name].get_arg_dtype() in [AbsDType.complex, AbsTensor] : 
                raise NotImplementedError
            args_lengths[arg_name] = wrapper.evaluate(model, z3_instance, attr="len")
            if args_lengths[arg_name] is None :
                args_lengths[arg_name] = random_gen_arr_len()
            else :
                args_lengths[arg_name] = clip_unbound_val(args_lengths[arg_name].as_long(), max = MAX_ARR_LEN)
            args_values[arg_name] = [wrapper.evaluate(model, z3_instance, attr = "value", idx = i) for i in range(args_lengths[arg_name])]
            for i in range(len(args_values[arg_name])) :
                if args_values[arg_name][i] is None :
                    args_values[arg_name][i] = random_gen(args_types[arg_name].get_arg_dtype())
                else :
                    args_values[arg_name][i] = clip_unbound_val(to_conc_py_val(args_values[arg_name][i]), max = MAX_VALUE)
            return args_values[arg_name]  # Returning only the values, not the length
        else :
            raise NotImplementedError(f"Unsupported type {z3_arg.range()}")
        
def process_len(
        args_types : Dict[str, Union[AbsDType, AbsTensor, AbsLiteral]],
        args_lengths : Dict[str, Optional[int]],
        solver : z3.Solver,
        noise : float = 0.0,
        allow_zero_length_rate : float = 0.5,
        ) -> List[z3.ExprRef] :
    
    constrs = []
    for arg_name, arg_type in args_types.items() : 
        if isinstance(arg_type, (AbsTensor, AbsIter)) :
            args_lengths[arg_name] = None
            ## gen default constr 
            len_sym = arg_type.z3()(arg_name).rank
            len_constr = length_default_constraints(len_sym) \
                            if should_generate_noise(allow_zero_length_rate) \
                            else length_not_zero_constraints(len_sym)
            constrs.append(len_constr)
            ## gen noise 
            if should_generate_noise(noise) :
                len_noise = gen_random_int_constr(len_sym, 0, MAX_ARR_LEN)
                constrs.append(len_noise)
    
    # Solving 
    solver.add(constrs)
    if not is_solver_solvable(solver) :
        if SMT_LOG.getEffectiveLevel() <= logging.DEBUG:
            SMT_LOG.debug(f"Solver is not solvable with {solver.assertions()}")
        return None 
    
    model = solver.model()
    constrs.clear()
    for z3_arg in model.decls() :
        sym_to_conc(model, z3_arg, args_types, {}, args_lengths)
    
    for arg_name in args_lengths.keys() :
        if args_lengths[arg_name] is None :
            args_lengths[arg_name] = random.randint(0, MAX_ARR_LEN)

    for arg_name, len_val in args_lengths.items() :
        constrs.append(args_types[arg_name].z3()(arg_name).rank == len_val)
        if isinstance(args_types[arg_name], AbsTensor) :
            assert isinstance(len_val, int) and len_val <= MAX_ARR_LEN, f"Invalid length {len_val}"
            constrs.append(check_numel_constr(args_types[arg_name].z3()(arg_name).shape, len_val))
    
    return constrs

def process_model(solver, args_types : Dict[str, Any]) : 

    args_lengths = {}
    args_values = {}

    if not is_solver_solvable(solver):
        if SMT_LOG.getEffectiveLevel() <= logging.DEBUG:
            SMT_LOG.debug(f"Solver is not solvable with {solver.assertions()}")
        return None
    
    model = solver.model()
    if SMT_LOG.getEffectiveLevel() <= logging.DEBUG:
        SMT_LOG.debug(f"MODEL ---> {model}")
    model_decls = sorted(model.decls(), key=lambda d: d.name())
    for z3_arg in model_decls :
        sym_to_conc(model, z3_arg, args_types, args_values, args_lengths)

    for key, value in args_values.items() : 
        if isinstance(value, dict) and "tensor" in value.keys() :
            args_values[key] = AbsTensor(**value["tensor"])
    return args_values, args_lengths

def gen_val(num_of_try, *args, **kwargs) -> Optional[Dict[str, Any]] : 
    values = None 
    tries = 0
    num_of_try = int(num_of_try * kwargs["noise_prob"])
    while values is None and tries <= num_of_try :
        values = _gen_val(*args, **kwargs)
        tries += 1
    return values

def _gen_val(
          args_types : Dict[str, Union[AbsDType, AbsTensor, AbsLiteral]],
          constrs : List[Callable] = [], 
          noise_prob : float = 0.0,
          allow_zero_length_rate : float = 0.5,
          allow_zero_rate : float = 0.5,
          constraints : List[z3.ExprRef] = [],
          api_name : str = "",
          ) -> Optional[Dict[str, Any]] : 
    """ 
    Gen failed -> return None 
    """
    args_lengths = {}
    ## activate constrs
    constrs = [constr(z3objs={
                        name : abs.z3()(name)
                        for name, abs in args_types.items()
                    }) for constr in constrs]
    solver = init_solver()
    solver.add(constrs)
    len_rules = process_len(args_types, args_lengths, solver, noise=noise_prob, allow_zero_length_rate=allow_zero_length_rate)
    if len_rules is None :
        SMT_LOG.debug(f"rank related rules cannot be satisfied")
        return None 
    # solve all rank and len varaibles_first 
    help_rules = gen_default_constr(args_types, args_lengths, allow_zero_rate=allow_zero_rate)
    noises = add_noises(args_types, args_lengths, noise_prob)
    all_rules = len_rules + help_rules + noises + constraints
    if all_rules : 
        solver.add(all_rules)
    if SMT_LOG.getEffectiveLevel() <= logging.DEBUG:
        SMT_LOG.debug(f"---> Trying to solve: {api_name} ~ {solver.assertions()}")
    results = process_model(solver, args_types)
    if results is not None :
        args_values, args_lengths = results
        for arg_name in args_types.keys() : 
            if args_values.get(arg_name, None) is None :
                args_values[arg_name] = random_gen(args_types[arg_name], args_lengths.get(arg_name))
    else : 
        args_values = None
    return args_values

def init_solver() -> z3.Solver :
    solver = z3.Solver()
    solver.set(timeout=10000)
    return solver

def is_solver_solvable(solver: z3.Solver) -> bool:
    return solver.check() in [z3.sat] ## FIXME : How to deal with z3.unknown

def solve(solver : z3.Solver, 
          args_types : Dict[str, Any]) -> \
                    Tuple[Dict[str, Any], Dict[str, Optional[int]]] : 
    
    args_values, args_lengths = process_model(solver, args_types)
    return args_values, args_lengths

def random_gen_arr_len() -> int :
    return random.randint(0, MAX_ARR_LEN)
def _random_gen(datatype):
    """
    Randomly generate a value based on the given datatype.
    """
    if datatype == AbsDType.bool:
        return random.choice([True, False])
    elif datatype == AbsDType.int:
        return random.randint(0, 100)  # Adjust range as needed
    elif datatype == AbsDType.float:
        return random.uniform(0, 1)
    elif datatype == AbsDType.str:
        return ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    elif isinstance(datatype, AbsLiteral):
        return random.choices(datatype.choices)[0]
    else:
        raise NotImplementedError(f"Unsupported datatype {datatype}")

def random_gen(abs, length=None):
    """
    Materialize an abstract data type into a concrete value.
    """
    if isinstance(abs, AbsTensor):
        if length is None :
            length = random_gen_arr_len()
        shape = [random_gen_arr_len() for _ in range(length)]
        dtype = random.choice(DTYPE_GEN_ALL)
        return AbsTensor(dtype=dtype, shape=shape)
    elif isinstance(abs, AbsIter) : 
        if length is None :
            length = random_gen_arr_len()
        arg_type = abs.get_arg_dtype()
        return [random_gen(arg_type) for _ in range(length)]
    elif isinstance(abs, (AbsDType, AbsLiteral)):
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

# def assign_otheres(args_values : Dict[str, Any], 
#                    args_types : Dict[str, Union[AbsDType, AbsTensor, AbsLiteral]], 
#                    args_lengths : Dict[str, Optional[int]]) -> Dict[str, Any]:
    
#     for arg_name in args_types.keys() :
#         if args_values.get(arg_name) is None :
#             if isinstance(args_types[arg_name], AbsTensor) :
#                 args_values[arg_name] = RandomGenerator.materalrize_abs(args_types[arg_name])
#             if arg_name in args_lengths.keys() and args_lengths[arg_name] is not None :
#                 args_values[arg_name] = [RandomGenerator.materalrize_abs(args_types[arg_name].get_arg_dtype()) for _ in range(args_lengths[arg_name])]
#             else :
#                 args_values[arg_name] = RandomGenerator.materalrize_abs(args_types[arg_name])

#     return args_values


# def gen_sufficient_condition(equations, suff_conds_needed : Dict[str, bool]) -> List[z3.ExprRef] :
#     """
#     Generate sufficient condition for the given equation
#     a[r1] == b[r2] -> r1 >= 0 and r1 < len(a) and r2 >= 0 and r2 < len(b)
#     """     
#     res = []   
#     if any(z3.is_app_of(equations, op) for op in COMPARISON_KINDS) : 
#         res.extend(_gen_sufficient_condition(equations, suff_conds_needed))
#     elif any(z3.is_app_of(equations, op) for op in BOOL_KINDS) : 
#         for equation in equations.children() : 
#             res.extend(gen_sufficient_condition(equation, suff_conds_needed))
#     return res

# def _gen_sufficient_condition(equation, suff_conds_needed : Dict[str, bool]) -> List[z3.ExprRef] :
#     """
#     Generate sufficient condition for the given equation
#     a[r1] == b[r2] -> r1 >= 0 and r1 < len(a) and r2 >= 0 and r2 < len(b)
#     """        

#     res = []
#     if any(z3.is_app_of(equation, op) for op in COMPARISON_KINDS) : 
#         for subscript in equation.children() : # left, right
#             if z3.is_app_of(subscript, z3.Z3_OP_SELECT) :
#                 subscript_tensor, subscript_idx = subscript.children()
#                 subscript_len = gen_len_obj(subscript_tensor, suff_conds_needed)
#                 if subscript_len is None : continue
#                 subscript_sufficient = z3.And(subscript_idx >= 0, subscript_idx < subscript_len)
#                 res.append(subscript_sufficient)
#     return res

def to_conc_py_val(z3_obj: z3.ExprRef) -> Any:
    if isinstance(z3_obj, z3.IntNumRef):
        return z3_obj.as_long()
    elif isinstance(z3_obj, z3.RatNumRef):
        return float(z3_obj.as_decimal(prec=7).rstrip('?'))
    elif isinstance(z3_obj, z3.BoolRef):
        return bool(z3_obj)
    else :
        raise NotImplementedError

# def process_tensor_rank(z3_arg: z3.FuncDecl, model: z3.Model) -> int:
#     rank = model.evaluate(TensorZ3.rank(model[z3_arg]))
#     return rank.as_long()