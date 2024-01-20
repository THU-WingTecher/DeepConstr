from functools import partial
import logging
import random
import z3
from abstract.dtype import DTYPE_GEN_ALL, DTYPE_NOT_SUPPORTED, AbsDType, AbsIter, AbsLiteral, DType
from neuri.logger import SMT_LOG
from specloader.materalize import RandomGenerator
from neuri.abstract.tensor import AbsTensor 
from typing import Dict, Any, List, Literal, Optional, Tuple, Union
from specloader import  BOOL_POOLS, MAX_ARR_LEN, MAX_SHAPE_SUM, MAX_VALUE, MIN_VALUE, OP_POOLS, Z3DTYPE, Z3TENSOR, CONSTRAINTS, DEFAULT_PREMISES, check_numel_constr, \
    gen_arr_len_z3, COMPARISON_KINDS, BOOL_KINDS, gen_len_obj, length_default_constraints, length_not_zero_constraints, pos_constraints, z3_funcs
from specloader.irs import SYM_LEN
import operator

__NOISE_MAX_VAL__ = 10
__NOISE_MIN_VAL__ = -3

def gen_dtype_constraints(arg_name : str, not_supported_dtypes : List[DType]) -> z3.ExprRef :
    
    assert not_supported_dtypes, "DTYPE_NOT_SUPPORTED not defined"
    constr = z3_funcs.not_in(
        Z3TENSOR.dtype(AbsTensor.z3()(arg_name)), 
        [
        dtype.z3() for dtype in not_supported_dtypes 
    ]
    )
    return constr

DEFAULT_DTYPE_CONSTR : Dict[str, z3.ExprRef] = {
    "torch" : partial(gen_dtype_constraints, not_supported_dtypes=DTYPE_NOT_SUPPORTED.get("torch")),
    "tensorflow" : partial(gen_dtype_constraints, not_supported_dtypes=DTYPE_NOT_SUPPORTED.get("tensorflow")),
}

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
    if min is None : min = __NOISE_MIN_VAL__
    if max is None : max = __NOISE_MAX_VAL__
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
        shape_var = Z3TENSOR.shape(arg_type.z3()(arg_name))
        for idx in range(args_length) :
            if should_generate_noise(noise_prob):
                # Generate noise for each dimension
                noise = gen_random_int_constr(shape_var[idx], 0, __NOISE_MAX_VAL__)
                noises.append(noise)

        if should_generate_noise(noise_prob):
            # Generate dtype noise
            dtype_var = Z3TENSOR.dtype(arg_type.z3()(arg_name))
            dtype_noise = gen_radnom_list_choice_constr(dtype_var, [dtype.z3() for dtype in DType], [operator.ne])
            noises.append(dtype_noise)

    elif isinstance(arg_type, AbsIter):
        list_var = arg_type.z3()(arg_name)
        for idx in range(args_length) :
            if should_generate_noise(noise_prob):
                # Generate noise for each dimension
                noise = gen_random_int_constr(list_var[idx], __NOISE_MIN_VAL__, __NOISE_MAX_VAL__)
                noises.append(noise)

    elif isinstance(arg_type, AbsDType) : 
        if should_generate_noise(noise_prob) : 
            if arg_type in [AbsDType.bool] :

                dtype_noise = gen_radnom_list_choice_constr(arg_type.z3()(arg_name), [True, False])
            elif arg_type in [AbsDType.int, AbsDType.float, AbsDType.complex] :
                dtype_noise = gen_random_int_constr(arg_type.z3()(arg_name), __NOISE_MIN_VAL__, __NOISE_MAX_VAL__)
            else : # AbsDType.str 
                dtype_noise = []
                pass # dont generate noise for str 
            noises.append(dtype_noise)

    return noises

def gen_default_constr(
        args_types : Dict[str, Union[AbsDType, AbsTensor, AbsLiteral]],
        args_lengths : Dict[str, Optional[int]],
        allow_zero_rate : float = 0.5,
                        ) -> List[z3.ExprRef] :
    rules = []
    for arg_name in args_types.keys() :
        if isinstance(args_types[arg_name], AbsTensor) :
            if should_generate_noise(allow_zero_rate) : 
                include_zero = True
            else :
                include_zero = False
            rules.append(
                pos_constraints(Z3TENSOR.shape(args_types[arg_name].z3()(arg_name)),
                                                args_lengths[arg_name],
                                                include_zero)
                )
        elif isinstance(args_types[arg_name], AbsLiteral) :
            rules.append(z3_funcs.in_(args_types[arg_name].z3()(arg_name), args_types[arg_name].choices))
        else :
            continue 
    return rules

def process_len(args_types : Dict[str, Union[AbsDType, AbsTensor, AbsLiteral]],
              solver : z3.Solver,
              noise : float = 0.0,
              allow_zero_length_rate : float = 0.5,
              ) -> Tuple[List[z3.ExprRef], Dict[str, int]] :
    
    constrs = []
    res = []
    args_len : Dict[str, int] = {}
    for arg_name, arg_type in args_types.items() : 
        if isinstance(arg_type, (AbsTensor, AbsIter)) :
            args_len[arg_name] = None
            ## gen default constr 
            is_tensor = isinstance(arg_type, AbsTensor)
            len_sym = Z3TENSOR.rank(arg_type.z3()(arg_name)) if is_tensor else gen_arr_len_z3(arg_name)
            len_constr = length_default_constraints(len_sym) \
                            if should_generate_noise(allow_zero_length_rate) \
                            else length_not_zero_constraints(len_sym)
            ## gen noise 
            if should_generate_noise(noise) :
                len_noise = gen_random_int_constr(len_sym, 0, MAX_ARR_LEN)
                constrs.extend([len_constr, len_noise])
    
    # Solving 
    solver.add(constrs)
    if not is_solver_solvable(solver) :
        if SMT_LOG.getEffectiveLevel() <= logging.DEBUG:
            SMT_LOG.debug(f"Solver is not solvable with {solver.assertions()}")
        return None 
    
    model = solver.model()
    
    for z3_arg in model.decls() :
        arg_name = z3_arg.name()
        if arg_name.endswith('len') : 
            arg_name = arg_name.split('_')[0]
            args_len[arg_name] = model[z3_arg].as_long()
            res.append(gen_arr_len_z3(arg_name) == args_len[arg_name])
        elif z3_arg.range() == Z3TENSOR :
            arg_name = z3_arg.name()
            args_len[arg_name] = process_tensor_rank(z3_arg, model)
            arg_type = args_types[arg_name]
            z3_object = arg_type.z3()(arg_name)         
            res.append(z3.And(Z3TENSOR.rank(z3_object) == args_len[arg_name], 
                              check_numel_constr(Z3TENSOR.shape(z3_object), args_len[arg_name])))
    
    for arg_name in args_len.keys() :
        if args_len[arg_name] is None :
            args_len[arg_name] = random.randint(1, MAX_ARR_LEN)
    return res, args_len

def gen_val(
          args_types : Dict[str, Union[AbsDType, AbsTensor, AbsLiteral]],
          rules : List["Rule"] = [], 
          noise_prob : float = 0.0,
          allow_zero_length_rate : float = 0.5,
          allow_zero_rate : float = 0.5,
          constraints : List[z3.ExprRef] = [],
          api_name : str = "",
          ) -> Optional[Dict[str, Any]] : 
    """ 
    Gen failed -> return None 
    """
    
    solver = init_solver()
    len_results = process_len(args_types, solver, noise=noise_prob, allow_zero_length_rate=allow_zero_length_rate)
    if len_results is None :
        return None 
    len_rules, args_lengths = len_results
    # solve all rank and len varaibles_first 
    help_rules = gen_default_constr(args_types, args_lengths, allow_zero_rate=allow_zero_rate)
    noises = add_noises(args_types, args_lengths, noise_prob)
    solver.add(
        rules +
        len_rules +
        help_rules +
        noises +
        CONSTRAINTS + 
        constraints
    )
    if SMT_LOG.getEffectiveLevel() <= logging.DEBUG:
        SMT_LOG.debug(f"---> Trying to solve: {api_name} ~ {help_rules + CONSTRAINTS + constraints}")
    results = process_model(solver, args_types)
    if results is not None :
        args_values, args_lengths = results
        args_values = assign_otheres(args_values, args_types, args_lengths)
    else : 
        args_values = None
    return args_values

def init_solver() -> z3.Solver :
    clear_solver()
    solver = z3.Solver()
    solver.set(timeout=10000)
    solver.add(DEFAULT_PREMISES)
    return solver
def clear_solver() : 
    SYM_LEN.clear()
    CONSTRAINTS.clear()

def gen_sufficient_condition(equations, suff_conds_needed : Dict[str, bool]) -> List[z3.ExprRef] :
    """
    Generate sufficient condition for the given equation
    a[r1] == b[r2] -> r1 >= 0 and r1 < len(a) and r2 >= 0 and r2 < len(b)
    """     
    res = []   
    if any(z3.is_app_of(equations, op) for op in COMPARISON_KINDS) : 
        res.extend(_gen_sufficient_condition(equations, suff_conds_needed))
    elif any(z3.is_app_of(equations, op) for op in BOOL_KINDS) : 
        for equation in equations.children() : 
            res.extend(gen_sufficient_condition(equation, suff_conds_needed))
    return res
def _gen_sufficient_condition(equation, suff_conds_needed : Dict[str, bool]) -> List[z3.ExprRef] :
    """
    Generate sufficient condition for the given equation
    a[r1] == b[r2] -> r1 >= 0 and r1 < len(a) and r2 >= 0 and r2 < len(b)
    """        

    res = []
    if any(z3.is_app_of(equation, op) for op in COMPARISON_KINDS) : 
        for subscript in equation.children() : # left, right
            if z3.is_app_of(subscript, z3.Z3_OP_SELECT) :
                subscript_tensor, subscript_idx = subscript.children()
                subscript_len = gen_len_obj(subscript_tensor, suff_conds_needed)
                if subscript_len is None : continue
                subscript_sufficient = z3.And(subscript_idx >= 0, subscript_idx < subscript_len)
                res.append(subscript_sufficient)
    return res
def to_py(z3_obj: z3.ExprRef) -> Any:
    if isinstance(z3_obj, z3.IntNumRef):
        return z3_obj.as_long()
    elif isinstance(z3_obj, z3.RatNumRef):
        return float(z3_obj.as_decimal(prec=7).rstrip('?'))
    elif isinstance(z3_obj, z3.BoolRef):
        return bool(z3_obj)
    else :
        raise NotImplementedError

def is_solver_solvable(solver: z3.Solver) -> bool:
    return solver.check() in [z3.sat] ## FIXME : How to deal with z3.unknown

def solve(solver : z3.Solver, 
          args_types : Dict[str, Any]) -> \
                    Tuple[Dict[str, Any], Dict[str, Optional[int]]] : 
    
    
    args_values, args_lengths = process_model(solver, args_types)
    return args_values, args_lengths

def process_tensor_rank(z3_arg: z3.FuncDecl, model: z3.Model) -> int:
    rank = model.evaluate(Z3TENSOR.rank(model[z3_arg]))
    return rank.as_long()

def sym_to_conc(model, z3_arg, args_types, args_values, args_lengths) : 
    """
    fill the args_values and args_lengths based on model of solver
    """

    arg_name = z3_arg.name() 
    if arg_name.endswith('len') : 
        key = arg_name.split('_')[0]
        args_lengths[key] = model[z3_arg].as_long()
    if arg_name not in args_types.keys() :
        return None
    z3_instance = model[z3_arg]
    if z3_arg.range() == Z3TENSOR :
        dtype = model.evaluate(Z3TENSOR.dtype(z3_instance))
        rank = model.evaluate(Z3TENSOR.rank(z3_instance))
        if rank == 0 : 
            shape = [] 
        else :
            shape = [model.evaluate(Z3TENSOR.shape(z3_instance)[i]).as_long() for i in range(rank.as_long())]
        args_lengths[arg_name] = rank
        args_values[arg_name] = AbsTensor(shape=shape, 
                                            dtype=DType.from_str(str(dtype)))
    elif z3_arg.range() == z3.IntSort() :
        args_values[arg_name] = assign(z3_instance.as_long())
    elif z3_arg.range() == z3.RealSort() :
        args_values[arg_name] = assign(float(z3_instance.as_decimal(prec=7).replace('?','')))
    elif z3_arg.range() == z3.BoolSort() :
        args_values[arg_name] = bool(z3_instance)
    elif z3_arg.range() == z3.StringSort() :
        args_values[arg_name] = str(z3_instance).replace("\"", "").replace("\'", "")
    elif z3_arg.range().kind() == z3.Z3_ARRAY_SORT :
        length = model.evaluate(gen_arr_len_z3(arg_name)).as_long()
        args_values[arg_name] = [assign(to_py(model.evaluate(z3_instance[i]))) for i in range(length)]
        args_lengths[arg_name] = length
    else :
        args_lengths[arg_name] = None
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

    return args_values, args_lengths

def assign(value) : 
    if value <= MIN_VALUE : 
        return MIN_VALUE
    elif value >= MAX_VALUE : 
        return MAX_VALUE
    else :
        return value 

def is_equiv_constraint(A : z3.ExprRef, B : z3.ExprRef) -> bool :
    s = z3.Solver()
    s.add(z3.Not(z3.Implies(A, B)))
    if s.check() == z3.unsat:
        s.reset()
        s.add(z3.Not(z3.Implies(B, A)))
        if s.check() == z3.unsat:
            return True
    return False

def is_implies(A : z3.ExprRef, B : z3.ExprRef) -> bool :
    s = z3.Solver()
    s.push()
    s.add(z3.And(A,z3.Not(B)))
    if s.check() == z3.unsat: return True 
    s.pop()
    s.add(z3.And(B,z3.Not(A)))
    if s.check() == z3.unsat: return True 
    return False 

def assign_otheres(args_values : Dict[str, Any], 
                   args_types : Dict[str, Union[AbsDType, AbsTensor, AbsLiteral]], 
                   args_lengths : Dict[str, Optional[int]]) -> Dict[str, Any]:
    
    for arg_name in args_types.keys() :
        if args_values.get(arg_name) is None :
            if isinstance(args_types[arg_name], AbsTensor) :
                args_values[arg_name] = RandomGenerator.materalrize_abs(args_types[arg_name])
            if arg_name in args_lengths.keys() and args_lengths[arg_name] is not None :
                args_values[arg_name] = [RandomGenerator.materalrize_abs(args_types[arg_name].get_arg_dtype()) for _ in range(args_lengths[arg_name])]
            else :
                args_values[arg_name] = RandomGenerator.materalrize_abs(args_types[arg_name])

    return args_values