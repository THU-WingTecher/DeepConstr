from abstract.dtype import AbsDType, AbsIter, AbsLiteral
from specloader.materalize import RandomGenerator
from specloader import Z3TENSOR, BOOL_POOLS, OP_POOLS
from neuri.abstract.tensor import AbsTensor
from specloader.rule import Rule
from typing import Union, List, Tuple, Dict, Any, Optional
import copy
import random
import z3 

def classify_rules(rules : List[Rule]) -> Tuple[List[Rule], List[Rule], List[Rule], List[Rule]] :
    """ 
    Rules -> type_rules, type_match_rules, others, type_constraints
    """

    z3_rules = []
    non_z3_rules = []
    for rule in rules :
        if rule.activated : 
            continue 
        if rule.ast2z3.is_unsolverable() :
            non_z3_rules.append(rule)
        else :
            z3_rules.append(rule)
    
    return z3_rules, non_z3_rules

def divide(target : str) : 
    
    """
    txt : a or b -> a, b 
    txt : a and b -> a, b
    txt : a or (b or c) -> a, b or c
    txt : (a) or (b or c and (d and e)) -> a, b or c and (d and e)
    """
    parts = []
    split_hints = [' or ', ' and '] 
    split_mark = "##"
    if not any(hint in target for hint in split_hints) :
        return [] 
    for hint in split_hints :
        target = target.replace(hint, split_mark)
    
    for part in target.split(split_mark) :
        while part.count('(') != part.count(')') :
            if part.count('(') == 0 and part.count(')') == 0 : break
            if part.count('(') > part.count(')') :
                pos = part.index('(')
                part = part[:pos] + part[pos+1:]
                
            if part.count('(') < part.count(')') :
                pos = part.rindex(')')
                part = part[:pos] + part[pos+1:]
        parts.append(part)
    return parts

def is_mergeable(rules : List[Rule]) : 
    
    if len(rules) < 2 :
        return False 
    
    z3_rules, non_z3_rules = classify_rules(rules) # z3_rules, type_rules, type_match_rules, others
    if len(non_z3_rules) > 0 :
        return False 
    return True 

def gen_opposed_rule(rule : Rule) : 
    """
    return (opposed_rule, orig_rule)"""
    from specloader import NOT_SIGN
    if rule.ast2z3.is_unsolverable() :
        for arg_name in rule.related_args :
            if not rule.ast2z3.is_in_types_map(arg_name) :
                continue
            for i, mapped_tupple in enumerate(rule.ast2z3.types_map[arg_name]) :
                mapped_tupple = list(mapped_tupple)
                mapped_tupple[0] = NOT_SIGN if mapped_tupple[0] is None else None
                rule.ast2z3.types_map[arg_name][i] = tuple(mapped_tupple)
        return rule
    else :
        opposed_rule = copy.deepcopy(rule) 
        opposed_rule.re_init(rule.export_opposed())
        return opposed_rule

##
def add_noises(args_types, args_lengths, noise):
    noises = []
    noise_dist = {arg_name: noise if isinstance(noise, float) else noise.get(arg_name, 0) for arg_name in args_types}

    for arg_name, arg_noise in noise_dist.items():
        if not arg_noise:
            continue
        noises.extend(
            gen_noise(arg_name, args_types[arg_name], args_lengths[arg_name])
        )
    return noises

def should_generate_noise(noise_prob):
    return random.random() < noise_prob

def gen_noise(arg_name, arg_type, arg_length, noise_prob):
    noises = []

    if isinstance(arg_type, AbsLiteral) and should_generate_noise(noise_prob):
        literal_noise = random.choice(BOOL_POOLS)(arg_type.z3()(arg_name), random.choice(arg_type.choices))
        noises.append(literal_noise)

    elif isinstance(arg_type, AbsTensor):
        # Assuming AbsTensor requires generation of shape, rank, and dtype noises
        shape_var = Z3TENSOR.shape(arg_type.z3()(arg_name))
        for dim in shape_var:
            if should_generate_noise(noise_prob):
                # Generate noise for each dimension
                noise = random.choice(OP_POOLS)(dim, random.randint(__MIN_VAL__, __MAX_VAL__))
                noises.append(noise)

        if should_generate_noise(noise_prob):
            # Generate rank noise
            rank_noise = random.choice(OP_POOLS)(arg_length, random.randint(__MIN_VAL__, __MAX_VAL__))
            noises.append(rank_noise)

        if len(arg_type.possible_dtypes) > 1 and should_generate_noise(noise_prob):
            # Generate dtype noise
            dtype_var = Z3TENSOR.dtype(arg_type.z3()(arg_name))
            dtype_noise = random.choice(BOOL_POOLS)(dtype_var, random.choice(arg_type.possible_dtypes).z3())
            noises.append(dtype_noise)

    elif isinstance(arg_type, AbsIter):
        # Generate noise for iterable types
        
        if should_generate_noise(noise_prob):
            iter_noise = random.choice(OP_POOLS)(arg_type.z3()(arg_name), random.materalrize_abs(arg_type.get_arg_dtype()))
            noises.append(iter_noise)

    elif isinstance(arg_type, AbsDType) and arg_type != AbsDType.str and should_generate_noise(noise_prob):
        # Generate noise for data types
        dtype_noise = random.choice(OP_POOLS)(arg_type.z3()(arg_name), random.materalrize_abs(arg_type))
        noises.append(dtype_noise)

    return noises


def gen_noise(arg_name, arg_type, arg_length, num_of_gen=1):
    noises = []
    noise_idx = 1

    def generate_base_noise(sym_idx):
        base_req = z3.And(sym_idx <= arg_length, sym_idx >= 0)
        return base_req

    if isinstance(arg_type, AbsLiteral):
        literal_noise = RandomGenerator.choice(BOOL_POOLS)(arg_type.z3()(arg_name), RandomGenerator.choice(arg_type.choices))
        noises.append(literal_noise)

    elif isinstance(arg_type, (AbsTensor, AbsIter)):
        is_tensor = isinstance(arg_type, AbsTensor)
        dtype = Z3TENSOR.dtype(arg_type.z3()(arg_name)) if is_tensor else None
        comparator_type = AbsDType.int if is_tensor else arg_type.get_arg_dtype()
        op_pool = OP_POOLS if comparator_type != AbsDType.bool else BOOL_POOLS

        for _ in range(num_of_gen):
            sym_idx = AbsDType.int.z3()(f'n_{noise_idx}')
            noise_idx += 1
            noise = z3.And(generate_base_noise(sym_idx), RandomGenerator.choice(op_pool)(arg_type.z3()(arg_name)[sym_idx], RandomGenerator.materalrize_abs(comparator_type)))
            noises.append(noise)

        if is_tensor and len(arg_type.possible_dtypes) > 1:
            dtype_noise = RandomGenerator.choice(BOOL_POOLS)(dtype, RandomGenerator.choice(arg_type.possible_dtypes).z3())
            noises.append(dtype_noise)

        rank_noise = RandomGenerator.choice(op_pool)(arg_length, RandomGenerator.materalrize_abs(AbsDType.int))
        noises.append(rank_noise)

    elif isinstance(arg_type, AbsDType) and arg_type != AbsDType.str:
        dtype_noise = RandomGenerator.choice(OP_POOLS if arg_type != AbsDType.bool else BOOL_POOLS)(arg_type.z3()(arg_name), RandomGenerator.materalrize_abs(arg_type))
        noises.append(dtype_noise)

    return noises


        if isinstance(args_types[arg_name], (AbsTensor, AbsIter)):
            selected = random.sample(gen, int(len(gen) * arg_noise))
            noises.extend(selected)
        else:  # AbsLiteral, AbsDType
            if RandomGenerator.random_choice_int((1, 10)) <= arg_noise * 10:
                noises.extend(gen)

    return noises

####
def gen_noise(arg_name,
              arg_type : Union[AbsDType, AbsIter, AbsLiteral],
              arg_length,
              num_of_gen : Optional[int] = 1
              ) -> Union["z3.ExprRef", bool] :
            
    noises = []
    noise_idx=1
    if isinstance(arg_type, AbsLiteral) :
        noises.append(RandomGenerator.choice(BOOL_POOLS)(arg_type.z3()(arg_name), RandomGenerator.choice(arg_type.choices)))

    elif isinstance(arg_type, AbsTensor) :
        shape_var = Z3TENSOR.shape(arg_type.z3()(arg_name))
        comparator_type = AbsDType.int
        dtype_comps = arg_type.possible_dtypes
        dtype_var = Z3TENSOR.dtype(arg_type.z3()(arg_name))
        for i in range(num_of_gen) :
            sym_idx = AbsDType.int.z3()(f'n_{noise_idx}')
            noise_idx+=1
            base_req = z3.And(sym_idx <= arg_length, sym_idx >= 0)
            noises.append(
                z3.And(base_req,
                RandomGenerator.choice(OP_POOLS)(shape_var[sym_idx], RandomGenerator.materalrize_abs(comparator_type))
                ))
        if len(dtype_comps) > 1 :
            noises.append(RandomGenerator.choice(BOOL_POOLS)(dtype_var, RandomGenerator.choice(dtype_comps).z3()))
        noises.append(RandomGenerator.choice(OP_POOLS)(arg_length, RandomGenerator.materalrize_abs(comparator_type))) # rank noise 

    elif isinstance(arg_type, AbsIter) :
        comparator_type_abs = arg_type.get_arg_dtype()
        op_pool = OP_POOLS if comparator_type_abs != AbsDType.bool else BOOL_POOLS
        curop = RandomGenerator.choice(op_pool)
        arr_var = arg_type.z3()(arg_name)
        for i in range(num_of_gen) :
            sym_idx = z3.Int(f'n_{noise_idx}')
            noise_idx+=1
            base_req = z3.And(sym_idx <= arg_length, sym_idx >= 0)
            noises.append(
                z3.And(base_req,
                RandomGenerator.choice(OP_POOLS)(arr_var[sym_idx], RandomGenerator.materalrize_abs(comparator_type_abs))
                ))
        noises.append(RandomGenerator.choice(OP_POOLS)(arg_length, RandomGenerator.materalrize_abs(AbsDType.int))) # rank noise 

    elif isinstance(arg_type, AbsDType) :
        if not arg_type == AbsDType.str : 
            op_pool = OP_POOLS if arg_type != AbsDType.bool else BOOL_POOLS
            curop = RandomGenerator.choice(op_pool)
            comparator = RandomGenerator.materalrize_abs(arg_type)
            noise = curop(arg_type.z3()(arg_name), comparator)
            noises.append(noise)

    return noises

def add_noises(
        args_types : Dict[str, Union[AbsDType, AbsTensor, AbsIter, AbsLiteral]],
        args_lengths : Dict[str, Optional["z3.ExprRef"]],
        noise : Union[float, Dict[str,float]]
        ) -> List["z3.ExprRef"] :
    noises = []
    noise_dist = {}
    if type(noise) != dict : 
        noise_dist = {arg_name : noise for arg_name in args_types.keys()}
    else :
        noise_dist = noise

    for arg_name in args_types.keys() :
        if noise_dist[arg_name] : # != 0
            if isinstance(args_types[arg_name], AbsTensor) :
                gen = gen_noise(arg_name, args_types[arg_name], args_lengths[arg_name])
                num_elements_to_select = int(len(gen)*noise_dist[arg_name]) # 0.1 -> 10% of elements
                selected = random.sample(gen, num_elements_to_select)
                noises.extend(selected)
            elif isinstance(args_types[arg_name], AbsIter) :
                gen = gen_noise(arg_name, args_types[arg_name], args_lengths[arg_name])
                num_elements_to_select = int(len(gen)*noise_dist[arg_name])# 0.1 -> 10% of elements
                selected = random.sample(gen, num_elements_to_select)
                noises.extend(selected)
            else : # AbsLiteral, AbsDType
                if RandomGenerator.random_choice_int((1,10)) <= noise_dist[arg_name] * 10 :# 0.1 -> gen nosie of 10% probability
                    noise = gen_noise(arg_name, args_types[arg_name], args_lengths[arg_name])
                    noises.extend(noise)
    return noises