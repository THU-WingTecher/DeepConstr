from abstract.dtype import AbsDType, AbsIter, AbsLiteral
from specloader.materalize import RandomGenerator
from specloader import BOOL_POOLS, OP_POOLS
from neuri.constrinf.smt_funcs import TensorZ3
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
        shape_var = TensorZ3.shape(arg_type.z3()(arg_name))
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
            dtype_var = TensorZ3.dtype(arg_type.z3()(arg_name))
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