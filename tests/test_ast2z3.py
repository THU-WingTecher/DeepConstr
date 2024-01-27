import ast
import random
import string
import traceback
import operator as op 
import z3
from typing import *
from neuri.abstract.dtype import AbsDType, DType
from neuri.constrinf import gen_inst_with_records
from neuri.constrinf.ast2z3 import ast2z3

def test_with_given_constraints(constrs, arg_names, dtypes) :

    all_results = []
    func_name = 'test'
    for constr in constrs:
        result = "error"
        try :
            result = ast2z3(arg_names, dtypes, constr, func_name).convert()
            # print(result)
        except :
            print(traceback.format_exc())
            print(constr, func_name)
        print(f"Constraint: {constr}\nMimic Z3: {result}\n")


def test_whole_constraints(dir_path = None ) :

    if dir_path is None :
        dir_path = "/artifact/data/constraints"
    records = gen_inst_with_records(dir_path)
    for record in records : 
        arg_names = record['args']['name']
        dtypes = record['args']['dtype']
        rule_txts = [r['txt'] for r in record['rules']]
        for constr in rule_txts:
            result = "error"
            try :
                converter  = ast2z3(arg_names, dtypes, constr, record['name'])
                result = converter.convert()
            except :
                print(traceback.format_exc())
                print(constr, record['name'])
            print(f"Constraint: {constr}\nMimic Z3: {result}\n")

test_whole_constraints()
#  Example Usage : test_with_given_constraints
# arg_names = ['a', 'b']
# dtypes = [load_z3_const('a','int', is_array=True), load_z3_const('b','int', is_array=True)]
# test_constraints = [
#     "all((a[i]>1 and a[i]<4) for i in range(len(a[2:])))",
#     "((i>1 and i<4) for i in a[2:])",
#     'a[-1] > b[-2]', 'a[:-1] == b[1:]'
# ]
# test_with_given_constraints(test_constraints, arg_names, dtypes)