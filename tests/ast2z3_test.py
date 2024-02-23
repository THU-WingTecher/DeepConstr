import traceback
from typing import *
from neuri.constrinf import gen_inst_with_records, make_record_finder
from neuri.constrinf.ast2z3 import Ast2z3

def test_with_given_constraints(constrs, arg_names, dtypes) :

    all_results = []
    func_name = 'test'
    for constr in constrs:
        result = "error"
        try :
            converter = Ast2z3(arg_names, dtypes, constr, func_name)
            print(f"{func_name}-Constraint: {constr}")
            result = converter.convert()
            print(f"Z3: {result}\n")
            print(f"suff conds : {converter.pretty_flags()}\n")
        except :
            print(traceback.format_exc())
            print(constr, func_name)
        print(f"Constraint: {constr}\nMimic Z3: {result}\n")


def test_whole_constraints(dir_path = None ) :

    if dir_path is None :
        dir_path = "/artifact/data/constraints"
    # stop_func = "tf.raw_ops.TruncateMod"
    # idx = [i for i, r in enumerate(gen_inst_with_records(dir_path)) if r['name'] == stop_func][0]
    cnt = 0
    records = gen_inst_with_records(dir_path)
    for i, record in enumerate(records) : 
        # if i < idx : continue
        arg_names = record['args']['name']
        dtypes = record['args']['dtype']
        for constr in record['rules']:
            cnt+=1
            result = "error"
            converter  = Ast2z3(arg_names, dtypes, constr['txt'], record['name'])
            print(f"{record['name']}-Constraint: {constr['txt']}")
            result = converter.convert()
            if "type(" not in constr['txt'] and "isinstance" not in constr['txt'] and result is None : 
                print("stop")
            print(f"Z3: {result}\n")
            print(f"suff conds : {converter.pretty_flags()}\n")

    print(f"NL_CONSTR ------> SMT CONSTR : {cnt} CASES TEST COMPLETED")

if __name__ == "__main__" :
     
    #  (len(mat2.shape) == 2) or ((out.shape == [mat1.shape[0], mat2.shape[1]]) or (len(input) == len(mat1)))

    from neuri.constrinf.smt_funcs import load_z3_const
    from neuri.abstract.dtype import AbsDType
    from neuri.abstract.tensor import AbsTensor
    # test_whole_constraints()
    #  Example Usage : test_with_given_constraints
    arg_names = ['a', 'b','c']
    dtypes = [
        [AbsDType.int.to_iter()],
        [AbsDType.int.to_iter()],
        [AbsTensor.to_iter()],
        ]
    test_constraints = [
        "all((a[i]>1 and a[i]<4) for i in a[2:])",
        "c[0].shape[0] == b[0]",
        'a[-1] > b[-2]'
    ]
    test_with_given_constraints(test_constraints, arg_names, dtypes)