import traceback
from typing import *

import hydra
from omegaconf import DictConfig
from neuri.constrinf import _process_record, gen_inst_with_records, make_record_finder
from neuri.constrinf.ast2z3 import Ast2z3
from neuri.constrinf.executor import Executor
from neuri.materialize import Model
from tests.smt import test_smt

def test_with_given_constraints(constrs, arg_names, dtypes) :

    all_results = []
    func_name = 'test'
    for constr in constrs:
        result = "error"
        try :
            converter = Ast2z3(arg_names, dtypes, constr, func_name)
            print(f"{func_name}-Constraint: {constr}")
            result = converter.convert()
            all_results.append(result)
            print(f"Z3: {result}\n")
            print(f"suff conds : {converter.pretty_flags()}\n")
        except :
            print(traceback.format_exc())
            print(constr, func_name)
        print(f"Constraint: {constr}\nMimic Z3: {result}\n")
    return all_results

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
    
@hydra.main(version_base=None, config_path="../neuri/config/", config_name="main")
def main(cfg: DictConfig):
    from neuri.constrinf.smt_funcs import load_z3_const
    from neuri.abstract.dtype import AbsDType
    from neuri.abstract.dtype import AbsTensor
    ## whole constraints testing ##
    # test_whole_constraints()

    ## target constr testing ##
    """
    Example Usage : test_with_given_constraints
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
    test_smt(arg_names, [d[0] for d in dtypes], constrs, noise_prob=0.3)
    """
    # model = Model.init(
    #     "torch", backend_target="torchcomp"
    # )
    # executor = Executor(model, parallel = cfg["train"]["parallel"])
    # record = _process_record("/artifact/data/records/torch/Tensor/abs_-0.yaml")
    # arg_names = record['args']['name']
    # test_constraints = [
    #     "dtype(self) == complex",
    # ]
    # constrs = test_with_given_constraints(test_constraints, arg_names, record["args"]["dtype_obj"])
    # executor.execute(
    #     1, constrs, record=record
    # )
    arg_names = ['a', 'b','c','d']
    dtypes = [
        [AbsTensor()],
        [AbsTensor()],
        [AbsTensor.to_iter()],
        [AbsDType.str],
        ]
    test_constraints = [
        # "all(a.shape[i] == b.shape[i] or a.shape[i] == 1 or b.shape[i] == 1 for i in range(-1, -min(len(a.shape), len(b.shape))-1, -1))",
        "-len(a.shape) <= 3 < len(a.shape)",
        # "a.shape == b.shape",
        # "d in ['batch', 'layer', 'instance']",
        # "d < len(self.shape) for d in c"
        # "all(i > len(c) and i < len(c) for i in a)",
        # "alld > len(c)",
        # "all((a[i]>1 and a[i]<4) for i in a[2:])",
        # "c[0].shape[0] == b[0]",
        # 'a[-1] > b[-2]'
    ]
    for i in range(10) :
        constrs = test_with_given_constraints(test_constraints, arg_names, dtypes)
        test_smt(arg_names, [d[0] for d in dtypes], constrs, noise_prob=0.3)


if __name__ == "__main__" :
     main()
    #  (len(mat2.shape) == 2) or ((out.shape == [mat1.shape[0], mat2.shape[1]]) or (len(input) == len(mat1)))

