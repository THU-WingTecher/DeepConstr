import argparse
import multiprocessing as mp
import os
import pickle

import z3

from nnsmith.autoinf.inference.const import DATA_DIR, GEN_DIR
from nnsmith.autoinf.inference.invocations import inst_augmented_invocations
from nnsmith.logger import AUTOINF_LOG
from nnsmith.autoinf.inference.utils import equivalent
from nnsmith.autoinf.instrument.categorize import gen_inst_with_records
from nnsmith.autoinf.instrument.op import OpInstance


def solve_inst(inst: OpInstance, dump_dir: str):
    # print(inst.name_index, "start", flush=True)
    library = "torch" if "torch" in inst.name_index else "tf"
    validity_dir = os.path.join(dump_dir, f"{library}_rules_validity")
    shape_valid, constraint_valid = True, True
    symb_dict = dict()
    for symb in inst.I + inst.A:
        exec(f"symb_dict['{symb}'] = z3.Int('{symb}')")
    type_transfer_rules = inst.type_transfer_expressions(symb_dict)
    for o in inst.O:
        if len(type_transfer_rules[o]) == 0:
            shape_valid = False
            break
    fail_records = inst_augmented_invocations(
        inst, "fail", os.path.join(dump_dir, f"{library}_augmented_records")
    )
    for fail_record in fail_records:
        concrete_dict = dict()
        for i, val in enumerate(fail_record):
            concrete_dict[f"s{i}"] = val
        predicates = inst.requires_expressions(concrete_dict)
        rejectable = False
        for predicate in predicates:
            if equivalent(predicate, False):
                rejectable = True
                break
        if not rejectable:
            constraint_valid = False
            break
    with open(os.path.join(validity_dir, f"{inst.name_index}.pkl"), "wb") as f:
        pickle.dump(
            [shape_valid and constraint_valid, shape_valid, constraint_valid], f
        )
    # print(inst.name_index, "complete", flush=True)
    AUTOINF_LOG.info(f"{inst.name_index} complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_dir", type=str, default=GEN_DIR)
    parser.add_argument("--parallel", type=int, default=16)
    parser.add_argument("--library", nargs="+", default=["tf", "torch"])
    args = parser.parse_args()
    for library in args.library:
        record_dir = os.path.join(DATA_DIR, f"{library}_records")
        validity_dir = os.path.join(args.dump_dir, f"{library}_rules_validity")
        os.system(f"rm {validity_dir} -r")
        os.makedirs(validity_dir)
        gen_inst_records = gen_inst_with_records(
            data_dir=record_dir,
            int_policy="fix_dim",
        )
        p = mp.Pool(args.parallel)
        for i_op, (inst, _) in enumerate(gen_inst_records):
            # solve_inst(inst)
            p.apply_async(solve_inst, (inst, args.dump_dir))
        p.close()
        p.join()
