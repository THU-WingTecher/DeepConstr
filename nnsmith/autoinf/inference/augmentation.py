import argparse
import multiprocessing as mp
import os
import pickle
from copy import deepcopy
from functools import partial
from random import randint

import tensorflow as tf
import torch

from nnsmith.autoinf.inference.const import GEN_DIR, ROOT_DIR
from nnsmith.autoinf.inference.invocations import input_validity_test
from nnsmith.logger import AUTOINF_LOG
from nnsmith.autoinf.inference.utils import (
    compare_array_diff,
    make_list,
    tf_config_gpu,
    time_limit,
)
from nnsmith.autoinf.instrument.categorize import gen_inst_with_records
from nnsmith.autoinf.instrument.op import OpInstance
from nnsmith.autoinf.instrument.utils import (
    data_type_str,
    get_ret_list,
    tensors_from_numpys,
)

skipMutationAPI = [
    "torch._C._nn.fractional.max_pool3d",
    "torch.Tensor.flatten",
    "torch.nn.functional.embedding_bag",
    "torch.histogramdd",
    "torch.is_close",
]

skip_mutation_operator = [
    "tf.raw_ops.MaxPoolGrad-954",
    "tf.raw_ops.MaxPool3DGrad-1787",
    "tf.raw_ops.ComplexAbs-1795",
    "tf.raw_ops.MatrixDiagV3-970",
    "tf.raw_ops.Reshape-135",
    "torch.stack-1773",
    "torch.stack-1752",
    "torch.native_batch_norm-889",
    "torch.cat-2178",
]


class OpDatabase:
    def __init__(self, DB_Name=None):
        from collections import defaultdict

        if DB_Name != None:
            with open(DB_Name, "rb") as f:
                self.DB = pickle.load(f)
        else:
            self.DB = defaultdict(set)
        self.related = []
        self.unrelated = []
        self.aliases = []

    def Add(self, inputs, outputs):
        if outputs == None:
            self.DB["fail"].add(tuple(inputs))
        else:
            self.DB["success"].add((tuple(inputs), tuple(outputs)))

    def check_duplicate_sym(self, sym_set: int) -> bool:
        for (i, j) in self.aliases:
            if (sym_set & (1 << i)) != 0 and (sym_set & (1 << j)) != 0:
                return True
        return False

    def InvocationCount(self, type=None) -> int:
        count = 0
        if type == None or type == "success":
            count += len(self.DB["success"])
        if type == None or type == "fail":
            count += len(self.DB["fail"])
        return count

    def validity_check(self) -> bool:
        opin_len = None
        for entry in self.DB["success"]:
            if not isinstance(entry[0], tuple):
                return False
            for item in entry[0]:
                if not isinstance(item, int) and item != None:
                    # if not isinstance(item, int):
                    return False
            if not isinstance(entry[1], tuple):
                return False
            for item in entry[1]:
                if not isinstance(item, int) and item != None:
                    # if not isinstance(item, int):
                    return False
            if opin_len == None:
                opin_len = len(entry[0])
            elif len(entry[0]) != opin_len:
                return False
        return True

    def analyze_symbol(self):
        record_list = list(self.DB["success"])
        record_count = len(record_list)
        input_len = len(record_list[0][0])
        output_len = max([len(item[1]) for item in record_list])
        for i in range(0, input_len - 1):
            for j in range(i + 1, input_len):
                valid = True
                for (inputs, outputs) in record_list:
                    if inputs[i] != inputs[j]:
                        valid = False
                        break
                if valid:
                    self.aliases.append((i, j))
        for _ in range(output_len):
            self.related.append(set())
            self.unrelated.append(set())
        for i in range(0, record_count):
            for j in range(len(record_list[i][0])):
                if not isinstance(record_list[i][0][j], int):
                    for k in range(len(record_list[i][1])):
                        self.unrelated[k].add(j)
        for i in range(0, record_count):
            for j in range(i + 1, record_count):
                diff = compare_array_diff(record_list[i][0], record_list[j][0])
                cmp_len = min(len(record_list[i][1]), len(record_list[j][1]))
                if 0 < len(diff) <= 3:
                    for k in range(cmp_len):
                        if record_list[i][1][k] != record_list[j][1][k]:
                            for argnum in diff:
                                self.related[k].add(argnum)
                if len(diff) == 1:
                    for k in range(cmp_len):
                        if record_list[i][1][k] == record_list[j][1][k]:
                            self.unrelated[k].add(diff[0])

    def display(self, type=None):
        display_list = ["success", "fail"] if type == None else [type]
        for display_key in display_list:
            print(display_key + ":")
            for (inputs, outputs) in self.DB[display_key]:
                print(inputs, outputs)

    def dump(self, DB_Name):
        with open(DB_Name, "wb") as f:
            pickle.dump(self.DB, f)


def mutateTarget(
    apiname: str,
    inst: OpInstance,
    input_symb_2_value: dict,
    mutateMask: int,
    delta: int,
    library: str = "torch",
):
    mutateCount = len(input_symb_2_value.keys())
    for argNum in range(mutateCount):
        if mutateMask & (1 << argNum):
            input_symb_2_value[f"s{argNum}"] += delta
    return input_validity_test(inst, input_symb_2_value, library)


def add_new_input(
    OpDB: OpDatabase, inst: OpInstance, input_dict: dict, output_len: int, library: str
):
    inputs, outputs = input_validity_test(inst, input_dict, library)
    if not (outputs != None and len(outputs) != output_len):
        OpDB.Add(inputs, outputs)


def mutate(apiname: str, OpDB: OpDatabase, inst: OpInstance, record, library: str):
    mutateCount = len(record[0].keys())
    output_len = len(record[1].keys())
    # 1 symbol inequality
    for i in range(mutateCount):
        # shape > 0 are trivial rules so we ignore them
        if f"s{i}" in inst.I:
            continue
        input_dict = deepcopy(record[0])
        input_dict[f"s{i}"] = 0
        add_new_input(OpDB, inst, input_dict, output_len, library)
        input_dict[f"s{i}"] = -2  # avoid using -1 as trial value
        add_new_input(OpDB, inst, input_dict, output_len, library)
    # 2 symbol inequality
    for i in range(mutateCount):
        for j in range(i + 1, mutateCount):
            input_dict = deepcopy(record[0])
            if input_dict[f"s{i}"] == input_dict[f"s{j}"]:
                input_dict[f"s{j}"] += 1
                add_new_input(OpDB, inst, input_dict, output_len, library)
            input_dict[f"s{i}"], input_dict[f"s{j}"] = (
                input_dict[f"s{j}"],
                input_dict[f"s{i}"],
            )
            add_new_input(OpDB, inst, input_dict, output_len, library)
    if mutateCount <= 8:
        # try all possible subsets
        for mutateMask in range(1, (1 << mutateCount)):
            inputs, outputs = mutateTarget(
                apiname, inst, deepcopy(record[0]), mutateMask, 2, library=library
            )
            if not (outputs != None and len(outputs) != output_len):
                OpDB.Add(inputs, outputs)
                # InvocationDB.Add(apiname, i_op, inputs, outputs)
    # 1-element subsets
    if mutateCount <= 100:
        for mutateID in range(mutateCount):
            for delta in range(1, 4):
                inputs, outputs = mutateTarget(
                    apiname,
                    inst,
                    deepcopy(record[0]),
                    (1 << mutateID),
                    delta,
                    library=library,
                )
                if not (outputs != None and len(outputs) != output_len):
                    OpDB.Add(inputs, outputs)
                # if outputs != None:
                # InvocationDB.Add(apiname, i_op, inputs, outputs)
    # 2-element subsets
    if mutateCount <= 50:
        for id1 in range(mutateCount):
            for id2 in range(id1 + 1, mutateCount):
                for delta in range(1, 3):
                    inputs, outputs = mutateTarget(
                        apiname,
                        inst,
                        deepcopy(record[0]),
                        (1 << id1) | (1 << id2),
                        delta,
                        library=library,
                    )
                    if not (outputs != None and len(outputs) != output_len):
                        OpDB.Add(inputs, outputs)


def generate_inst_invocations(
    opid: int, inst: OpInstance, records: list, library: str, aug_dir: str
):
    apiname = inst.name
    filename = str(inst.name_index)
    OpDB = OpDatabase()
    inst.make_tensors_numeric()
    for record in records:
        inputs, outputs = list(record[0].values()), list(record[1].values())
        OpDB.Add(inputs, outputs)
    OpDB.dump(os.path.join(aug_dir, f"{filename}.pkl"))
    if f"{apiname}-{opid}" in skip_mutation_operator:
        return
    # print(f"{apiname}-{opid} start", flush=True)
    mutated = False
    try:
        with time_limit(30):
            for record in records:
                inputs, outputs = list(record[0].values()), list(record[1].values())
                if apiname not in skipMutationAPI:
                    if mutated and OpDB.InvocationCount(type="success") >= 100:
                        continue
                    mutate(apiname, OpDB, inst, record, library=library)
                    mutated = True
    except:
        pass
    if OpDB.validity_check():
        # print(f"{apiname}-{opid} complete", flush=True)
        AUTOINF_LOG.info(f"{apiname}-{opid} complete!")
        OpDB.dump(os.path.join(aug_dir, f"{filename}.pkl"))
    else:
        AUTOINF_LOG.warning(f"{apiname}-{opid} failed!")


def build_database(library: str, parallel: int, dump_dir: str):
    records_dir = os.path.join(ROOT_DIR, f"data/{library}_records")
    aug_dir = os.path.join(dump_dir, f"{library}_augmented_records")
    os.system(f"rm {aug_dir} -r")
    os.makedirs(aug_dir)
    assert os.path.isdir(records_dir), "record not exist"
    # global InvocationDB
    # InvocationDB = InvocationDatabase(from_DB=False)
    p = mp.Pool(parallel)
    gen_inst_records = gen_inst_with_records(
        data_dir=records_dir,
        int_policy="fix_dim",
    )
    for i_op, (inst, records) in enumerate(gen_inst_records):
        p.apply_async(
            generate_inst_invocations, (i_op, inst, records, library, aug_dir)
        )
        # generate_inst_invocations(i_op, inst, records, library, aug_dir)
    p.close()
    p.join()
    AUTOINF_LOG.info(f"{library} augmentation finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_dir", type=str, default=GEN_DIR)
    parser.add_argument("--library", nargs="+", default=["tf", "torch"])
    parser.add_argument("--parallel", type=int, default=16)
    args = parser.parse_args()
    tf_config_gpu()
    for lib in args.library:
        build_database(lib, args.parallel, args.dump_dir)
    # assert InvocationDB.validity_check()
    # InvocationDB.analyze_symbol()
    # InvocationDB.display()
    # InvocationDB.summary()
