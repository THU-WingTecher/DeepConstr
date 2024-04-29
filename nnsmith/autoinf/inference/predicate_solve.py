import argparse
import multiprocessing as mp
import os
import pickle
import time
from copy import deepcopy

import z3

from nnsmith.autoinf.inference.const import GEN_DIR
from nnsmith.autoinf.inference.invocations import input_validity_test
from nnsmith.logger import AUTOINF_LOG
from nnsmith.autoinf.inference.tree import ArithExpNode, ArithExpTree, TreeDatabase
from nnsmith.autoinf.inference.utils import equivalent, mask_to_list, popcount

TreeDB: TreeDatabase = TreeDatabase(os.path.join(GEN_DIR, "tree_simple.pkl"), 5, 5)

hard_oplist = []
hard_apiset = set()


class RuleDatabase:
    def __init__(self, argCount):
        self.argCount = argCount
        self.ruleset = list()

    def Add(self, rule, sign):
        if len(self.ruleset) > 50:
            return
        for i in range(self.argCount):
            exec(f"s{i} = z3.Int('s{i}')")
        if sign == "==":
            e = eval(rule) == 0
        elif sign == ">":
            e = eval(rule) > 0
        elif sign == ">=":
            e = eval(rule) >= 0
        else:
            raise f"Unknown sign {sign}"
        self.ruleset.append((e, rule, sign))
        prunable = True
        while prunable:
            prunable = False
            expr = True
            for rule, _, __ in self.ruleset:
                expr = z3.And(expr, rule)
            del_item = None
            for i in range(len(self.ruleset) - 1, 0, -1):
                del_expr = True
                for j, (rule, _, __) in enumerate(self.ruleset):
                    if i != j:
                        del_expr = z3.And(del_expr, rule)
                if equivalent(expr, del_expr):
                    prunable = True
                    del_item = self.ruleset[i]
                    break
            if del_item != None:
                self.ruleset.remove(del_item)

    def Count(self):
        return len(self.ruleset)


def filter_hard(record_dir: str):
    global hard_oplist, hard_apiset
    hard_oplist, hard_apiset = [], set()
    for filename in os.listdir(record_dir):
        with open(os.path.join(record_dir, filename), "rb") as f:
            info_dict = pickle.load(f)
        if len(info_dict["fail"]) > 0:
            hard_oplist.append(filename)
            apiname = filename.split("-")[0]
            hard_apiset.add(apiname)


tmp_store = []


def dfs(cur, count, lim, symlist):
    if cur == count:
        res = 0
        for i in range(count):
            res |= 1 << symlist[i]
        tmp_store.append(res)
    else:
        st = 0 if len(symlist) == 0 else symlist[-1] + 1
        for nxt in range(st, lim):
            dfs(cur + 1, count, lim, symlist + [nxt])


sym_set_table = dict()


def gen_sym_set(mask: int, count: int):
    if (mask, count) not in sym_set_table:
        global tmp_store
        lim = popcount(mask)
        tmp_store = []
        dfs(0, count, lim, [])
        res = tuple(tmp_store)
        # for submask in range(1, mask):
        # if popcount(submask) == count:
        # res.append(submask)
        sym_set_table[(mask, count)] = res
    return sym_set_table[(mask, count)]


def remap(rule: str, index: list) -> str:
    for i, ind in enumerate(index):
        rule = rule.replace(f"s{i}", f"tmp{ind}")
    rule = rule.replace("tmp", "s")
    return rule


def inspect_all_records(
    valid_list: list, invalid_list: list, sym_list: list, tree: ArithExpTree, sign: str
) -> bool:
    for inputs in valid_list:
        val = []
        for argNum in sym_list:
            val.append(inputs[argNum])
        res = eval(f"tree.evaluate(val) {sign} 0")
        if not res:
            return False
    if sign == "==":
        return True
    for inputs in invalid_list:
        val = []
        for argNum in sym_list:
            val.append(inputs[argNum])
        res = eval(f"tree.evaluate(val) {sign} 0")
        if not res:
            return True
    return False


skip_solve = ["torch.Tensor.sum_to_size-5.pkl", "torch.Tensor.sum_to_size-3.pkl"]


def solve_inst(filename: str, info: dict, predicate_dir: str):
    if filename in skip_solve:
        return
    global TreeDB
    valid_list, invalid_list = [], []
    for (inputs, _) in list(info["success"]):
        valid_list.append(list(inputs))
    for inputs in list(info["fail"]):
        invalid_list.append(list(inputs))
    dd = {"rules": []}
    input_len = len(valid_list[0])
    RuleDB = RuleDatabase(input_len)
    timeout = 1000
    tree_tried = 0
    r_count = 0
    start_time = time.time()
    # constant constraint test
    for i in range(input_len):
        val = valid_list[0][i]
        valid = True
        for inputs in valid_list:
            if inputs[i] != val:
                valid = False
                break
        if valid:
            RuleDB.Add(f"s{i}-{val}", "==")
    for (depth, ArgSet, i) in TreeDB.genTreeList(5, min(input_len, 5)):
        tree_tried += 1
        if time.time() - start_time > timeout:
            break
        tree: ArithExpTree = TreeDB.getTree(depth, ArgSet, i)
        if not tree.is_simple():
            continue
        if tree.op not in ["+", "-", None]:
            continue
        sym_count = popcount(ArgSet)
        for sym_set in gen_sym_set((1 << input_len) - 1, sym_count):
            sym_list = mask_to_list(sym_set)
            sign_list = ["==", ">", ">="] if depth <= 1 else ["=="]
            for sign in sign_list:
                if inspect_all_records(valid_list, invalid_list, sym_list, tree, sign):
                    new_constraint = remap(tree.display(), sym_list)
                    RuleDB.Add(new_constraint, sign)
        if RuleDB.Count() >= 50:
            break
    for (_, rule, sign) in RuleDB.ruleset:
        dd["rules"].append((rule, sign))
    end_time = time.time()
    dd["time"] = end_time - start_time
    dd["tree_tried"] = tree_tried
    with open(f"{predicate_dir}/{filename}", "wb") as f:
        pickle.dump(dd, f)
    AUTOINF_LOG.info(f"{filename} complete!")


def solve(record_dir: str, predicate_dir: str, parallel: int):
    os.system(f"rm {predicate_dir} -r")
    os.makedirs(predicate_dir)
    p = mp.Pool(parallel)
    for filename in hard_oplist:
        api_name = filename.split("-")[0]
        with open(os.path.join(record_dir, filename), "rb") as f:
            info = pickle.load(f)
        # print(filename)
        p.apply_async(solve_inst, (filename, info, predicate_dir))
        # solve_inst(filename, info)
    p.close()
    p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_dir", type=str, default=GEN_DIR)
    parser.add_argument("--parallel", type=int, default=16)
    parser.add_argument("--library", nargs="+", default=["tf", "torch"])
    args = parser.parse_args()
    for library in args.library:
        record_dir = os.path.join(args.dump_dir, f"{library}_augmented_records")
        predicate_dir = os.path.join(args.dump_dir, f"{library}_input_predicates")
        filter_hard(record_dir)
        solve(record_dir, predicate_dir, args.parallel)
