import argparse
import multiprocessing as mp
import os
import pickle
import time

import z3

from nnsmith.autoinf.inference.augmentation import OpDatabase
from nnsmith.autoinf.inference.const import GEN_DIR
from nnsmith.logger import AUTOINF_LOG
from nnsmith.autoinf.inference.tree import ArithExpNode, ArithExpTree, TreeDatabase
from nnsmith.autoinf.inference.utils import equivalent, mask_to_list, popcount, wrap_time
from nnsmith.autoinf.instrument.op import OpInstance

TreeDB: TreeDatabase = TreeDatabase(os.path.join(GEN_DIR, "tree.pkl"), 5, 5)


class RuleDatabase:
    def __init__(self, argCount):
        self.E = True
        self.rule_count = 0
        self.argCount = argCount

    def Add(self, rule) -> bool:
        if self.rule_count >= 10:
            return False
        for i in range(self.argCount):
            exec(f"s{i} = z3.Int('s{i}')")
        o = z3.Int("o")
        e = o == eval(rule)
        E_with_e = z3.And(self.E, e)
        if not equivalent(E_with_e, self.E):
            self.E = E_with_e
            self.rule_count += 1
            return True
        else:
            return False


"""
Rules format:
- name: apiname (str)
- opid: operator id (int)
- output_rank: rank of output tensor (int)
- invocation_count: #invocation
- time: (float)
- output_rules: [
  {
    o_id: index of output (int)
    tree_decay: (list)
    rule_count: (int)
    index: related index list
    rules: [(str), (str), (str), (str), (str)]
  },
  ...
  ]
"""

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
    if count == 0:
        return [0]
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


def solve_inst(filename: str, OpDB: OpDatabase, dump_dir: str):
    global TreeDB
    record_list = list(OpDB.DB["success"])
    record_count = len(record_list)
    input_len = len(record_list[0][0])
    output_len = len(OpDB.unrelated)
    # rule_info creation
    rule_info = dict()
    rule_info["output_rank"] = output_len
    rule_info["invocation_count"] = record_count
    rule_info["output_rules"] = []
    # Preliminaries
    hard = False
    valid_tree_set = [[] for _ in range(output_len)]
    timeout = 1000
    tree_tried = 0
    start_time = time.time()
    # input rank check

    for (depth, ArgSet, i) in TreeDB.genTreeList(5, min(input_len, 5)):
        tree_tried += 1
        tree = TreeDB.getTree(depth, ArgSet, i)
        sym_count = popcount(ArgSet)
        happen = False
        for o_id in range(output_len):
            if len(valid_tree_set[o_id]) >= 1:
                continue
            happen = True
            for sym_set in gen_sym_set((1 << input_len) - 1, sym_count):
                if OpDB.check_duplicate_sym(sym_set):
                    continue
                sym_list = mask_to_list(sym_set)
                # if (index[o_id][1] & tree.ArgSet) != tree.ArgSet: continue
                valid = True
                at_least_one_record = False
                for record_id in range(record_count):
                    if o_id >= len(record_list[record_id][1]):
                        continue
                    expectedVal = record_list[record_id][1][o_id]
                    # val = defaultdict(int)
                    val = []
                    haszero = False
                    for i, argNum in enumerate(sym_list):
                        val.append(record_list[record_id][0][argNum])
                        if record_list[record_id][0][argNum] <= 0:
                            haszero = True
                    if not haszero:
                        at_least_one_record = True
                        if tree.evaluate(val) != expectedVal:
                            valid = False
                            break
                if valid and at_least_one_record:
                    valid_tree_set[o_id].append((tree, sym_set))
        if not happen:
            break
        if time.time() - start_time > timeout:
            break
    # input rank
    for o_id in range(output_len):
        if len(valid_tree_set[o_id]) >= 1:
            continue
        valid = True
        for record_id in range(record_count):
            if o_id >= len(record_list[record_id][1]):
                continue
            if record_list[record_id][1][o_id] != input_len:
                valid = False
                break
        if valid:
            valid_tree_set[o_id].append((input_len, input_len))
    for o_id in range(output_len):
        dd = {
            "o_id": o_id,
            "rule_count": len(valid_tree_set[o_id]),
            "tree_tried": tree_tried,
            "rules": [],
        }
        RuleDB = RuleDatabase(input_len)
        for (tree, sym_set) in valid_tree_set[o_id]:
            if isinstance(tree, int):
                dd["rules"].append(str(tree))
            else:
                term = remap(tree.display(), mask_to_list(sym_set))
                dd["rules"].append(term)
        rule_info["output_rules"].append(dd)
    end_time = time.time()
    rule_info["time"] = end_time - start_time
    AUTOINF_LOG.info(f"{filename} shape inference complete!")
    with open(f"{dump_dir}/{filename}", "wb") as f:
        pickle.dump(rule_info, f)
    # if hard == True:
    # with lock:
    # open(f'{InformationRootDir}/IO-hard-{date}', 'a').write(filename + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_dir", type=str, default=GEN_DIR)
    parser.add_argument("--parallel", type=int, default=16)
    parser.add_argument("--library", nargs="+", default=["tf", "torch"])
    args = parser.parse_args()

    for library in args.library:
        rule_dir = os.path.join(args.dump_dir, f"{library}_shape_rules")
        record_dir = os.path.join(args.dump_dir, f"{library}_augmented_records")
        if os.path.isdir(rule_dir):
            os.system(f"rm {rule_dir} -r")
        os.makedirs(rule_dir, exist_ok=True)
        # if cfg_zerofilter:
        #     filelist, _filelist = [], os.listdir(InvocDBDir)
        #     for filename in _filelist:
        #         info = None
        #         try:
        #             with open(os.path.join(IORuleDir, filename), "rb") as f:
        #                 info = pickle.load(f)
        #         except:
        #             pass
        #         if info == None:
        #             filelist.append(filename)
        #             continue
        #         flag = True
        #         for item in info["output_rules"]:
        #             if item["rule_count"] == 0:
        #                 flag = False
        #                 break
        #         if not flag:
        #             filelist.append(filename)
        # else:
        #     filelist = os.listdir(InvocDBDir) if len(specify_inst) == 0 else specify_inst
        # print(len(filelist))
        p = mp.Pool(args.parallel)
        for filename in os.listdir(record_dir):
            api_name = filename.split("-")[0]
            # if specify_op != [] and api_name not in specify_op:
            # continue
            OpDB: OpDatabase = OpDatabase(DB_Name=os.path.join(record_dir, filename))
            OpDB.analyze_symbol()
            p.apply_async(solve_inst, (filename, OpDB, rule_dir))
            # solve_inst(filename, OpDB, rule_dir)
    p.close()
    p.join()
