import os
import pickle
from collections import defaultdict

from neuri.inference.const import *
from neuri.inference.tree import gen_tree_from_string


def gen_type_transfer_trees(inst):
    rule_filename = f"{inst.name_index}.pkl"
    lib = "torch" if "torch" in inst.name_index else "tf"
    with open(os.path.join(InformationRootDir, f"{lib}-date-version"), "r") as f:
        dt = f.read()[:-1]
    type_transfer_rules = defaultdict(list)
    type_transfer_dbg_info = ""
    type_transfer_rule_dir = os.path.join(
        InformationRootDir, f"IO-rules-{lib}-{DataMode}-{dt}"
    )
    try:
        with open(os.path.join(type_transfer_rule_dir, rule_filename), "rb") as f:
            type_transfer_info = pickle.load(f)
        for o_id in range(type_transfer_info["output_rank"]):
            info_rules = type_transfer_info["output_rules"][o_id]
            for rule in info_rules["rules"][:3]:
                type_transfer_rules[f"o{o_id}"].append(gen_tree_from_string(rule))
                type_transfer_dbg_info += f"o{o_id}: {rule}\n"
    except:
        type_transfer_dbg_info += "no inferred rules\n"
    return type_transfer_rules, type_transfer_dbg_info


def gen_requires_trees(inst):
    rule_filename = f"{inst.name_index}.pkl"
    lib = "torch" if "torch" in inst.name_index else "tf"
    with open(os.path.join(InformationRootDir, f"{lib}-date-version"), "r") as f:
        dt = f.read()[:-1]
    requires_rules = []
    requires_dbg_info = ""
    input_constraint_rule_dir = os.path.join(
        InformationRootDir, f"input-rules-{lib}-{dt}"
    )
    try:
        with open(os.path.join(input_constraint_rule_dir, rule_filename), "rb") as f:
            input_constraint_info = pickle.load(f)
        for (rule, sign) in input_constraint_info["rules"]:
            if not enable_inequality and sign != "==":
                continue
            requires_rules.append((gen_tree_from_string(rule), sign))
            requires_dbg_info += f"{rule} {sign} 0\n"
    except:
        requires_dbg_info += "no inferred rules\n"
    return requires_rules, requires_dbg_info


def gen_nnsmith_rules(inst):
    lib = "torch" if "torch" in inst.name_index else "tf"
    with open(os.path.join(InformationRootDir, f"{lib}-date-version"), "r") as f:
        dt = f.read()[:-1]
    try:
        with open(
            os.path.join(
                InformationRootDir,
                f"nnsmith_rules-{lib}-{dt}",
                f"{inst.name_index}.pkl",
            ),
            "rb",
        ) as f:
            res = pickle.load(f)
    except:
        res = []
    return res


def shape_transfer_valid(inst) -> bool:
    lib = "torch" if "torch" in inst.name_index else "tf"
    with open(os.path.join(InformationRootDir, f"{lib}-date-version"), "r") as f:
        dt = f.read()[:-1]
    judge_result_dir = os.path.join(InformationRootDir, f"rule_validity-{lib}-{dt}")
    try:
        with open(os.path.join(judge_result_dir, f"{inst.name_index}.pkl"), "rb") as f:
            valid = pickle.load(f)[1]
    except Exception as e:
        valid = False
    return valid


def constraint_valid(inst) -> bool:
    lib = "torch" if "torch" in inst.name_index else "tf"
    with open(os.path.join(InformationRootDir, f"{lib}-date-version"), "r") as f:
        dt = f.read()[:-1]
    judge_result_dir = os.path.join(InformationRootDir, f"rule_validity-{lib}-{dt}")
    try:
        with open(os.path.join(judge_result_dir, f"{inst.name_index}.pkl"), "rb") as f:
            valid = pickle.load(f)[2]
    except Exception as e:
        valid = False
    return valid


def infer_failure(inst) -> bool:
    lib = "torch" if "torch" in inst.name_index else "tf"
    with open(os.path.join(InformationRootDir, f"{lib}-date-version"), "r") as f:
        dt = f.read()[:-1]
    judge_result_dir = os.path.join(InformationRootDir, f"rule_validity-{lib}-{dt}")
    try:
        with open(os.path.join(judge_result_dir, f"{inst.name_index}.pkl"), "rb") as f:
            valid = pickle.load(f)[0]
    except Exception as e:
        valid = False
    return False if valid else True


def judge_failure(inst) -> bool:
    if len(inst.nnsmith_rules_list) > 0:
        return False
    return infer_failure(inst)
    """
    for o in inst.O:
        if len(inst.type_transfer_rules[o]) == 0:
            return True
    is_hard = False
    with open(
        os.path.join(InformationRootDir, f"hard_list-{cfg_library}-{date}.pkl"), "rb"
    ) as f:
        info = pickle.load(f)
    if f"{inst.name_index}.pkl" in info:
        is_hard = True
    if is_hard and len(inst.requires_rules) == 0:
        return True
    return False
    """
