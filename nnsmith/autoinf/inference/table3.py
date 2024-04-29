import argparse
import os
import pickle

from rich.console import Console
from rich.table import Table

from nnsmith.autoinf.inference.const import DATA_DIR, GEN_DIR, ROOT_DIR
from nnsmith.autoinf.instrument.categorize import gen_inst_with_records
from nnsmith.autoinf.instrument.op import OpInstance

BLACKLIST = [
    # PyTorch
    # value randomness
    "torch.rand_like",
    "torch.randn_like",
    "torch.randint_like",
    "torch.Tensor.random_",
    "torch.Tensor.uniform_",
    "torch.empty_like",
    "torch.Tensor.normal_",
    "torch.Tensor.new_empty",
    "torch.Tensor.new_empty_strided",
    "torch.dropout",
    "torch.native_dropout",
    "torch.nn.functional.dropout",
    "torch.nn.functional.dropout1d",
    "torch.nn.functional.dropout2d",
    "torch.nn.functional.dropout3d",
    "torch.nn.functional.feature_alpha_dropout",
    # unlock when preprocessing filters out dynamic output numbers.
    "torch.Tensor.unbind",
    "torch.unbind",
    "torch.Tensor.split",
    "torch.split",
    # some special cases
    "torch.gather",
    "torch.Tensor.resize_as_",  # resize_as_ can't be represented in the JIT at the moment ...
    "torch.Tensor.rename",
    "torch.Tensor.rename_",
    "torch.Tensor.requires_grad_",
    "torch.searchsorted",  # sorter has value constraints but the crash needs to be triggered by a big value.
    "torch.native_batch_norm",  # crash when input constraint is violated.
    "torch.Tensor.sum_to_size",  # some odd dtype transfer
    # TensorFlow
    "tf.raw_ops.Unique",
]


def filtered(inst: OpInstance) -> bool:
    if inst.name in BLACKLIST:
        return True
    invoke_str = inst.invoke_str({k: None for k in inst.A})
    if "torch.sort(" in invoke_str and "stable=True" in invoke_str:
        # stable sort could lead to crash | https://github.com/pytorch/pytorch/issues/91420
        return True
    return False


def aug_rcount(record_path: str) -> int:
    rcount = 0
    for filename in os.listdir(record_path):
        with open(os.path.join(record_path, filename), "rb") as f:
            info_dict = pickle.load(f)
            rcount += len(info_dict["success"]) + len(info_dict["fail"])
    return rcount


def aug_success(record_path: str) -> int:
    cnt = 0
    for filename in os.listdir(record_path):
        with open(os.path.join(record_path, filename), "rb") as f:
            info_dict = pickle.load(f)
            if len(info_dict["fail"]) > 0:
                cnt += 1
    return cnt


def raw_count(record_path: str) -> int:
    gen_inst_records = gen_inst_with_records(data_dir=record_path, int_policy="fix_dim")
    api_set = set()
    acount, pacount, rcount = 0, 0, 0
    for (inst, records) in gen_inst_records:
        if not filtered(inst):
            if inst.name not in api_set:
                api_set.add(inst.name)
                acount += 1
            pacount += 1
            rcount += len(records)
    return acount, pacount, rcount


def inf_count(record_path: str):
    gen_inst_records = gen_inst_with_records(data_dir=record_path, int_policy="fix_dim")
    success_api, success_op = set(), 0
    for (inst, _) in gen_inst_records:
        if not filtered(inst):
            if not inst.infer_failed():
                success_op += 1
                success_api.add(inst.name)
    return len(success_api), success_op


def fuzz_count(record_path: str):
    with open(os.path.join(record_path, "op_rej.txt")) as f:
        rej_op_inst = f.readlines()
        rej_op_inst = set([x.strip() for x in rej_op_inst])
    with open(os.path.join(record_path, "op_used.txt")) as f:
        used_op_inst = f.readlines()
        used_op_inst = set([x.strip() for x in used_op_inst])
    rej_op_apis = set([x.split("-")[0] for x in rej_op_inst])
    used_op_apis = set([x.split("-")[0] for x in used_op_inst])
    return len(used_op_apis) - len(rej_op_apis), len(used_op_inst) - len(rej_op_inst)


def table_fmt(v: int) -> str:
    return "{:,}".format(v)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule_dir", type=str, default=DATA_DIR)
    parser.add_argument("--fuzzing_dir", type=str, default=GEN_DIR)
    args = parser.parse_args()

    table = Table(title="Table 3: # API/partial operator/record at different stages")
    table.add_column("")
    table.add_column("#API (PT)")
    table.add_column("#API (TF)")
    table.add_column("#Partial Op. (PT)")
    table.add_column("#Partial Op. (TF)")
    table.add_column("#Record (PT)")
    table.add_column("#Record (TF)")

    table.add_row(
        "Collected", "758", "248", "-", "-", table_fmt(63136), table_fmt(33973)
    )

    tf_raw_path = os.path.join(DATA_DIR, "tf_records")
    tf_acount, tf_pacount, tf_raw_rcount = raw_count(tf_raw_path)
    torch_raw_path = os.path.join(DATA_DIR, "torch_records")
    torch_acount, torch_pacount, torch_raw_rcount = raw_count(torch_raw_path)
    table.add_row(
        "Filtering",
        table_fmt(torch_acount),
        table_fmt(tf_acount),
        table_fmt(torch_pacount),
        table_fmt(tf_pacount),
        table_fmt(torch_raw_rcount),
        table_fmt(tf_raw_rcount),
    )

    tf_aug_path = os.path.join(args.rule_dir, "tf_augmented_records")
    tf_aug_rcount = aug_rcount(tf_aug_path)
    torch_aug_path = os.path.join(args.rule_dir, "torch_augmented_records")
    torch_aug_rcount = aug_rcount(torch_aug_path)

    table.add_row(
        "Augment.",
        table_fmt(torch_acount),
        table_fmt(tf_acount),
        table_fmt(torch_pacount),
        table_fmt(tf_pacount),
        table_fmt(torch_aug_rcount),
        table_fmt(tf_aug_rcount),
    )

    torch_api_inf, torch_op_inf = inf_count(torch_raw_path)
    tf_api_inf, tf_op_inf = inf_count(tf_raw_path)

    table.add_row(
        "Inference",
        table_fmt(torch_api_inf),
        table_fmt(tf_api_inf),
        table_fmt(torch_op_inf),
        table_fmt(tf_op_inf),
        "-",
        "-",
    )

    torch_fuzz_path = os.path.join(args.fuzzing_dir, "torch-neuri-i-n1")
    torch_api_fuzz_bottom, torch_op_fuzz_bottom = fuzz_count(torch_fuzz_path)
    tf_fuzz_path = os.path.join(args.fuzzing_dir, "tensorflow-neuri-i-n1")
    tf_api_fuzz_bottom, tf_op_fuzz_bottom = fuzz_count(tf_fuzz_path)

    table.add_row(
        "Fuzz_bot",
        table_fmt(torch_api_fuzz_bottom),
        table_fmt(tf_api_fuzz_bottom),
        table_fmt(torch_op_fuzz_bottom),
        table_fmt(tf_op_fuzz_bottom),
        "-",
        "-",
    )

    console = Console()
    console.print(table)
