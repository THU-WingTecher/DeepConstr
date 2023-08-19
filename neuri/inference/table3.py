import os
import pickle

from rich.console import Console
from rich.table import Table

from neuri.inference.const import GEN_DIR, ROOT_DIR
from neuri.instrument.categorize import gen_inst_with_records
from neuri.instrument.op import OpInstance

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


def table_fmt(v: int) -> str:
    return "{:,}".format(v)


if __name__ == "__main__":
    table = Table(title="Table 3: # API/partial operator/record at different stages")
    table.add_column("")
    table.add_column("#API (PT)")
    table.add_column("#API (TF)")
    table.add_column("#Partial Op. (PT)")
    table.add_column("#Partial Op. (TF)")
    table.add_column("#Record (PT)")
    table.add_column("#Record (TF)")

    table.add_row("Collected", "-", "-", "-", "-", "-", "-")

    tf_raw_path = os.path.join(ROOT_DIR, "neuri/data/tf_records")
    tf_acount, tf_pacount, tf_raw_rcount = raw_count(tf_raw_path)
    torch_raw_path = os.path.join(ROOT_DIR, "neuri/data/torch_records")
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

    tf_aug_path = os.path.join(GEN_DIR, "tf_augmented_records")
    tf_aug_rcount = aug_rcount(tf_aug_path)
    torch_aug_path = os.path.join(GEN_DIR, "torch_augmented_records")
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

    console = Console()
    console.print(table)
