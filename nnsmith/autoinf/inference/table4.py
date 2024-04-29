import argparse
import os
import pickle

from rich.console import Console
from rich.table import Table

from nnsmith.autoinf.inference.const import DATA_DIR, GEN_DIR

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
    "torch.sort",
]


def table_fmt(v: int) -> str:
    return "{:,}".format(v)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule_dir", type=str, default=DATA_DIR)
    args = parser.parse_args()

    table = Table(title="Table 4: Number of inferred shape propagation rules")
    table.add_column("")
    table.add_column("<1s")
    table.add_column("<10s")
    table.add_column("<100s")
    table.add_column("<1000s")
    table.add_column("Timeout")
    table.add_column("Unsat.")

    library = "torch"

    s1, s10, s100, s1000, timeout, unsat = 0, 0, 0, 0, 0, 0
    shape_dir = os.path.join(args.rule_dir, f"{library}_shape_rules")
    for filename in os.listdir(shape_dir):
        if filename.split("-")[0] in BLACKLIST:
            continue
        with open(os.path.join(shape_dir, filename), "rb") as f:
            info_dict = pickle.load(f)
        t = info_dict["time"]
        success = True
        for info in info_dict["output_rules"]:
            if info["rule_count"] == 0:
                success = False
                break
        if success:
            if t <= 1:
                s1 += 1
            elif t <= 10:
                s10 += 1
            elif t <= 100:
                s100 += 1
            else:
                s1000 += 1
        else:
            if t <= 100:
                unsat += 1
            else:
                timeout += 1

    table.add_row(
        "NeuRI",
        table_fmt(s1),
        table_fmt(s1 + s10),
        table_fmt(s1 + s10 + s100),
        table_fmt(s1 + s10 + s100 + s1000),
        table_fmt(timeout),
        table_fmt(unsat),
    )

    s1, s10, s100, s1000, timeout, unsat = 0, 0, 0, 0, 0, 0
    rosette_dir = os.path.join(GEN_DIR, f"{library}_rosette")
    for filename in os.listdir(rosette_dir):
        if filename.split("-")[0] in BLACKLIST:
            continue
        with open(os.path.join(rosette_dir, filename), "rb") as f:
            info_dict = pickle.load(f)
        if info_dict["status"] == "success":
            t = info_dict["time"]
            if t <= 1:
                s1 += 1
            elif t <= 10:
                s10 += 1
            elif t <= 100:
                s100 += 1
            else:
                s1000 += 1
        elif info_dict["status"] == "timeout":
            timeout += 1
        else:
            unsat += 1

    table.add_row(
        "Rosette",
        table_fmt(s1),
        table_fmt(s1 + s10),
        table_fmt(s1 + s10 + s100),
        table_fmt(s1 + s10 + s100 + s1000),
        table_fmt(timeout),
        table_fmt(unsat),
    )

    console = Console()
    console.print(table)
