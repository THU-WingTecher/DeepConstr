from nnsmith.autoinf.instrument.instr_tf_old import get_all_supported_ops

if __name__ == "__main__":
    name_2_op = get_all_supported_ops(
        "/home/jinjun/fastd/code/autoinf/autoinf/instrument/xla-compilable-ops.md"
    )

    with open("tf_api_full_name.txt", "w") as f:
        for name, op in name_2_op.items():
            print(f"{op.__module__}.{op.__name__}", file=f)
