if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str)
    args = parser.parse_args()

    with open(os.path.join(args.root, "op_rej.txt")) as f:
        rej_op_inst = f.readlines()
        rej_op_inst = set([x.strip() for x in rej_op_inst])
    with open(os.path.join(args.root, "op_used.txt")) as f:
        used_op_inst = f.readlines()
        used_op_inst = set([x.strip() for x in used_op_inst])

    print(f"# Used Partial Op = {len(used_op_inst)}")
    print(f"# Rejected Partial Op = {len(rej_op_inst)}")
    print(f"# Used & Unrejected Partial Op = {len(used_op_inst - rej_op_inst)}")

    rej_op_apis = set([x.split("-")[0] for x in rej_op_inst])
    used_op_apis = set([x.split("-")[0] for x in used_op_inst])
    print(f"# Used APIs = {len(used_op_apis)}")
    print(f"# Rejected APIs = {len(rej_op_apis)}")
    print(f"# Used & Unrejected APIs = {len(used_op_apis - rej_op_apis)}")
