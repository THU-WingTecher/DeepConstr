import multiprocessing as mp
import os
import pickle
import subprocess
import tempfile
from copy import deepcopy
from pathlib import Path


def src_filter(filename):
    if "tensorflow/compiler" not in filename:
        return True
    if "bazel-out" in filename:
        return True
    return False


def analyze_lcov(lcov_data):
    file_covs = lcov_data.split("end_of_record\n")[:-1]
    ret = {
        # 'filename': {
        #    'lines': [identifiers],
        #    'branches': [identifiers],
        #    'lf': # lines total,
        #    'bf': # branches total,
        # }
    }

    # SF: source file
    # sperated by end_of_record

    # FN: <line number>,<function name>

    for file_cov in file_covs:
        cov_lines = file_cov.split("\n")
        while cov_lines[0].startswith("TN:"):
            cov_lines = cov_lines[1:]
        assert cov_lines[0].startswith("SF:")
        filename = cov_lines[0][3:]
        if src_filter(filename):
            continue

        lines = set()
        branches = set()

        # LH: # line hits
        # FNH: # function hits | note we consider source-level functions. e.g., templates are one function.
        # BRH: # branch hits
        # NOTE: LH might (slightly) != len(lines). we only consider len(lines).

        n_line_total = 0  # LF: # lines
        n_branch_total = 0  # BRF: # branches

        lf = 0
        brf = 0

        for cov_line in cov_lines[1:]:
            cov_line = cov_line.rstrip("\n")
            if cov_line.startswith("DA:"):
                # DA: <line number>,<execution count> for each instrumented line
                line_number, exec_count = cov_line[3:].split(",")
                if exec_count != "0":
                    lines.add(int(line_number))
                n_line_total += 1
            elif cov_line.startswith("BRDA:"):
                # BRDA: <line number>,<block number>,<branch number>,<taken>
                line_number, block_number, branch_number, taken = cov_line[5:].split(
                    ","
                )
                if taken != "-" and taken != "0":
                    branches.add(line_number + ":" + block_number + ":" + branch_number)
                n_branch_total += 1
            # elif cov_line.startswith('LH:'):
            #     n_line_hit = int(cov_line[3:])
            elif cov_line.startswith("LF:"):
                lf = int(cov_line[3:])
            # elif cov_line.startswith('FNH:'):
            #     n_func_hit = int(cov_line[4:])
            elif cov_line.startswith("BRF:"):
                brf = int(cov_line[4:])
            else:
                pass
        # # BRF might be even smaller than branches you hit. we conservatively use n_branch_total.
        n_branch_total = max(brf, n_branch_total)
        # Similarly
        n_line_total = max(lf, n_line_total)

        assert (
            len(lines) <= n_line_total
        ), f"{len(lines)} <= {n_line_total} in {filename}"
        assert (
            len(branches) <= n_branch_total
        ), f"{len(branches)} <= {n_branch_total} in {filename}"

        ret[filename] = {
            "lines": lines,
            "branches": branches,
            "lf": n_line_total,
            "bf": n_branch_total,
        }

    return ret


def get_cmd_output(cmd_list):
    return subprocess.run(cmd_list, stdout=subprocess.PIPE, check=True).stdout.decode()


def check_profraw(fname: str):
    for n in fname.split(".")[:-1]:
        if not n.isdigit():
            return False
    return True


def merge_cov(current, rhs, hint=""):
    # {
    #     'lines': lines,
    #     'branches': branches,
    #     'lf': n_line_total,
    #     'bf': n_branch_total,
    # }

    for k in set(current.keys()).union(set(rhs.keys())):
        if k not in current:
            current[k] = rhs[k]
            continue
        elif k not in rhs:
            continue
        else:
            current[k]["lines"] = current[k]["lines"].union(rhs[k]["lines"])
            current[k]["branches"] = current[k]["branches"].union(rhs[k]["branches"])

            if current[k]["lf"] != rhs[k]["lf"]:
                print(
                    f'[WARNING] total line {current[k]["lf"]} != {rhs[k]["lf"]} in {k} '
                    + hint
                )
                current[k]["lf"] = max(current[k]["lf"], rhs[k]["lf"])

            if current[k]["bf"] != rhs[k]["bf"]:
                print(
                    f'[WARNING] total branch {current[k]["bf"]} != {rhs[k]["bf"]} in {k} '
                    + hint
                )
                current[k]["bf"] = max(current[k]["bf"], rhs[k]["bf"])

    return current


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=str, required=True, help="Folder to all the tests."
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=8,
        help="Number of parallel jobs for processing profraws.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="How many models evaluted in each profraw?",
    )

    args = parser.parse_args()

    assert os.path.exists(args.root)
    cov_root = os.path.join(args.root, "coverage")

    profraws = [p for p in Path(cov_root).rglob("*.info") if check_profraw(p.name)]
    assert len(profraws) > 0, f"No profraws found in {cov_root}"
    # rank by
    profraws.sort(key=lambda f: float(f.name.rsplit(".", 1)[0]))

    merged_cov_path = os.path.join(cov_root, "merged_cov.pkl")

    with tempfile.TemporaryDirectory() as tmpdirname:

        def process_one_profraw(profraw_path: Path):
            with open(os.path.join(cov_root, profraw_path.name), "r") as f:
                lcov_data = f.read()
            return analyze_lcov(lcov_data)
            """
            profdata_path = os.path.join(tmpdirname, profraw_path.name + ".profdata")
            # llvm-prof-data
            profdata_cmd = (
                f"{llvm_profdata} merge -sparse {profraw_path} -o {profdata_path}"
            )
            assert 0 == os.system(profdata_cmd), profdata_cmd
            llvmcov_cmd = f"{llvm_cov} export -instr-profile={profdata_path} -format=lcov {lib_expr}"
            print(llvmcov_cmd)
            lcov_data = get_cmd_output(llvmcov_cmd.split(" "))
            """

        merged_cov = {
            # 'time': time point. { # ! Not time duration.
            #   'n_model': # how many models got successfully executed?
            #   'merged_cov': coverage dictionary.
            # }
        }

        current_cov = {}
        with mp.Pool(args.parallel) as pool:
            for i, res in enumerate(pool.imap(process_one_profraw, profraws)):
                profraw_name = profraws[i].name
                current_time = float(profraw_name.rsplit(".", 1)[0])

                merged_cov[current_time] = {}
                merged_cov[current_time][
                    "n_model"
                ] = len(profraws)  # inaccurate for the last.
                current_cov = merge_cov(current_cov, res, f"merging {profraw_name}")
                merged_cov[current_time]["merged_cov"] = deepcopy(current_cov)

        with open(merged_cov_path, "wb") as fp:
            pickle.dump(merged_cov, fp)
