"""
Given the directory containing all tests. We replay the test execution and record coverage in LLVM profraw format.
The intermediate tests can be saved using fuzz.save_test={{DIR_TO_SAVE}}.
"""
import json
import multiprocessing as mp
import os
import pickle
import subprocess
import sys
print(sys.path)

from neuri.autoinf.inference.const import ROOT_DIR
from neuri.util import mkdir

results = dict()


def collect():
    for root, dirs, files in os.walk("./bazel-out"):
        for name in files:
            filename = os.path.join(root, name)
            os.system(f"cp {filename} ./gcno/{name}")


def clear_gcda():
    os.system(f"rm ./gcno/*.gcda")


def move_gcno(id_path):
    os.system(f"cp {ROOT_DIR}/experiments/gcno ./{id_path}-workspace/gcno -r")


def move_gcda(id_path):
    for root, dirs, files in os.walk(f"./{id_path}-workspace/bazel-out"):
        for name in files:
            filename = os.path.join(root, name)
            os.system(f"cp {filename} ./{id_path}-workspace/gcno/{name}")


def gen_lcov(output_path, id_path):
    conda_prefix = os.getenv("CONDA_PREFIX")
    gcov_tool = os.path.join(conda_prefix, "bin/x86_64-conda-linux-gnu-gcov")
    os.system(
        f"lcov --capture --directory ./{id_path}-workspace/gcno --output-file {output_path} --rc lcov_branch_coverage=1 --gcov-tool {gcov_tool} 1>/dev/null 2>/dev/null"
    )


def coverage_collect(output_path, id_path):
    if not os.path.exists(f"{id_path}-workspace/gcno"):
        move_gcno(id_path)
        move_gcda(id_path)
    gen_lcov(output_path, id_path)


def batched(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


def model_exec(
    test_paths, model_type, backend_type, backend_target, output_path, id_path
):
    model_paths = []
    for test_path in test_paths:
        for file in os.listdir(test_path):
            if file.startswith("model"):
                model_paths.append(os.path.join(test_path, file))
                break

    # print(len(model_paths))
    gcda_save_path = f"{id_path}-workspace"
    os.system(f"rm {gcda_save_path} -r")
    os.makedirs(gcda_save_path)
    trial_arguments = [
        "python3",
        f"{ROOT_DIR}/neuri/cli/model_exec.py",
        "model.type=" + model_type,
        "backend.type=" + backend_type,
        "backend.target=" + backend_target,
        f"model.path={model_paths}",
    ]
    p = subprocess.Popen(
        trial_arguments,  # Show all output
        cwd=os.path.join(os.getcwd(), gcda_save_path),
    )
    p.communicate()
    exit_code = p.returncode

    if exit_code != 0:
        print(
            f"==> model_exec crashed when generating {output_path}! => EXIT CODE {exit_code}"
        )
        return

    coverage_collect(output_path, id_path)
    os.system(f"rm {gcda_save_path} -r")
    # else:
    # cov_file = ''
    # os.system(f'cp {cov_file} {profraw_path}')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=str, required=True, help="Folder to all the tests."
    )
    parser.add_argument("--batch_size", type=int, default=1000, help="")
    parser.add_argument(
        "--parallel", type=int, default=8, help="Number of process for execution."
    )

    args = parser.parse_args()

    time2path = {}
    for dir in os.listdir(args.root):
        if dir != "coverage":
            time2path[float(dir)] = os.path.join(args.root, dir)

    time_stamps = sorted(time2path.keys())
    batches = list(batched(time_stamps, args.batch_size))

    print(f"=> Number of batches: {len(batches)} of size {args.batch_size}")

    cov_save = os.path.join(args.root, "coverage")

    # name = args.root.split("/")[-1]
    # cov_save = os.path.join("/ColossalTitan/yuyao-data/", name, "coverage")

    mkdir(cov_save)
    clear_gcda()

    def batch_exec(batch):
        batch_paths = [time2path[time] for time in batch]
        profraw_path = os.path.join(cov_save, f"{max(batch)}.info")
        model_exec(
            batch_paths, "tensorflow", "xla", "cpu", profraw_path, str(max(batch))
        )

    with mp.Pool(processes=args.parallel) as pool:
        pool.map(batch_exec, batches)
