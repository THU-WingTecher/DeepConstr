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

results = dict()
from pathlib import Path
from tqdm import tqdm
ROOT_DIR = Path(__file__).parent.parent.parent.parent
def mkdir(dir: os.PathLike, yes=False):
    # if os.path.exists(dir):
    #     decision = ""
    #     if yes:
    #         decision = "y"
    #     while decision.lower() not in ["y", "n"]:
    #         CORE_LOG.warning(
    #             "Report folder already exists. Press [Y/N] to continue or exit..."
    #         )
    #         decision = input()
    #     if decision.lower() == "n":
    #         CORE_LOG.error(f"{dir} already exist... Remove it or use a different name.")
    #         raise RuntimeError("Folder already exists")
    #     else:
    #         shutil.rmtree(dir)

    os.makedirs(dir)
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
    test_paths, output_path, id_path, package
):  
    script_name = "prog.py"
    model_paths = [f"{os.path.join(test_path,script_name)}" for test_path in test_paths]

    print(len(model_paths))
    if package != "torch" :
        gcda_save_path = f"{id_path}-workspace"
        os.system(f"rm {gcda_save_path} -r")
        os.makedirs(gcda_save_path)
    for model_path in tqdm(model_paths) :
        trial_arguments = [
            "python3",
            f"{model_path}"
        ]
        print(f"running command {trial_arguments} ...")
        if package != "torch" : 
            p = subprocess.Popen(
                trial_arguments,  # Show all output
                cwd=os.path.join(os.getcwd(), gcda_save_path),
            )
        else :
            copied_env = os.environ.copy()
            copied_env["LLVM_PROFILE_FILE"] = output_path
            p = subprocess.Popen(
                trial_arguments,  # Show all output
                env=copied_env,
            )
        p.communicate()
        exit_code = p.returncode

        if exit_code != 0:
            print(
                f"==> model_exec crashed when generating {output_path}! => EXIT CODE {exit_code}"
            )
            return
    if package != "torch" :
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
    parser.add_argument(
        "--package", type=str,default='torch', required=True, help="Folder to all the tests."
    )
    parser.add_argument("--batch_size", type=int, default=1000, help="")
    parser.add_argument(
        "--parallel", type=int, default=8, help="Number of process for execution."
    )

    args = parser.parse_args()

    time2path = {}
    for dir in os.listdir(args.root):
        if dir != "coverage" and not dir.endswith('json'):
            time2path[float(dir)] = os.path.join(args.root, dir)

    batch_size = len(time2path) // int(args.parallel)
    time_stamps = sorted(time2path.keys())
    batches = list(batched(time_stamps, batch_size))

    print(f"=> Number of batches: {len(batches)} of size {batch_size}")

    cov_save = os.path.join(args.root, "coverage")

    if not os.path.exists(cov_save) :
        mkdir(cov_save)
    clear_gcda()
    
    def batch_exec(batch):
        batch_paths = [time2path[time] for time in batch]
        format = f"{max(batch)}.info" if args.package != 'torch' else f"{max(batch)}.profraw"
        profraw_path = os.path.join(cov_save, format)
        model_exec(
            batch_paths, profraw_path, str(max(batch)), args.package
        )

    with mp.Pool(processes=args.parallel) as pool:
        pool.map(batch_exec, batches)
