"""
Given the directory containing all tests. We replay the test execution and record coverage in LLVM profraw format.
The intermediate tests can be saved using fuzz.save_test={{DIR_TO_SAVE}}.
"""
import re
import json
import multiprocessing as mp
import os
import pickle
from queue import Empty
import subprocess
import sys
import time
import random, string
results = dict()
from pathlib import Path
from tqdm import tqdm
import shutil
ROOT_DIR = "/DeepConstr"

def generate_random_string(length):
    """Generate a random string of fixed length."""
    letters = string.ascii_letters + string.digits
    return ''.join(random.choice(letters) for _ in range(length))

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


# def clear_gcda():
#     os.system(f"rm ./gcno/*.gcda")

def clear_gcda():
    """
    Remove all .gcda files from the ./gcno directory.
    """
    gcda_dir = "./gcno"
    for root, dirs, files in os.walk(gcda_dir):
        for file in files:
            if file.endswith(".gcda"):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    # print(f"Removed {file_path}")
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")

def move_gcno(id_path):
    """
    Copy the gcno directory to a new location specified by id_path.
    """
    src_dir = os.path.join(ROOT_DIR, "experiments/gcno")
    dest_dir = f"./{id_path}-workspace/gcno"
    
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    
    try:
        shutil.copytree(src_dir, dest_dir)
        # print(f"Copied {src_dir} to {dest_dir}")
    except Exception as e:
        print(f"Error copying {src_dir} to {dest_dir}: {e}")

# def move_gcno(id_path):
#     os.system(f"cp {ROOT_DIR}/experiments/gcno ./{id_path}-workspace/gcno -r")
def move_gcda(id_path):
    """
    Copy all files from the bazel-out directory to the gcno directory under the specified id_path workspace.
    """
    src_base_dir = f"./{id_path}-workspace/bazel-out"
    dest_dir = f"./{id_path}-workspace/gcno"
    
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for root, dirs, files in os.walk(src_base_dir):
        for name in files:
            src_file = os.path.join(root, name)
            dest_file = os.path.join(dest_dir, name)
            try:
                shutil.copy(src_file, dest_file)
                # print(f"Copied {src_file} to {dest_file}")
            except Exception as e:
                print(f"Error copying {src_file} to {dest_file}: {e}")

# def move_gcda(id_path):
#     for root, dirs, files in os.walk(f"./{id_path}-workspace/bazel-out"):
#         for name in files:
#             filename = os.path.join(root, name)
#             os.system(f"cp {filename} ./{id_path}-workspace/gcno/{name}")


def gen_lcov(output_path, id_path):
    conda_prefix = os.getenv("CONDA_PREFIX")
    gcov_tool = os.path.join(conda_prefix, "bin/x86_64-conda-linux-gnu-gcov")
    print(f"Using gcov tool: {gcov_tool}")
    print("running lcov command . . .")
    print(f"lcov --capture --directory ./{id_path}-workspace/gcno --output-file {output_path} --rc lcov_branch_coverage=1 --gcov-tool {gcov_tool} 1>/dev/null 2>/dev/null")
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

def change_to_legal(strings, file_name) :
    input_name = file_name.replace(".py", "p")
    input_keys = pickle.load(open('/DeepConstr/exp/doctor/torch/torch.acos/9f8c1249b47e5caf2bc52976ac1ccab57c3f363f.p', 'rb')).keys()
    input_keys_strs = ", ".join(input_keys)
    strings = strings.replace("forward(self, data)", f"forward(self, {input_keys_strs})").replace("eag = model(data)", "")
    pattern = r'(return\s+torch\.\w+\(\*\*data\))'

    # Function to replace '**data' with '**kwargs' in a string
    def replace_data_with_kwargs(text):
        return re.sub(pattern, lambda m: m.group(0).replace('**data', input_keys_strs), text)
    updated_text =replace_data_with_kwargs(strings)
    return updated_text

def merge_files(file_paths, save_dir) : 
    temp_script_path = os.path.join(save_dir, generate_random_string(10)+".py")
    with open(temp_script_path, 'w') as file:
        for file_path in file_paths:
            with open(file_path, 'r') as f:
                file.write("try : \n")
                file.write(change_to_legal("\n".join(["   "+content for content in f.read().split("\n")]), os.path.basename(file_path)))
                file.write("\n")
                file.write("except : \n")
                file.write("    pass\n")
    return temp_script_path

def script_exec(
    test_paths, output_path, id_path, package, batch_size, script_name="prog.py"
):  
    script_paths = []
    save_dir = os.path.join(os.getcwd(), "temp")
    if not os.path.exists(save_dir) :
        os.makedirs(save_dir)
    if package != "torch" :
        gcda_save_path = f"{id_path}-workspace"
        if os.path.exists(gcda_save_path) :
            print("gen new id path")
            id_path = random.randint(0, 1000000)
            gcda_save_path = f"{id_path}-workspace"
        # os.system(f"rm {gcda_save_path} -r")
        os.makedirs(gcda_save_path)

    model_paths = [f"{os.path.join(test_path,script_name)}" \
                   if not test_path.endswith(".py") else f"{test_path}" \
                    for test_path in test_paths]
    batched_paths = batched(model_paths, batch_size)
    for batch in tqdm(batched_paths) : 
        temp_script_path = merge_files(batch, save_dir)
        trial_arguments = [
            "python3",
            f"{temp_script_path}"
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
        # os.system(f"rm {temp_script_path}")
        p.communicate()
        exit_code = p.returncode

        # if exit_code != 0:
        #     print(
        #         f"==> model_exec crashed when generating with {output_path}! => EXIT CODE {exit_code}"
        #     )
        #     return
    print("collecting coverage . . .")
    if package != "torch" :
        coverage_collect(output_path, id_path)
        print("clear temporary files . . .")
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
    # parser.add_argument(
    #     "--tool", type=str, default='normlal', required=True, help="Folder to all the tests."
    # )
    # parser.add_argument("--batch_size", type=int, default=1000, help="")
    # parser.add_argument(
    #     "--parallel", type=int, default=1, help="Number of process for execution."
    # )

    args = parser.parse_args()

    time2path = {}
    for i, dir in enumerate(os.listdir(args.root)):
        if dir != "coverage" and not dir.endswith('json') and not dir.endswith('csv') and not dir.endswith('p') and not dir.endswith('.e') and not dir.startswith('gen_order') and not dir.endswith('record') and not dir.endswith('config'):
            # time2path[float(dir)] = os.path.join(args.root, dir)
            time2path[float(i)] = os.path.join(args.root, dir)
    time_stamps = sorted(time2path.keys())
    num_of_batch = 1
    batch_size = len(time_stamps) // num_of_batch
    print(f"Total number of tests: {len(time_stamps)}, each batch size: {batch_size}")
    cov_save = os.path.join(args.root, "coverage")
    if not os.path.exists(cov_save) :
        mkdir(cov_save)
    clear_gcda()
    id_path = f"{max(time_stamps)}.info" if args.package != 'torch' else f"{max(time_stamps)}.profraw"
    output_path = os.path.join(cov_save, id_path)
    script_exec(
        [time2path[time] for time in time_stamps], output_path, random.randint(0, 100000), args.package, batch_size=batch_size
    )
    # def batch_exec(batch):
    #     batch_paths = [time2path[time] for time in batch]
    #     format = f"{max(batch)}.info" if args.package != 'torch' else f"{max(batch)}.profraw"
    #     profraw_path = os.path.join(cov_save, format)
    #     model_exec(
    #         batch_paths, profraw_path, str(max(batch)), args.package
    #     )

    # with mp.Pool(processes=args.parallel) as pool:
    #     pool.map(batch_exec, batches)

