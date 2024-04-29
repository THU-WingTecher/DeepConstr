"""
Given the directory containing all tests. We replay the test execution and record coverage in LLVM profraw format.
The intermediate tests can be saved using fuzz.save_test={{DIR_TO_SAVE}}.
"""
import multiprocessing as mp
import os
import subprocess
import sys
from nnsmith.util import mkdir


def batched(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


def model_exec(test_paths, model_type, backend_type, backend_target, profraw_path):
    model_paths = []
    for test_path in test_paths:
        for file in os.listdir(test_path):
            if file.startswith("model"):
                model_paths.append(os.path.join(test_path, file))
                break
    activation_command = "source /opt/conda/etc/profile.d/conda.sh && conda activate " + "cov" + " && "
    # model_paths = model_paths[:len(model_paths)//10]
    batches = batched(model_paths, n=500)
    for model_paths in batches :
        arguments = [
            "python3",
            "neuri/cli/model_exec.py",
            "model.type=" + model_type,
            "backend.type=" + backend_type,
            "backend.target=" + backend_target,
            f"'model.path={model_paths}'",
        ]
        full_command = activation_command + " ".join(arguments)
        copied_env = os.environ.copy()
        copied_env["LLVM_PROFILE_FILE"] = profraw_path

        p = subprocess.Popen(
            full_command,  # Show all output
            env=copied_env,
            shell=True,
            executable='/bin/bash',
        )
        p.communicate()
        exit_code = p.returncode

        if exit_code != 0:
            print(
                f"==> model_exec crashed when generating {profraw_path}! => EXIT CODE {exit_code}"
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=str, required=True, help="Folder to all the tests."
    )
    parser.add_argument("--batch_size", type=int, default=10000, help="")
    parser.add_argument("--model_type", type=str, required=True, help="Model type.")
    parser.add_argument("--backend_type", type=str, required=True, help="Backend type.")
    parser.add_argument(
        "--backend_target", type=str, default="cpu", help="Say `cpu` or `cuda`."
    )
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
    if not os.path.exists(cov_save) :
        mkdir(cov_save)

    def batch_exec(batch):
        batch_paths = [time2path[time] for time in batch]
        profraw_path = os.path.join(cov_save, f"{max(batch)}.profraw")
        model_exec(
            batch_paths,
            args.model_type,
            args.backend_type,
            args.backend_target,
            profraw_path,
        )

    with mp.Pool(processes=args.parallel) as pool:
        pool.map(batch_exec, batches)
