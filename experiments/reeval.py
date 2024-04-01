import json
import subprocess
import os
from typing import Literal
import hydra
import concurrent.futures
import subprocess
import threading
import multiprocessing
from experiments.evaluate_models import model_exec, batched
from experiments.evaluate_apis import activate_conda_environment, process_profraw, process_lcov, tf_batch_exec, torch_batch_exec
from experiments.evaluate_tf_models import tf_model_exec, clear_gcda
from neuri.logger import DTEST_LOG

# Load the JSON file

BASELINES = ["symbolic-cinit", "neuri", "constrinf", "constrinf_2"]
FIXED_FUNC = None #"torch.sin"#"tf.cos"#"torch.sin"
cov_parallel = 1

def collect_cov(root, model_type, backend_type, batch_size=100, backend_target="cpu", parallel=8):
    """
    Collects coverage data after fuzzing.
    
    :param root: Folder to all the tests.
    :param model_type: Model type used in fuzzing.
    :param backend_type: Backend type used in fuzzing.
    :param batch_size: Size of each batch for processing.
    :param backend_target: Backend target (cpu or cuda).
    :param parallel: Number of processes for execution.
    """

    time2path = {}
    for dir in os.listdir(root):
        if dir != "coverage":
            time2path[float(dir)] = os.path.join(root, dir)

    time_stamps = sorted(time2path.keys())
    # batch_size = len(time2path)
    # batches = [time_stamps[i:i + batch_size] for i in range(0, len(time_stamps), batch_size)]
    batches = [time_stamps] # no need to batch
    # print(f"=> Number of batches: {len(batches)} of size {batch_size}")
    print(f"=> Number of batches: {len(batches)} of size {len(time_stamps)}")

    cov_save = os.path.join(root, "coverage")
    if not os.path.exists(cov_save):
        os.mkdir(cov_save)
    if model_type == "tensorflow" :
        clear_gcda()
        batch_exec = tf_batch_exec
    elif model_type == "torch" :
        batch_exec = torch_batch_exec
    
    for batch in batches:
        batch_exec(batch, time2path, cov_save, model_type, backend_type, backend_target)

def run(api_name, baseline, config, task : Literal["fuzz", "cov"] = "cov"):
    """
    Runs the fuzzing process for a given API and baseline with the specified configuration.
    Captures and displays output in real-time.
    
    :param api_name: The name of the API to fuzz.
    :param baseline: The baseline method to use.
    :param config: Configuration parameters for the fuzzing process.
    :param max_retries: Maximum number of retries in case of failure.
    """
    tries = 0
    print(f"Running {task} API {api_name} with baseline {baseline}")
    test_pool = [FIXED_FUNC, api_name]
    test_pool_modified = '-'.join(test_pool)
    if config['mgen']['max_nodes'] is not None :
        max_nodes = config['mgen']['max_nodes']
    else :
        max_nodes = 3
    save_path = f"{os.getcwd()}/{config['exp']['save_dir']}/{config['model']['type']}-{baseline}-n{max_nodes}-{test_pool_modified}.models"
    def execute_command(command):
        """
        Executes a given command and prints its output in real-time.
        """
        print("Running\n", command)
        p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        p.communicate()
        exit_code = p.returncode
        if exit_code != 0:
            return 
    if config['model']['type'] == "tensorflow":
        if baseline == "constrinf":
            RECORD = config["mgen"]["record_path"]
        elif baseline == "constrinf_2":
            RECORD = config["mgen"]["record_path"].replace("records", "only_acc")
        else:
            RECORD = os.path.join(os.getcwd(), "data", "tf_records")
    elif config['model']['type'] == "torch":
        if baseline == "constrinf":
            RECORD = config["mgen"]["record_path"]
        elif baseline == "constrinf_2":
            RECORD = config["mgen"]["record_path"].replace("records", "only_acc")
        else:
            RECORD = os.path.join(os.getcwd(), "data", "torch_records")
    # Construct the command to run fuzz.py
    fuzz_command = f"PYTHONPATH=$(pwd):$(pwd)/neuri python neuri/cli/fuzz.py " \
                   f"fuzz.time={config['fuzz']['time']} " \
                   f"mgen.record_path={RECORD} " \
                   f"fuzz.root=$(pwd)/{config['exp']['save_dir']}/{config['model']['type']}-{baseline}-n{max_nodes}-{test_pool_modified} " \
                   f"fuzz.save_test={save_path} " \
                   f"model.type={config['model']['type']} backend.type={config['backend']['type']} filter.type=\"[nan,dup,inf]\" " \
                    f"debug.viz=true hydra.verbose=fuzz fuzz.resume=true " \
                   f"mgen.method={baseline.split('_')[0]} mgen.max_nodes={max_nodes} mgen.test_pool=\"{test_pool}\""

    if task == "fuzz":
        # while tries < 3:
        execute_command(fuzz_command)
            # tries += 1
    elif task == "cov":
        print(f"Collect Cov for {api_name} with baseline {baseline}")
        print("Activate Conda env -cov")
        activate_conda_environment("cov")
        collect_cov(root=save_path,
                    model_type=config['model']['type'],
                    backend_type=config['backend']['type'],
                    batch_size=100,  # or other desired default
                    backend_target="cpu",  # or config-specified
                    parallel=cov_parallel)  # or other desired default
        if config['model']['type'] == "torch":
            process_profraw(save_path)
        elif config['model']['type'] == "tensorflow":
            process_lcov(save_path)
        else :
            raise NotImplementedError
    else :
        raise NotImplementedError

@hydra.main(version_base=None, config_path="../neuri/config", config_name="main")
def main(cfg) : 
    from neuri.cli.train import get_completed_list
    from experiments.summarize_merged_cov import parse_directory_name
    # from experiments.summarize_merged_cov import parse_directory_name

    """
    totally, cfg['exp']['parallel'] * cov_parallel * len(BASELINES) process will be craeted
    """
    #load csv 
    global FIXED_FUNC
    if cfg["model"]["type"] == "torch" :
        FIXED_FUNC = "torch.sin"
    elif cfg["model"]["type"] == "tensorflow" :
        FIXED_FUNC = "tf.cos"
    else :
        raise NotImplementedError
    csv_paths = [
        "/artifact/experiments/results/20240401-093149.csv"
        ]
    columns = []
    retrain_list = set()
    refuzz_list = set()
    for csv_path in csv_paths :
        with open(csv_path, "r") as f:
            for i, line in enumerate(f.readlines()) :
                if i==0 :
                    columns.extend(line.split(","))
                    for i in range(len(columns)) :
                        if columns[i] == "symbolic" : columns[i] = "symbolic-cinit"
                    continue 
                row = line.split(",")
                for i, col in enumerate(row) :
                    if col.replace(".","").isdigit() :
                        is_dicol = float(col.strip())
                        if is_dicol <= 0.0 :
                            api_name = row[0].replace(".models","")
                            test_pool = [FIXED_FUNC, api_name]
                            test_pool_modified = '-'.join(test_pool)
                            if cfg['mgen']['max_nodes'] is not None :
                                max_nodes = cfg['mgen']['max_nodes']
                            else :
                                max_nodes = 3
                            save_path = f"{os.getcwd()}/{cfg['exp']['save_dir']}/{cfg['model']['type']}-{columns[i]}-n{max_nodes}-{test_pool_modified}.models"
                            if len(os.listdir(save_path)) <= 1 :
                                # print(columns[i], row[0], os.listdir(save_path))
                                refuzz_list.add((columns[i], row[0].replace(".models",""), "fuzz"))
                            retrain_list.add((columns[i], row[0].replace(".models",""), "cov"))
    # for dir_name in os.listdir(os.path.join(os.getcwd(),cfg["exp"]["save_dir"])) :
    #     if dir_name.endswith("models") :
    #         test_list = os.listdir(os.path.join(os.getcwd(),cfg["exp"]["save_dir"],dir_name))
    #         baseline, name = parse_directory_name(dir_name)
    #         name = name.replace('.models','')

    #         if "coverage" in test_list or max(map(float, test_list)) > 500 :
    #             pass
    #             # print(len(test_list), sep="")
    #             # if name in api_names :
    #             #     print(f"Remove {name} from test list")
    #             #     api_names.remove(name)
    #         else :
    #             print(max(map(float, test_list)) if test_list else 0)
    #             refuzz_list.add((baseline, name, "fuzz"))
    #             print(f"keep test {name}, {baseline}")
    #         retrain_list.add((baseline, name, "cov"))
             
    retrain_list = sorted(list(retrain_list), key=lambda x: x[1])
    refuzz_list = sorted(list(refuzz_list), key=lambda x: x[1])
    print(retrain_list)
    print(refuzz_list)
    print("retrain", len(retrain_list), "refuzz", len(refuzz_list))

    # with concurrent.futures.ProcessPoolExecutor(max_workers=cfg["exp"]["parallel"]) as executor:
    #     # Pass necessary arguments to the api_worker function
    #     futures = [executor.submit(run, api, baseline, cfg, task) for baseline, api, task in refuzz_list]
    #     for future in concurrent.futures.as_completed(futures):
    #         try:
    #             result = future.result()
    #         except Exception as e:
    #             print(f"An error occured: {e}")
    with concurrent.futures.ProcessPoolExecutor(max_workers=cfg["exp"]["parallel"]) as executor:
        # Pass necessary arguments to the api_worker function
        futures = [executor.submit(run, api, baseline, cfg, task) for baseline, api, task in retrain_list]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                print(f"An error occured: {e}")

if __name__ == "__main__":
    main()
