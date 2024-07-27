import json
import subprocess
import os
from typing import Literal
import hydra
import concurrent.futures
import subprocess
from experiments.evaluate_models import model_exec, batched
from experiments.evaluate_tf_models import tf_model_exec, clear_gcda
from nnsmith.util import parse_timestr
import shutil
import time
import random
# Load the JSON file

cov_parallel = 1

def activate_conda_environment(env_name):
    """
    Activates the specified conda environment.

    :param env_name: Name of the conda environment to activate.
    """
    activation_command = "source /opt/conda/etc/profile.d/conda.sh && conda activate " + env_name
    subprocess.run(activation_command, shell=True, executable='/bin/bash')

def torch_batch_exec(batch, time2path, cov_save, model_type, backend_type, backend_target, *args, **kwargs):
    batch_paths = [time2path[time] for time in batch]
    profraw_path = os.path.join(cov_save, f"{max(batch)}.profraw")
    model_exec(
        batch_paths,
        model_type,
        backend_type,
        backend_target,
        profraw_path,
    )

def tf_batch_exec(batch, time2path, cov_save, model_type, backend_type, backend_target, root_path):
    batch_paths = [time2path[time] for time in batch]
    profraw_path = os.path.join(cov_save, f"{max(batch)}.info")
    tf_model_exec(
        batch_paths,
        model_type,
        backend_type,
        backend_target,
        profraw_path,
        root_path.replace('/','.')
    )

def process_profraw(path):

    activation_command = "source /opt/conda/etc/profile.d/conda.sh && conda activate " + "cov" + " && "
    arguments = [
        "python",
        "experiments/process_profraws.py",
        f"--root {path}",
        "--llvm-config-path $(which llvm-config-14)",
        '--instrumented-libs "$(pwd)/build/pytorch-cov/build/lib/libtorch_cpu.so" "$(pwd)/build/pytorch-cov/build/lib/libtorch.so"',
        f"--batch-size 0 --parallel {cov_parallel}",
    ]
    full_command = activation_command + " ".join(arguments)

    p = subprocess.Popen(
        full_command,  # Show all output
        shell=True,
        executable='/bin/bash',
    )

    p.communicate()
    exit_code = p.returncode

    if exit_code != 0:
        print(
            f"==> process_profraw crashed when generating {os.path.join(path, 'coverage','merged_cov.pkl')}! => EXIT CODE {exit_code}"
        )    
def process_lcov(path):
    activation_command = "source /opt/conda/etc/profile.d/conda.sh && conda activate " + "cov" + " && "
    arguments = [
        "python",
        "experiments/process_lcov.py",
        f"--root {path}",
        f"--batch-size 0 --parallel {cov_parallel}",
    ]
    full_command = activation_command + " ".join(arguments)
    print(full_command)
    p = subprocess.Popen(
        full_command,  # Show all output
        shell=True,
        executable='/bin/bash',
    )

    p.communicate()
    exit_code = p.returncode

    if exit_code != 0:
        print(
            f"==> process_lcov crashed when generating {os.path.join(path, 'coverage','merged_cov.pkl')}! => EXIT CODE {exit_code}"
        )
def parallel_eval(api_list, BASELINES, config, task):
    """
    Runs fuzzing processes in parallel for each combination of API and baseline.
    
    :param api_list: List of APIs to fuzz.
    :param baseline_list: List of baseline methods to use.
    :param config: Configuration parameters for the fuzzing process.
    """

    with concurrent.futures.ProcessPoolExecutor(max_workers=config["exp"]["parallel"]) as executor:
        # Pass necessary arguments to the api_worker function
        futures = [executor.submit(api_worker, api, config, BASELINES, task) for api in api_list]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                print(f"An error occured: {e}")

def api_worker(api, cfg, BASELINES, task):
    """
    Top-level worker function for multiprocessing, handles each API task.
    """
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(BASELINES)) as executor:
        # Pass necessary arguments to the api_worker function
        futures = [executor.submit(run, api, baseline, cfg, task) for baseline in BASELINES]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                print(f"An error occured: {e}")
    # Trigger drawing after completing all baseline tasks for the current API
    # run_draw_script(api, cfg, BASELINES)

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
    # print(f"=> Number of batches: {len(batches)} of size {len(time_stamps)}")

    cov_save = os.path.join(root, "coverage")
    if not os.path.exists(cov_save):
        os.mkdir(cov_save)
    if model_type == "tensorflow" :
        clear_gcda()
        batch_exec = tf_batch_exec
    elif model_type == "torch" :
        batch_exec = torch_batch_exec
    
    for batch in batches:
        batch_exec(batch, time2path, cov_save, model_type, backend_type, backend_target, root)

    del_test_cases(root)

def del_test_cases(root) :
    for dir in os.listdir(root):
        if dir != "coverage":
            os.system(f"rm -r {os.path.join(root, dir)}")

def cal_time(time, baseline) :
    parsed_time = parse_timestr(time)
    num_of_baselines = len(baseline.split("+"))
    time = parsed_time // len(num_of_baselines)
    return str(int(time))+"s"

def acetest_run(api_name, config, save_path) : 
    time_to_seconds = parse_timestr(config["fuzz"]["time"])
    os.makedirs(save_path, exist_ok=True)
    package = config["model"]["type"] 
    if config["model"]["type"] == "tensorflow" :
        package = "tf"
    return f"python main.py --test_round=100000 --mode=cpu_ori --framework={package} --work_path={save_path} --target_api={api_name} --save_non_crash=true --api_timeout={time_to_seconds}"

def move_files_to_directory(source_dirs, target_dir):
    """
    Move files from source directories to a target directory.

    :param source_dirs: List of directories to move files from.
    :param target_dir: Directory to move files to.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for dir in source_dirs:
        if not os.path.exists(dir):
            print(f"Source directory does not exist: {dir}")
            continue

        for file_name in os.listdir(dir):
            file_path = os.path.join(dir, file_name)
            if os.path.isfile(file_path):
                shutil.move(file_path, target_dir)
                print(f"Moved {file_name} to {target_dir}")
    
def script_cov(api_name, config, save_path, baseline) : 
    from experiments.collect_cov_by_script import script_exec
    package = config["model"]["type"] 
    if config["model"]["type"] == "tensorflow" :
        package = "tf"

    time2path = {}
    if baseline in ["acetest"] :
        save_path = os.path.join(save_path, f"output_{package}_0", api_name, "non_crash")
    elif baseline == "doctor" : 
        save_path = os.path.join(os.path.dirname(save_path), api_name)
    for i, dir in enumerate(os.listdir(save_path)):
        if dir != "coverage" and not dir.endswith('json') and not dir.endswith('csv') and not dir.endswith('p') and not dir.endswith('.e') and not dir.startswith('gen_order') and not dir.endswith('record') and not dir.endswith('config'):
            # time2path[float(dir)] = os.path.join(save_path, dir)
            time2path[float(i)] = os.path.join(save_path, dir)
    time_stamps = sorted(time2path.keys())
    num_of_batch = 2
    batch_size = len(time_stamps) // num_of_batch
    print(f"Total number of tests: {len(time_stamps)}, each batch size: {batch_size}")
    cov_save = os.path.join(save_path, "coverage")
    if not os.path.exists(cov_save) :
        os.mkdir(cov_save)
    clear_gcda()
    _id = max(time_stamps) if time_stamps else 0
    id_path = f"{_id}.info" if package != 'torch' else f"{_id}.profraw"
    output_path = os.path.join(cov_save, id_path)
    script_exec(
        [time2path[time] for time in time_stamps], output_path, random.randint(0, 100000), package, batch_size=batch_size
    )
    return save_path

def run(api_name, baseline, config, task : Literal["fuzz", "cov"] = "cov", dirname=None, fuzz_time=None, run_cov=True):
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
    test_pool = [api_name]
    test_pool_modified = '-'.join(test_pool)
    if config['mgen']['max_nodes'] is not None :
        max_nodes = config['mgen']['max_nodes']
    else :
        max_nodes = 3
    
    root_save_path = gen_save_path(api_name, baseline, config) if dirname is None else dirname
    save_path = f"{root_save_path}.models"
    def execute_command(command):
        """
        Executes a given command and prints its output in real-time.
        """
        print("Running\n", command)
        if baseline in ["acetest", "doctor"] :
            cwd = "/DeepConstr/ACETest/Tester/src"
        else :
            cwd = os.getcwd()
        time.sleep(7)
        with subprocess.Popen(command, cwd = cwd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as p:
            for line in p.stdout:
                print(line, end='')
            p.wait(timeout=60*11)
            exit_code = p.returncode
            if exit_code != 0:
                print(f"Command failed with exit code {exit_code}")
            else:
                print("Command executed successfully!")

    if config['model']['type'] == "tensorflow":
        if baseline == "deepconstr":
            RECORD = config["mgen"]["record_path"]
        elif baseline == "deepconstr_2":
            RECORD = config["mgen"]["record_path"].replace("records", "only_acc")
        else:
            RECORD = os.path.join(os.getcwd(), "data", "tf_records")
    elif config['model']['type'] == "torch":
        if baseline == "deepconstr":
            RECORD = config["mgen"]["record_path"]
        elif baseline == "deepconstr_2":
            RECORD = config["mgen"]["record_path"].replace("records", "only_acc")
        else:
            RECORD = os.path.join(os.getcwd(), "data", "torch_records")
    # Construct the command to run fuzz.py
    if fuzz_time is None :
        fuzz_time = config["fuzz"]["time"]
    if baseline in ["acetest"] :
        fuzz_command = acetest_run(api_name, config, save_path)
    elif baseline in ["doctor"] :
        print("We need to run doctor and move test files to here")
        pass
    else : 
        fuzz_command = f"PYTHONPATH=$(pwd):$(pwd)/nnsmith:$(pwd)/deepconstr python nnsmith/cli/fuzz.py " \
                    f"fuzz.time={fuzz_time} " \
                    f"mgen.record_path={RECORD} " \
                    f"fuzz.root={root_save_path} " \
                    f"fuzz.save_test={save_path} " \
                    f"model.type={config['model']['type']} backend.type={config['backend']['type']} filter.type=\"[nan,dup,inf]\" " \
                        f"debug.viz=true hydra.verbose=fuzz fuzz.resume=false " \
                    f"mgen.method={baseline.split('_')[0]} mgen.max_nodes={max_nodes} mgen.test_pool=\"{test_pool}\""
    if task == "cov" : 
        print(f"Collect Cov for {api_name} with baseline {baseline}")
        print("Activate Conda env -cov")
        activate_conda_environment("cov")
        if baseline in ["acetest", "doctor"] :
            save_path = script_cov(api_name, config, save_path, baseline)
            pass

        else :
            collect_cov(root=save_path,
                        model_type=config['model']['type'],
                        backend_type=config['backend']['type'],
                        batch_size=100,  # or other desired default
                        backend_target="cpu",  # or config-specified
                        parallel=cov_parallel)  # or other desired default
        if config['model']['type'] == "torch":
            print("running process_profraw", save_path)
            time.sleep(5)
            process_profraw(save_path)
        elif config['model']['type'] == "tensorflow":
            process_lcov(save_path)
        else :
            raise NotImplementedError
    elif task == "fuzz":
        if "+" in baseline :
            baselines = baseline.split("+")
            fuzz_time = cal_time(config['fuzz']['time'], baseline)
            for base in baselines :
                run(api_name, base.strip(), config, task = "fuzz", dirname=gen_save_path(api_name, baseline, config), fuzz_time = fuzz_time, run_cov=False)
        else :
            if baseline != "doctor" :
                execute_command(fuzz_command)
        if run_cov :
            run(api_name, baseline, config, task = "cov")
    else :
        raise NotImplementedError

def load_from_csvs(csv_paths) :   
    columns = []
    recollect_list = set()
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
                        col = float(col.strip())
                        if col == 0 : 
                            refuzz_list.add((columns[i], row[0].replace(".models",""), "fuzz"))
                            recollect_list.add((columns[i], row[0].replace(".models",""), "cov"))

    return recollect_list, refuzz_list

def gen_save_path(api_name, baseline, cfg) :
    test_pool = [api_name]
    test_pool_modified = '-'.join(test_pool)
    if cfg['mgen']['max_nodes'] is not None :
        max_nodes = cfg['mgen']['max_nodes']
    else :
        max_nodes = 5
    if "+" in baseline :
        root_save_path = f"{os.getcwd()}/{cfg['exp']['save_dir']}/{baseline}-{test_pool_modified}"
    else :
        root_save_path = f"{os.getcwd()}/{cfg['exp']['save_dir']}/{cfg['model']['type']}-{baseline}-n{max_nodes}-{test_pool_modified}"
    return root_save_path


def load_api_names_from_data(record_path, pass_rate) : 
    from deepconstr.gen.record import make_record_finder  
    records = make_record_finder(
        path=record_path,
        pass_rate=pass_rate,
    )
    return [
        record.get("name") for record in records
    ]

def load_api_names_from_json(path) :
    with open(path, 'r') as file:
        api_sets = json.load(file)
    return list(set(api_sets))
# Run the script for each API set

def need_to_collect_cov(api_name, base_line, cfg) : 
    save_path = gen_save_path(api_name, base_line, cfg)+".models"
    cov_dir_path = os.path.join(save_path, "coverage", "merged_cov.pkl")
    return not os.path.exists(cov_dir_path)

def need_to_gen_testcases(api_name, base_line, cfg) : 
    save_path = gen_save_path(api_name, base_line, cfg)+".models"
    if os.path.exists(save_path) :
        test_list = os.listdir(save_path)
        if "coverage" in test_list : return False
        if test_list : 
            if isinstance( cfg["fuzz"]["time"], str):
                timeout_s = parse_timestr( cfg["fuzz"]["time"])
            assert isinstance(
                 timeout_s, int
            ), "`fuzz.time` must be an integer (with `s` (default), `m`/`min`, or `h`/`hr`)."
            if max(map(float, test_list)) > timeout_s-60 : return False
    return True

def load_nnsmith_api_names(model_type) :
    if model_type == "torch" :
        with open("./data/torch_nnsmith.json", "r") as f :
            api_names = json.load(f)
    elif model_type == "tensorflow" :
        with open("./data/tf_nnsmith.json", "r") as f :
            api_names = json.load(f)
    return api_names
    
@hydra.main(version_base=None, config_path="../nnsmith/config", config_name="main")
def main(cfg) : 
    # from nnsmith.cli.train import get_completed_list
    # from experiments.summarize_merged_cov import exclude_intestable
    """
    task = ['deepconstr', 'neuri', 'acetest', 'symbolic-cinit', 'deepconstr_2']
    if gives multiple tasks connecting with the charactor "+", it will fuzz and collect coverage them in together.
    ex) task = ['deepconstr+neuri', 'symbolic-cinit+deepconstr_2']
    """
    # model_type=cfg["model"]["type"]
    # backend_type=cfg["backend"]["type"]
    # root="/DeepConstr/exp/deepconstr_1/tf/tensorflow-deepconstr-n1-tf.raw_ops.Abs.models"
    # collect_cov(root, model_type, backend_type, batch_size=100, backend_target="cpu", parallel=8)
    recollect_list = set()
    refuzz_list = set()
    # api_names = load_api_names_from_data(cfg["mgen"]["record_path"], cfg["mgen"]["pass_rate"])
    # api_names = list(set(api_names))
    nnsmith_api_list = load_nnsmith_api_names(cfg["model"]["type"])
    with open(cfg["exp"]["targets"], "r") as f :
        api_names = json.load(f)
    if cfg["exp"]["mode"] is not None :
        if cfg["exp"]["mode"] == "fix" :
            assert len(cfg["exp"]["baselines"]) == 1, "Fix mode only support single baseline" 
            refuzz_list = [(cfg["exp"]["baselines"][0], name , "fuzz") for name in api_names]
            recollect_list = [(cfg["exp"]["baselines"][0], name, "cov") for name in api_names]

    else :
        for baseline in cfg["exp"]["baselines"] : 
            for api_name in api_names :
                if baseline not in ["acetest", "doctor"] :
                    if baseline in ["symbolic-cinit"] : 
                        if api_name not in nnsmith_api_list :
                            continue 
                    if need_to_gen_testcases(api_name, baseline, cfg) :
                        refuzz_list.add((baseline, api_name, "fuzz"))
                else :
                    refuzz_list.add((baseline, api_name, "fuzz"))
                refuzz_list.add((baseline, api_name, "fuzz"))
                recollect_list.add((baseline, api_name, "cov"))
                
    recollect_list = sorted(list(recollect_list), key=lambda x: x[1])
    refuzz_list = sorted(list(refuzz_list), key=lambda x: x[1])
    all_tasks = refuzz_list + recollect_list
    print("recollect", len(recollect_list), "refuzz", len(refuzz_list))
    with concurrent.futures.ProcessPoolExecutor(max_workers=cfg["exp"]["parallel"]) as executor:
        # Pass necessary arguments to the api_worker function
        futures = [executor.submit(run, api, baseline, cfg, task) for baseline, api, task in all_tasks]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                print(f"An error occured: {e}")

if __name__ == "__main__":
    main()
