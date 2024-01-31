import json
import subprocess
import os
import hydra
import concurrent.futures

# Load the JSON file

BASELINES = ["symbolic-cinit", "neuri", "constrinf"]
FIXED_FUNC = "Slice"


def collect_cov(api_name, cfg):
    #execute file is at folder1/api_name.models/
    #execute file is at folder2/api_name.models/
    print(f"dealing with {api_name}")
    if cfg["model"]["type"] == "torch" :
        for method in BASELINES :
            command = (f"cd /artifact && "
                        "bash collect_cov.sh "
                        f"{cfg['mgen']['max_nodes']} "
                        f"{cfg['model']['type']} "
                        f"{method} "
                        f"{','.join([FIXED_FUNC, api_name])} "
                        f"{cfg['exp']['parallel']} "
            )
            process = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if process.stderr:
                print("Error:", process.stderr)
            else:
                print("Output:", process.stdout)

    else : #tensorflow
        pass
        # folder1 = f"$(pwd)/{folder1}/{api_name}.models"
        # folder2 = f"$(pwd)/{folder2}/{api_name}.models"
        # working_directory = "/artifact"
        # pythonpath = f"{working_directory}/:{working_directory}/neuri"
        # env = os.environ.copy()
        # env['PYTHONPATH'] = pythonpath
        # command_neuri1 = (
        #     f"python3 experiments/evaluate_tf_models.py --root {folder1} --parallel $(nproc)"
        # )
        # command_neuri2 = (
        #     f"python3 experiments/process_lcov.py --root {folder1} --parallel $(nproc)"
        # )

        # # Execute the command
        # command_richerm1 = (
        #     f"python experiments/evaluate_richerm_models.py --root {folder2} --parallel $(nproc) --package tensorflow"
        # )
        # command_richerm2 = (
        #     f"python python3 experiments/process_lcov.py --root {folder2} --parallel $(nproc)"
        # )
    return process.stdout, process.stderr

def run_draw_script(api_name : str, cfg):

    print(f"drawing with {api_name}")
    folder_names = [f"$(pwd)/gen/{cfg['model']['type']}-{method}-n{cfg['mgen']['max_nodes']}-{'-'.join([FIXED_FUNC, api_name])}.models/coverage " for method in BASELINES]
    command = (
        f"""python experiments/viz_merged_cov.py """
        f"""--folders {' '.join([f"{fnm}" for fnm in folder_names])} """
        f"""--tags {' '.join([f"{nm}" for nm in BASELINES])} """
        f"--name {api_name}"
    )
    subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def run_fuzz_script(api_name : str, cfg):
    print(f"dealing with {api_name}")
    for method in BASELINES :
        command = (f"cd /artifact && "
                    "bash fuzz.sh "
                    f"{cfg['mgen']['max_nodes']} "
                    f"{method} "
                    f"{cfg['model']['type']} "
                    f"{cfg['backend']['type']} "
                    f"{cfg['fuzz']['time']} "
                    f"{','.join([FIXED_FUNC, api_name])} "
        )
        process = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    if process.stderr:
        print("Error:", process.stderr)
    else:
        print("Output:", process.stdout)
    return process.stdout, process.stderr

def load_api_names_from_data(record_path, pass_rate) : 
    from neuri.constrinf import make_record_finder  
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
    return api_sets
# Run the script for each API set

@hydra.main(version_base=None, config_path="../neuri/config", config_name="main")
def main(cfg) : 
    func_name = run_fuzz_script #run_fuzz_script
    # api_names = load_api_names_from_json("/artifact/temp.json")
    api_names = load_api_names_from_data(cfg["mgen"]["record_path"], cfg["mgen"]["pass_rate"])
    with concurrent.futures.ProcessPoolExecutor(max_workers=cfg["exp"]["parallel"]) as executor:
        futures = [executor.submit(func_name, api_name, cfg) for api_name in api_names]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                print(f"An error occured: {e}")

    # func_name = collect_cov
    # with concurrent.futures.ProcessPoolExecutor(max_workers=cfg["exp"]["parallel"]) as executor:
    #     futures = [executor.submit(func_name, api_name, cfg) for api_name in api_names]
    #     for future in concurrent.futures.as_completed(futures):
    #         try:
    #             result = future.result()
    #         except Exception as e:
    #             print(f"An error occured: {e}")
    #     # collect_cov(api_name, cfg)
    # for api_name in api_names :
    #     run_draw_script(api_name, cfg)

if __name__ == "__main__":
    main()
