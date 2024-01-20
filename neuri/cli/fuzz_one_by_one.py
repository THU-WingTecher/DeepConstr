import json
import subprocess
import os
import hydra
# Load the JSON file

BASELINES = ["symbolic-cinit", "neuri", "constrinf"]
FIXED_FUNC = "Slice"


def collect_cov(api_name, cfg):
    #execute file is at folder1/api_name.models/
    #execute file is at folder2/api_name.models/
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
        

    else : #tensorflow
        folder1 = f"$(pwd)/{folder1}/{api_name}.models"
        folder2 = f"$(pwd)/{folder2}/{api_name}.models"
        working_directory = "/artifact"
        pythonpath = f"{working_directory}/:{working_directory}/neuri"
        env = os.environ.copy()
        env['PYTHONPATH'] = pythonpath
        if package == "tensorflow" :
            
            command_neuri1 = (
                f"python3 experiments/evaluate_tf_models.py --root {folder1} --parallel $(nproc)"
            )
            command_neuri2 = (
                f"python3 experiments/process_lcov.py --root {folder1} --parallel $(nproc)"
            )

            # Execute the command
            command_richerm1 = (
                f"python experiments/evaluate_richerm_models.py --root {folder2} --parallel $(nproc) --package tensorflow"
            )
            command_richerm2 = (
                f"python python3 experiments/process_lcov.py --root {folder2} --parallel $(nproc)"
            )
        process = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if process.stderr:
            print("Error:", process.stderr)
        else:
            print("Output:", process.stdout)
    return process.stdout, process.stderr

def run_draw_script(api_name : str, cfg):

    commands = (
        f"""python experiments/viz_merged_cov.py --root {folder2} """,
        f"""--folders {' '.join([f"$(pwd)/gen/{fnm}" for fnm in folder_names])} """,
        f"""--tags {' '.join([f"{nm}" for nm in BASELINES])} """,
        f"--name '{api_name}'"
    )
    subprocess.run(plot, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)

def run_fuzz_script(api_name : str, cfg):

    for method in BASELINES :
        command = (f"cd /artifact && "
                    "bash fuzz.sh "
                    f"{cfg['mgen']['max_nodes']} "
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
# Run the script for each API set

@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg) : 
    # json_file_path = cfg['eval']['test_pool']
    api_names = load_api_names_from_data(cfg["mgen"]["record_path"], cfg["mgen"]["pass_rate"])
    for api_name in api_names :
        run_fuzz_script(api_name, cfg)
        # collect_cov(api_name, cfg)
        # run_draw_script(api_name, cfg)

if __name__ == "__main__":
    main()