import json
import subprocess
import os
import hydra
# Load the JSON file


# Function to run the fuzz.py script
def eval_cov_one_by_one(api_name, folder1, folder2, package='torch'):
    command = f"cd /artifact && bash eva_cov.sh {api_name} {folder1} {folder2}"
    process = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    if process.stderr:
        print("Error:", process.stderr)
    else:
        print("Output:", process.stdout)


def _eval_cov_one_by_one(api_name, folder1, folder2, package='torch'):
    #execute file is at folder1/api_name.models/
    #execute file is at folder2/api_name.models/
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
        
    elif package == "torch" :
        command_neuri1 = (
            f"python experiments/evaluate_models.py  --root {folder1} --model_type torch --backend_type torchjit --parallel $(nproc)"
        )
        command_neuri2 = (
            f"""python experiments/process_profraws.py --root {folder1} """,
            f"""--llvm-config-path $(which llvm-config-14) """,
            f"""--instrumented-libs "$(pwd)/build/pytorch-cov/build/lib/libtorch_cpu.so" "$(pwd)/build/pytorch-cov/build/lib/libtorch.so" """,
            f"--parallel $(nproc)"
        )
        command_richerm1 = (
            f"python experiments/evaluate_richerm_models.py --root {folder2} --parallel $(nproc) --package torch"
        )
        command_richerm2 = (
            f"""python experiments/process_profraws.py --root {folder2} """,
            f"""--llvm-config-path $(which llvm-config-14) """,
            f"""--instrumented-libs "$(pwd)/build/pytorch-cov/build/lib/libtorch_cpu.so" "$(pwd)/build/pytorch-cov/build/lib/libtorch.so" """,
            f"--parallel $(nproc)"
        )

    process = subprocess.run(command_neuri1, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    print(process.stderr)
    process = subprocess.run(command_neuri2, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    print(process.stderr)
        

    process = subprocess.run(command_richerm1, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    print(process.stderr)
    process = subprocess.run(command_richerm2, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    print(process.stderr)

    plot = (
        f"""python experiments/viz_merged_cov.py --root {folder2} """,
        f"""--folders {folder1}/coverage {folder2}/coverage """,
        """--tags '\textsc{neuri}' '\textsc{us}' """,
        f"--name '{api_name}'"
    )
    subprocess.run(plot, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)

# python3 experiments/evaluate_richerm_models.py --root $(pwd)/gen/complete_shape_constraints --parallel $(nproc)
# python3 experiments/process_lcov.py --root $(pwd)/gen/complete_shape_constraints --parallel $(nproc)

def run_fuzz_script(api_name, cfg):
    # Extract parameters

    if cfg['eval_each']['model'] == "torch" :
        BACKEND="torchjit"
        RECORD="$(pwd)/data/torch_records"
    elif cfg['eval_each']['model'] == "tensorflow" :
        BACKEND="xla"
        RECORD="$(pwd)/data/tf_records"
    # Create the command to run
    working_directory = "/artifact"
    pythonpath = f"{working_directory}/:{working_directory}/neuri"
    command = (
        f"python neuri/cli/fuzz.py fuzz.time={cfg['eval_each']['time']} "
        f"fuzz.root=$(pwd)/{cfg['eval_each']['save_folder']}/{api_name} "
        f"fuzz.pool=['{api_name}'] "
        f"mgen.record_path={RECORD} "
        f"fuzz.save_test=$(pwd)/{cfg['eval_each']['save_folder']}/{api_name}.models "
        f"model.type={cfg['eval_each']['model']} backend.type={BACKEND} filter.type=\"[nan,dup,inf]\" "
        f"debug.viz=true hydra.verbose=fuzz fuzz.resume=true "
        f"mgen.method={cfg['eval_each']['method']} mgen.max_nodes={cfg['eval_each']['max_nodes']}"
    )
    env = os.environ.copy()
    env['PYTHONPATH'] = pythonpath
    # Execute the command
    process = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    return process.stdout, process.stderr


# Run the script for each API set
@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg) : 
    json_file_path = cfg['eval']['test_pool']
    with open(json_file_path, 'r') as file:
        api_sets = json.load(file)
    for api_name in api_sets :
        if cfg['eval']['type'] == 'test' :
            stdout, stderr = run_fuzz_script(api_name, cfg)
            if stdout:
                print(f"Output for {api_name}:\n{stdout.decode()}")
            if stderr:
                print(f"Error for {api_name}:\n{stderr.decode()}")
        elif cfg['eval']['type'] == 'cov' :
            eval_cov_one_by_one(api_name, cfg['eval']['neuri'], cfg['eval']['us'], cfg['eval']['package'])
if __name__ == "__main__":
    main()