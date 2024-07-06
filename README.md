# Introduction

This repository contains the implementation of DeepConstr.

### Get Ready

Before you start, please make sure you have [Docker](https://docs.docker.com/engine/install/) installed.
To check the installation:
```bash
docker --version # Test docker availability
```
Get Docker image from Docker Hub
```bash
docker pull gwihwan/artifact-issta24:latest
``` 
Navigate to the DeepConstr project directory.
```bash
cd ../DeepConstr
```
### Start fuzzing

You can start fuzzing with the `fuzz.py` script.

> [!NOTE]
>
> **Command usage of**: `python nnsmith/cli/fuzz.py`
>
> **Arguments**:
> - `mgen.max_nodes`: the number of operators in each generated graph.
> - `mgen.method`: approach of generated constraints, choose from `["deepconstr", "neuri", "symbolic-cinit"]`.
> - `model.type`: generated model type, choose from `["tensorflow", "torch"]`.
> - `backend.type`: generated backend type, choose from `["xla", "torchjit"]`.
> - `fuzz.time`: fuzzing time in formats such as `4h`, `1m`, `30s`.
> - `mgen.record_path`: the directory that constraints are saved, such as `$(pwd)/data/records/torch`.
> - `fuzz.save_test`: the directory that generated test cases are saved, such as `$(pwd)/bugs/${model.type}-${mgen.method}-n${mgen.max_nodes}`.
> - `fuzz.root`: the directory that buggy test cases are saved, such as `$(pwd)/bugs/${model.type}-${mgen.method}-n${mgen.max_nodes}-buggy`.
> - `mgen.test_pool`(Optional): specific API for fuzzing. If not specified, fuzzing will be conducted across all prepared APIs.
>
> **Outputs**:
> The buggy test cases will be saved in the directory specified by `fuzz.root`, while every generated test case will be saved in the directory specified by `fuzz.save_test`.

#### Quick start for PyTorch

First, activate the conda environment created for this project.
```bash 
conda activate std
``` 

For PyTorch, you can specify the APIs to be tested by setting the `mgen.test_pool` argument, such as `[torch.abs,torch.add]`. For example, following code will fuzz `torch.abs` and `torch.add` for 15 minutes.
```bash
PYTHONPATH=$(pwd):$(pwd)/deepconstr:$(pwd)/nnsmith python nnsmith/cli/fuzz.py fuzz.time=15m \
mgen.record_path=/DeepConstr/data/records/torch fuzz.root=/DeepConstr/outputs/torch-deepconstr-n5-torch.abs-torch.add \
fuzz.save_test=/DeepConstr/outputs/torch-deepconstr-n5-torch.abs-torch.add.models \
model.type=torch backend.type=torchcomp filter.type=[nan,dup,inf] \
debug.viz=true hydra.verbose=['fuzz'] fuzz.resume=true \
mgen.method=deepconstr mgen.max_nodes=5 mgen.test_pool=[torch.abs,torch.add] mgen.pass_rate=10
```
If the `mgen.test_pool` is not specified, the program will fuzz all APIs that deepconstr supports. Following code will fuzz all APIs that deepconstr support for 4 hours.
```bash
PYTHONPATH=$(pwd):$(pwd)/deepconstr:$(pwd)/nnsmith python nnsmith/cli/fuzz.py fuzz.time=4h \
mgen.record_path=/DeepConstr/data/records/torch \
fuzz.root=/DeepConstr/outputs/torch-deepconstr-n5-torch.abs-torch.add \
fuzz.save_test=/DeepConstr/outputs/torch-deepconstr-n5-torch.abs-torch.add.models \
model.type=torch backend.type=torchcomp filter.type=[nan,dup,inf] debug.viz=true hydra.verbose=['fuzz'] fuzz.resume=true mgen.method=deepconstr mgen.max_nodes=5 mgen.pass_rate=10
```

#### Quick start for TensorFlow
First, activate the conda environment created for this project.
```bash
conda activate std
```
Then, execute the following commands to start fuzzing. Following code will fuzz all APIs that deepconstr supports for 4 hours.
```bash 
PYTHONPATH=$(pwd):$(pwd)/deepconstr:$(pwd)/nnsmith python nnsmith/cli/fuzz.py fuzz.time=4h \
mgen.record_path=/DeepConstr/data/records/tf \
fuzz.root=/DeepConstr/outputs/tensorflow-deepconstr-n5- fuzz.save_test=/DeepConstr/outputs/tensorflow-deepconstr-n5-.models \
model.type=tensorflow backend.type=xla filter.type=[nan,dup,inf] \
debug.viz=true hydra.verbose=['fuzz'] \
fuzz.resume=true mgen.method=deepconstr mgen.max_nodes=5 mgen.pass_rate=10
```

#### Generate python code

The test case of deepconstr is saved as the format of `gir.pkl`. To convert the `git.pkl` into python code, you can utilize below code. You can specify the code with the option of compiler. For now, we support "torchcomp" compiler with pytorch. You can use following code to convert the `gir.pkl` which is saved at `code_saved_dir` into python code.

```bash
python nnsmith/materialize/torch/program.py ${code_saved_dir} torchcomp
```

# Extract Constraints

### Setup Instructions

1. (optional) If you are not using docker, install required packages:
```bash 
pip install -r requirements.txt
```
2. Generate a `.env` file in your workspace directory `$(pwd)/.env` and populate it with your specific values:
- OpenAI API Key:
```OPENAI_API_KEY1 ='sk-********'```
- Proxy Setting (Optional):
```MYPROXY ='166.***.***.***:****'```

3. Testing Your Configuration: After setting your environment variables, you can verify your configuration by running:
```bash
PYTHONPATH=$(pwd):$(pwd)/deepconstr:$(pwd)/nnsmith python tests/proxy.py
# INFO    llm    - Output(Ptk12-OtkPtk9) : 
# Hello! How can I assist you today? 
# Time cost : 1.366152286529541 seconds 
```
If configured correctly, you will receive a response from the OpenAI API, such as: "Hello! How can I assist you today?"

### Start Extraction
You can extract constraints by running `deepconstr/train/run.py` script.

> [!NOTE]
>
> **Command usage of**: `python deepconstr/train/run.py`
>
> **Important Arguments**:
> - `tran.target`: Specifies the API name or path to extract. This can be a single API name (e.g., `"torch.add"`), a list containing multiple API names (e.g., `["torch.add", "torch.abs"]`), or a JSON file path containing the list.
> - `train.retrain`: A boolean value that determines whether to reconduct constraint extraction. If set to false, the tool will only collect APIs that haven't been extracted. If set to true, the tool collects all APIs except those where the pass rate exceeds the preset target pass rate (`train.pass_rate`).
> - `train.pass_rate`: The target pass rate to filter out APIs that have a pass rate higher than this target.
> - `train.parallel`: The number of parallel processes used to validate the constraints. We do not recommend to set this argument to 1.
> - `train.record_path`: The path where the extracted constraints are saved. This directlry should be the same as the `mgen.record_path` in the fuzzing step.
> - `hydra.verbose`: Set the logging level of Hydra for specific modules ("smt", "train", "convert", "constr", "llm", etc). If you want to see all the log messages, you can set it to `True`.
> - `train.num_eval`: The number of evaluations performed to validate the constraints (default: 500).
> - `model.type`: Choose from `["tensorflow", "torch"]`.
> - `backend.type`: Choose from `["xla", "torchjit"]`.
>
> **Other Arguments**:
> For additional details, refer to the values under train at `/DeepConstr/nnsmith/config/main.yaml`.
>
> **Outputs**:
> - `$(pwd)/${train.record_path}/torch` if `model.type` is `torch`
> - `$(pwd)/${train.record_path}/tf` if `model.type` is `tensorflow`


#### Quick Start :

Please set your `train.record_path` to the desired location that you want to store. For instance, `$(pwd)/repro/records/torch`

##### for PyTorch 
Below command will extract constraints from `"torch.add","torch.abs"`. The extracted constraints are stored to `$(pwd)/repro/records/torch`. We recommand to set `train.parallel` to larger than 1.

```bash
PYTHONPATH=/DeepConstr/:/DeepConstr/nnsmith/:/DeepConstr/deepconstr/:$PYTHONPATH \
python deepconstr/train/run.py train.record_path=repro/records/torch backend.type=torchcomp \
model.type=torch hydra.verbose=train train.parallel=1 train.num_eval=500 \
train.pass_rate=95 hydra.verbose=['train'] \
train.retrain=false train.target='["torch.add","torch.abs"]'
```
By specifying the path to a JSON file, you can target a specific set of APIs for processing. This JSON file should contain a list of API names.

```bash
PYTHONPATH=/DeepConstr/:/DeepConstr/nnsmith/:/DeepConstr/deepconstr/:$PYTHONPATH \
python deepconstr/train/run.py train.record_path=repro/records/torch backend.type=torchcomp \
model.type=torch hydra.verbose=train train.parallel=1 train.num_eval=500 \
train.pass_rate=95 hydra.verbose=['train'] \
train.retrain=false train.target='/your/json/path'
```

##### for TensorFlow 

Below command will extract constraints from `"tf.add", "tf.abs"`. The extracted constraints are stored to `$(pwd)/repro/records/tf`. 

```bash
PYTHONPATH=/DeepConstr/:/DeepConstr/nnsmith/:/DeepConstr/deepconstr/:$PYTHONPATH \
python deepconstr/train/run.py train.record_path=repro/records/tf backend.type=xla \
model.type=tensorflow hydra.verbose=train train.parallel=1 train.num_eval=300 train.pass_rate=95 hydra.verbose=['train'] \
train.retrain=false train.target='["tf.add","tf.abs"]'
```

##### For NumPy( new added during rebuttal)
```bash
PYTHONPATH=/DeepConstr/:/DeepConstr/nnsmith/:/DeepConstr/deepconstr/:$PYTHONPATH \
python deepconstr/train/run.py train.record_path=repro/records/numpy backend.type=numpy \
model.type=numpy hydra.verbose=train train.parallel=1 train.num_eval=300 train.pass_rate=95 hydra.verbose=['train'] \
train.retrain=false train.target='["numpy.add"]'
```
# Reproduce Experiments

### Comparative Experiment (RQ1) 

#### Check trained operators( table 1)

You can inspect the number of trained APIs by executing the following commands:

```bash 
python experiments/apis_overview.py /DeepConstr/data/records
# Number of trained tf apis:  258
# Number of trained torch apis:  843
```

#### Coverage Comparison Experiment
> [!NOTE]
> For this step, you need first download PyTorch and TensorFlow and compile them 
> Or you can pull our docker containor(Work in Progress)

We have four baselines for conducting experiments. Additionally, approximately 700 operators (programs) require testing for PyTorch and 150 operators for TensorFlow. Given that each operator needs to be tested for 15 minutes, completing the experiment will be time-intensive. To expedite the process, we recommend using the `exp.parallel` argument to enable multiple threads during the experiment. The experiment results will be saved in the folder specified by `exp.save_dir`.

##### for PyTorch 

First, change the environment to the conda environment created for this project. We strongly recommend to set `exp.parallel` larger than 1.

```bash
conda activate cov
```

```bash
PYTHONPATH=/DeepConstr/:/DeepConstr/nnsmith/:/DeepConstr/deepconstr/:$PYTHONPATH \
python experiments/evaluate_apis.py \
exp.save_dir=exp/torch mgen.record_path=$(pwd)/data/records/torch/ mgen.pass_rate=0.05 model.type=torch backend.type=torchjit fuzz.time=15m exp.parallel=1 mgen.noise=0.8 exp.targets=/DeepConstr/data/torch_dc_neuri.json exp.baselines="['deepconstr', 'neuri', 'symbolic-cinit', 'deepconstr_2']"
```

<!-- for testing acetest
```bash
PYTHONPATH=/DeepConstr/:/DeepConstr/nnsmith/:/DeepConstr/deepconstr/:$PYTHONPATH python experiments/evaluate_apis.py exp.save_dir= mgen.max_nodes=1 mgen.pass_rate=0.05 model.type=torch backend.type=torchjit fuzz.time=5m exp.parallel=1 mgen.noise=0.8 exp.targets=/DeepConstr/data/tf_dc_acetest.json exp.baselines=['acetest']
``` -->
##### for TensorFlow 

<!-- First, change the environment to the conda environment created for this project.
```bash
conda activate cov
``` -->

```bash
PYTHONPATH=/DeepConstr/:/DeepConstr/nnsmith/:/DeepConstr/deepconstr/:$PYTHONPATH \
python experiments/evaluate_apis.py \
exp.save_dir=exp/tf mgen.record_path=$(pwd)/data/records/tf/ mgen.pass_rate=0.05 model.type=tensorflow backend.type=xla fuzz.time=15m exp.parallel=1 mgen.noise=0.8 exp.targets=/DeepConstr/data/tf_dc_neuri.json exp.baselines=['deepconstr']
```

<!-- for testing acetest
```bash
PYTHONPATH=/DeepConstr/:/DeepConstr/nnsmith/:/DeepConstr/deepconstr/:$PYTHONPATH python experiments/evaluate_apis.py exp.save_dir=exp/aceteonstr_1/tf mgen.max_nodes=1 mgen.records/tf mgen.max_nodes=1 mgen.record=onstr_1/tf mgen.max_nodes=1 mgen.record_path=$(pwd)/data/records/tf/ mgen.pass_rate=0.05 model.type=tensorflow backend.type=xla fuzz.time=5m exp.parallel=64 mgen.noise=0.8 exp.targets=/DeepConstr/data/tf_dc_acetest.json exp.baselines=['acetest']
``` -->


##### Summarize the results

Specify the folder name that you used in a previous experiment. Use the -o option to name the output file. The final experiment results will be saved in the path that is specified through -o.

For example, to specify a folder named pt_gen and save the results to pt_gen.csv, use the following command:
```bash
python experiments/summarize_merged_cov.py -f exp/torch -o torch_exp -p deepconstr -k torch
# Result will be saved at /DeepConstr/results/torch_exp.csv
python experiments/summarize_merged_cov.py -f exp/tf -o tf_exp -p deepconstr -k tf
# Result will be saved at /DeepConstr/results/tf_exp.csv
```

##### When encounters with unnormal values 

Occasionally, you may encounter abnormal coverage values, such as 0. In such cases, please refer to the list of abnormal values saved at `$(pwd)/results/unnormal_val*`. To address these issues, re-run the experiment with the following adjustments to your arguments: `mode=fix exp.targets=$(pwd)/results/unnormal_val*`.

```bash
PYTHONPATH=/DeepConstr/:/DeepConstr/nnsmith/:/DeepConstr/deepconstr/:$PYTHONPATH \
python experiments/evaluate_apis.py \
exp.save_dir=pt_gen mgen.record_path=$(pwd)/data/records/torch/ mgen.pass_rate=0 model.type=torch backend.type=torchjit fuzz.time=15m exp.parallel=1 mgen.noise=0.8 exp.targets=/DeepConstr/results/unnormal_val_deepconstr_torch.json
```

### Constraint Assessment (RQ2) 

You can review the overall scores of constraints by executing the following script:
You can look into the overall scores of constraints by running below scripts.
```bash
python experiments/eval_constr.py
```
This script will automatically gather the constraints from the  default locations(`/DeepConstr/data/records/`). The resulting plots will be saved at`/DeepConstr/results/5_dist_tf.png` for TensorFlow and `/DeepConstr/results/5_dist_torch.png` for PyTorch.

### Bug finding evidence (RQ3)

To ensure the anonymity of this artifact, we are currently withholding the list of identified bugs.