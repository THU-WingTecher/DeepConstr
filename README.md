# Introduction

This is the implementation of DeepConstr.

# Bug finding evidence (RQ3)

To fully support anonymous of this artifact, we temporarily don't present bug list.

### Start fuzzing

> [!NOTE]
>
> **Command usage of**: `./fuzz.sh NSIZE METHOD MODEL BACKEND TIME`
>
> **Arguments**:
> - `NSIZE`: the number of operators in each generated graph.
> - `METHOD`: in `["deepconstr", "neuri", "symbolic-cinit"]`.
> - `MODEL`: in `["tensorflow", "torch"]`.
> - `BACKEND`: in `["xla", "torchjit"]`.
> - `TIME`: fuzzing time in formats like `4h`, `1m`, `30s`.
> - `POOL`(Optional): Fuzz specific api. if not specified, conducts the fuzzing on the whole apis prepared.
>
> **Outputs**:
> - `$(pwd)/outputs/${MODEL}-${METHOD}-n${NSIZE}-{POOL}.models`: the generated test-cases (models)

#### For PyTorch

>
```bash
source ./env_std.sh
./fuzz.sh 5 deepconstr     torch torchcomp 4h # test all apis that deepconstr supports
./fuzz.sh 5 deepconstr     torch torchcomp 4h torch.abs,torch.add # test torch.abs, torch.add
./fuzz.sh 5 deepconstr     torch torchcomp 4h torch.abs,torch.add,torch.acos # test torch.abs, torch.add, torch.acos
```

#### For TensorFlow

>
```bash
source ./env_std.sh
./fuzz.sh 5 deepconstr     tensorflow xla 4h # test all apis that deepconstr supports
./fuzz.sh 5 deepconstr     tensorflow xla 4h tf.abs,tf.add # test tf.abs, tf.add
./fuzz.sh 5 deepconstr     tensorflow xla 4h tf.abs,tf.add,tf.acos # test tf.abs, tf.add, tf.acos
```

# Extract Constraints

### Settings


### Select test operator
We strongly recommend to use virtual environment using anaconda.
For detail, you can refer [!this]()
1. install needed libraries. 
```bash 
pip install -r requirements.txt
```
2. setting needed configuration
generate .env file on the workspace(~/artifact/.env) 
and set your values.

    (1) Set openai key, ```OPENAI_API_KEY ='sk-********'```

    (2) Set proxy(Optional), ```MYPROXY ='166.111.***.***:****'```

    (3) Testing, after setting, you can check your setting by running ```python 
    python tests/proxy.py```

3. start extraction(WIP)
> [!NOTE]
>
> **Command usage of**: `./fuzz.sh tran.target METHOD MODEL BACKEND TIME`
>
> **Important Arguments**:
> - `tran.target`: api name or path to extract. It can be either single api name(e.g, "torch.add") or a list contaning multiple api names(e.g, "['torch.add','torch.abs']"), or a json file path containing the list(e.g, "['torch.add','torch.abs']").
> - `train.retrain`: boolean value that deteremine whether re-conduct constraint extraction, if it is set false, the tool only collect api that doesn't extracted. if is set true, the tool collect all apis except for pass rate exceeds pre-set target pass rate(i.e, `train.pass_rate`) 
> - `train.pass_rate`: the target pass rate to filter out apis that have pass rate higher than the target pass rate.
> - `train.parallel`: the number of parallel processes to validate the constraints.
> - `train.record_path`: the path to record the extracted constraints.
> - `hydra.verbose`: the logging level of hydra of certain mlogger("smt", "train", "convert", "constr", "llm").
> - `train.num_eval`: the number of evaluation to validate the constraints(default: 500).
> - `model.type`: in `["tensorflow", "torch"]`.
> - `backend.type`: in `["xla", "torchjit"]`.
> **Other Arguments**:
> refer to the values under train at `/artifact/nnsmith/config/main.yaml` for more information.
> **Outputs**:
> - `$(pwd)/${train.record_path}/torch` if `model.type` is `torch`
> - `$(pwd)/${train.record_path}/tf` if `model.type` is `tensorflow`

#### Quick Start (Not tested yet):

##### for PyTorch 
```bash
PYTHONPATH=/artifact/:/artifact/nnsmith/:/artifact/deepconstr/:$PYTHONPATH python deepconstr/train/run.py train.record_path=test/records/torch backend.type=torchcomp model.type=torch hydra.verbose=train train.parallel=1 train.eval_asset=100 train.pass_rate=95 hydra.verbose=['train'] train.retrain=true train.target='["torch.add","torch.abs"]'
```

##### for TensorFlow 
```bash
PYTHONPATH=/artifact/:/artifact/nnsmith/:/artifact/deepconstr/:$PYTHONPATH python deepconstr/train/run.py train.record_path=test/records/tf backend.type=xla model.type=tensorflow hydra.verbose=train train.parallel=1 train.eval_asset=100 train.pass_rate=95 hydra.verbose=['train'] train.retrain=true train.target='["tf.add","tf.abs"]'
```

# Reproduct Experiments