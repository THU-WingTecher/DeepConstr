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
./fuzz.sh 5 deepconstr     tensorflow xla 4h torch.abs,torch.add # test torch.abs, torch.add
./fuzz.sh 5 deepconstr     tensorflow xla 4h torch.abs,torch.add,torch.acos # test torch.abs, torch.add, torch.acos
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

- PyTorch 
    
```bash 
python train.py
```
