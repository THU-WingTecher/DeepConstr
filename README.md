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
3. start extraction 

- PyTorch 
    
```bash 
python train.py
```
1. Type information extraction 
- 

- [ ] sliced list logic check,
- [ ] isinstance, type implementation
- [ ] dtype rule prioritizing
- [ ] type rule prioritizing logic while rule solving 
- [ ] *args operation with yaml module 
- [ ] transfer train codes and make it multicoreable
- [ ] DB revision(type, dtype, ...)
- [ ] retrain apis and reevaluate 


# Towards More Sound and Complete Constraints for Deep Learning Framework Testing

## Commands

Specific API testing
```bash
./fuzz.sh 5 neuri          torch torchjit 30m Slice,torch.acos
```

```bash
source ./env_std.sh
./fuzz.sh 5 neuri          torch torchjit 4h
./fuzz.sh 5 neuri-i        torch torchjit 4h
./fuzz.sh 5 neuri-r        torch torchjit 4h
./fuzz.sh 5 symbolic-cinit torch torchjit 4h # NNSmith
./fuzz.sh 1  neuri         torch torchjit 4h
./fuzz.sh 9  neuri         torch torchjit 4h
./fuzz.sh 13 neuri         torch torchjit 4h
```

#### For TensorFlow

```bash
source ./env_std.sh
./fuzz.sh 5 neuri          tensorflow xla 4h
./fuzz.sh 5 neuri-i        tensorflow xla 4h
./fuzz.sh 5 neuri-r        tensorflow xla 4h
./fuzz.sh 5 symbolic-cinit tensorflow xla 4h # NNSmith
```

### S2: Collect coverage

#### For PyTorch

```bash
source ./env_cov.sh
python experiments/evaluate_models.py  --root $(pwd)/gen/torch-neuri-n5.models          --model_type torch --backend_type torchjit --parallel $(nproc)
python experiments/evaluate_models.py  --root $(pwd)/gen/torch-neuri-i-n5.models        --model_type torch --backend_type torchjit --parallel $(nproc)
python experiments/evaluate_models.py  --root $(pwd)/gen/torch-neuri-r-n5.models        --model_type torch --backend_type torchjit --parallel $(nproc)
python experiments/evaluate_models.py  --root $(pwd)/gen/torch-symbolic-cinit-n5.models --model_type torch --backend_type torchjit --parallel $(nproc)
python experiments/evaluate_models.py  --root $(pwd)/gen/torch-neuri-n1.models          --model_type torch --backend_type torchjit --parallel $(nproc)
python experiments/evaluate_models.py  --root $(pwd)/gen/torch-neuri-n9.models          --model_type torch --backend_type torchjit --parallel $(nproc)
python experiments/evaluate_models.py  --root $(pwd)/gen/torch-neuri-n13.models         --model_type torch --backend_type torchjit --parallel $(nproc)
# Compute coverage
python experiments/process_profraws.py --root $(pwd)/gen/torch-neuri-n5.models    \
                                       --llvm-config-path $(which llvm-config-14) \
                                       --instrumented-libs "$(pwd)/build/pytorch-cov/build/lib/libtorch_cpu.so" "$(pwd)/build/pytorch-cov/build/lib/libtorch.so" \
                                        --batch-size 1000 --parallel $(nproc)
python experiments/process_profraws.py --root $(pwd)/gen/torch-neuri-i-n5.models  \
                                       --llvm-config-path $(which llvm-config-14) \
                                       --instrumented-libs "$(pwd)/build/pytorch-cov/build/lib/libtorch_cpu.so" "$(pwd)/build/pytorch-cov/build/lib/libtorch.so" \
                                        --batch-size 1000 --parallel $(nproc)
python experiments/process_profraws.py --root $(pwd)/gen/torch-neuri-r-n5.models  \
                                       --llvm-config-path $(which llvm-config-14) \
                                       --instrumented-libs "$(pwd)/build/pytorch-cov/build/lib/libtorch_cpu.so" "$(pwd)/build/pytorch-cov/build/lib/libtorch.so" \
                                        --batch-size 1000 --parallel $(nproc)
python experiments/process_profraws.py --root $(pwd)/gen/torch-symbolic-cinit-n5.models \
                                       --llvm-config-path $(which llvm-config-14)       \
                                       --instrumented-libs "$(pwd)/build/pytorch-cov/build/lib/libtorch_cpu.so" "$(pwd)/build/pytorch-cov/build/lib/libtorch.so" \
                                        --batch-size 1000 --parallel $(nproc)
python experiments/process_profraws.py --root $(pwd)/gen/torch-neuri-n1.models    \
                                       --llvm-config-path $(which llvm-config-14) \
                                       --instrumented-libs "$(pwd)/build/pytorch-cov/build/lib/libtorch_cpu.so" "$(pwd)/build/pytorch-cov/build/lib/libtorch.so" \
                                        --batch-size 1000 --parallel $(nproc)
python experiments/process_profraws.py --root $(pwd)/gen/torch-neuri-n9.models    \
                                       --llvm-config-path $(which llvm-config-14) \
                                       --instrumented-libs "$(pwd)/build/pytorch-cov/build/lib/libtorch_cpu.so" "$(pwd)/build/pytorch-cov/build/lib/libtorch.so" \
                                        --batch-size 1000 --parallel $(nproc)
python experiments/process_profraws.py --root $(pwd)/gen/torch-neuri-n13.models   \
                                       --llvm-config-path $(which llvm-config-14) \
                                       --instrumented-libs "$(pwd)/build/pytorch-cov/build/lib/libtorch_cpu.so" "$(pwd)/build/pytorch-cov/build/lib/libtorch.so" \
                                        --batch-size 1000 --parallel $(nproc)
```

#### For TensorFlow

```bash
source ./env_cov.sh
python3 experiments/evaluate_tf_models.py --root $(pwd)/gen/tensorflow-neuri-n5.models --parallel $(nproc)
python3 experiments/evaluate_tf_models.py --root $(pwd)/gen/tensorflow-neuri-i-n5.models --parallel $(nproc)
python3 experiments/evaluate_tf_models.py --root $(pwd)/gen/tensorflow-neuri-r-n5.models --parallel $(nproc)
python3 experiments/evaluate_tf_models.py --root $(pwd)/gen/tensorflow-symbolic-cinit-n5.models --parallel $(nproc)
# Compute coverage
python3 experiments/process_lcov.py --root $(pwd)/gen/tensorflow-neuri-n5.models --parallel $(nproc)
python3 experiments/process_lcov.py --root $(pwd)/gen/tensorflow-neuri-i-n5.models --parallel $(nproc)
python3 experiments/process_lcov.py --root $(pwd)/gen/tensorflow-neuri-r-n5.models --parallel $(nproc)
python3 experiments/process_lcov.py --root $(pwd)/gen/tensorflow-symbolic-cinit-n5.models --parallel $(nproc)
```


## Bug finding evidence (RQ3)

See [links to real-world bug reports](https://demo.hedgedoc.org/uVYLcZVSSseXknO6lQY0Mg).


### S3: Checkout the results

#### Table 1 & 2

```bash
# PyTorch
python experiments/genstat.py --root $(pwd)/gen/torch-neuri-n5
python experiments/genstat.py --root $(pwd)/gen/torch-neuri-i-n5
python experiments/genstat.py --root $(pwd)/gen/torch-neuri-r-n5
python experiments/genstat.py --root $(pwd)/gen/torch-symbolic-cinit-n5

# TensorFlow
python experiments/genstat.py --root $(pwd)/gen/tensorflow-neuri-n5
python experiments/genstat.py --root $(pwd)/gen/tensorflow-neuri-i-n5
python experiments/genstat.py --root $(pwd)/gen/tensorflow-neuri-r-n5
python experiments/genstat.py --root $(pwd)/gen/tensorflow-symbolic-cinit-n5
```

Check the terminal output for the results.

#### Figure 6 (a)

```bash
python experiments/viz_merged_cov.py --folders             \
        $(pwd)/gen/torch-symbolic-cinit-n5.models/coverage \
        $(pwd)/gen/torch-neuri-r-n5.models/coverage        \
        $(pwd)/gen/torch-neuri-i-n5.models/coverage        \
        $(pwd)/gen/torch-neuri-n5.models/coverage          \
    --tags '\textsc{NNSmith}' '\textsc{NeuRI}$^r$' '\textsc{NeuRI}$^i$'  '\textsc{NeuRI}'
```

Check images under `./results/branch_cov-time.png` for the results.

#### Figure 6 (b)

```bash
python experiments/viz_merged_cov.py --folders                  \
        $(pwd)/gen/tensorflow-symbolic-cinit-n5.models/coverage \
        $(pwd)/gen/tensorflow-neuri-r-n5.models/coverage        \
        $(pwd)/gen/tensorflow-neuri-i-n5.models/coverage        \
        $(pwd)/gen/tensorflow-neuri-n5.models/coverage          \
    --tags '\textsc{NNSmith}' '\textsc{NeuRI}$^r$' '\textsc{NeuRI}$^i$'  '\textsc{NeuRI}'
```

Check images under `./results/branch_cov-time.png` for the results.

#### Figure 6 (c)

```bash
python experiments/viz_merged_cov.py --folders             \
        $(pwd)/gen/torch-neuri-n1.models/coverage          \
        $(pwd)/gen/torch-neuri-n5.models/coverage          \
        $(pwd)/gen/torch-neuri-n9.models/coverage          \
        $(pwd)/gen/torch-neuri-n13.models/coverage         \
    --tags '\#Node 1' '\#Node 5' '\#Node 9' '\#Node 13'
```

Check images under `./results/branch_cov-time.png` for the results.

## Evaluating Rule Inference (RQ2)

### S1: Tree & rule generation

```bash
source ./env_std.sh
python3 neuri/autoinf/inference/tree.py
python3 neuri/autoinf/inference/augmentation.py
python3 neuri/autoinf/inference/shape_solve.py
python3 neuri/autoinf/inference/predicate_solve.py
python3 neuri/autoinf/inference/nnsmith_reuse.py
python3 neuri/autoinf/inference/rule_validity.py
python3 neuri/autoinf/inference/rosette_solve.py
```

Rules will be stored in `gen/`.

### S2: 1-Node test-cases generation

```bash
RULE_DIR=$(pwd)/gen ./fuzz.sh 1 neuri-i   torch      torchjit 4h
RULE_DIR=$(pwd)/gen ./fuzz.sh 1 neuri-i   tensorflow xla      4h
```
