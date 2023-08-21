# NeuRI: Diversifying DNN Generation via Inductive Rule Inference

<p align="center">
    <a href="https://arxiv.org/abs/2302.02261"><img src="https://img.shields.io/badge/arXiv-2302.02261-b31b1b.svg">
    <a href="https://github.com/ise-uiuc/neuri-artifact/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
</p>

Welcome to the artifact repository of the NeuRI paper which is accepted by ESEC/FSE 2023.


## Bug finding evidence (RQ3)

See [lists of bug reports](docs/rq3-bug-reports.md).


## Get ready for running experiments!

> [!IMPORTANT]
> 
> **General test-bed requirements**
> - **OS**: A Linux System with Docker Support;
> - **Hardware**: X86/X64 CPU; 16GB RAM; 512GB Storage; Good Network to GitHub and Docker Hub;

### S1: Docker installation

> **Note**
> 
> Before you start, please make sure you have [Docker](https://docs.docker.com/engine/install/) installed.
>
> To check the installation:
> ```bash
> docker --version # Test docker availability
> # Docker version 20.10.12, build e91ed5707e
> ```

```bash
docker pull ganler/neuri-fse23-ae:latest

# Run docker image
docker run -it --name ${USER}-neuri ganler/neuri-fse23-ae
# Now, you will "get into" the image like entering a virtual machine.
# By using this command, you will "get into" the image like entering a virtual machine.
# The session will be kept under the name "${USER}-neuri"

# Inside the image;
cd /artifact
git remote set-url origin https://github.com/ise-uiuc/neuri-artifact.git
git pull origin main
```

### S2: Kick the tires

In this section, we will quickly try to run some scripts in docker to see if everything works.

```bash
source env_std.sh # Use a virtual environment
bash kick_tire.sh # 40 seconds
```

Congratulations if you can run it without errors!

## Evaluating Coverage (RQ1)

### S1: Start fuzzing and cache test-cases

We will use `./fuzz.sh` to generate the test-cases.

> [!NOTE]
> 
> **Command usage of**: `./fuzz.sh NSIZE METHOD MODEL BACKEND TIME`
> 
> **Arguments**:
> - `NSIZE`: the number of operators in each generated graph.
> - `METHOD`: in `["neuri", "neuri-i", "neuri-r", "symbolic-cinit"]`.
> - `MODEL`: in `["tensorflow", "torch"]`.
> - `BACKEND`: in `["xla", "torchjit"]`.
> - `TIME`: fuzzing time in formats like `4h`, `1m`, `30s`.
>
> **Outputs**:
> - `$(pwd)/gen/${MODEL}-${DATE}-${METHOD}-n${NSIZE}.models`: the generated test-cases (models)

### S2: Collect coverage

### S3: Draw figures


## Evaluating Rule Inference (RQ2)

> **Warning** 
> **Experiment dependency**:
> You need to first finish the last section (RQ1) to continue this section.

> **Note** 
> Please run command `source ./env_std.sh` in the root directory to install necessary libraries and configure environment variables.

```bash
# in neuri/autoinf/inference
python3 tree.py
python3 augmentation.py
python3 shape_solve.py
python3 predicate_solve.py
python3 nnsmith_reuse.py
python3 rule_validity.py
python3 table3.py
```

```bash
# in neuri/autoinf/inference
python3 rosette_solve.py
python3 table4.py
```


## Learning More

- Pre-print: [![](https://img.shields.io/badge/arXiv-2302.02261-b31b1b.svg)](https://arxiv.org/abs/2302.02261)
- NeuRI is being merged into [NNSmith](https://github.com/ise-uiuc/nnsmith)
