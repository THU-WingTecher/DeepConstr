#!/bin/bash
set -e
set -x

cd "$(dirname "$0")" || exit 1

pip install -r ../requirements.txt

# Build Torch
bash ./build_torch_cov_yourself.sh
# Building TF is just too complicated so let's use a pre-built wheel
pip install gcovr
conda install -c conda-forge libgcc-ng==12.2.0 -y
conda install -c conda-forge gxx==12.2.0 -y
pip install ./tensorflow-2.12.0-cp38-cp38-linux_x86_64.whl
