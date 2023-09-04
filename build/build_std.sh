#!/bin/bash
set -e
set -x

cd "$(dirname "$0")" || exit 1

pip install -r ../requirements.txt
pip install tensorflow==2.12.0
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu
