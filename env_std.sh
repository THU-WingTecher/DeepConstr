#!/bin/bash

VENV_NAME="std"

activate () {
    # shellcheck source=/dev/null
    conda activate "$VENV_NAME"
}

activate