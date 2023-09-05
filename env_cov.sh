#!/bin/bash

VENV_NAME="cov"

activate () {
    # shellcheck source=/dev/null
    conda activate "$VENV_NAME"
}

activate