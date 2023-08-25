#!/bin/bash

VENV_NAME="cov"

[ -d "$(dirname "$0")/${VENV_NAME}" ] || python3 -m venv "$(dirname "$0")/${VENV_NAME}"
activate () {
    # shellcheck source=/dev/null
    . "$(dirname "$0")/${VENV_NAME}/bin/activate"
    PYTHONPATH="$(dirname "$0")/neuri:$PYTHONPATH"
    export PYTHONPATH
}

activate