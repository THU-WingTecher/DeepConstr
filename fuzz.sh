#!/bin/bash

set -x

cd "$(dirname "$0")"

echo "NSIZE: $1";
echo "METHOD: $2";
echo "MODEL: $3";
echo "BACKEND: $4";
echo "TIME: $5";

NSIZE="$1"
METHOD="$2"
MODEL="$3"
BACKEND="$4"
TIME="$5"
TESTPOOL="$6"
TESTPOOL_MODIFIED="${TESTPOOL//,/-}"
# assert non-empty above
if [ -z "$NSIZE" ] || [ -z "$METHOD" ] || [ -z "$MODEL" ] || [ -z "$TIME" ]; then
    echo "Usage: $0 NSIZE METHOD MODEL BACKEND TIME"
    exit 1
fi

# set environment variables CUDA_VISIBLE_DEVICES=""
export CUDA_VISIBLE_DEVICES=""
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

if [ $MODEL = "tensorflow" ]; then
    if [ $METHOD = "constrinf" ]; then
        RECORD="$(pwd)/data/constraints/tf"
    else 
        RECORD="$(pwd)/data/tf_records"
    fi
elif [ $MODEL = "torch" ]; then
    if [ $METHOD = "constrinf" ]; then
        RECORD="$(pwd)/data/constraints/torch"
    else 
        RECORD="$(pwd)/data/torch_records"
    fi
else
    echo "MODEL must be tensorflow or torch"
    exit 1
fi

mkdir $(pwd)/gen -p

# attempt at most 2 times
for i in {1..2}
do
    echo "Attempt $i"
    PYTHONPATH=$(pwd):$(pwd)/neuri python neuri/cli/fuzz.py fuzz.time=${TIME} \
                                            mgen.record_path=${RECORD} \
                                            fuzz.root=$(pwd)/gen/${MODEL}-${METHOD}-n${NSIZE}-${TESTPOOL_MODIFIED} \
                                            fuzz.save_test=$(pwd)/gen/${MODEL}-${METHOD}-n${NSIZE}-${TESTPOOL_MODIFIED}.models \
                                            model.type=${MODEL} backend.type=${BACKEND} filter.type="[nan,dup,inf]" \
                                            debug.viz=true hydra.verbose=fuzz fuzz.resume=true \
                                            mgen.method=${METHOD} mgen.max_nodes=${NSIZE} mgen.test_pool="[${TESTPOOL}]" 
    if [ $? -eq 0 ]; then
        echo "Fuzzing succeeded after $i attempts."
        break
    else
        echo "Fuzzing crashed with exit code $?.  Respawning.." >&2
        sleep 0.5
    fi
done

echo "WAS RUNNING:"
PYTHONPATH=$(pwd):$(pwd)/neuri python neuri/cli/fuzz.py fuzz.time=${TIME} \
                                        mgen.record_path=${RECORD} \
                                        fuzz.root=$(pwd)/gen/${MODEL}-${METHOD}-n${NSIZE}-${TESTPOOL_MODIFIED} \
                                        fuzz.save_test=$(pwd)/gen/${MODEL}-${METHOD}-n${NSIZE}-${TESTPOOL_MODIFIED}.models \
                                        model.type=${MODEL} backend.type=${BACKEND} filter.type="[nan,dup,inf]" \
                                        debug.viz=true hydra.verbose=fuzz fuzz.resume=true \
                                        mgen.method=${METHOD} mgen.max_nodes=${NSIZE} mgen.test_pool="[${TESTPOOL}]" 
