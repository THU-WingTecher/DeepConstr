cd "$(dirname "$0")"

echo "NSIZE: $1";
echo "MODEL: $2"; # torch / tensorflow
echo "METHOD: $3"; # torch / tensorflow
echo "TESTPOOL: $4"; # torch.acos,Slice, ... (Tuple[str, ...])
echo "PARALLEL: $5";

NSIZE="$1"
MODEL="$2"
METHOD="$3"
TESTPOOL="$4"
PARALLEL="$5"
TESTPOOL_MODIFIED="${TESTPOOL//,/-}"

if [ $MODEL = "tensorflow" ]; then
    BACKEND="xla"
elif [ $MODEL = "torch" ]; then
    BACKEND="torchjit"
else
    echo "MODEL must be tensorflow or torch"
    exit 1
fi


## NNSMITTH ##
echo "evaluating nnsmith..."
export PYTHONPATH=$(pwd):$(pwd)/neuri
python experiments/evaluate_models.py  --root $(pwd)/gen/${MODEL}-${METHOD}-n${NSIZE}-${TESTPOOL_MODIFIED}.models          --model_type ${MODEL} --backend_type ${BACKEND} --parallel ${PARALLEL}
python experiments/process_profraws.py --root $(pwd)/gen/${MODEL}-${METHOD}-n${NSIZE}-${TESTPOOL_MODIFIED}.models    \
                                       --llvm-config-path $(which llvm-config-14) \
                                       --instrumented-libs "$(pwd)/build/pytorch-cov/build/lib/libtorch_cpu.so" "$(pwd)/build/pytorch-cov/build/lib/libtorch.so" \
                                        --batch-size 1000 --parallel ${PARALLEL}

python experiments/viz_merged_cov.py --name ${TESTPOOL_MODIFIED} --folders                  \
        $(pwd)/gen/${MODEL}-symbolic-cinit-n${NSIZE}-${TESTPOOL_MODIFIED}.models/coverage        \
        $(pwd)/gen/${MODEL}-neuri-n${NSIZE}-${TESTPOOL_MODIFIED}.models/coverage \
        $(pwd)/gen/${MODEL}-deepconstr-n${NSIZE}-${TESTPOOL_MODIFIED}.models/coverage        \
    --tags '\textsc{NNSmith}' '\textsc{NeuRI}$^r$' '\textsc{ConstrInf}$^i$'

# echo "evaluating finished..."
# python3 experiments/evaluate_tf_models.py --root $(pwd)/gen/tensorflow-neuri-n2.models --parallel $(nproc)
# python3 experiments/process_lcov.py --root $(pwd)/gen/tensorflow-neuri-n2.models --parallel $(nproc)

# python3 experiments/evaluate_richerm_models.py --root $(pwd)/gen/complete_shape_constraints --parallel $(nproc)
# python3 experiments/process_lcov.py --root $(pwd)/gen/complete_shape_constraints --parallel $(nproc)

# python3 experiments/evaluate_richerm_models.py --root $(pwd)/gen/simple_shape_constraints --parallel $(nproc)
# python3 experiments/process_lcov.py --root $(pwd)/gen/simple_shape_constraints --parallel $(nproc)

# python3 experiments/evaluate_richerm_models.py --root $(pwd)/gen/sim_len_shape_constraints --parallel $(nproc)
# python3 experiments/process_lcov.py --root $(pwd)/gen/sim_len_shape_constraints --parallel $(nproc)
# python experiments/evaluate_models.py  --root $(pwd)/gen/torch-neuri-n5.models          --model_type torch --backend_type torchjit --parallel $(nproc)

# python experiments/viz_merged_cov.py --folders             \
#         $(pwd)/gen/tensorflow-neuri-n1.models/coverage \
#         $(pwd)/gen/torch-neuri-r-n5.models/coverage        \
#         $(pwd)/gen/torch-neuri-i-n5.models/coverage        \
#         $(pwd)/gen/torch-neuri-n5.models/coverage          \
#     --tags '\textsc{NNSmith}' '\textsc{NeuRI}$^r$' '\textsc{NeuRI}$^i$'  '\textsc{NeuRI}'
