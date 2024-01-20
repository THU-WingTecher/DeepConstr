source ./env_cov.sh
set -x

cd "$(dirname "$0")"

echo "APINAME: $1";
APINAME="$1"
NEURI_FOLDER="$2"
OUR_FOLDER="$3"

echo "Start testing ${APINAME}..."
echo "evaluating neuri..."
# python experiments/evaluate_models.py  --root $(pwd)/exp_data/${NEURI_FOLDER}/${APINAME}.models         --model_type torch --backend_type torchcomp --parallel $(nproc)
python neuri/materialize/torch/program.py --root exp_data/${NEURI_FOLDER}/${APINAME}.models
python3 experiments/evaluate_richerm_models.py --root $(pwd)/exp_data/${NEURI_FOLDER}/${APINAME}.models --parallel $(nproc) --package torch
python experiments/process_profraws.py --root $(pwd)/${NEURI_FOLDER}/${APINAME}.models    \
                                       --llvm-config-path $(which llvm-config-14) \
                                       --instrumented-libs "/artifact/build/pytorch-cov/build/lib/libtorch_cpu.so" "/artifact/build/pytorch-cov/build/lib/libtorch.so" \
                                        --batch-size 1000 --parallel $(nproc)

echo "evaluating us..."
python3 experiments/evaluate_richerm_models.py --root $(pwd)/exp_data/${OUR_FOLDER}/${APINAME}.models --parallel $(nproc) --package torch
python experiments/process_profraws.py --root $(pwd)/exp_data/${OUR_FOLDER}/${APINAME}.models    \
                                       --llvm-config-path $(which llvm-config-14) \
                                       --instrumented-libs "/artifact/build/pytorch-cov/build/lib/libtorch_cpu.so" "/artifact/build/pytorch-cov/build/lib/libtorch.so" \
                                        --batch-size 1000 --parallel $(nproc)

python experiments/viz_merged_cov.py --folders             \
        $(pwd)/exp_data/${NEURI_FOLDER}/${APINAME}.models/coverage        \
        $(pwd)/exp_data/${OUR_FOLDER}/${APINAME}.models/coverage \
    --tags '\textsc{neuri}' '\textsc{us}'\
    --name ${APINAME}_final_test
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
