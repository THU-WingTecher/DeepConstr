import os

ONNX_EXTERNAL_DATA_DIR_SUFFIX = "-mlist"
NNSMITH_ORT_INTRA_OP_THREAD = int(os.getenv("NNSMITH_ORT_INTRA_OP_THREAD", 1))
NNSMITH_BUG_PATTERN_TOKEN = "${PATTERN}"


def onnx2external_data_dir(onnx_file):
    return onnx_file + ONNX_EXTERNAL_DATA_DIR_SUFFIX


# can be changing variables (drag and cntrl + shift + l)
# save directory : torchcomp-fuzz-0221_record
# save directory : torchcomp-fuzz-0221
""" API TEST
python experiments/evaluate_apis.py \
exp.save_dir=pt_gen_constr mgen.record_path=$(pwd)/data/records/torch/ mgen.pass_rate=0.05 model.type=torch backend.type=torchjit fuzz.time=15m exp.parallel=32 mgen.noise=0.8 exp.baselines=['deepconstr']
"""
""" API TEST
python experiments/evaluate_apis.py exp.save_dir=tf_gen_constr mgen.record_path=/artifact/data/records/tf/ mgen.pass_rate=0.05 model.type=tensorflow backend.type=xla fuzz.time=15m exp.parallel=32 mgen.noise=0.8
"""

""" FUZZING 
python neuri/cli/fuzz.py fuzz.time=24h mgen.record_path=data/records/torch fuzz.root=bugs/torchcomp-deepconstr-0414 filter.type=\[\'nan\',\'dup\',\'inf\'\] backend.type=torchcomp model.type=torch fuzz.save_test=bugs/torchcomp-deepconstr-0414_record debug.viz=true hydra.verbose=fuzz fuzz.resume=false mgen.method=deepconstr mgen.max_nodes=3 mgen.pass_rate=10
./fuzz.sh 5 deepconstr          torch torchcomp 4h
"""
"""FUZZING NEURI
python neuri/cli/fuzz.py fuzz.time=24h mgen.record_path=data/torch_records fuzz.root=bugs/torchcomp-neuri-fuzz-0221 filter.type=\[\'nan\',\'dup\',\'inf\'\] backend.type=torchcomp model.type=torch fuzz.save_test=bugs/torchcomp-neuri-fuzz-0221_record debug.viz=true hydra.verbose=fuzz fuzz.resume=false mgen.method=neuri mgen.max_nodes=3 

"""
""" to_reproduce_code 
python neuri/materialize/torch/program.py /artifact/exp/ torchcomp
"""

"""train
PYTHONPATH=/artifact/neuri/:/artifact/:$PYTHONPATH python neuri/cli/train.py train.record_path=data/records/torch backend.type=torchcomp model.type=torch hydra.verbose=train train.parallel=10 train.eval_asset=300

tf

PYTHONPATH=/artifact/neuri/:/artifact/:$PYTHONPATH python neuri/cli/train.py train.record_path=data/records/tf backend.type=xla model.type=tensorflow hydra.verbose=train train.resume=false train.parallel=1 train.eval_asset=150 temp.start=0 temp.end=50 train.target=/artifact/data/tf_overall_apis.json
"""

"""process prfraw
python experiments/process_profraws.py --root $(pwd)/experiments/torch_default  \
                                       --llvm-config-path $(which llvm-config-14) \
                                       --instrumented-libs "$(pwd)/build/pytorch-cov/build/lib/libtorch_cpu.so" "$(pwd)/build/pytorch-cov/build/lib/libtorch.so" \
                                        --batch-size 1 --parallel $(nproc)
"""