import os

ONNX_EXTERNAL_DATA_DIR_SUFFIX = "-mlist"
NNSMITH_ORT_INTRA_OP_THREAD = int(os.getenv("NNSMITH_ORT_INTRA_OP_THREAD", 1))
NNSMITH_BUG_PATTERN_TOKEN = "${PATTERN}"


def onnx2external_data_dir(onnx_file):
    return onnx_file + ONNX_EXTERNAL_DATA_DIR_SUFFIX


""" API TEST
python experiments/evaluate_apis.py  mgen.record_path=$(pwd)/data/constraints/torch/ mgen.pass_rate=0.05 mgen.max_nodes=5 model.type=torch backend.type=torchjit fuzz.time=20m exp.parallel=6 mgen.noise=0.4
"""

""" FUZZING 
python neuri/cli/fuzz.py fuzz.time=120m mgen.record_path=data/constraints/torch fuzz.root=gen/torch-constrinf-n3 filter.type=\[\'nan\',\'dup\',\'inf\'\] backend.type=torchjit model.type=torch fuzz.save_test=gen/torch-constrinf-n3.models debug.viz=true hydra.verbose=fuzz fuzz.resume=false mgen.method=constrinf mgen.max_nodes=3

./fuzz.sh 5 constrinf          torch torchcomp 4h
"""
