import os

ONNX_EXTERNAL_DATA_DIR_SUFFIX = "-mlist"
NNSMITH_ORT_INTRA_OP_THREAD = int(os.getenv("NNSMITH_ORT_INTRA_OP_THREAD", 1))
NNSMITH_BUG_PATTERN_TOKEN = "${PATTERN}"


def onnx2external_data_dir(onnx_file):
    return onnx_file + ONNX_EXTERNAL_DATA_DIR_SUFFIX


# can be changing variables (drag and cntrl + shift + l)
# save directory : torchcomp-fuzz-0220_2_record
# save directory : torchcomp-fuzz-0220_2
""" API TEST
python experiments/evaluate_apis.py  mgen.record_path=$(pwd)/data/constraints/torch/ mgen.pass_rate=0.05 mgen.max_nodes=5 model.type=torch backend.type=torchjit fuzz.time=20m exp.parallel=6 mgen.noise=0.4
"""

""" FUZZING 
python neuri/cli/fuzz.py fuzz.time=1h mgen.record_path=data/constraints/torch fuzz.root=bugs/torchcomp-fuzz-0220_2 filter.type=\[\'nan\',\'dup\',\'inf\'\] backend.type=torchcomp model.type=torch fuzz.save_test=bugs/torchcomp-fuzz-0220_2_record debug.viz=true hydra.verbose=fuzz fuzz.resume=false mgen.method=constrinf mgen.max_nodes=3

./fuzz.sh 5 constrinf          torch torchcomp 4h
"""

""" to_reproduce_code 
python neuri/materialize/torch/program.py /artifact/gen/torch-constrinf-n3 torchcomp
"""