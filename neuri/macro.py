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
exp.save_dir=exp mgen.record_path=$(pwd)/data/records/torch/ mgen.pass_rate=0.05 mgen.max_nodes=5 model.type=torch backend.type=torchjit fuzz.time=10m exp.parallel=1 mgen.noise=0.4
"""

""" FUZZING 
python neuri/cli/fuzz.py fuzz.time=24h mgen.record_path=data/constraints/torch fuzz.root=bugs/torchcomp-constrinf-fuzz-0221 filter.type=\[\'nan\',\'dup\',\'inf\'\] backend.type=torchcomp model.type=torch fuzz.save_test=bugs/torchcomp-constrinf-fuzz-0221_record debug.viz=true hydra.verbose=fuzz fuzz.resume=false mgen.method=constrinf mgen.max_nodes=3 mgen.pass_rate=0.6

./fuzz.sh 5 constrinf          torch torchcomp 4h
"""
"""FUZZING NEURI
python neuri/cli/fuzz.py fuzz.time=24h mgen.record_path=data/torch_records fuzz.root=bugs/torchcomp-neuri-fuzz-0221 filter.type=\[\'nan\',\'dup\',\'inf\'\] backend.type=torchcomp model.type=torch fuzz.save_test=bugs/torchcomp-neuri-fuzz-0221_record debug.viz=true hydra.verbose=fuzz fuzz.resume=false mgen.method=neuri mgen.max_nodes=3 mgen.pass_rate=0.6

"""
""" to_reproduce_code 
python neuri/materialize/torch/program.py /artifact/bugs/torchcomp-neuri-fuzz-0221_record torchcomp
"""

"""train
python neuri/cli/train.py train.record_path=data/constraints/torch backend.type=torchcomp model.type=torch hydra.verbose=train train.resume=false
"""