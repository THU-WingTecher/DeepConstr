import copy
import os
import hydra
from omegaconf import DictConfig

import torch
from torch import Tensor
import yaml

from neuri.autoinf.instrument.collect import parse_torch_sigs
from neuri.constrinf import _process_record
from neuri.constrinf.executor import Executor
from neuri.error import UnsolverableError
from neuri.logger import TRAIN_LOG
from neuri.materialize import Model
from neuri.abstract.dtype import materalize_dtypes

def load_executor(model_type, backend_target, parallel):
    ModelType = Model.init(
        model_type, backend_target
    )
    executor = Executor(ModelType, parallel = parallel)
    return executor

def transform_record_for_saving(record: dict) -> dict:
    """
    Transform the record dictionary to the original format expected for saving.

    Args:
        record (dict): The modified record dictionary.

    Returns:
        dict: The transformed record dictionary suitable for saving.
    """
    transformed = {}
    for key, value in record.items():
        if key == 'args' :
            transformed[key] = {}
            for k, v in value.items():
                if k in ['dtype_obj', 'value'] :
                    pass 
                else :
                    transformed[key][k] = v
        elif key == "rules" :
            ## logic for deserialize rules
            pass 
        elif key == "outputs" :
            ## outputs are only for placeholder
            pass
        else:
            transformed[key] = value
    return transformed

def deal_special_case(record) :
    special_records = {
        "torch.align_tensors" : {
            "error" : NotImplementedError
    },
        "torch.functional.align_tensors" : {
            "error" : NotImplementedError
    },
        "torch.block_diag" : {
            "is_pos" : ["tensors"]
    },
        "torch.functional.block_diag" : {
            "is_pos" : ["tensors"]
    },
        "torch.broadcast_tensors" : {
            "is_pos" : ["tensors"]
    },
        "torch.functional.broadcast_tensors" : {
            "is_pos" : ["tensors"]
    },
        "torch.cartesian_prod" : {
            "is_pos" : ["tensors"]
    },
        "torch.functional.cartesian_prod" : {
            "is_pos" : ["tensors"]
    },
        "torch.meshgrid" : {
            "error" : NotImplementedError
    },
        "torch.functional.meshgrid" : {
            "error" : NotImplementedError
    },
        "torch._C._nn.unflatten_dense_tensors" : {
            "error" : UnsolverableError
    },
    }
    if special_records.get(record["name"], None) is not None :
        new_data = special_records[record["name"]]
        for key, value in new_data.items() :
            if key == "error" :
                record[key] = value
            elif key == "is_pos" :
                record["args"]["is_pos"] = value


def torch_prepare(save_dir, executor):
    import torch.jit.supported_ops

    ops_doc_lines = torch.jit.supported_ops.__doc__.splitlines()
    """
    1. tensor_methods: `Tensor.method_name(...`
    2. torch_funcs: `torch.func_name(...`
    """
    skip_startwith_kws = []
    skip_in_kws = [
        ".detach",
        ".save",
        ".item",
        ".dim",
        ".to",
        ".set",
        ".clone",
        ".device",
        ".cpu",
        ".cuda",
        ".tensor",
    ]
    tensor_methods = {}
    torch_funcs = {}
    tot_doc_lines = len(ops_doc_lines)
    i_line = 0
    prev_op_name = None
    while i_line < tot_doc_lines:
        line = ops_doc_lines[i_line].strip()
        if not line:
            i_line += 1
            continue
        # read wanted lines
        if any(
            [line.startswith(skip_kw) for skip_kw in skip_startwith_kws]
        ) or any([skip_kw in line for skip_kw in skip_in_kws]):
            i_line += 1
            continue
        is_tensor_method = line.startswith("Tensor.")
        is_torch_func = line.startswith("torch.")
        if is_tensor_method or is_torch_func:
            # get specification of a whole op/func
            op_spec = []
            while i_line < tot_doc_lines:
                line = ops_doc_lines[i_line].strip()
                op_spec.append(line)
                if "->" in line:
                    break
                i_line += 1
            op_spec = " ".join(op_spec)
            # parse op spec
            op_name = op_spec.split("(")[0]
            if op_name in tensor_methods or op_name in torch_funcs:
                i_line += 1
                continue
            if is_tensor_method:
                op_name = f"torch.{op_name}"
            if prev_op_name == op_name :
                cnt+=1 
            else :
                cnt = 0
            obj = eval(op_name)
            if callable(obj):
                op_args = op_spec.split("(")[1].split(")")[0]
                op_rets = op_spec.split("-> ")[1]
                if not (
                    ("Tensor" in op_args or is_tensor_method)
                    and "Tensor" in op_rets
                    # and "out : Tensor" not in op_args
                ):
                    # both inputs and outputs have tensors
                    i_line += 1
                    continue
                if is_torch_func:
                    torch_funcs[op_name] = obj
                else:
                    tensor_methods[op_name] = obj
                save_path = os.path.join(save_dir, f"{op_name.replace('.', '/')}-{cnt}.yaml")
                prev_op_name = op_name
                if os.path.exists(save_path):
                    i_line += 1
                    continue
                TRAIN_LOG.info(f"deal with {save_path}")
                record = gen_record_for_operator(op_name, op_args, is_tensor_method)
                # if "torch.split" in record["name"] :
                #     pass
                # else :
                #     i_line += 1
                #     continue
                record['args']['dtype_obj'] = [materalize_dtypes(dtype) for dtype in record['args']['dtype']]
                record['outputs'] = {'value': []} # Placeholder for the output values
                deal_special_case(record)
                constr = []
                ntimes = executor.parallel
                results = executor.execute(ntimes, constr, record=record) 
                illegal_cnt = 0
                for res in results : 
                    if res is None : 
                        illegal_cnt+=1
                        continue 
                    if res[0] == False : 
                        if res[1].error_type in [TypeError, NotImplementedError] :
                            TRAIN_LOG.info(f"  (Ignored: {obj = } from {op_name = } is illegal({res[1]})")
                            record["error"] = str(res[1].error_type)
                            illegal_cnt+=1
                if res is None :
                    res = (False, None)
                if illegal_cnt > ntimes * 0.8 :
                    TRAIN_LOG.warning(f"  (Ignored: {obj = } from {op_name = } is illegal({res[1]}) N_ILLEGAL : {illegal_cnt}")
                else :
                    TRAIN_LOG.info(f"SELECTED  {op_name = } from {op_args = } is legal({res[1]})")
                save_record(transform_record_for_saving(record), save_path)
            else:
                TRAIN_LOG.info(f"  (Ignored: {obj = } from {op_name = } is not callable")
        # end if
        i_line += 1
    # end while
    TRAIN_LOG.info(f"end of torch_prepare, {len(tensor_methods)} tensor methods, {len(torch_funcs)} torch funcs, cur_line {i_line}")
    return tensor_methods, torch_funcs

def save_record(record, path) :
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    
    with open(path, 'w') as file:
        yaml.dump(record, file)

def custom_split(input_string):
    # Stores the parts of the string split by commas outside brackets
    parts = []
    current_part = []  # Stores characters for the current part being processed
    bracket_stack = []  # Keeps track of open brackets to ensure matching pairs
    
    # Mapping of closing and opening brackets
    brackets = {']': '[', '}': '{', ')': '('}
    
    for char in input_string:
        if char in "[{(":
            bracket_stack.append(char)
        elif char in "]})" and bracket_stack:
            if bracket_stack[-1] == brackets[char]:
                bracket_stack.pop()
            else:
                # Handle mismatched bracket scenario
                raise ValueError("Mismatched brackets in input string")
        
        if char == ',' and not bracket_stack:
            # If we're not inside brackets, split here
            parts.append(''.join(current_part))
            current_part = []
        else:
            current_part.append(char)
    
    # Add the last part if there's any
    if current_part:
        parts.append(''.join(current_part))
    
    return parts


def gen_record_for_operator(op_name, args_str, is_tensor_method, package="torch") :
    """
    save records of an API to a folder
    API_name/hash_value.pkl

    {
        'name': 'torch.add',
        'args': {
            'name': ['arg_0', 'arg_1', 'alpha'],
            'dtype' : [number, List[int], Optional[str], ...]
            'is_pos': [True, True, False],
            'required': [True, True, False],
            'value' : [None, None, None],
        },
        'package': 'torch',
        'pass_rate': 0, 
        'rules' : {}

    }
    """
    is_pos_kwargs = [ "self", "tensors"]
    # args_str --> name : dtype=1, Optional
    records = {
        'name': op_name,
        "args": {
            "name": [],
            "dtype" : [],
            "is_pos": [],
            "required": [],
            "value": [],
        },
        "package": package,
        "pass_rate": 0,
        "rules" : {}
    }
    names = []
    dtypes = []
    is_pos = []
    required_list = []
    if is_tensor_method :
        names.append("self")
        dtypes.append("Tensor")
        is_pos.append(True)
        required_list.append(True)
    for arg_str in custom_split(args_str) :
        if arg_str :
            arg_name = arg_str.split(':')[0].strip()
            arg_dtype = arg_str.split(':')[1].split('=')[0].strip()
            if 'Optional' in arg_str or "=" in arg_str :
                required = False
            else :
                required = True
            names.append(arg_name)
            dtypes.append(arg_dtype)
            is_pos.append(True if arg_name in is_pos_kwargs else False)
            required_list.append(required)
    records['args']['name'] = names
    records['args']['dtype'] = dtypes
    records['args']['is_pos'] = is_pos
    records['args']['required'] = required_list
    records['args']['value'] = [None] * len(names)
    return records

@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):
    executor = load_executor(cfg["model"]["type"], "cpu", cfg["train"]["parallel"])
    if cfg["model"]["type"] == "torch":
        torch_prepare(cfg["train"]["root"], executor)
    else :
        raise NotImplementedError


# if int(os.getenv("INSTR", "0")):
if __name__ == "__main__":
    main()
    # import sys 
    # dir = sys.argv[1] 
    # for root, dirs, files in os.walk(dir):
    #     for file in files:
    #         if file.endswith(".yaml"):
    #             with open(os.path.join(root, file), 'r') as f:
    #                 record = yaml.safe_load(f)
    #             if "tensors" in record["args"]["name"] :
    #                 print(record)

    # for name, func in tensor_methods.items():
    #     instrument(name, func)
    # for name, func in torch_funcs.items():
    #     instrument(name, func)

    # from hanging_threads import start_monitoring

    # start_monitoring(seconds_frozen=10, test_interval=100)
