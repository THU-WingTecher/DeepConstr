import copy
import os
import hydra
from omegaconf import DictConfig

import torch
from torch import Tensor
import yaml
import tensorflow as tf 
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

def transfer_older_record_to_newer(record: dict) -> dict:
    new = {
        "args" : { 
            "is_pos" : [],
            "dtype" : [],
            "name" : [],
            "required" : [],
        },
        "name" : None,
        "package" : None,
        "pass_rate" : 0,
    }
    new["name"] = record["title"]
    new["args"]["name"] = list(record["constraints"].keys())
    new["args"]["dtype"] = list([a["dtype"] for a in record["constraints"].values()])
    new["args"]["required"] = list([a["required"] for a in record["constraints"].values()])
    new["args"]["is_pos"] = [False for _ in range(len(new["args"]["name"]))]
    new["args"]["value"] = [None for _ in range(len(new["args"]["name"]))]

    assert record["package"] == "tf"
    new["package"] = "tensorflow"
    return new

def build_record_from_sig(api) :
    def dtype_infer(name, default) :
        if name in ["name"] :
            return "str"
        elif name in ["input", "x", "y"] :
            return "tensor" 
        elif param.default != inspect.Parameter.empty :
            return type(param.default)
        else :
            print(name, default)
            return None
    import inspect 
    sig = inspect.signature(api)
    names = []
    dtypes = []
    is_pos = []
    required_list = []
    for param in sig.parameters.values() :
        names.append(param.name)
        if dtype_infer(param.name, param.default) is None :
            continue
        dtypes.append(param.annotation)
        is_pos.append(True)
        required_list.append(param.default == inspect.Parameter.empty)
    print(names, dtypes, is_pos, required_list)
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
            "error" : str(NotImplementedError)
    },
        "torch.functional.align_tensors" : {
            "error" : str(NotImplementedError)
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
            "error" : str(NotImplementedError)
    },
        "torch.functional.meshgrid" : {
            "error" : str(NotImplementedError)
    },
        "torch._C._nn.unflatten_dense_tensors" : {
            "error" : str(UnsolverableError)
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
    legal = 0
    illegal = 0
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
    n_ = 0
    prev_op_name = None
    while i_line < tot_doc_lines:
        line = ops_doc_lines[i_line].strip()
        if not line:
            i_line += 1
            continue
        is_tensor_method = line.startswith("Tensor.")
        is_torch_func = line.startswith("torch.")
        # i_line += 1
        # read wanted lines
        if any(
            [line.startswith(skip_kw) for skip_kw in skip_startwith_kws]
        ) or any([skip_kw in line for skip_kw in skip_in_kws]):
            i_line += 1
            continue
        if is_tensor_method or is_torch_func:
            n_+=1
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
                cnt+=1 
            else :
                cnt = 0
            if is_tensor_method:
                op_name = f"torch.{op_name}"
            obj = eval(op_name)
            if callable(obj):
                op_args = op_spec.split("(")[1].split(")")[0]
                op_rets = op_spec.split("-> ")[1]
                if not (
                    ("Tensor" in op_args or is_tensor_method)
                    # and "Tensor" in op_rets
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
                if os.path.exists(save_path):
                    i_line += 1
                    continue
                TRAIN_LOG.info(f"deal with {save_path}")
                record = gen_record_for_operator(op_name, op_args, is_tensor_method)
                res, record = check_trainable(record, executor, save_path)
                if res :
                    legal+=1
                    save_record(transform_record_for_saving(record), save_path)
                else :
                    illegal+=1
            else:
                TRAIN_LOG.info(f"  (Ignored: {obj = } from {op_name = } is not callable")
        # end if
        i_line += 1
    # end while
    TRAIN_LOG.info(f"end of torch_prepare, {legal} legal methods, {illegal} illegal funcs, cur_line {i_line}, all {n_} operators")
    return tensor_methods, torch_funcs

def tf_prepare(save_dir, executor, datapath="/artifact/data/tf_nnsmith.json"):
    legal, illegal, notfound = 0, 0, 0
    notfounds = []
    illegals = []
    with open(datapath, "r") as f:
        overall_apis = yaml.safe_load(f)
    
    for api in overall_apis:
        nm_to_path = api.replace(".", "/")
        load_path = os.path.join(save_dir, f"{nm_to_path}.yaml")
        save_path = os.path.join("cleaned", f"{nm_to_path}-{0}.yaml")
        if not os.path.exists(load_path):
            TRAIN_LOG.warning(f"{load_path} not found")
            notfounds.append(api)
            continue
        with open(load_path, 'r') as f:
            record = yaml.safe_load(f)
            res, record = check_trainable(record, executor, load_path)
            if res :
                legal+=1
                save_record(transform_record_for_saving(record), save_path)
            else :
                TRAIN_LOG.info(f"Ignored: {record['name'] = }")
                illegals.append(api)
                illegal+=1
    print(illegals)
    print(notfounds)
    TRAIN_LOG.info(f"end of tf_prepare, {legal} legal methods, {illegal} illegal funcs, {notfound} not found")
    return legal, illegal

def check_trainable(record, executor, save_path, *args, **kwargs) : 

    record = transfer_older_record_to_newer(record)
    record['args']['dtype_obj'] = [materalize_dtypes(dtype) for dtype in record['args']['dtype']]
    record['args']['value'] = [None] * len(record['args']['name'])
    record['outputs'] = {'value': []} # Placeholder for the output values
    deal_special_case(record)
    constr = []
    ntimes = 100
    results = executor.execute(ntimes, constr, record=record) 
    illegal_cnt = 0
    legal = 0
    for res in results : 
        if res is None : 
            illegal_cnt+=1
            continue 
        if res[0] == False : 
            if res[1].error_type in [TypeError, NotImplementedError] :
                # TRAIN_LOG.info(f"  (Ignored: {obj = } from {op_name = } is illegal({res[1]})")
                record["error"] = str(res[1].error_type)
                illegal_cnt+=1
    if res is None :
        res = (False, None)
    if illegal_cnt > ntimes * 0.8 :
        TRAIN_LOG.warning(f"  (Ignored: {record['name'] = } from {record['args'] = } is illegal({res[1]}) N_ILLEGAL : {illegal_cnt}")
        return False, record
    else :
        TRAIN_LOG.info(f"SELECTED  {record['name'] = } from {record['args'] = } is legal({res[1]})")
        return True, record

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
    Used for the generate records by cli/prepare.py
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
    special_kws = [ # kwarg name -> we don't support this object
        "dtype",
        "memory_format",
        "layout"
    ]
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
            if arg_name in special_kws : # dtype arg name related data type is all wrong
                arg_dtype = "None"
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
    elif cfg["model"]["type"] == "tensorflow":
        tf_prepare(cfg["train"]["root"], executor)
    else :
        raise NotImplementedError

if __name__ == "__main__":
    main()
    # import sys 
    # dir = sys.argv[1] 
    # legal = 0
    # illegal = 0
    # operators = []
    # for root, dirs, files in os.walk(dir):
    #     for file in files:
    #         if file.endswith(".yaml"):
    #             name = file.split(".")[0].split("-")[0]
    #             if name in operators :
    #                 continue
    #             with open(os.path.join(root, file), 'r') as f:
    #                 record = yaml.safe_load(f)
    #             if record.get("error") is not None :
    #                 illegal+=1
    #             else :
    #                 operators.append(name)
    #                 legal+=1
    # print(f"legal {legal} illegal {illegal}, unique operator : {len(operators)}") 
