# Utilities for using and be used by autoinf.
from os import PathLike
import os
from typing import Any, Dict, List, Tuple, Type

import yaml
from deepconstr.logger import CONVERT_LOG

RANDOM_FILTER_NAME = "random"
BUG_FILTER_NAME = "exist_bug"
BUG_OPS = ["torch.Tensor.unsqueeze_"]
RANDOM_OPS = [
    # PyTorch
    # value randomness
    "torch.rand_like",
    "torch.randn_like",
    "torch.randint_like",
    "torch.Tensor.random_",
    "torch.Tensor.uniform_",
    "torch.empty_like",
    "torch.Tensor.normal_",
    "torch.Tensor.new_empty",
    "torch.Tensor.new_empty_strided",
    "torch.dropout",
    "torch.native_dropout",
    "torch.nn.functional.dropout",
    "torch.nn.functional.dropout1d",
    "torch.nn.functional.dropout2d",
    "torch.nn.functional.dropout3d",
    "torch.nn.functional.feature_alpha_dropout",
    # TensorFlow
    "tf.raw_ops.Unique", 
]
BLACKLIST = [
    # unlock when preprocessing filters out dynamic output numbers.
    "torch.Tensor.unbind",
    "torch.unbind",
    "torch.Tensor.split",
    "torch.split",
    # some special cases
    "torch.gather",
    "torch.Tensor.resize_as_",  # resize_as_ can't be represented in the JIT at the moment ...
    "torch.Tensor.rename",
    "torch.Tensor.rename_",
    "torch.Tensor.requires_grad_",
    "torch.searchsorted",  # sorter has value constraints but the crash needs to be triggered by a big value.
    "torch.native_batch_norm",  # crash when input constraint is violated.
    "torch.Tensor.sum_to_size",  # some odd dtype transfer

]

uncompatiable_api_marks = ["tensor element", "bug", "no function"]

class Record(Dict): pass

def transform_record_for_saving(record: Record) -> dict:
    """
    Transform the record dictionary to the original format expected for saving.

    Args:
        record (dict): The modified record dictionary.

    Returns:
        dict: The transformed record dictionary suitable for saving.
    """
    transformed = {}
    for key, value in record.items():
        if key == 'args':
            transformed[key] = {k : v for k, v in record[key].items() if k not in ["value", "dtype_obj"]}
        elif key in ["outputs"] :
            pass
        elif key == "rules" :
            transformed[key] = list(value)
        else:
            transformed[key] = value
    return transformed

def record_args_info(record, values) : 
    for i_arg, arg_name, in enumerate(record['args']['name']) :
        record['args']['value'][i_arg] = values.get(arg_name, None)

def is_special_apis(record) : 
    if record.get("skipped", False) :
        skipped_reason = record.get("skipped_reason", " ").strip().lower()
        if any([mark in skipped_reason for mark in uncompatiable_api_marks]) :
            return True

    return False 

def del_related_rule(record, args_idx_to_filter) :
    names = [record['args'].get('name', [])[idx] for idx in args_idx_to_filter]
    for rule in record.get("rules", []):
        for name in names :
            if name in rule[0]["txt"] :
                record['rules'].remove(rule)

    for rule in record.get("rules", []):
        for name in names :
            if name in rule[0]["target"]["choosen_dtype"] :
                #delete the key, item pair 
                del rule[0]["target"]["choosen_dtype"][name]
def filter_record(record, filter) :
    args_idx_to_filter = []
    if filter is None :
        return record
    for key in filter : 
        if key == "args" : 
            names = record[key].get('name', [])
            for to_filter in filter.get(key, []) :
                if names and to_filter in names: 
                    args_idx_to_filter.append(names.index(to_filter))
            del_related_rule(record, args_idx_to_filter)
            for idx in sorted(args_idx_to_filter, reverse=True) :
                for ele in record[key] :
                    if idx < len(record[key][ele]) :
                        record[key][ele].pop(idx)
        else :
            pass #other filter rules 
    
    return record


def load_yaml(path) :
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def process_record(file_path: str, test_pool: list = [], filter = []) -> dict:
    """
    Process a single file and return the configuration as a record dictionary.
    """
    from deepconstr.grammar.dtype import materalize_dtypes  
    record = {}
    record = load_yaml(file_path)
    if test_pool and record["name"].split("-")[0] not in test_pool:
        return None
    if "out_in_args" in filter : 
        record = filter_record(record, {"args" : ["out"]})
    record['args']['dtype_obj'] = [materalize_dtypes(dtype) for dtype in record['args']['dtype']]
    record['args']['value'] = [None] * len(record['args']["name"]) # Placeholder for the input values
    record['outputs'] = {'value': []} # Placeholder for the output values
    if record.get('rules', None) is None :
        record['rules'] = []
    return record
# Step 2: Define the traversal function

def gen_inst_with_records(data_dir: str, filter: List = [], test_pool: List = []):
    """
    Traverse directories to process files and yield configuration records.
    """
    import os
    for root, _, files in os.walk(data_dir):
        for file in files:
            path = os.path.join(root, file)
            record = process_record(path, test_pool, filter)
            if record:
                yield record


def save_record(record, path) :
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    
    transformed = transform_record_for_saving(record)
    with open(path, 'w') as file:
        yaml.dump(transformed, file)
        
def make_record_finder(
    path: PathLike = None,
    pass_rate: float = 0.8,
    filter: List = [],
    test_pool: List = [],
) -> List[Dict[str, Any]]:
    
    from deepconstr.train.constr import convert_constr_to_executable
    gen_inst_records = gen_inst_with_records(data_dir=path, filter=filter, test_pool=test_pool)

    records = []
    total_rec = 0
    skipped_err = 0
    unsupported_constr = 0
    skipped_blacklist = 0
    skipped_unenough_psrate = 0
    blacklisted = set()
    rule_cnt = {"cnt" : 0}
    if test_pool : CONVERT_LOG.info(f"testing {test_pool}")
    for record in gen_inst_records:
        total_rec+=1
        if test_pool :
            if record['name'] not in test_pool or record['name'] in ["torch.sin", "tf.cos"] :
                continue 
        else : # when test_pool is defined; we don't check others
            if is_special_apis(record) :
                CONVERT_LOG.info(f"unsupported constraint --> Skip {record['name']}")
                unsupported_constr+=1
                continue
            if record.get('skipped') is not None :
                CONVERT_LOG.info(f"skipped key --> Skip {record['name']}")
                skipped_err+=1
                continue
            if record.get('pass_rate') is None or record['pass_rate'] < pass_rate :
                CONVERT_LOG.info(f"low pass_rate[thr:{pass_rate}] {record.get('pass_rate')}")
                skipped_unenough_psrate+=1
                continue
            if RANDOM_FILTER_NAME in filter and record['name'] in RANDOM_OPS:  # black list
                if record['name'] not in blacklisted:
                    CONVERT_LOG.warning(f"Blacklist operator {record['name']} found!")
                    blacklisted.add(record['name'])
                skipped_blacklist += 1
                continue
            if BUG_FILTER_NAME in filter and record['name'] in BUG_OPS:  # black list
                if record['name'] not in blacklisted:
                    CONVERT_LOG.warning(f"Blacklist operator {record['name']} found!")
                    blacklisted.add(record['name'])
                skipped_blacklist += 1
                continue

        CONVERT_LOG.info(f"Loading record name : {record['name']}")
        record['constraints'] = convert_constr_to_executable(record, rule_cnt)
        records.append(record)

    skipped_rec = skipped_err + skipped_unenough_psrate + skipped_blacklist + unsupported_constr
    CONVERT_LOG.info(
        f"Got {len(records)} records of {total_rec} records with total {rule_cnt['cnt']} rules."
    )
    CONVERT_LOG.info(f"Filtered {skipped_rec} records from {total_rec} initial set.")
    CONVERT_LOG.info(
        f" Skipped : {skipped_err} Lower_psrate : {skipped_unenough_psrate} black_list : {skipped_blacklist} Unsupporting : {unsupported_constr}"
    )

    return records

def count_sub_constraints() :
    pass



if __name__ == "__main__" :
    import sys 
    record_path = "/DeepConstr/data/records/torch/nn/functional/adaptive_max_pool2d_with_indices-0.yaml" 
    if len(sys.argv) > 1 :
        record_path = sys.argv[1]
    
    record = process_record(record_path)
    add_default_rules(record)