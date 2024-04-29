# Utilities for using and be used by autoinf.
from os import PathLike
from typing import Any, Dict, List, Tuple, Type
from deepconstr.utils import load_yaml
from deepconstr.logger import CONVERT_LOG
# from neuri.specloader.rule import gen_rule
BLACKLIST = [

]

uncompatiable_api_marks = ["tensor element", "bug", "no function"]


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

def process_record(file_path: str, test_pool: list = [], filter : Dict[str, List[str]] = {"args" : ["out"]}) -> dict:
    """
    Process a single file and return the configuration as a record dictionary.
    """
    from deepconstr.grammar.dtype import materalize_dtypes  
    record = {}
    record = load_yaml(file_path)
    if test_pool and record["name"].split("-")[0] not in test_pool:
        return None
    record = filter_record(record, filter)
    record['args']['dtype_obj'] = [materalize_dtypes(dtype) for dtype in record['args']['dtype']]
    record['args']['value'] = [None] * len(record['args']["name"]) # Placeholder for the input values
    record['outputs'] = {'value': []} # Placeholder for the output values
    if record.get('rules', None) is None :
        record['rules'] = []
    return record
# Step 2: Define the traversal function

def gen_inst_with_records(data_dir: str, test_pool: list = []):
    """
    Traverse directories to process files and yield configuration records.
    """
    import os
    for root, _, files in os.walk(data_dir):
        for file in files:
            path = os.path.join(root, file)
            record = process_record(path, test_pool)
            if record:
                yield record

def make_record_finder(
    path: PathLike = None,
    pass_rate: float = 0.8,
    test_pool: List = [],
) -> List[Dict[str, Any]]:
    
    from deepconstr.train.constr import convert_constr_to_executable
    gen_inst_records = gen_inst_with_records(data_dir=path, test_pool=test_pool)

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
            if record['name'] in BLACKLIST:  # black list
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
