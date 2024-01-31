# Utilities for using and be used by autoinf.
from os import PathLike
from typing import Any, Dict, List, Tuple, Type
from neuri.abstract.op import *
from neuri.logger import AUTOINF_LOG
# from neuri.specloader.rule import gen_rule
from neuri.specloader.utils import load_yaml
from neuri.constrinf.ast2z3 import Ast2z3
def gen_inst_with_records(
    data_dir: str,
    test_pool: List = [],
):
    """
    yield: record dict
        name : api_name 
        args :
            names : List[str] := names
            dtypes : List[dtype] := dtypes
            is_pos : List[bool] := whether the arg is positioned??
            values : args_values
        outputs :
            values List[Any](mainly AbsTensor):
        constraints : List[Rule]
    """
    from neuri.specloader.materalize import materalize_dtypes
    for root, _, files in os.walk(data_dir):
        for file in files : 
            record = {}
            path = os.path.join(root, file)
            cfg = load_yaml(path)
            record['name'] = cfg['title']
            if test_pool :
                if record['name'] not in test_pool :
                    continue
            record['pass_rate'] = cfg['pass_rate']
            if 'constraints' in cfg.keys() :
                record['args'] = {'name' : [arg_name for arg_name in cfg['constraints'].keys()], 
                                'is_pos' : [False] * len(cfg['constraints'].keys()), 
                                'value' : [None] * len(cfg['constraints'].keys()),
                                'dtype' : [None] * len(cfg['constraints'].keys()),
                                }
                record['rules'] = cfg['rules']
                record['outputs'] = {
                    'value' : [],
                }
                for i_name, name in enumerate(record['args']['name']) :
                    record['args']['dtype'][i_name] = materalize_dtypes(cfg['constraints'][name]['dtype'])   
                yield record         
            else : 
                AUTOINF_LOG.warning(f"no constraints in {path}")

def convert_rule_to_executable(record, rule_cnt) -> List["z3.Exr"] : 

    chosn_dtypes = {} 
    exec_rules = []
    # for i_arg, arg_name, in enumerate(record['args']['name']) : ## FIXME -> gen diff constr depend on diff dtype( adding suff_conds is enough)
    #     if len(record['args']['dtype'][i_arg]) > 0 :
    #         chosn_dtypes[arg_name] = record['args']['dtype'][i_arg][0]
    #     else :
    #         chosn_dtypes[arg_name] = record['args']['dtype'][i_arg]

    for rule in record['rules'] :
        AUTOINF_LOG.debug(f"rule : {rule['txt']}")
        converter = Ast2z3(record['args']['name'], record['args']['dtype'], rule, record['name'])
        rule = converter.convert()
        if rule is None : 
            AUTOINF_LOG.warning(f"rule generation Failed : {rule}")
            # raise ValueError(f"rule generation Failed : {rule}")
            continue
        
        AUTOINF_LOG.debug(f"{converter.pretty_flags()}")
        exec_rules.append(rule)

    rule_cnt["cnt"] += len(exec_rules)
    AUTOINF_LOG.info(f"{len(exec_rules)} rules are generated")
    return exec_rules

def make_record_finder(
    path: PathLike = None,
    pass_rate: float = 0.8,
    test_pool: List = [],
) -> List[Dict[str, Any]]:
    
    from neuri.autoinf import BLACKLIST
    gen_inst_records = gen_inst_with_records(data_dir=path, test_pool=test_pool)

    records = []
    total_rec = 0
    skipped_err = 0
    skipped_blacklist = 0
    skipped_unenough_psrate = 0
    blacklisted = set()
    rule_cnt = {"cnt" : 0}
    if test_pool : AUTOINF_LOG.info(f"testing {test_pool}")
    for record in gen_inst_records:
        total_rec+=1
        if test_pool :
            if record['name'] not in test_pool :
                continue 
        else : # when test_pool is defined; we don't check others
            if record.get('skipped') is not None :
                AUTOINF_LOG.info(f"skipped key --> Skip {record['name']}")
                skipped_err+=1
                continue
            if record.get('pass_rate') is None or record['pass_rate'] < pass_rate :
                AUTOINF_LOG.info(f"low pass_rate[thr:{pass_rate}] {record.get('pass_rate')}")
                skipped_unenough_psrate+=1
                continue
            if record['name'] in BLACKLIST:  # black list
                if record['name'] not in blacklisted:
                    AUTOINF_LOG.warning(f"Blacklist operator {record['name']} found!")
                    blacklisted.add(record['name'])
                skipped_blacklist += 1
                continue

        AUTOINF_LOG.info(f"Loading record name : {record['name']}")
        record['constraints'] = convert_rule_to_executable(record, rule_cnt)
        records.append(record)

    skipped_rec = skipped_err + skipped_unenough_psrate + skipped_blacklist
    AUTOINF_LOG.info(
        f"Got {len(records)} records of {total_rec} records with total {rule_cnt['cnt']} rules."
    )
    AUTOINF_LOG.info(f"Filtered {skipped_rec} records from {total_rec} initial set.")
    AUTOINF_LOG.info(
        f" Skipped : {skipped_err} Lower_psrate : {skipped_unenough_psrate} black_list : {skipped_blacklist}"
    )

    return records
