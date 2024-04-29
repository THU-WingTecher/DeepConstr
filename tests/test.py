import os
import sys
from typing import Any, Dict, List
from nnsmith.autoinf import AutoInfOpBase
from nnsmith.autoinf.instrument.op import OpInstance
from nnsmith.abstract.dtype import AbsTensor, materalize_dtypes
from nnsmith.logger import LOGGER
from deepconstr.gen.solve import gen_val
from nnsmith.specloader.utils import load_yaml

def proxy_check(cfg) : 
    from nnsmith.deepconstr.inferencer import Inferencer
    inferencer = Inferencer(cfg['llm']['settings'])
    inferencer.change_to_gpt3()
    prompts = 'hello'
    results = inferencer.inference(prompts)
    print(results)

def gen_record(func='raw_ops.QuantizeAndDequantizeV4') :
    
    record={}
    cfg = load_yaml(f"{os.getcwd()}/data/constraints/{func.replace('.','/')}.yaml")
    record['name'] = cfg['title']
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
            # if len(types_dict[name]) > 0 : 
            #     types_dict[name] = types_dict[name][0] 
    return record

def convert_rule_to_executable(record) -> List["z3.Exr"] : 

    chosn_dtypes = {} 
    exec_rules = []
    for i_arg, arg_name, in enumerate(record['args']['name']) :
        if len(record['args']['dtype'][i_arg]) > 0 :
            chosn_dtypes[arg_name] = record['args']['dtype'][i_arg][0]
        else :
            chosn_dtypes[arg_name] = record['args']['dtype'][i_arg]

    for rule in record['rules'] :
        print(f"rule : {rule['txt']}")
        rule = gen_rule(rule['target'],rule['cot'], rule['txt'], {name : dtype for name, dtype in zip(record['args']['name'], record['args']['dtype'])},
                                    ) # multiple dtype list
        if rule is None : continue
        rule.ast2z3.set_args_types(chosn_dtypes) # only one dtypes 
        c1 = rule.ast2z3.gen_constraints()
        exec_rules.append(c1)
    return exec_rules

def test_whole_rules(data_dir = None) : 
    from nnsmith.deepconstr import gen_inst_with_records
    if data_dir == None : 
        dir_path = os.path.join(os.getcwd(), 'data/constraints')

    records = gen_inst_with_records(dir_path)
    for record in records : 
        arg_names = record['args']['name']
        rule_txts = [r['txt'] for r in record['rules']]
        ##test function(arg_names, rule_txts)

def read_rule_file(path) : 

    res = []  
    with open(path, 'r') as f :
        rules = f.readlines()
    while rules : 
        rule_set = []
        rule = rules.pop(0)
        while rules and len(rule.strip()) != 0 :
            rule_set.append(rule.strip()) 
            rule = rules.pop(0)
        if rule_set : 
            res.append(rule_set)  

    return res

def add_info_to_error_DB(cfg) : 
    #load error DB
    import sys, os 
    from train.utils import save_cfg

    sys.path.append(os.getcwd())
    db_path = "/home/guihuan/LLM/data/DB_error.yaml"
    db = load_yaml(db_path)
    print(f"current DB : {db}")
    target = cfg['target']
    cot = cfg['cot']
    answers = cfg['answers']
    LOGGER.info(f'adding \ntarget : {target}\ncot : {cot}\nanswers : {answers}\n to DB')
    db[target] = {} 
    db[target]['cot'] = cot
    db[target]['answers'] = answers
    save_cfg(db, db_path)


def test_solving(rules, record) : # return the values params and input tensor shape&dtype

    chosn_dtypes = {} 
    for i_arg, arg_name, in enumerate(record['args']['name']) :
        if len(record['args']['dtype'][i_arg]) > 0 :
            chosn_dtypes[arg_name] = record['args']['dtype'][i_arg][0]
        else :
            chosn_dtypes[arg_name] = record['args']['dtype'][i_arg]

    values = gen_val(chosn_dtypes, rules, )
    print(f'values : {values}')
    for i_arg, arg_name, in enumerate(record['args']['name']) :
        record['args']['value'][i_arg] = values[arg_name]
    
    return record 

def gen_op_inst(record : Dict[str, Any]) : 
    
    import tensorflow as tf
    inst = OpInstance(record)
    output_info = inst.execute(
        symb_2_value=None, 
        tensor_from_numpy=tf.convert_to_tensor,
        numpy_from_tensor=lambda x: x.numpy(),
        is_tensor=lambda x: isinstance(x, tf.Tensor),
        func=eval(inst.name)
    )
    inst.add_output_arg(*output_info)

    # print(inst.invoke_str(attr_map=attr_map))
    # inst.materialize(eval(inst.name), attr_map)
    opbase = AutoInfOpBase(inst, {
        sym : inst.input_symb_2_value[sym] for sym in inst.A
    })
    opbase.execute()

def gen_record_finder(path : str , pass_rate=0.8) :
    from nnsmith.deepconstr import make_record_finder 
    record_finder = make_record_finder(path, pass_rate)
    return record_finder

def test_insert_node(record_finder, model_type, backend_type) : 
    from nnsmith.graph_gen import ConstrInf 
    from nnsmith.materialize import Model
    dump_ops = []
    ModelType = Model.init(
        model_type, backend_type
    )
    gen = ConstrInf([], record_finder, ModelType)
    cnt=0
    num_of_tries = 200
    container = []
    try :
        for i in range(num_of_tries) :
            result = gen.try_autoinf_insert_forward()
            if result : 
                cnt += 1
            else : 
                container.append(gen.load_err_msg())
        print(f"cnt : {cnt} all_tries : {num_of_tries}")
    except :
        print(f"cnt : {cnt} all_tries : {num_of_tries}")
        print('\n'.join(container))
    # gen.try_autoinf_insert_backward()
# tensor_from_numpy_torch = lambda x: torch.from_numpy(x)
# numpy_from_tensor_torch = lambda x: x.detach().cpu().numpy() if x.requires_grad else x.cpu().numpy()
# is_tensor_torch = lambda x: isinstance(x, torch.Tensor)

import hydra
from omegaconf import DictConfig
@hydra.main(version_base=None, config_path="../neuri/config", config_name="main")
def main(cfg: DictConfig):
    proxy_check(cfg)
    # if cfg['task']['type'] == 1 :
    #     proxy_check(cfg)
    # elif cfg['task']['type'] in [3, 'db']:
    #     add_info_to_error_DB(cfg['db'])
    # elif cfg['task']['type'] == 5 : # with cfg['paths']['rule'] rules
    #     # Solve rule of the func.yaml file" 
    #     record = gen_record(cfg['func'])
    #     rules = test_rule_parsing([r["txt"] for r in record['rules']], record)     
    #     test_solving(rules, record)
    # elif cfg['task']['type'] == 6 : # with real rules
    #     record = gen_record(cfg['func'])
    #     rules = convert_rule_to_executable(record)
    #     updated_record = test_solving(rules, record)
    #     op_inst = gen_op_inst(updated_record)
    # elif cfg['task']['type'] == 7 :
    #     record_finder = gen_record_finder(cfg['paths']['rule'], 1)
    #     test_insert_node(record_finder, cfg['model_type'], cfg['backend_type'])
if __name__ == "__main__" :
    main()