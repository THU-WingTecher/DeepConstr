import os
import hydra
from omegaconf import DictConfig
import yaml
from deepconstr.gen.record import save_record, transform_record_for_saving
from deepconstr.train.executor import Executor
from deepconstr.error import UnsolverableError
from nnsmith.logger import TRAIN_LOG
from nnsmith.materialize import Model
from deepconstr.grammar.dtype import materalize_dtypes

def load_executor(model_type, backend_target, parallel):
    ModelType = Model.init(
        model_type, backend_target
    )
    executor = Executor(ModelType, parallel = parallel)
    return executor

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

def tf_prepare(save_dir, executor, datapath="/DeepConstr/data/tf_nnsmith.json"):
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

def check_trainable(record, executor, ntimes=30, *args, **kwargs) : 

    record['args']['dtype_obj'] = [materalize_dtypes(dtype) for dtype in record['args']['dtype']]
    record['args']['value'] = [None] * len(record['args']['name'])
    record['outputs'] = {'value': []} # Placeholder for the output values
    deal_special_case(record)
    constr = []
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
    if illegal_cnt > ntimes * 0.8 :
        TRAIN_LOG.warning(f"InValid: {record['name'] = } from {record['args'] = } is illegal({res[1]}) N_ILLEGAL : {illegal_cnt}")
        not_required_indices = [i for i in range(len(record['args']['required'])) if not record['args']['required'][i]]
        if not_required_indices : 
            index = not_required_indices[-1]
            TRAIN_LOG.info(f" removing {record['args']['name'][index]} from {record['name']}")
            for key in record['args'].keys() :
                record['args'][key].pop(index)
            TRAIN_LOG.info(f"current arguments : {record['args']['name']}")
            return check_trainable(record, executor, ntimes)
        else :
            return False, record 
    else :
        TRAIN_LOG.debug(f"Valid  {record['name'] = } from {record['args'] = } is legal({res[1]})")
        if "error" in record.keys() :
            del record["error"]
        return True, record

@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):
    executor = load_executor(cfg["model"]["type"], "cpu", cfg["train"]["parallel"])
    if cfg["model"]["type"] == "torch":
        ## generate torch type information from torch.jit.supported_ops
        torch_load_from_doc(cfg["train"]["root"], executor)
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
