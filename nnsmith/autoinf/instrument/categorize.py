import copy
import json
import os

# import dill as pickle
import pickle
import tempfile
import traceback
from typing import Any, Dict, List

from tqdm import tqdm

from nnsmith.autoinf.instrument.op import OpInstance
from nnsmith.autoinf.instrument.utils import get_ret_list


def categorize_records_to_insts(
    records: List[Dict[str, Any]],
    masked_arg_names: List[str] = None,
    int_policy: str = "fix_dim",
    framework: str = "none",
) -> Dict[str, Dict[str, List[Any]]]:
    """
    Return: {
        "op_name-<index>": {
            "records": [(input_symb_2_value, output_symb_2_value), ...],
        }
    }
    """
    op_dict: Dict[str, Dict[str, Any]] = {}
    for i_record, record in enumerate(records):
        op = None
        # test compiler compability
        try:
            if framework == "torch":
                import torch

                op = OpInstance(record, int_policy, masked_arg_names, torch.dtype)
                input_tensors = op.concrete_input_tensors(
                    None, lambda x: torch.from_numpy(copy.deepcopy(x))
                )
                op_func = eval(op.name)

                def forward(*inputs):
                    fn = op.materialize(op_func, op.input_symb_2_value)
                    return fn(*inputs)

                exported = torch.jit.trace(forward, input_tensors)
                ret = exported(*input_tensors)
            elif framework == "tensorflow":
                import tensorflow as tf

                for gpu in tf.config.list_physical_devices("GPU"):
                    tf.config.experimental.set_memory_growth(gpu, True)

                op = OpInstance(record, int_policy, masked_arg_names, tf.dtypes.DType)
                if (
                    not op.output_tensors
                ):  # TODO temporary fix; should be removed after collecting new data
                    raise ValueError(f"op has no output tensor")
                input_tensors = op.concrete_input_tensors(None, tf.convert_to_tensor)
                op_func = eval(op.name)
                fn = op.materialize(op_func, op.input_symb_2_value)
                traced = tf.function(jit_compile=True)(fn).get_concrete_function(
                    *input_tensors
                )
                rets = get_ret_list(traced(*input_tensors))
                for ret in rets:
                    if tf.is_tensor(ret) and "qint" in str(ret.dtype):
                        raise ValueError(f"{ret.dtype} tensor is not supported")
                with tempfile.TemporaryDirectory() as tmpdirname:
                    tf.saved_model.save(traced, tmpdirname)
        except Exception as e:
            print(f"[Compilation, Saving, Corner case checking] failed: {e}")
            print(traceback.format_exc())
            continue

        op_hash = op.signature_hash
        if op_hash not in op_dict:
            op_dict[op_hash] = {
                "records": [op],
                "hashes": [op.tensor_shape_hash],
            }
        else:
            op_info = op_dict[op_hash]
            record_hash = op.record_hash
            if record_hash not in op_info["hashes"]:
                op_info["records"].append(op)
                op_info["hashes"].append(record_hash)
    # end for

    ret_dict: Dict[str, Dict[str, Any]] = {}
    for i_op, (op_hash, op_info) in enumerate(op_dict.items()):
        name_index = f"{op_info['records'][0].name}-{i_op}"
        for r in op_info["records"]:
            setattr(r, "name_index", name_index)
        ret_dict[name_index] = {
            "records": op_info["records"],
        }
    return ret_dict


def int_policy_to_subdir(int_policy: str):
    if int_policy == "fix_dim":
        return "cat_fix_dim"
    elif int_policy == "symb" or int_policy == "fix":
        return f"cat_{int_policy}_int"
    else:
        raise ValueError(f"Unknown {int_policy}")


def categorize(
    input_dir: str,
    int_policy: str = "fix_dim",
    skip_apis: List[str] = None,
    select_apis: List[str] = None,
    framework: str = "none",
):
    output_subdir = int_policy_to_subdir(int_policy)
    skip_apis = [a.replace(".", "_") for a in skip_apis] if skip_apis else []
    select_apis = [a.replace(".", "_") for a in select_apis] if select_apis else []
    api_files = os.listdir(input_dir)
    output_dir_path = os.path.join(input_dir, output_subdir)
    os.makedirs(output_dir_path, exist_ok=True)
    num_apis, num_insts, num_records = 0, 0, 0
    api_meta_info: Dict[str, Dict[str, Any]] = {}
    for api_file in tqdm(api_files):
        if not api_file.endswith(".pkl"):
            continue
        api_name = api_file[: -len(".pkl")]
        if (
            skip_apis
            and api_name in skip_apis
            or select_apis
            and api_name not in select_apis
        ):
            print(f"skipping {api_file} ...", flush=True)
            continue
        print(f"Processing {api_file} ...", flush=True)
        output_path = os.path.join(output_dir_path, api_file)
        # if os.path.exists(output_path):
        #     continue
        with open(os.path.join(input_dir, api_file), "rb") as f:
            records = pickle.load(f)
        if records is None:
            continue
        insts_dict = categorize_records_to_insts(
            records, int_policy=int_policy, framework=framework
        )
        if not insts_dict:
            continue
        with open(output_path, "wb") as f:
            pickle.dump(insts_dict, f)

        num_apis += 1
        num_insts += len(insts_dict)
        num_records_per_inst = [
            len(inst_info["records"]) for inst_info in insts_dict.values()
        ]
        tot_records = sum(num_records_per_inst)
        num_records += tot_records
        api_meta_info[api_name] = {
            "num_insts": len(insts_dict),
            "num_records": num_records_per_inst,
        }
    print(
        f"Total {num_apis} APIs, {num_insts} instances, {num_records} records.",
        flush=True,
    )
    with open(os.path.join(output_dir_path, "meta_info.json"), "w") as f:
        json.dump(api_meta_info, f, indent=2)


def gen_inst_with_records(
    data_dir: str,
    int_policy: str = "fix_dim",
    skip_apis: List[str] = None,
    select_apis: List[str] = None,
    record_as_tuple: bool = True,
):
    """
    yield:
        OpInstance, a list of records of an instance
    """
    cat_info_subdir = int_policy_to_subdir(int_policy)
    cat_dir_path = os.path.join(data_dir, cat_info_subdir)
    skip_apis = [a.replace(".", "_") for a in skip_apis] if skip_apis else []
    select_apis = [a.replace(".", "_") for a in select_apis] if select_apis else []
    insts_files = filter(lambda x: x.endswith(".pkl"), os.listdir(cat_dir_path))
    for inst_file in insts_files:
        inst_file_path = os.path.join(cat_dir_path, inst_file)
        inst_path_name = inst_file[: -len(".pkl")]
        if (
            skip_apis
            and inst_path_name in skip_apis
            or select_apis
            and inst_path_name not in select_apis
        ):
            # print(f"skipping {api_path = }", flush=True)
            continue
        with open(inst_file_path, "rb") as f:
            insts_dict = pickle.load(f)
        if not insts_dict:
            continue
        api_name = list(insts_dict.keys())[0].split("-")[0]
        for i_inst in range(len(insts_dict)):
            name_index = f"{api_name}-{i_inst}"
            inst_info = insts_dict[name_index]
            first_inst = inst_info["records"][0]
            # yield inst_info["records"]
            tuple_list = []
            for r in inst_info["records"]:
                if record_as_tuple:
                    tuple_list.append(r.abs_record)
                else:
                    tuple_list.append(r)
            yield first_inst, tuple_list


if __name__ == "__main__":
    import sys

    input_dir = sys.argv[1]
    int_policy = sys.argv[2]  # bool(int(sys.argv[2]))
    framework = sys.argv[3]

    categorize(
        input_dir=input_dir,
        int_policy=int_policy,
        skip_apis=[
            # "CollectiveReduceV2",
        ],
        select_apis=[
            # "torch.abs",
            # 'AvgPool',
            # "Mul",
        ],
        framework=framework,
    )
