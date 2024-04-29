import copy
import os
import pickle
import traceback
from functools import partial
from typing import Any, Callable, Dict, List

import numpy as np
# from natsort import natsorted
from tqdm import tqdm

from nnsmith.autoinf.instrument.utils import (
    data_type_str,
    get_ret_list,
    inputs_from_record,
    tensors_from_numpys,
)


def parse_torch_sigs() -> Dict[str, List[List[List[Any]]]]:
    import re

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
    op_2_arglists: Dict[str, List[List[List[Any]]]] = {}
    tot_doc_lines = len(ops_doc_lines)
    i_line = 0
    while i_line < tot_doc_lines:
        line = ops_doc_lines[i_line].strip()
        if not line:
            i_line += 1
            continue
        # read wanted lines
        if any([line.startswith(skip_kw) for skip_kw in skip_startwith_kws]) or any(
            [skip_kw in line for skip_kw in skip_in_kws]
        ):
            i_line += 1
            continue
        is_tensor_method = line.startswith("Tensor.")
        is_torch_func = line.startswith("torch.")
        if is_tensor_method or is_torch_func:
            # get specification of a whole op/func
            op_spec_list = []
            while i_line < tot_doc_lines:
                line = ops_doc_lines[i_line].strip()
                op_spec_list.append(line)
                if "->" in line:
                    break
                i_line += 1
            op_spec = " ".join(op_spec_list)
            # parse op spec
            op_name = op_spec.split("(")[0]
            if is_tensor_method:
                op_name = f"torch.{op_name}"
            op_args = op_spec.split("(")[1].split(")")[0]
            op_rets = op_spec.split("-> ")[1]
            if not (
                ("Tensor" in op_args or is_tensor_method)
                and "Tensor" in op_rets
                and "out : Tensor" not in op_args
            ):
                # both inputs and outputs have tensors
                i_line += 1
                continue
            obj = eval(op_name)
            if callable(obj):
                """get default arg values"""
                name_value_list: List[List[Any]] = []
                if is_tensor_method:
                    name_value_list.append(["self"])
                for i_arg_spec, arg_spec_str in enumerate(op_spec_list):
                    name_value: List[Any] = []
                    _kw_name = re.findall(r"(\w+)\ ", arg_spec_str)
                    if (
                        len(_kw_name) == 0
                        and i_arg_spec == 0
                        and is_tensor_method
                        and len(op_spec_list) == 1
                    ):
                        continue
                    elif len(_kw_name) != 1:
                        print(f"{len(_kw_name) = }")
                    kw_name = _kw_name[0]
                    if i_arg_spec == 0 and is_torch_func and kw_name == "self":
                        kw_name = "input"
                    name_value.append(kw_name)
                    _kw_type_value = re.findall(r":\ (.+)=([^>]+)[,)]", arg_spec_str)
                    if len(_kw_type_value) == 1:
                        _kw_type, _kw_value = _kw_type_value[0]
                        if _kw_type == "str":
                            kw_value = _kw_value
                        else:
                            kw_value = eval(_kw_value)
                        name_value.append(kw_value)
                    elif "Optional" in arg_spec_str:
                        name_value.append(None)
                    # op specific patch rules
                    if not (  # exclude these args
                        op_name == "torch.einsum"
                        and kw_name == "path"
                        or op_name.endswith(".flatten")
                        and kw_name == "out_dim"
                        # or op_name == "torch.Tensor.contiguous"
                        # and kw_name == "memory_format"
                    ):
                        name_value_list.append(name_value)
                # end for i_arg_spec
                # add these extra args
                if op_name == "torch.nonzero" or op_name == "torch.Tensor.nonzero":
                    name_value_list.append(["as_tuple", False])
                elif op_name == "torch.empty_like":
                    name_value_list.append(["requires_grad", False])

                if not (  # exclude these signatures
                    op_name == "torch.repeat_interleave"
                    and len(name_value_list) < 4
                    or op_name.split(".")[-1] in ["mean", "prod", "sum"]
                    and len(name_value_list) == 2
                ):
                    sigs: List[List[List[Any]]] = op_2_arglists.get(op_name, [])
                    sigs.append(name_value_list)
                    op_2_arglists[op_name] = sigs
            else:
                pass
                # print(f"  (Ignored: {obj = } from {op_name = } is not callable or has no default arg values")
        # end if
        i_line += 1
    # end while
    return op_2_arglists


def parse_tf_sigs(
    ops_file_path: str = "autoinf/instrument/xla-compilable-ops.md",
) -> Dict[str, List[List[List[Any]]]]:
    import tensorflow as tf

    with open(ops_file_path, "r") as f:
        lines = f.readlines()
    skip_in_kws = []
    opname_2_sigs = {}
    """get valid ops"""
    for line in lines:
        # '| `Abs`                                       | `T={bfloat16,double,float,half,int16,int32,int64,int8}`      |\n'
        splitted = line.split("`")
        if len(splitted) <= 2:
            print(f"skipped: {line}")
            continue
        last_name = splitted[1]
        if any([kw in last_name for kw in skip_in_kws]):
            continue
        op_name = f"tf.raw_ops.{last_name}"
        try:
            func = eval(op_name[len("") :])
        except Exception as e:
            print(f"Error when eval {op_name}: {e}")
            continue
        if not callable(func):
            raise RuntimeError(f"{func} is not callable")
        else:
            sigs = []
            opname_2_sigs[op_name] = [sigs]
            for i_arg, arg in enumerate(func.__signature__.parameters.values()):
                name_value = [arg.name]
                if arg.default is not arg.empty:
                    name_value.append(arg.default)
                sigs.append(name_value)

    # end for line
    return opname_2_sigs


opname_2_sigs = None


def patch_record(
    record: Dict[str, Any],
    is_tensor: Callable,
    tensor_from_numpy: Callable,
    tensor_to_numpy: Callable,
) -> Dict[str, Any]:
    if "out" in record["args"]["name"]:
        return
    name = record["name"]
    sigs = opname_2_sigs.get(name, [])
    if not sigs:
        raise RuntimeError(f"Cannot find {name} in opname_2_sigs")
    args, input_kwargs = inputs_from_record(
        record, tensor_from_numpy, random_value=False
    )
    # deal with some torch inconsistencies
    if name == "torch.sum" and "axis" in input_kwargs:
        input_kwargs["dim"] = input_kwargs.pop("axis")
    if name == "torch.zeros_like" and "requires_grad" in input_kwargs:
        input_kwargs.pop("requires_grad")
    if "flatten" in name:  # replace str identified dim
        _len_args = len(args)
        for i_arg in range(_len_args):
            ri = _len_args - 1 - i_arg
            if isinstance(args[ri], str):
                args.pop(ri)
        for k, v in input_kwargs:
            if isinstance(v, str):
                input_kwargs.pop(k)
    """select matched signature and default values"""
    for sig in sigs:
        # sig: List[List[Any]]
        arg_names: List[str] = []
        kwargs: Dict[str, Any] = {}
        if len(sig) < len(args):
            continue
        valid = True
        for i_arg, argname_value in enumerate(sig):
            arg_names.append(argname_value[0])
            if len(argname_value) == 2:
                kwargs[argname_value[0]] = argname_value[1]
            elif len(argname_value) == 1:
                # no default value
                # argname = argname_value[0]
                if i_arg >= len(args) and argname_value[0] not in input_kwargs:
                    valid = False
                    break
        else:
            if not all(k in arg_names for k in input_kwargs):
                valid = False
            # for k in input_kwargs:
            #     if k not in arg_names:
            #         arg_names.append(k)
            # e.g. torch.nonzero, as_tuple=False
        if valid:
            break
    else:
        from IPython import embed

        print(f"{sig = }")
        print(f'{record["args"]["name"] = }')
        print(f"Cannot find a valid signature for {name}", flush=True)
        # embed()
        # raise RuntimeError(f"Cannot find a valid signature for {name}")

    """compute args again"""
    kwargs.update(input_kwargs)
    if len(args) > len(arg_names):
        arg_names += [f"unknown_{i}" for i in range(len(args) - len(arg_names))]
    arg_is_pos = [True] * len(args) + [False] * (len(arg_names) - len(args))
    arg_tensor_shapes = [None] * len(arg_is_pos)
    arg_values = [None] * len(arg_is_pos)
    for i_arg, arg_name in enumerate(arg_names):
        try:
            arg = args[i_arg] if i_arg < len(args) else kwargs[arg_name]
        except Exception as e:
            raise RuntimeError(
                f"Getting arg has error: {i_arg = } , {arg_name = }, {e = }"
            )
        if isinstance(arg, tuple):
            arg = list(arg)
        if is_tensor(arg):
            arg_tensor_shapes[i_arg] = list(arg.shape)
            arg_values[i_arg] = tensor_to_numpy(arg)
        elif isinstance(arg, list) and all(is_tensor(arg_i) for arg_i in arg):
            arg_tensor_shapes[i_arg] = [list(arg_i.shape) for arg_i in arg]
            arg_values[i_arg] = [tensor_to_numpy(arg_i) for arg_i in arg]
        else:
            arg_values[i_arg] = arg
    # TODO temporary fix for torch.mul
    # out_shapes = record["outputs"]["tensor_shape"]
    # for i_shape, shape in enumerate(out_shapes):
    #     if isinstance(shape, list) and len(shape) > 0 and not isinstance(shape[0], list):
    #         out_shapes[i_shape] = [int(s) for s in shape]
    patched_record = {
        "name": name,
        "args": {
            "name": arg_names,
            "is_pos": arg_is_pos,
            "tensor_shape": arg_tensor_shapes,
            "value": arg_values,
            "type": [data_type_str(arg) for arg in arg_values],
        },
        "outputs": {
            "name": record["outputs"]["name"],
            "tensor_shape": record["outputs"]["tensor_shape"],
            "type": [data_type_str(v) for v in record["outputs"]["value"]],
            "value": record["outputs"]["value"],
        },
    }
    """test runnable again"""
    try:
        args, kwargs = inputs_from_record(patched_record, tensor_from_numpy)
        ret = eval(name)(*args, **kwargs)
    except Exception as e:
        print(e, flush=True)
        print(traceback.format_exc(), flush=True)
        patched_record = None
    return patched_record


def preprocess_record(record: Dict[str, Any]):
    return record
    for i_out, out in enumerate(record["outputs"]["value"]):
        if (isinstance(out, list) or isinstance(out, tuple)) and __builtins__["all"](
            isinstance(x, torch.Tensor) for x in out
        ):
            record["outputs"]["is_tensor"][i_out] = True
            record["outputs"]["tensor_dtype"][i_out] = [str(x.dtype) for x in out]
            record["outputs"]["tensor_shape"][i_out] = [list(x.shape) for x in out]
            record["outputs"]["value"][i_out] = type(out)([x_i for x_i in out])
    return record


def collect_api(
    input_dir: str,
    only_test_first: bool = True,
    repeat_times: int = 10,
    is_tensor: Callable = None,
    tensor_from_numpy: Callable = None,
    tensor_to_numpy: Callable = None,
    test_record_independently: bool = True,
    framework: str = "torch",
) -> List[Dict[str, Any]]:
    """read in all records"""
    records: List[Dict[str, Any]] = []
    record_paths: List[str] = []
    input_record_files = natsorted(os.listdir(input_dir))
    for i_rec, record_file in enumerate(input_record_files):
        if not record_file.endswith(".pkl"):
            continue
        record_path = os.path.join(input_dir, record_file)
        print(f"  load and rerun {record_path = }")
        with open(record_path, "rb") as f:
            try:
                record = pickle.load(f)
                record = preprocess_record(record)
                if not record:
                    continue
            except Exception as e:
                print(f"Exception when loading {record_path}. Skip...")
                print(traceback.format_exc())
                continue
        """test 0: re-runnable"""
        try:
            args, kwargs = inputs_from_record(record, tensor_from_numpy)
            ret = eval(record["name"])(*args, **kwargs)
            records.append(record)
            record_paths.append(record_path)
        except Exception as e:
            print(traceback.format_exc())
            continue
        # print(f"  runnable test end: {record_path}", flush=True)
    # end for record_file
    """test api"""
    if not records:
        return "No records"
    name = record["name"]
    func = eval(name)
    if test_record_independently:
        """test 1: (runnable and) stateless"""
        print(f"  enter test 1", flush=True)
        valid_map_test1 = [True] * len(records)
        for i_record, record in enumerate(records):
            record_path = record_paths[i_record]
            print(f"  test 1: {record_path = }", flush=True)
            try:
                ret_tensors = None
                for i_repeat in range(repeat_times):
                    # print(f"    test 1: {i_repeat = }  {len(args) = } {record_path = }", flush=True)
                    args, kwargs = inputs_from_record(record, tensor_from_numpy)
                    ret = func(*args, **kwargs)
                    ret_list = get_ret_list(ret)
                    ret_tensors_this = [
                        tensor_to_numpy(x) for x in ret_list if is_tensor(x)
                    ]
                    if ret_tensors is None:
                        ret_tensors = ret_tensors_this
                    else:
                        if not all(
                            np.allclose(
                                ret_tensors[i], ret_tensors_this[i], equal_nan=True
                            )
                            for i in range(len(ret_tensors))
                        ):
                            raise Exception("not stateless")
                # end for i_repeat
            except Exception as e:
                valid_map_test1[i_record] = False
                print(f"Test 1: try {name} with {record_path} failed: {e}", flush=True)
                print(traceback.format_exc())
                # raise e
        # end for i_record
        records = [r for i, r in enumerate(records) if valid_map_test1[i]]
        record_paths = [r for i, r in enumerate(record_paths) if valid_map_test1[i]]

        """test 2: output shape is independent of input values"""
        print(f"  enter test 2", flush=True)
        valid_map_test2 = [True] * len(records)
        for i_record, record in enumerate(records):
            record_path = record_paths[i_record]
            print(f"  test 2: {record_path = }", flush=True)
            try:
                ret_tensors = None
                for i_repeat in range(repeat_times):
                    # print(f"    test 2: {i_repeat = }  {len(args) = }", flush=True)
                    args, kwargs = inputs_from_record(
                        record, tensor_from_numpy, random_value=True
                    )
                    ret = func(*args, **kwargs)
                    ret_list = get_ret_list(ret)
                    ret_tensors_this = [x for x in ret_list if is_tensor(x)]
                    if ret_tensors is None:
                        ret_tensors = ret_tensors_this
                    else:
                        if len(ret_tensors) != len(ret_tensors_this) or any(
                            ret_tensors[i].shape != ret_tensors_this[i].shape
                            for i in range(len(ret_tensors))
                        ):
                            raise Exception("not output shape independent")
                    # print(f"    test 2: {i_repeat = }  finished", flush=True)
                # end for i_repeat
            except Exception as e:
                valid_map_test2[i_record] = False
                print(f"Test 2: try {name} with {record_path} failed: {e}", flush=True)
                print(traceback.format_exc())
                # raise e
        # end for i_record
        records = [r for i, r in enumerate(records) if valid_map_test2[i]]
        record_paths = [r for i, r in enumerate(record_paths) if valid_map_test2[i]]
    else:
        is_valid = True
        err_msg: str = None
        """test 1: (runnable and) stateless"""
        print(f"  enter test 1", flush=True)
        for i_record, record in enumerate(records):
            record_path = record_paths[i_record]
            try:
                ret_tensors = None
                for i_repeat in range(repeat_times):
                    # print(f"    test 1: {i_repeat = }  {len(args) = } {record_path = }", flush=True)
                    try:
                        args, kwargs = inputs_from_record(record, tensor_from_numpy)
                        ret = func(*args, **kwargs)
                    except Exception as e:
                        print(f"Test 1: Exec {name} failed: {e}", flush=True)
                        print(traceback.format_exc())
                        is_valid = False
                        err_msg = record_path
                        break
                    ret_list = get_ret_list(ret)
                    ret_tensors_this = [
                        tensor_to_numpy(x) for x in ret_list if is_tensor(x)
                    ]
                    if ret_tensors is None:
                        ret_tensors = ret_tensors_this
                    else:
                        if not all(
                            np.allclose(
                                ret_tensors[i], ret_tensors_this[i], equal_nan=True
                            )
                            for i in range(len(ret_tensors))
                        ):
                            print(f"Test 1: {name} is not stateless", flush=True)
                            is_valid = False
                            err_msg = record_path
                            break
                # end for i_repeat
                if only_test_first or not is_valid:
                    break
            except Exception as e:
                print(f"Test 1: try {name} with {record_path} failed: {e}", flush=True)
                print(traceback.format_exc())
                raise e
        # end for i_record
        if not is_valid:
            return err_msg
        """test 2: output shape is independent of input values"""
        print(f"  enter test 2", flush=True)
        for i_record, record in enumerate(records):
            record_path = record_paths[i_record]
            # print(f"  test 2: {record_path = }", flush=True)
            try:
                ret_tensors = None
                for i_repeat in range(repeat_times):
                    # print(f"    test 2: {i_repeat = }  {len(args) = }", flush=True)
                    try:
                        args, kwargs = inputs_from_record(
                            record, tensor_from_numpy, random_value=True
                        )
                        ret = func(*args, **kwargs)
                    except Exception as e:
                        print(f"Test 2: Run {name} failed: {e}", flush=True)
                        print(traceback.format_exc())
                        is_valid = False
                        err_msg = record_path
                        break
                    ret_list = get_ret_list(ret)
                    ret_tensors_this = [x for x in ret_list if is_tensor(x)]
                    if ret_tensors is None:
                        ret_tensors = ret_tensors_this
                    else:
                        if len(ret_tensors) != len(ret_tensors_this) or any(
                            ret_tensors[i].shape != ret_tensors_this[i].shape
                            for i in range(len(ret_tensors))
                        ):
                            print(f"{name} is not output shape independent", flush=True)
                            is_valid = False
                            err_msg = record_path
                            break
                    # print(f"    test 2: {i_repeat = }  finished", flush=True)
                # end for i_repeat
                if only_test_first or not is_valid:
                    break
            except Exception as e:
                print(f"Test 2: try {name} with {record_path} failed: {e}", flush=True)
                print(traceback.format_exc())
                raise e
        # end for i_record
        if not is_valid:
            return err_msg
    # end if test_record_independently
    """patch records"""
    print(f"  enter patching", flush=True)
    patched_records = []
    while records:
        patched_record = patch_record(
            records.pop(),
            is_tensor=is_tensor,
            tensor_from_numpy=tensor_from_numpy,
            tensor_to_numpy=tensor_to_numpy,
        )
        if patched_record:
            patched_records.append(patched_record)
    return patched_records


def collect(
    input_dir: str,
    output_dir: str,
    is_tensor: Callable = None,
    tensor_from_numpy: Callable = None,
    tensor_to_numpy: Callable = None,
    skip_apis: List[str] = None,
    select_apis: List[str] = None,
    framework: str = "torch",
):
    skip_apis = [a.replace(".", "_") for a in skip_apis] if skip_apis else []
    select_apis = [a.replace(".", "_") for a in select_apis] if select_apis else []
    api_dirs = natsorted(os.listdir(input_dir))
    os.makedirs(output_dir, exist_ok=True)
    num_records, num_apis = 0, 0
    for api_dir in tqdm(api_dirs):
        if (
            skip_apis
            and api_dir in skip_apis
            or select_apis
            and api_dir not in select_apis
        ):
            print(f"skipping {api_dir = }", flush=True)
            continue
        # TODO skip existing
        api_path = os.path.join(input_dir, api_dir)
        output_path = os.path.join(output_dir, f"{api_dir}.pkl")
        if os.path.exists(output_path):
            print(f"skipping existed {output_path = }", flush=True)
            continue
        print(f"start {api_path = }", flush=True)
        try:
            records = collect_api(
                input_dir=api_path,
                only_test_first=False,
                repeat_times=3,
                is_tensor=is_tensor,
                tensor_from_numpy=tensor_from_numpy,
                tensor_to_numpy=tensor_to_numpy,
            )
        except Exception as e:
            print(f"Exception when collecting {api_dir}", flush=True)
            print(traceback.format_exc())
            # embed()
            # raise e
            continue
        if records and not isinstance(records, str):
            with open(output_path, "wb") as f:
                pickle.dump(records, f)
            num_records += len(records)
            num_apis += 1
        else:
            print(f"Collect {api_dir} failed because of {records}", flush=True)
        print(f"finish {api_path = }", flush=True)
    print(f"{num_apis = }")
    print(f"{num_records = }")


def collect_torch(input_dir, out_dir):
    import torch

    def torch_allclose_wo_name(x, y):
        return torch.allclose(
            x.rename(None).cpu(), y.rename(None).cpu(), equal_nan=True
        )

    global opname_2_sigs
    opname_2_sigs = parse_torch_sigs()
    print(f"{len(opname_2_sigs) = }")

    collect(
        input_dir=input_dir,  # TODO
        output_dir=out_dir,
        is_tensor=lambda x: isinstance(x, torch.Tensor),
        tensor_from_numpy=partial(
            tensors_from_numpys,
            tensor_from_numpy=lambda x: torch.from_numpy(copy.deepcopy(x)),
        ),
        tensor_to_numpy=lambda x: x.detach().cpu().clone().numpy(),
        skip_apis=[
            "torch.isclose",
            "torch__C__linalg_linalg_ldl_solve",
            # "torch_native_batch_norm",
        ],
        select_apis=[
            # "torch.Tensor.gather",
            # "torch.Tensor.uniform_",
            # "torch__C__linalg_linalg_ldl_solve",
        ],
    )


def collect_tf(input_dir, out_dir):
    import tensorflow as tf

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    def tf_allclose(x, y):
        return tf.experimental.numpy.allclose(x, y)

    global opname_2_sigs
    opname_2_sigs = parse_tf_sigs()
    print(f"{len(opname_2_sigs) = }")

    collect(
        input_dir=input_dir,  # TODO
        output_dir=out_dir,
        is_tensor=tf.is_tensor,
        tensor_from_numpy=partial(
            tensors_from_numpys,
            tensor_from_numpy=tf.convert_to_tensor,
        ),
        tensor_to_numpy=lambda x: x.numpy(),
        skip_apis=[
            "CollectiveReduceV2",
            "FusedBatchNormV3",
            "FusedBatchNormGradV3",
            "MatrixTriangularSolve",
            "Svd",
            "DenseBincount",  # enter patching: free(): invalid next size (fast)
            "Fill",
            "StatelessTruncatedNormalV2",
            "StatelessRandomUniformV2",
            "StatelessRandomNormalV2",
        ],
        select_apis=[
            # "Mul",
        ],
        framework="tensorflow",
    )


if __name__ == "__main__":
    import sys

    # from hanging_threads import start_monitoring

    # start_monitoring(seconds_frozen=10, test_interval=100)

    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    framework = sys.argv[3]

    if framework == "torch":
        import torch

        collect_torch(in_dir, out_dir)
    elif framework == "tensorflow":
        import tensorflow as tf

        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)

        collect_tf(in_dir, out_dir)
