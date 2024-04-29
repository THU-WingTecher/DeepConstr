import logging
import os

import tensorflow as tf  # type: ignore
from IPython import embed

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").disabled = True

import inspect as _inspect

from tensorflow import config, is_tensor, python, raw_ops

for gpu in config.list_physical_devices("GPU"):
    config.experimental.set_memory_growth(gpu, True)

# instrumentation


def get_all_supported_ops(ops_file_path: str):
    with open(ops_file_path, "r") as f:
        lines = f.readlines()
    skip_in_kws = []
    name_2_op_params = {}
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
            func = eval(op_name[len("tf.") :])
        except Exception as e:
            print(f"Error when eval {op_name}: {e}")
            continue
        if not callable(func):
            raise RuntimeError(f"{func} is not callable")
        else:
            name_2_op_params[op_name] = (
                func,
                list(func.__signature__.parameters.values()),
            )
    # end for line
    return name_2_op_params


_aiinstr_name_2_op_params = None
_aiinstr_is_instrumenting = False
_aiinstr_allow_recursion = False

_aiinstr_ori_exec = python.pywrap_tfe.TFE_Py_FastPathExecute
"""
_result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, # ignore it
        "ApproximateEqual", # tf.raw_ops.xxx
        name, # ignore it
        x, y, # input args (w/o default values)
        "tolerance", # name of kwarg
        tolerance # value of kwarg
    )
"""
import os


def wrapper(*args, **kwargs):
    __builtins__["print"](f"before exec:\n{args = }\n{kwargs = }", flush=True)
    # assert not kwargs
    # assert len(args) > 3
    global _aiinstr_allow_recursion, _aiinstr_is_instrumenting
    if not _aiinstr_allow_recursion and _aiinstr_is_instrumenting:
        return _aiinstr_ori_exec(*args, **kwargs)

    import pickle
    import traceback

    # out_dir = "/home/jinjun/data/autoinf/tf_hists_0113_0"
    out_dir = "tf_hists"
    out_api_dir = os.path.join(out_dir, args[1])
    os.makedirs(out_api_dir, exist_ok=True)
    if len(os.listdir(out_api_dir)) >= 1_000_000:
        return _aiinstr_ori_exec(*args, **kwargs)

    if not _aiinstr_allow_recursion:
        _aiinstr_is_instrumenting = True
    save_record = True
    op_name = f"tf.raw_ops.{args[1]}"
    op_params = _aiinstr_name_2_op_params[op_name]
    """save input args into instr_kwargs"""
    try:
        instr_kwargs = {}
        arg_names = []
        i_arg = 3
        for i_p, param in enumerate(op_params[1]):
            if i_arg >= len(args):
                break
            arg_names.append(param.name)
            if param.default is _inspect._empty:
                instr_kwargs[param.name] = args[i_arg]
                i_arg += 1
            else:
                assert (
                    param.name == args[i_arg]
                ), f"{param.name = } != {args[i_arg] = } ({i_arg = })"
                instr_kwargs[param.name] = args[i_arg + 1]
                i_arg += 2
        # end for param
        arg_is_pos = [False] * len(instr_kwargs)
        arg_is_tensor = [False] * len(arg_is_pos)
        arg_tensor_shapes = [None] * len(arg_is_pos)
        arg_values = [None] * len(arg_is_pos)
        for i_arg, arg_name in enumerate(arg_names):
            arg = instr_kwargs[arg_name]
            # if isinstance(arg, Tensor) and not hasattr(arg, "ndim"):
            if is_tensor(arg) and len(arg.shape) == 0:
                arg = arg.numpy().item()
                if i_arg < len(args):
                    args[i_arg] = arg
                else:
                    instr_kwargs[arg_name] = arg
            if is_tensor(arg):
                arg_is_tensor[i_arg] = True
                arg_tensor_shapes[i_arg] = list(arg.shape)
                arg_values[i_arg] = arg.numpy()
            elif (
                isinstance(arg, list) or isinstance(arg, __builtins__["tuple"])
            ) and all(is_tensor(x) for x in arg):
                arg_is_tensor[i_arg] = True
                arg_tensor_shapes[i_arg] = [list(x.shape) for x in arg]
                arg_values[i_arg] = type(arg)([x.numpy() for x in arg])
            else:
                arg_values[i_arg] = arg
        # end for arg_name
        save_record = any(arg_is_tensor)

    except Exception as e:
        save_record = False
        __builtins__["print"](
            f"Error when saving inputs of {op_name}:\n{e}", flush=True
        )
        traceback.print_exc()

    """execute the function"""
    try:
        ret = _aiinstr_ori_exec(*args, **kwargs)
    except Exception as e:
        if not _aiinstr_allow_recursion:
            _aiinstr_is_instrumenting = False
        __builtins__["print"](f"Error when exec {op_name}:\n{e}", flush=True)
        traceback.print_exc()
        # embed()
        raise e

    if not save_record:
        if not _aiinstr_allow_recursion:
            _aiinstr_is_instrumenting = False
        return ret

    """save outputs"""
    try:
        if isinstance(ret, __builtins__["tuple"]):
            ret_list = list(ret)
        elif not isinstance(ret, list):
            ret_list = [ret]
        else:
            ret_list = ret

        out_names = [f"o_{i}" for i in __builtins__["range"](len(ret_list))]
        out_is_tensor = [False] * len(ret_list)
        out_tensor_shapes = [None] * len(ret_list)
        out_values = [None] * len(ret_list)
        for i_out, out_name in enumerate(out_names):
            out = ret_list[i_out]
            if is_tensor(out) and len(out.shape) == 0:
                out = out.numpy().item()
            if is_tensor(out):
                out_is_tensor[i_out] = True
                out_tensor_shapes[i_out] = list(out.shape)
                out_values[i_out] = out.numpy()
            elif (
                isinstance(out, list) or isinstance(out, __builtins__["tuple"])
            ) and __builtins__["all"](is_tensor(x) for x in out):
                out_is_tensor[i_out] = True
                out_tensor_shapes[i_out] = [list(x.shape) for x in out]
                out_values[i_out] = type(out)([x.numpy() for x in out])
            else:
                out_values[i_out] = out

        save_record = __builtins__["any"](out_is_tensor)

    except Exception as e:
        save_record = False
        __builtins__["print"](
            f"Error when saving outputs of {op_name}:\n{e}", flush=True
        )
        traceback.print_exc()

    if not save_record:
        if not _aiinstr_allow_recursion:
            _aiinstr_is_instrumenting = False
        return ret

    """save the record"""
    try:
        import hashlib

        record = {
            "name": op_name,
            "args": {
                "name": arg_names,
                "is_pos": arg_is_pos,
                "is_tensor": arg_is_tensor,
                "tensor_shape": arg_tensor_shapes,
                "value": arg_values,
            },
            "outputs": {
                "name": out_names,
                "is_tensor": out_is_tensor,
                "tensor_shape": out_tensor_shapes,
                "value": out_values,
            },
        }
        list_to_hash = []
        for i_arg, arg_value in enumerate(arg_values):
            list_to_hash.append(f"{arg_names[i_arg]}:")
            if arg_is_tensor[i_arg]:
                list_to_hash.append(str(arg_tensor_shapes[i_arg]))
            else:
                list_to_hash.append(str(arg_value))
        list_to_hash.append("->")
        for i_out, out_value in enumerate(out_values):
            list_to_hash.append(f"{out_names[i_out]}:")
            if out_is_tensor[i_out]:
                list_to_hash.append(str(out_tensor_shapes[i_out]))
            else:
                list_to_hash.append(str(out_value))
        str_to_hash = ",".join(list_to_hash)
        hash_value = hashlib.md5(str_to_hash.encode()).hexdigest()
        out_file = os.path.join(out_api_dir, f"{hash_value}.pkl")
        if os.path.exists(out_file):
            __builtins__["print"](f"File {out_file} already exists", flush=True)
        else:
            with open(out_file, "wb") as f:
                pickle.dump(record, f)
    except Exception as e:
        __builtins__["print"](
            f"Error when saving record of {op_name}:\n{e}", flush=True
        )
        traceback.print_exc()

    if not _aiinstr_allow_recursion:
        _aiinstr_is_instrumenting = False
    # __builtins__["print"](f'before  return', flush=True)
    return ret


if int(os.getenv("INSTR", "0")):
    # if __name__ == "__main__":
    _aiinstr_name_2_op_params = get_all_supported_ops(
        "/home/jinjun/code/autoinf/autoinf/instrument/xla-compilable-ops.md"
    )
    __builtins__["print"](f"{len(_aiinstr_name_2_op_params) = }", flush=True)
    setattr(python.pywrap_tfe, "TFE_Py_FastPathExecute", wrapper)
    __builtins__["print"](f"Autoinf instrumentation done", flush=True)
    # embed()
