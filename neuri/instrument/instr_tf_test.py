import logging
import os

import tensorflow as tf  # type: ignore

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").disabled = True

from tensorflow import config, is_tensor, raw_ops

# instrumentation
for gpu in config.list_physical_devices("GPU"):
    config.experimental.set_memory_growth(gpu, True)


def get_all_supported_ops(ops_file_path: str):
    with open(ops_file_path, "r") as f:
        lines = f.readlines()
    skip_in_kws = []
    name_2_op = {}
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
            name_2_op[op_name] = func
    # end for line
    return name_2_op


import inspect as builtin_inspect
import os
import sys

import tensorflow.python as python

is_instrumenting = False
allow_recursion = False
_this_tf_mod = sys.modules[__name__]


def instrument(name, raw_ops_func, out_dir):

    import functools

    # name: tf.raw_ops.Abs
    last_name = name.split(".")[-1]
    name_for_path = last_name  # name.replace(".", "_")

    module_obj = eval(raw_ops_func.__module__[len("tensorflow.") :])
    func_name = raw_ops_func.__name__
    inner_func = getattr(module_obj, func_name)
    sigs = builtin_inspect.signature(inner_func)

    @functools.wraps(inner_func)
    def wrapper(*args, **kwargs):
        """convert all args to kwargs"""
        for i_param, (param_name, param) in enumerate(sigs.parameters.items()):
            if i_param < len(args):
                kwargs[param_name] = args[i_param]
            else:
                break
        args = []

        global is_instrumenting
        global allow_recursion
        if not allow_recursion and is_instrumenting:
            return inner_func(*args, **kwargs)

        out_api_dir = os.path.join(out_dir, name_for_path)
        os.makedirs(out_api_dir, exist_ok=True)
        if len(os.listdir(out_api_dir)) > 1_000_000:
            return inner_func(*args, **kwargs)

        import pickle
        import traceback

        if not allow_recursion:
            is_instrumenting = True
        save_record = True
        """save input args"""
        try:
            arg_names = [f"arg_{i}" for i in __builtins__["range"](len(args))] + sorted(
                kwargs.keys()
            )
            if "@@@" in arg_names:
                save_record = False
            else:
                arg_is_pos = [True] * len(args) + [False] * len(kwargs)
                arg_is_tensor = [False] * len(arg_is_pos)
                arg_tensor_shapes = [None] * len(arg_is_pos)
                arg_values = [None] * len(arg_is_pos)
                for i_arg, arg_name in enumerate(arg_names):
                    arg = args[i_arg] if i_arg < len(args) else kwargs[arg_name]
                    # if isinstance(arg, Tensor) and not hasattr(arg, "ndim"):
                    if is_tensor(arg) and len(arg.shape) == 0:
                        arg = arg.numpy().item()
                        if i_arg < len(args):
                            args[i_arg] = arg
                        else:
                            kwargs[arg_name] = arg
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

                save_record = any(arg_is_tensor)

        except Exception as e:
            save_record = False
            __builtins__["print"](
                f"Error when saving inputs of {inner_func}:\n{e}", flush=True
            )
            traceback.print_exc()

        """execute the function"""
        try:
            ret = inner_func(*args, **kwargs)
        except Exception as e:
            if not allow_recursion:
                is_instrumenting = False
            __builtins__["print"](f"Error when exec {inner_func}:\n{e}", flush=True)
            traceback.print_exc()
            # embed()
            raise e

        if not save_record:
            if not allow_recursion:
                is_instrumenting = False
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
                f"Error when saving outputs of {inner_func}:\n{e}", flush=True
            )
            traceback.print_exc()

        if not save_record:
            if not allow_recursion:
                is_instrumenting = False
            return ret

        """save the record"""
        try:
            import hashlib

            record = {
                "name": name,
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
                f"Error when saving record of {inner_func}:\n{e}", flush=True
            )
            traceback.print_exc()

        if not allow_recursion:
            is_instrumenting = False
        return ret

    # end def wrapper
    setattr(wrapper, "func_name", name)
    setattr(module_obj, func_name, wrapper)
    setattr(raw_ops, last_name, wrapper)
    exported_names = list(
        getattr(inner_func, "_tf_api_names", [])
    )  # ('math.acosh', 'acosh')
    for exp_name in exported_names:
        ori_func = eval(exp_name)
        last_dot_pos = exp_name.rfind(".")
        if last_dot_pos == -1:
            exp_mod = _this_tf_mod
        else:
            exp_mod = getattr(_this_tf_mod, exp_name[:last_dot_pos])
        ori_func_name = exp_name[last_dot_pos + 1 :]
        setattr(exp_mod, ori_func_name, wrapper)

    # from IPython import embed; embed()


if int(os.getenv("INSTR", "0")):
    # if __name__ == "__main__":
    # out_dir = "/home/jinjun/data/autoinf/tf_hists_0114_0"
    out_dir = "tf_hists"
    name_2_op = get_all_supported_ops(
        "/home/jinjun/code/autoinf/autoinf/instrument/xla-compilable-ops.md"
    )
    # print(f"{len(name_2_op) = }", flush=True)
    __builtins__["print"](f"{len(name_2_op) = }", flush=True)
    for name, func in name_2_op.items():
        instrument(name, func, out_dir)

    # print("Done", flush=True)
    __builtins__["print"](f"Autoinf instrumentation done", flush=True)
