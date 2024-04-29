import os

import torch


def get_all_supported_ops():
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
    tensor_methods = {}
    torch_funcs = {}
    tot_doc_lines = len(ops_doc_lines)
    i_line = 0
    while i_line < tot_doc_lines:
        line = ops_doc_lines[i_line].strip()
        if not line:
            i_line += 1
            continue
        # read wanted lines
        if __builtins__["any"](
            [line.startswith(skip_kw) for skip_kw in skip_startwith_kws]
        ) or __builtins__["any"]([skip_kw in line for skip_kw in skip_in_kws]):
            i_line += 1
            continue
        is_tensor_method = line.startswith("Tensor.")
        is_torch_func = line.startswith("torch.")
        if is_tensor_method or is_torch_func:
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
                i_line += 1
                continue
            obj = eval(op_name)
            if callable(obj):
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
                if is_torch_func:
                    torch_funcs[op_name] = obj
                else:
                    tensor_methods[op_name] = obj
            else:
                print(f"  (Ignored: {obj = } from {op_name = } is not callable")
        # end if
        i_line += 1
    # end while
    return tensor_methods, torch_funcs


"""
save records of an API to a folder
API_name/hash_value.pkl

{
    'name': 'torch.add',
    'args': {
        'name': ['arg_0', 'arg_1', 'alpha'],
        'is_pos': [True, True, False],
        'is_tensor': [True, True, False],
        'tensor_dtype': ['torch.float32', 'torch.float32', None],
        'tensor_shape': [[2, 3], [2, 3], None],
        'value': [np.array(...), np.array(...), 1.0],
    },
    'outputs': {
        'name': ['o_0'],
        'is_tensor': [True],
        'tensor_dtype': ['torch.float32'],
        'tensor_shape': [[2, 3]],
        'value': [np.array(...)],
    },
}
"""

is_instrumenting = False


def instrument(name, func):
    name_for_path = name.replace(".", "_")

    def wrapper(*args, **kwargs):
        global is_instrumenting
        if is_instrumenting:
            return func(*args, **kwargs)

        out_dir = "/coconut/autoinf/hists_1123_0"
        # out_dir = "hists"
        out_api_dir = os.path.join(out_dir, name_for_path)
        os.makedirs(out_api_dir, exist_ok=True)
        if len(os.listdir(out_api_dir)) > 1_000_000:
            return func(*args, **kwargs)

        import pickle
        import traceback

        is_instrumenting = True
        save_record = True
        """save input args"""
        try:
            arg_names = [f"arg_{i}" for i in __builtins__["range"](len(args))] + sorted(
                kwargs.keys()
            )
            if "out" in arg_names:
                save_record = False
            else:
                arg_is_pos = [True] * len(args) + [False] * len(kwargs)

                arg_is_tensor = [False] * len(arg_is_pos)
                arg_tensor_dtypes = [None] * len(arg_is_pos)
                arg_tensor_shapes = [None] * len(arg_is_pos)
                arg_values = [None] * len(arg_is_pos)
                for i_arg, arg_name in enumerate(arg_names):
                    arg = args[i_arg] if i_arg < len(args) else kwargs[arg_name]
                    if isinstance(arg, torch.Tensor) and arg.ndim == 0:
                        arg = arg.item()
                        if i_arg < len(args):
                            args[i_arg] = arg
                        else:
                            kwargs[arg_name] = arg
                    if isinstance(arg, torch.Tensor):
                        arg_is_tensor[i_arg] = True
                        arg_tensor_dtypes[i_arg] = str(arg.dtype)
                        arg_tensor_shapes[i_arg] = list(arg.shape)
                        arg_values[i_arg] = arg.detach().cpu().numpy()
                    elif (
                        isinstance(arg, list) or isinstance(arg, tuple)
                    ) and __builtins__["all"](isinstance(x, torch.Tensor) for x in arg):
                        arg_is_tensor[i_arg] = True
                        arg_tensor_dtypes[i_arg] = [str(x.dtype) for x in arg]
                        arg_tensor_shapes[i_arg] = [list(x.shape) for x in arg]
                        arg_values[i_arg] = type(arg)(
                            [x.detach().cpu().numpy() for x in arg]
                        )
                    else:
                        arg_values[i_arg] = arg

                save_record = __builtins__["any"](arg_is_tensor)

        except Exception as e:
            save_record = False
            print(f"Error when saving inputs of {func}:\n{e}", flush=True)
            traceback.print_exc()

        """execute the function"""
        try:
            ret = func(*args, **kwargs)
        except Exception as e:
            is_instrumenting = False
            print(f"Error when exec {func}:\n{e}", flush=True)
            traceback.print_exc()
            raise e

        if not save_record:
            is_instrumenting = False
            return ret

        """save outputs"""
        try:
            if isinstance(ret, tuple):
                ret_list = list(ret)
            elif not isinstance(ret, list):
                ret_list = [ret]
            else:
                ret_list = ret

            out_names = [f"o_{i}" for i in __builtins__["range"](len(ret_list))]
            out_is_tensor = [False] * len(ret_list)
            out_tensor_dtypes = [None] * len(ret_list)
            out_tensor_shapes = [None] * len(ret_list)
            out_values = [None] * len(ret_list)
            for i_out, out_name in enumerate(out_names):
                out = ret_list[i_out]
                if isinstance(out, torch.Tensor) and out.ndim == 0:
                    out = out.item()
                if isinstance(out, torch.Tensor):
                    out_is_tensor[i_out] = True
                    out_tensor_dtypes[i_out] = str(out.dtype)
                    out_tensor_shapes[i_out] = list(out.shape)
                    out_values[i_out] = out.detach().cpu().numpy()
                elif (isinstance(out, list) or isinstance(out, tuple)) and __builtins__[
                    "all"
                ](isinstance(x, torch.Tensor) for x in out):
                    out_is_tensor[i_out] = True
                    out_tensor_dtypes[i_out] = [str(x.dtype) for x in out]
                    out_tensor_shapes[i_out] = [list(x.shape) for x in out]
                    out_values[i_out] = type(out)(
                        [x.detach().cpu().numpy() for x in out]
                    )
                else:
                    out_values[i_out] = out

            save_record = __builtins__["any"](out_is_tensor)

        except Exception as e:
            save_record = False
            print(f"Error when saving outputs of {func}:\n{e}", flush=True)
            traceback.print_exc()

        if not save_record:
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
                    "tensor_dtype": arg_tensor_dtypes,
                    "tensor_shape": arg_tensor_shapes,
                    "value": arg_values,
                },
                "outputs": {
                    "name": out_names,
                    "is_tensor": out_is_tensor,
                    "tensor_dtype": out_tensor_dtypes,
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
                print(f"File {out_file} already exists", flush=True)
            else:
                with open(out_file, "wb") as f:
                    pickle.dump(record, f)
        except Exception as e:
            print(f"Error when saving record of {func}:\n{e}", flush=True)
            traceback.print_exc()

        is_instrumenting = False
        return ret

    # end def wrapper
    setattr(wrapper, "func_name", name)

    last_dot_pos = name.rfind(".")
    module_obj = eval(name[:last_dot_pos])
    func_name = name[last_dot_pos + 1 :]
    setattr(module_obj, func_name, wrapper)


# if int(os.getenv("INSTR", "0")):
if __name__ == "__main__":

    tensor_methods, torch_funcs = get_all_supported_ops()
    print(f"{len(tensor_methods) = }")
    print(f"{len(torch_funcs) = }")

    for name, func in tensor_methods.items():
        instrument(name, func)
    for name, func in torch_funcs.items():
        instrument(name, func)

    # from hanging_threads import start_monitoring

    # start_monitoring(seconds_frozen=10, test_interval=100)
