import os
import pickle
from functools import partial

from neuri.inference.const import *
from neuri.instrument.op import OpInstance
from neuri.instrument.utils import get_ret_list, tensors_from_numpys


def input_validity_test(inst: OpInstance, inputs, library: str):
    input_symb_2_value = dict()
    if isinstance(inputs, dict):
        input_symb_2_value = inputs
    else:
        for i, value in enumerate(inputs):
            input_symb_2_value[f"s{i}"] = value
    input_list, output_list = list(input_symb_2_value.values()), None
    if library == "torch":
        import torch

        tensor_from_numpy = partial(
            tensors_from_numpys, tensor_from_numpy=torch.from_numpy
        )
        tensor_checker = torch.is_tensor
    elif library == "tf":
        import tensorflow as tf

        tensor_from_numpy = partial(
            tensors_from_numpys, tensor_from_numpy=tf.convert_to_tensor
        )
        tensor_checker = tf.is_tensor
    else:
        raise NotImplementedError
    try:
        args, kwargs = inst.input_args(input_symb_2_value, tensor_from_numpy)
        ret = eval(inst.name)(*args, **kwargs)
    except:
        ret = "Exception"
    if not (isinstance(ret, str) and ret == "Exception"):
        try:
            ret_list = [
                r.cpu().numpy() if tensor_checker(r) else r for r in get_ret_list(ret)
            ]
            output_list = list(inst.output_info(ret_list)[0].values())
        except:
            output_list = None
    return input_list, output_list
