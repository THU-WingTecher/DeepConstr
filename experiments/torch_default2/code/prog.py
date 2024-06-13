from typing import Dict
# import pickle
import torch
import torch.nn as nn
# from copy import deepcopy
# print("torch version: ",torch.__version__)


# inputs: Dict[str, torch.Tensor] = {'v0_0':torch.Tensor([1,2])}# 'v1_0': ..., 'v2_0': ...

# class Model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # nn.parameter.Parameter objects (with comments for shapes)

#         # nn.Module objects


#     def forward(self):
#         # v0_0: [1, 4], torch.float32
#         # return v0_0

# model = Model() #.to(torch.device("cpu"))
# copied = deepcopy(inputs)
# for k, v in inputs.items():
#     inputs[k] = v #.to(torch.device("cpu"))
# print('==== Eager mode ====')
# ret_eager = model(**inputs)

# print('==== JIT mode ====')
# exported = torch.jit.trace(model, example_kwarg_inputs=inputs)
# exported = torch.jit.optimize_for_inference(exported)
# ret_exported = exported(**copied)

# print('==== Check ====')
# for r1, r2 in zip(ret_eager, ret_exported):
#     if not torch.allclose(r1, r2, rtol=1e-2, atol=1e-3, equal_nan=True):
#         print("r1: ",r1,"r2: ",r2)
#         raise ValueError("Tensors are different.")
# print('OK!')



