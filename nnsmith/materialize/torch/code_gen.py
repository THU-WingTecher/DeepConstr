from functools import partial
from typing import Any, Dict, Type

import torch

from nnsmith.abstract.dtype import DTYPE_GEN_INTS
from nnsmith.abstract.op import *
from nnsmith.autoinf import AutoInfOpBase
from nnsmith.materialize import framework_operator_impl
from nnsmith.materialize.torch.dialect import Flatten, Linear, TorchReduceSum

# Code generation for operators.

# core dialect + some future PyTorch-only Operators.
TORCH_REALIZABLE_OPS = (
    FULL_OPERATOR_SETS["core"] + FULL_OPERATOR_SETS["torch"] + [AutoInfOpBase]
)
ALL_TORCH_OPS: List[Type[AbsOpBase]] = []

operator_impl = partial(framework_operator_impl, TORCH_REALIZABLE_OPS, ALL_TORCH_OPS)


def braces_template(n_curly: int, *args, **kwargs):
    def strr(x):
        if isinstance(x, str):
            return f"'{x}'"
        return str(x)

    return (
        "("
        + ", ".join(
            ["{}"] * n_curly
            + list(map(strr, args))
            + [f"{k}={strr(v)}" for k, v in kwargs.items()]
        )
        + ")"
    )


@operator_impl(Constant)
def gen_code(op: Constant):
    raise NotImplementedError("Should not call gen_code for Constant.")


@operator_impl(ReLU)
def gen_code(op: ReLU):
    return "torch.nn.ReLU()", True


@operator_impl(GELU)
def gen_code(op: GELU):
    return "torch.nn.GELU()", True


@operator_impl(LeakyReLU)
def gen_code(op: LeakyReLU):
    return f"torch.nn.LeakyReLU({op.negative_slope})", True


@operator_impl(PReLU)
def gen_code(op: PReLU):
    return "torch.nn.PReLU()", True


@operator_impl(Sigmoid)
def gen_code(op: Sigmoid):
    return "torch.nn.Sigmoid()", True


@operator_impl(Sin)
def gen_code(op: Sin):
    return "torch.sin({})", False


@operator_impl(Cos)
def gen_code(op: Cos):
    return "torch.cos({})", False


@operator_impl(Asin)
def gen_code(op: Asin):
    return "torch.asin({})", False


@operator_impl(Acos)
def gen_code(op: Acos):
    return "torch.acos({})", False


@operator_impl(Tan)
def gen_code(op: Tan):
    return "torch.tan({})", False


@operator_impl(Atan)
def gen_code(op: Atan):
    return "torch.atan({})", False


# Abs
@operator_impl(Abs)
def gen_code(op: Abs):
    return "torch.abs({})", False


@operator_impl(Where)
def gen_code(op: Where):
    return "torch.where" + braces_template(len(op.input_like)), False


@operator_impl(Add)
def gen_code(op: Add):
    return "torch.add" + braces_template(len(op.input_like)), False


@operator_impl(Sub)
def gen_code(op: Sub):
    return "torch.sub" + braces_template(len(op.input_like)), False


@operator_impl(Mul)
def gen_code(op: Mul):
    return "torch.mul" + braces_template(len(op.input_like)), False


@operator_impl(Div)
def gen_code(op: Div):
    # TODO
    if op.input_like[0] is None :
        return "torch.div", False
    return "torch.div" + braces_template(
        len(op.input_like),
        rounding_mode="floor" if op.input_like[0].dtype in DTYPE_GEN_INTS else None,
    ), False
    return lambda up, down: torch.div(
        up,
        down,
        rounding_mode="floor" if DType.from_torch(up.dtype) in DTYPE_GEN_INTS else None,
    )


@operator_impl(Max)
def gen_code(op: Max):
    return "torch.max" + braces_template(len(op.input_like)), False


@operator_impl(Min)
def gen_code(op: Min):
    return "torch.min" + braces_template(len(op.input_like)), False


@operator_impl(Equal)
def gen_code(op: Equal):
    return "torch.eq" + braces_template(len(op.input_like)), False


@operator_impl(Greater)
def gen_code(op: Greater):
    return "torch.gt" + braces_template(len(op.input_like)), False


@operator_impl(Less)
def gen_code(op: Less):
    return "torch.lt" + braces_template(len(op.input_like)), False


@operator_impl(And)
def gen_code(op: And):
    return "torch.logical_and" + braces_template(len(op.input_like)), False


@operator_impl(Or)
def gen_code(op: Or):
    return "torch.logical_or" + braces_template(len(op.input_like)), False


@operator_impl(Xor)
def gen_code(op: Xor):
    return "torch.logical_xor" + braces_template(len(op.input_like)), False


@operator_impl(Pow)
def gen_code(op: Pow):
    return "torch.pow" + braces_template(len(op.input_like)), False


# Floor
@operator_impl(Floor)
def gen_code(op: Floor):
    return "torch.floor" + braces_template(len(op.input_like)), False


# Ceil
@operator_impl(Ceil)
def gen_code(op: Ceil):
    return "torch.ceil" + braces_template(len(op.input_like)), False


@operator_impl(Clip)
def gen_code(op: Clip):
    if op.input_like[0] is None :
        return "torch.clip", False
    if op.input_like[0].dtype in DTYPE_GEN_FLOATS:
        return (
            "torch.clip" + braces_template(len(op.input_like), min=-1.5, max=1.5),
            False,
        )
        return lambda x: torch.clip(x, -1.5, 1.5)
    else:
        return "torch.clip" + braces_template(len(op.input_like), min=-1, max=1), False
        return lambda x: torch.clip(x, -1, 1)


@operator_impl(Round)
def gen_code(op: Round):
    return "torch.round" + braces_template(len(op.input_like)), False
    return torch.round


@operator_impl(Sqrt)
def gen_code(op: Sqrt):
    return "torch.sqrt" + braces_template(len(op.input_like)), False
    return torch.sqrt


@operator_impl(Log2)
def gen_code(op: Log2):
    return "torch.log2" + braces_template(len(op.input_like)), False
    return torch.log2


@operator_impl(Neg)
def gen_code(op: Neg):
    return "torch.neg" + braces_template(len(op.input_like)), False
    return torch.neg


@operator_impl(Softmax)
def gen_code(op: Softmax):
    if op.input_like[0] is None :
        return "torch.nn.Softmax", True
    return f"torch.nn.Softmax(dim={op.dim})", True
    return torch.nn.Softmax(dim=op.dim)


@operator_impl(MaxPool2d)
def gen_code(op: MaxPool2d):
    if op.input_like[0] is None :
        return "torch.nn.MaxPool2d", True
    return (
        f"torch.nn.MaxPool2d(kernel_size=({op.kernel_h_size}, {op.kernel_w_size}), stride={op.stride}, padding={op.padding})",
        True,
    )
    return torch.nn.MaxPool2d(
        kernel_size=(op.kernel_h_size, op.kernel_w_size),
        stride=op.stride,
        padding=op.padding,
    )


@operator_impl(AvgPool2d)
def gen_code(op: AvgPool2d):
    if op.input_like[0] is None :
        return "torch.nn.AvgPool2d", True
    return (
        f"torch.nn.AvgPool2d(kernel_size=({op.kernel_h_size}, {op.kernel_w_size}), stride={op.stride}, padding={op.padding})",
        True,
    )
    return torch.nn.AvgPool2d(
        kernel_size=(op.kernel_h_size, op.kernel_w_size),
        stride=op.stride,
        padding=op.padding,
    )


@operator_impl(Slice)
def gen_code(op: Slice):
    if op.input_like[0] is None :
        return "Slice", False
    reg = op.extra_attrs["region"]

    shape = op.input_like[0].shape
    dim_s = shape[op.extra_attrs["axis"]]
    start, end = op.start, op.end
    if reg in ["left", "mid"]:
        start -= dim_s
    # actual end would be 0, which is not really 'left'
    if reg == "left" and end < dim_s and end != Slice.INT_MAX:
        end -= dim_s
    s = tuple(
        slice(None, None) if i != op.extra_attrs["axis"] else slice(start, end, op.step)
        for i in range(op.extra_attrs["ndims"])
    )

    return "{}" + f"[{s}]", False


@operator_impl(Pad)
def gen_code(op: Pad):
    if op.input_like[0] is None :
        return "torch.nn.functional.pad", False
    if op.extra_attrs["type"] == "constant":
        # 0 easily cause division by zero...
        # 1 easily cause false positives (sqrt(1) = 0.99999... != 1 in ORT, so floor(sqrt(1))=0)
        return (
            "torch.nn.functional.pad"
            + braces_template(1, pad=op.padding_list, mode="constant", value=0.5),
            False,
        )
        return lambda x: torch.nn.functional.pad(
            x, op.padding_list, "constant", value=0.5
        )
    elif op.extra_attrs["type"] == "replicate" or op.extra_attrs["type"] == "reflect":
        return (
            "torch.nn.functional.pad"
            + braces_template(1, pad=op.padding_list, mode=op.extra_attrs["type"]),
            False,
        )
        return lambda x: torch.nn.functional.pad(
            x, op.padding_list, op.extra_attrs["type"]
        )


@operator_impl(Expand)
def gen_code(op: Expand):
    if op.input_like[0] is None :
        return "torch.Tensor.expand", False
    return (
        "{}.expand" + braces_template(0, *op.type_transfer(op.input_like)[0].shape),
        False,
    )


@operator_impl(BatchNorm2d)
def gen_code(op: BatchNorm2d):
    if op.input_like[0] is None :
        return "torch.nn.BatchNorm2d", True
    return f"torch.nn.BatchNorm2d(num_features={op.nfeat})", True
    return torch.nn.BatchNorm2d(num_features=op.nfeat)


@operator_impl(Conv1d)
def gen_code(op: Conv1d):
    if op.input_like[0] is None :
        return "torch.nn.Conv1d", True
    return (
        f"torch.nn.Conv1d(in_channels={op.in_channels}, out_channels={op.out_channels}, kernel_size={op.kernel_size}, stride={op.stride}, padding={op.padding}, dilation={op.dilation})",
        True,
    )
    return torch.nn.Conv1d(
        in_channels=op.in_channels,
        out_channels=op.out_channels,
        kernel_size=op.kernel_size,
        stride=op.stride,
        padding=op.padding,
        dilation=op.dilation,
    )


@operator_impl(NCHWConv2d)
def gen_code(op: NCHWConv2d):
    if op.input_like[0] is None :
        return "torch.nn.Conv2d", True
    return (
        f"torch.nn.Conv2d(in_channels={op.in_channels}, out_channels={op.out_channels}, kernel_size=({op.kernel_h_size}, {op.kernel_w_size}), stride={op.stride}, padding={op.padding}, dilation=({op.dilation_h}, {op.dilation_w}))",
        True,
    )
    return torch.nn.Conv2d(
        op.in_channels,
        op.out_channels,
        kernel_size=(op.kernel_h_size, op.kernel_w_size),
        stride=op.stride,
        padding=op.padding,
        dilation=(op.dilation_h, op.dilation_w),
    )


@operator_impl(Reshape)
def gen_code(op: Reshape):
    if op.input_like[0] is None :
        return "torch.Tensor.reshape", False
    return "{}.reshape" + braces_template(0, *op.target_shape), False
    return lambda x: x.reshape(*op.target_shape)


@operator_impl(Flatten)
def gen_code(op: Flatten):
    if op.input_like[0] is None :
        return "torch.Tensor.flatten", False
    return "{}.flatten()", False
    return lambda x: x.flatten()


@operator_impl(Transpose)
def gen_code(op: Transpose):
    if op.input_like[0] is None :
        return "torch.Tensor.transpose", False
    dim0, dim1 = op._init_swap_dims(op.input_like[0].shape)
    return "{}.transpose" + braces_template(0, dim0, dim1), False

    def f(x: torch.Tensor):
        dim0, dim1 = op._init_swap_dims(list(x.shape))
        return x.transpose(dim0, dim1)

    return f


# NearestInterp
@operator_impl(NearestInterp)
def gen_code(op: NearestInterp):
    if op.input_like[0] is None :
        return "torch.nn.functional.interpolate", False
    return (
        "torch.nn.functional.interpolate"
        + braces_template(1, size=op.size, mode="nearest"),
        False,
    )
    return lambda x: torch.nn.functional.interpolate(x, size=op.size, mode="nearest")


# LinearInterp
@operator_impl(LinearInterp)
def gen_code(op: LinearInterp):
    if op.input_like[0] is None :
        return "torch.nn.functional.interpolate", False
    return (
        "torch.nn.functional.interpolate"
        + braces_template(1, size=op.size, mode="linear"),
        False,
    )
    return lambda x: torch.nn.functional.interpolate(x, size=op.size, mode="linear")


# BilinearInterp
@operator_impl(BilinearInterp)
def gen_code(op: BilinearInterp):
    if op.input_like[0] is None :
        return "torch.nn.functional.interpolate", False
    return (
        "torch.nn.functional.interpolate"
        + braces_template(1, size=op.size, mode="bilinear"),
        False,
    )
    return lambda x: torch.nn.functional.interpolate(x, size=op.size, mode="bilinear")


@operator_impl(BicubicInterp)
def gen_code(op: BicubicInterp):
    if op.input_like[0] is None :
        return "torch.nn.functional.interpolate", False
    return (
        "torch.nn.functional.interpolate"
        + braces_template(1, size=op.size, mode="bicubic"),
        False,
    )
    return lambda x: torch.nn.functional.interpolate(x, size=op.size, mode="bicubic")


# TrilinearInterp
@operator_impl(TrilinearInterp)
def gen_code(op: TrilinearInterp):
    if op.input_like[0] is None :
        return "torch.nn.functional.interpolate", False
    return (
        "torch.nn.functional.interpolate"
        + braces_template(1, size=op.size, mode="trilinear"),
        False,
    )
    return lambda x: torch.nn.functional.interpolate(x, size=op.size, mode="trilinear")


@operator_impl(Squeeze)
def gen_code(op: Squeeze):
    if op.input_like[0] is None :
        return "torch.Tensor.squeeze", False
    if op.extra_attrs["reduce_dim"] is not None:
        return "{}.squeeze" + braces_template(0, op.extra_attrs["reduce_dim"]), False
        return lambda x: x.squeeze(op.extra_attrs["reduce_dim"])
    else:
        return "{}.squeeze()", False
        return lambda x: x.squeeze()


@operator_impl(TorchReduceSum)
def gen_code(op: TorchReduceSum):
    if op.input_like[0] is None :
        return "torch.Tensor.sum", False
    if op.extra_attrs["reduce_dim"] is not None:
        return "{}.sum" + braces_template(0, op.extra_attrs["reduce_dim"]), False
        return lambda x: x.sum(op.extra_attrs["reduce_dim"])
    return "{}.sum()", False
    return lambda x: x.sum()


# ReduceMin
@operator_impl(ReduceMin)
def gen_code(op: ReduceMin):
    if op.input_like[0] is None :
        return "torch.Tensor.min", False

    if op.extra_attrs["reduce_dim"] is not None:
        return (
            "{}.min" + braces_template(0, op.extra_attrs["reduce_dim"]) + ".values",
            False,
        )
        return lambda x: x.min(op.extra_attrs["reduce_dim"]).values
    return "{}.min()", False
    return lambda x: x.min()


# ReduceMax
@operator_impl(ReduceMax)
def gen_code(op: ReduceMax):
    if op.input_like[0] is None :
        return "torch.Tensor.max", False
    if op.extra_attrs["reduce_dim"] is not None:
        return (
            "{}.max" + braces_template(0, op.extra_attrs["reduce_dim"]) + ".values",
            False,
        )
        return lambda x: x.max(op.extra_attrs["reduce_dim"]).values
    return "{}.max()", False
    return lambda x: x.max()


# ReduceMean
@operator_impl(ReduceMean)
def gen_code(op: ReduceMean):
    if op.input_like[0] is None :
        return "torch.Tensor.mean", False
    if op.extra_attrs["reduce_dim"] is not None:
        return "{}.mean" + braces_template(0, op.extra_attrs["reduce_dim"]), False
        return lambda x: x.mean(op.extra_attrs["reduce_dim"])
    return "{}.mean()", False
    return lambda x: x.mean()


# ArgMin
@operator_impl(ArgMin)
def gen_code(op: ArgMin):
    if op.input_like[0] is None :
        return "torch.Tensor.argmin", False
    if op.extra_attrs["reduce_dim"] is not None:
        return "{}.argmin" + braces_template(0, op.extra_attrs["reduce_dim"]), False
        return lambda x: x.argmin(op.extra_attrs["reduce_dim"])
    return "{}.argmin()", False
    return lambda x: x.argmin()


# ArgMax
@operator_impl(ArgMax)
def gen_code(op: ArgMax):
    if op.input_like[0] is None :
        return "torch.Tensor.argmax", False
    if op.extra_attrs["reduce_dim"] is not None:
        return "{}.argmax" + braces_template(0, op.extra_attrs["reduce_dim"]), False
        return lambda x: x.argmax(op.extra_attrs["reduce_dim"])
    return "{}.argmax()", False
    return lambda x: x.argmax()


# Tril
@operator_impl(Tril)
def gen_code(op: Tril):
    if op.input_like[0] is None :
        return "torch.Tensor.tril", False
    return "{}.tril" + braces_template(0, op.diagonal), False
    return lambda x: x.tril(op.diagonal)


# Triu
@operator_impl(Triu)
def gen_code(op: Triu):
    if op.input_like[0] is None :
        return "torch.Tensor.triu", False
    return "{}.triu" + braces_template(0, op.diagonal), False
    return lambda x: x.triu(op.diagonal)


# Linear
@operator_impl(Linear)
def gen_code(op: Linear):
    if op.input_like[0] is None :
        return "torch.nn.Linear", True
    return f"torch.nn.Linear(in_features={op.ifeat}, out_features={op.ofeat})", True
    return torch.nn.Linear(in_features=op.ifeat, out_features=op.ofeat)


@operator_impl(Concat)
def gen_code(op: Concat):
    if op.input_like[0] is None :
        return "torch.cat", False
    axis = op.extra_attrs["axis"]
    return (
        "torch.cat" + "([" + ", ".join(["{}"] * len(op.input_like)) + f"], dim={axis})",
        False,
    )
    return lambda *args: torch.cat(args, dim=axis)


@operator_impl(Cast)
def gen_code(op: Cast):
    if op.input_like[0] is None :
        return "torch.Tensor.to", False
    return "{}.to" + braces_template(0, op.extra_attrs["to"].torch()), False
    return lambda x: x.to(dtype=op.extra_attrs["to"].torch())


@operator_impl(MatMul)
def gen_code(op: MatMul):
    if op.input_like[0] is None :
        return "torch.matmul", False
    return "torch.matmul" + braces_template(len(op.input_like)), False
    return torch.matmul


@operator_impl(AutoInfOpBase)
def gen_code(op: AutoInfOpBase):
    raise NotImplementedError("Should not call gen_code for AutoInfOpBase")
    return op.inst.materialize(eval(op.inst.name), op.attrs)
