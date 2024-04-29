from functools import partial
from typing import List, Type

import tensorflow as tf  # type: ignore
from keras import layers
import os
from nnsmith.abstract.op import *
from nnsmith.materialize import framework_operator_impl
from nnsmith.materialize.tensorflow.dialect import *

# core dialect + some future TensorFlow-only Operators.
TF_REALIZABLE_OPS = FULL_OPERATOR_SETS["core"] + FULL_OPERATOR_SETS["tensorflow"]
ALL_TF_OPS: List[Type[AbsOpBase]] = []

operator_impl = partial(framework_operator_impl, TF_REALIZABLE_OPS, ALL_TF_OPS)

"""Implement TensorFlow forward Callables for operator classes"""
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

# @operator_impl(Constant)
# def gen_code(op: Constant):
#     dtype = op.abs_tensor.dtype.tensorflow()
#     data = tf.cast(tf.random.normal(op.abs_tensor.shape), dtype)
#     return lambda: tf.constant(data, dtype=dtype)

@operator_impl(ReLU)
def gen_code(op: ReLU):
    return "tf.nn.relu" + braces_template(len(op.input_like)), False
    return tf.nn.relu

@operator_impl(GELU)
def gen_code(op: GELU):
    return "tf.nn.gelu" + braces_template(len(op.input_like)), False
    return tf.nn.gelu

@operator_impl(LeakyReLU)
def gen_code(op: LeakyReLU):
    return "tf.nn.leaky_relu" + braces_template(len(op.input_like)), False
    return tf.nn.leaky_relu


@operator_impl(Sigmoid)
def gen_code(op: Sigmoid):
    return "tf.sigmoid" + braces_template(len(op.input_like)), False
    return tf.sigmoid


@operator_impl(Cos)
def gen_code(op: Cos):
    return "tf.cos" + braces_template(len(op.input_like)), False
    return tf.cos


@operator_impl(Asin)
def gen_code(op: Asin):
    return "tf.asin" + braces_template(len(op.input_like)), False
    return tf.asin


@operator_impl(Acos)
def gen_code(op: Acos):
    return "tf.acos" + braces_template(len(op.input_like)), False
    return tf.acos

@operator_impl(Tan)
def gen_code(op: Tan):
    return "tf.tan" + braces_template(len(op.input_like)), False
    return tf.tan

@operator_impl(Atan)
def gen_code(op: Atan):
    return "tf.atan" + braces_template(len(op.input_like)), False
    return tf.atan


@operator_impl(Abs)
def gen_code(op: Abs):
    return "tf.abs" + braces_template(len(op.input_like)), False
    return tf.abs


@operator_impl(Where)
def gen_code(op: Where):
    return "tf.where" + braces_template(len(op.input_like)), False
    return tf.where


@operator_impl(Add)
def gen_code(op: Add):
    return "tf.add" + braces_template(len(op.input_like)), False
    return tf.add


@operator_impl(Sub)
def gen_code(op: Sub):
    return "tf.math.subtract" + braces_template(len(op.input_like)), False
    return tf.math.subtract


@operator_impl(Mul)
def gen_code(op: Mul):
    return "tf.multiply" + braces_template(len(op.input_like)), False
    return tf.multiply


@operator_impl(Div)
def gen_code(op: Div):
    return "tf.divide" + braces_template(len(op.input_like)), False
    return tf.divide


@operator_impl(Max)
def gen_code(op: Max):
    return "tf.maximum" + braces_template(len(op.input_like)), False
    return tf.maximum


@operator_impl(Min)
def gen_code(op: Min):
    return "tf.minimum" + braces_template(len(op.input_like)), False
    return tf.minimum


@operator_impl(Equal)
def gen_code(op: Equal):
    return "tf.equal" + braces_template(len(op.input_like)), False
    return tf.equal


@operator_impl(Greater)
def gen_code(op: Greater):
    return "tf.greater" + braces_template(len(op.input_like)), False
    return tf.greater


@operator_impl(Less)
def gen_code(op: Less):
    return "tf.less" + braces_template(len(op.input_like)), False
    return tf.less


@operator_impl(And)
def gen_code(op: And):
    return "tf.logical_and" + braces_template(len(op.input_like)), False
    return tf.logical_and


@operator_impl(Or)
def gen_code(op: Or):
    return "tf.logical_or" + braces_template(len(op.input_like)), False
    return tf.logical_or


@operator_impl(Xor)
def gen_code(op: Xor):
    return "tf.math.logical_xor" + braces_template(len(op.input_like)), False
    return tf.math.logical_xor


@operator_impl(Pow)
def gen_code(op: Pow):
    return "tf.pow" + braces_template(len(op.input_like)), False
    return tf.pow


@operator_impl(Floor)
def gen_code(op: Floor):
    return "tf.floor" + braces_template(len(op.input_like)), False
    return tf.floor


@operator_impl(Ceil)
def gen_code(op: Ceil):
    return "tf.math.ceil" + braces_template(len(op.input_like)), False
    return tf.math.ceil


@operator_impl(Clip)
def gen_code(op: Clip):
    if op.input_like[0] is None or op.input_like[0].dtype in DTYPE_GEN_FLOATS:
        return "tf.clip_by_value" + braces_template(len(op.input_like), -1.5, 1.5), False
        return lambda x: tf.clip_by_value(x, -1.5, 1.5)
    else:
        return "tf.clip_by_value" + braces_template(len(op.input_like), -1, 1), False
        return lambda x: tf.clip_by_value(x, -1, 1)


@operator_impl(Round)
def gen_code(op: Round):
    return "tf.round" + braces_template(len(op.input_like)), False
    return tf.round


@operator_impl(Sqrt)
def gen_code(op: Sqrt):
    return "tf.sqrt" + braces_template(len(op.input_like)), False
    return tf.sqrt


@operator_impl(Log2)
def gen_code(op: Log2):
    return "tf.experimental.numpy.log2" + braces_template(len(op.input_like)), False
    return tf.experimental.numpy.log2


@operator_impl(Neg)
def gen_code(op: Neg):
    return "tf.negative" + braces_template(len(op.input_like)), False
    return tf.negative


@operator_impl(Softmax)
def gen_code(op: Softmax):
    return "tf.nn.softmax" + braces_template(len(op.input_like), 
                                            # logits=op.input_like[0].shape,
                                            axis=op.dim), False

    return lambda x: tf.nn.softmax(
        logits=tf.ensure_shape(x, op.input_like[0].shape),
        axis=op.dim,
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
    idx = tuple(
        slice(None, None) if i != op.extra_attrs["axis"] else slice(start, end, op.step)
        for i in range(op.extra_attrs["ndims"])
    )
    return "{}" + f"[{idx}]", False
    return lambda x: x[idx]


@operator_impl(BatchNorm2d)
def gen_code(op: BatchNorm2d):
    if op.input_like[0] is None : 
        return "tf.keras.layers.BatchNormalization", False
    return "tf.keras.layers.BatchNormalization" + \
        braces_template(len(op.input_like), 
                        axis=1, 
                        dtype = repr(op.input_like[0].dtype.tensorflow()),
                        autocast=False), False 
    return layers.BatchNormalization(
        axis=1,
        dtype=op.input_like[0].dtype.tensorflow(),
        autocast=False,
    )  # NCHW


@operator_impl(Reshape)
def gen_code(op: Reshape):
    return "tf.reshape" + braces_template(len(op.input_like), op.target_shape), False
    def _reshape(x):
        return tf.reshape(x, op.target_shape)

    return _reshape


@operator_impl(Transpose)
def gen_code(op: Transpose):
    if op.input_like[0] is None : 
        return "tf.transpose", False
    aten = op.input_like[0]
    dim0, dim1 = op._init_swap_dims(aten.shape)
    perm = list(range(aten.ndims))
    perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
    return "tf.transpose" + braces_template(len(op.input_like),  perm=perm), False

    def _transpose(x: tf.Tensor):
        aten = op.input_like[0]
        dim0, dim1 = op._init_swap_dims(aten.shape)
        perm = list(range(aten.ndims))
        perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
        return tf.transpose(x, perm=perm)

    return _transpose


@operator_impl(Dense)
def gen_code(op: Dense):
    if op.input_like[0] is None :
        return "tf.keras.layers.Dense", True
    return "tf.keras.layers.Dense" + \
        braces_template(0, units=op.ofeat, 
                        dtype=repr(op.input_like[0].dtype.tensorflow()), 
                        autocast=False), True
    return layers.Dense(
        units=op.ofeat, dtype=op.input_like[0].dtype.tensorflow(), autocast=False
    )


@operator_impl(LocalRespNorm)
def gen_code(op: LocalRespNorm):
    if op.input_like[0] is None :
        return "tf.raw_ops.LRN", False
    return "tf.raw_ops.LRN" + \
        braces_template(0, input="{}",
                        depth_radius=op.depth_radius,
                        bias=op.extra_attrs["bias"],
                        alpha=op.extra_attrs["alpha"],
                        beta=op.extra_attrs["beta"],), False 
    def _lrn(x):
        return tf.raw_ops.LRN(
            input=x,
            depth_radius=op.depth_radius,
            bias=op.extra_attrs["bias"],
            alpha=op.extra_attrs["alpha"],
            beta=op.extra_attrs["beta"],
        )

    return _lrn


@operator_impl(NHWCConv2d)
def gen_code(op: NHWCConv2d):
    if op.input_like[0] is None :
        return "tf.nn.conv2d", False
    return "tf.nn.conv2d" + \
        braces_template(len(op.input_like),
                        strides=op.stride,
                        padding='"'+op.extra_attrs["padding"]+'"',
                        dilations=(op.dilation_h, op.dilation_w),), False 
    return lambda input, filters: tf.nn.conv2d(
        input=input,
        filters=filters,
        strides=op.stride,
        padding=op.extra_attrs["padding"],
        dilations=(op.dilation_h, op.dilation_w),
    )


@operator_impl(NHWCAtrousConv2d)
def gen_code(op: NHWCAtrousConv2d):
    if op.input_like[0] is None :
        return "tf.nn.atrous_conv2d", False
    return "tf.nn.atrous_conv2d" + \
        braces_template(len(op.input_like),
                        rate=op.rate,
                        padding='"'+op.extra_attrs["padding"]+'"',), False 
    return lambda value, filters: tf.nn.atrous_conv2d(
        value=value,
        filters=filters,
        rate=op.rate,
        padding=op.extra_attrs["padding"],
    )


@operator_impl(NHWCDepthwiseConv2d)
def gen_code(op: NHWCDepthwiseConv2d):
    if op.input_like[0] is None :
        return "tf.nn.depthwise_conv2d", False
    return "tf.nn.depthwise_conv2d" + \
        braces_template(len(op.input_like),
                        strides=(1, op.stride, op.stride, 1),
                        padding='"'+op.extra_attrs["padding"]+'"',
                        dilations=(op.dilation_h, op.dilation_w),), False 
    return lambda input, filter: tf.nn.depthwise_conv2d(
        input=input,
        filter=filter,
        strides=(1, op.stride, op.stride, 1),
        padding=op.extra_attrs["padding"],
        dilations=(op.dilation_h, op.dilation_w),
    )


@operator_impl(NHWCSeparableConv2d)
def gen_code(op: NHWCSeparableConv2d):
    if op.input_like[0] is None :
        return "tf.nn.separable_conv2d", False
    return "tf.nn.separable_conv2d" + \
        braces_template(len(op.input_like), \
                        strides=(1, op.stride, op.stride, 1), \
                        padding='"'+op.extra_attrs["padding"]+'"',\
                        dilations=(op.dilation_h, op.dilation_w)), False 
    return lambda input, depthwise_filter, pointwise_filter: tf.nn.separable_conv2d(
        input=input,
        depthwise_filter=depthwise_filter,
        pointwise_filter=pointwise_filter,
        strides=(1, op.stride, op.stride, 1),
        padding=op.extra_attrs["padding"],
        dilations=(op.dilation_h, op.dilation_w),
    )


@operator_impl(NHWCConv2dTranspose)
def gen_code(op: NHWCConv2dTranspose):
    if op.input_like[0] is None :
        return "tf.nn.conv2d_transpose", False
    return "tf.nn.conv2d_transpose" + \
        braces_template(len(op.input_like), \
                        output_shape=op.output_like[0].shape,\
                        strides=op.stride,\
                        padding='"'+op.extra_attrs["padding"]+'"'), False 
    return lambda input, filters: tf.nn.conv2d_transpose(
        input=input,
        filters=filters,
        output_shape=op.output_like[0].shape,
        strides=op.stride,
        padding=op.extra_attrs["padding"],
    )


@operator_impl(NHWCDepthToSpace)
def gen_code(op: NHWCDepthToSpace):
    if op.input_like[0] is None :
        return "tf.nn.depth_to_space", False
    return "tf.nn.depth_to_space" + \
        braces_template(len(op.input_like), block_size=op.block_size), False 
    return lambda input: tf.nn.depth_to_space(
        input=input,
        block_size=op.block_size,
    )


@operator_impl(NHWCSpaceToDepth)
def gen_code(op: NHWCSpaceToDepth):
    if op.input_like[0] is None :
        return "tf.nn.space_to_depth", False
    return "tf.nn.space_to_depth" + \
        braces_template(len(op.input_like), block_size=op.block_size), False 
    return lambda input: tf.nn.space_to_depth(
        input=input,
        block_size=op.block_size,
    )


@operator_impl(Gather)
def gen_code(op: Gather):
    if op.input_like[0] is None :
        return "tf.gather(tf.clip_by_value)", False
    axis = op.extra_attrs["axis"]
    clip_value_min, clip_value_max = 0, op.input_like[0].shape[axis] - 1

    def _gather(params, indices):
        indices = tf.clip_by_value(indices, clip_value_min, clip_value_max)
        return tf.gather(params, indices, axis=axis)
    # axis = op.extra_attrs["axis"]
    # # clip_value_min, clip_value_max = 0, op.input_like[0].shape[axis] - 1
    # # indices = tf.clip_by_value(op.extra_attrs['op_index'], clip_value_min, clip_value_max)
    indices = "tf.clip_by_value" + braces_template(1, clip_value_min, clip_value_max)
    return "tf.gather" + braces_template(len(op.input_like)-1, indices, axis=axis).replace('\'',''), False
    return _gather


@operator_impl(Squeeze)
def gen_code(op: Squeeze):
    if op.input_like[0] is None :
        return "tf.squeeze", False
    if op.extra_attrs["reduce_dim"] is not None:
        return "tf.squeeze" + braces_template(len(op.input_like), axis=op.extra_attrs["reduce_dim"]), False
        return lambda x: tf.squeeze(x, axis=op.extra_attrs["reduce_dim"])
    return "tf.squeeze" + braces_template(len(op.input_like)), False
    return lambda x: tf.squeeze(x)


@operator_impl(Unsqueeze)
def gen_code(op: Unsqueeze):
    if op.input_like[0] is None :
        return "tf.expand_dims", False
    SanityCheck.true(op.extra_attrs["expand_dim"] != None)
    return "tf.expand_dims" + braces_template(len(op.input_like), axis=op.extra_attrs["expand_dim"]), False 
    return lambda x: tf.expand_dims(x, axis=op.extra_attrs["expand_dim"])


@operator_impl(ReduceSum)
def gen_code(op: ReduceSum):
    if op.input_like[0] is None :
        return "tf.math.reduce_sum", False
    if op.extra_attrs["reduce_dim"] is not None:
        return "tf.math.reduce_sum" + braces_template(len(op.input_like), axis=op.extra_attrs["reduce_dim"]), False 
        return lambda x: tf.math.reduce_sum(x, axis=op.extra_attrs["reduce_dim"])
    return "tf.math.reduce_sum" + braces_template(len(op.input_like)), False
    return lambda x: tf.math.reduce_sum(x)


@operator_impl(ReduceMin)
def gen_code(op: ReduceMin):
    if op.input_like[0] is None :
        return "tf.math.reduce_min", False
    if op.extra_attrs["reduce_dim"] is not None:
        return "tf.math.reduce_min" + braces_template(len(op.input_like), axis=op.extra_attrs["reduce_dim"]), False 
        return lambda x: tf.math.reduce_min(x, axis=op.extra_attrs["reduce_dim"])
    return "tf.math.reduce_min" + braces_template(len(op.input_like)), False
    return lambda x: tf.math.reduce_min(x)


@operator_impl(ReduceMax)
def gen_code(op: ReduceMax):
    if op.input_like[0] is None :
        return "tf.math.reduce_max", False
    if op.extra_attrs["reduce_dim"] is not None:
        return "tf.math.reduce_max" + braces_template(len(op.input_like), axis=op.extra_attrs["reduce_dim"]), False 
        return lambda x: tf.math.reduce_max(x, axis=op.extra_attrs["reduce_dim"])
    return "tf.math.reduce_max" + braces_template(len(op.input_like)), False
    return lambda x: tf.math.reduce_max(x)


@operator_impl(ReduceMean)
def gen_code(op: ReduceMean):
    if op.input_like[0] is None :
        return "tf.math.reduce_mean", False
    if op.extra_attrs["reduce_dim"] is not None:
        return "tf.math.reduce_mean" + braces_template(len(op.input_like), axis=op.extra_attrs["reduce_dim"]), False 
        return lambda x: tf.math.reduce_mean(x, axis=op.extra_attrs["reduce_dim"])
    return "tf.math.reduce_mean" + braces_template(len(op.input_like)), False
    return lambda x: tf.math.reduce_mean(x)


@operator_impl(ReduceProd)
def gen_code(op: ReduceProd):
    if op.input_like[0] is None :
        return "tf.math.reduce_prod", False
    if op.extra_attrs["reduce_dim"] is not None:
        return "tf.math.reduce_prod" + braces_template(len(op.input_like), axis=op.extra_attrs["reduce_dim"]), False 
        return lambda x: tf.math.reduce_prod(x, axis=op.extra_attrs["reduce_dim"])
    return "tf.math.reduce_prod" + braces_template(len(op.input_like)), False
    return lambda x: tf.math.reduce_prod(x)


@operator_impl(ArgMin)
def gen_code(op: ArgMin):
    if op.input_like[0] is None :
        return "tf.math.argmin", False
    if op.extra_attrs["reduce_dim"] is not None:
        return "tf.math.argmin" + braces_template(len(op.input_like), axis=op.extra_attrs["reduce_dim"]), False 
        return lambda x: tf.math.argmin(x, axis=op.extra_attrs["reduce_dim"])
    return "tf.math.argmin" + braces_template(len(op.input_like)), False
    return lambda x: tf.math.argmin(x)


@operator_impl(ArgMin)
def gen_code(op: ArgMin):
    if op.input_like[0] is None :
        return "tf.math.argmin", False
    if op.extra_attrs["reduce_dim"] is not None:
        return "tf.math.argmax" + braces_template(len(op.input_like), axis=op.extra_attrs["reduce_dim"]), False 
        return lambda x: tf.math.argmax(x, axis=op.extra_attrs["reduce_dim"])
    return "tf.math.argmax" + braces_template(len(op.input_like)), False
    return lambda x: tf.math.argmax(x)


@operator_impl(Tril)
def gen_code(op: Tril):
    if op.input_like[0] is None :
        return "tf.experimental.numpy.tril", False
    return "tf.experimental.numpy.tril" + braces_template(len(op.input_like), k=op.diagonal), False 
    return lambda x: tf.experimental.numpy.tril(x, k=op.diagonal)


@operator_impl(Triu)
def gen_code(op: Triu):
    if op.input_like[0] is None :
        return "tf.experimental.numpy.triu", False
    return "tf.experimental.numpy.triu" + braces_template(len(op.input_like), k=op.diagonal), False 
    return lambda x: tf.experimental.numpy.triu(x, k=op.diagonal)


@operator_impl(Concat)
def gen_code(op: Concat):
    if op.input_like[0] is None :
        return "tf.concat", False
    axis = op.extra_attrs["axis"]
    return (
        "tf.concat" + "([" + ", ".join(["{}"] * len(op.input_like)) + f"], axis={axis})",
        False,
    )
    return lambda *args: tf.concat(args, axis=axis)


@operator_impl(Cast)
def gen_code(op: Cast):
    if op.input_like[0] is None :
        return "tf.cast", False
    return "tf.cast" + braces_template(len(op.input_like), dtype=repr(op.extra_attrs["to"].tensorflow())), False
    return lambda x: tf.cast(x, dtype=op.extra_attrs["to"].tensorflow())


@operator_impl(TFMatMul)
def gen_code(op: TFMatMul):
    return "tf.matmul" + braces_template(len(op.input_like)), False 
    return tf.matmul


@operator_impl(Reverse)
def gen_code(op: Reverse):
    if op.input_like[0] is None :
        return "tf.reverse", False
    return "tf.reverse" + braces_template(len(op.input_like), axis=op.extra_attrs["axis"]), False
    return lambda x: tf.reverse(x, axis=op.extra_attrs["axis"])


@operator_impl(Cholesky)
def gen_code(op: Cholesky):
    return "tf.linalg.cholesky" + braces_template(len(op.input_like)), False
    return tf.linalg.cholesky


@operator_impl(Eigh)
def gen_code(op: Eigh):
    return "tf.linalg.eigh" + braces_template(len(op.input_like)), False
    # return tf.linalg.eigh


# @operator_impl(Split)
# def gen_code(op: Split):
#     axis = op.extra_attrs["axis"]
#     num_splits = op.arity
#     return "tf.split" + braces_template(len(op.input_like), num_splits, axis=axis), False 
#     return lambda x: tf.split(x, num_splits, axis=axis)
#     return "tf.linalg.eigh" + braces_template(len(op.input_like)), False