import contextlib
import signal
from random import random

import numpy as np
from z3 import *


def wrap_time(func):
    def wrapped_func(*args, **kwargs):
        from time import time

        begin = time()
        func(*args, **kwargs)
        end = time()
        print(f"func {func.__name__}'s execution time: {end - begin}")

    return wrapped_func


def random_tensor(tensor_shape, tensor_dtype, library="torch"):
    if library == "torch":
        import torch

        if tensor_dtype == torch.bool:
            return torch.randint(0, 2, tensor_shape, dtype=tensor_dtype)
        elif tensor_dtype.is_floating_point or tensor_dtype.is_complex:
            return torch.rand(tensor_shape, dtype=tensor_dtype)
        else:
            return torch.randint(0, 10, tensor_shape, dtype=tensor_dtype)
    else:
        import tensorflow as tf

        if tensor_dtype == tf.bool:
            return tf.cast(
                tf.random.uniform(tensor_shape, minval=0, maxval=2, dtype=tf.int32),
                dtype=tf.bool,
            )
        elif tensor_dtype.is_floating:
            return tf.random.uniform(tensor_shape, dtype=tensor_dtype)
        elif tensor_dtype.is_complex:
            ftype = tf.float64 if tensor_dtype == tf.complex128 else tf.float32
            return tf.complex(
                tf.random.uniform(tensor_shape, dtype=ftype),
                tf.random.uniform(tensor_shape, dtype=ftype),
            )
        elif tensor_dtype == tf.string:
            return tf.convert_to_tensor(np.ones(tensor_shape, dtype=str))
        else:
            # print(tensor_dtype, flush=True)
            return tf.saturate_cast(
                tf.random.uniform(tensor_shape, minval=0, maxval=10, dtype=tf.int64),
                dtype=tensor_dtype,
            )


def make_iterable(target, library="torch"):
    import tensorflow as tf
    import torch

    tensor_checker = torch.is_tensor if library == "torch" else tf.is_tensor

    def flatten(target):
        if not hasattr(target, "__iter__") or tensor_checker(target):
            yield target
        else:
            for item in target:
                yield from flatten(item)

    return list(flatten(target))


def make_list(target, library="torch"):
    import tensorflow as tf
    import torch

    tensor_checker = torch.is_tensor if library == "torch" else tf.is_tensor
    assert not isinstance(target, str)

    def flatten(target):
        if tensor_checker(target):
            # print(target, flush=True)
            # print(target, list(target.shape))
            # assert(len(list(target.shape)) == 1)
            yield target.item()
        elif not hasattr(target, "__iter__"):
            yield target
        else:
            for item in target:
                yield from flatten(item)

    return list(flatten(target))


def compare_array_diff(l1, l2) -> list:
    if len(l1) != len(l2):
        return []
    ret = []
    for i in range(len(l1)):
        if l1[i] != l2[i]:
            ret.append(i)
    return ret


def probability(p: float) -> bool:
    return random() <= p


def equivalent(exp1, exp2, show=False, **keywords) -> bool:
    claim = exp1 == exp2
    s = Solver()
    s.set(**keywords)
    s.add(Not(claim))
    r = s.check()
    if r == unsat:
        return True
    else:
        return False


def str_equivalent(expr1, expr2) -> bool:
    s0, s1, s2, s3, s4, s5, s6, s7 = Ints("s0 s1 s2 s3 s4 s5 s6 s7")
    # print(expr1, expr2)
    e1, e2 = eval(expr1), eval(expr2)
    return equivalent(e1, e2)


def tf_config_gpu():
    import tensorflow as tf

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)


def popcount(num: int) -> int:
    return bin(num).count("1")


MaskListTable = dict()


def mask_to_list(mask: int):
    if mask in MaskListTable:
        return list(MaskListTable[mask])
    res = []
    pw, pos = 1, 0
    while pw <= mask:
        if mask & pw:
            res.append(pos)
        pw *= 2
        pos += 1
    MaskListTable[mask] = tuple(res)
    return res


class TimeoutException(Exception):
    pass


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
