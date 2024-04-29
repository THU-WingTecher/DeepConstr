from __future__ import annotations
from copy import deepcopy

import logging
import os
import pickle
from abc import ABC
from os import PathLike
from typing import Callable, Dict, List, Type

from nnsmith.abstract.dtype import AbsTensor, DType
from nnsmith.logger import TF_LOG
from nnsmith.materialize.tensorflow.code_gen import gen_code

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").disabled = True

import numpy as np
import tensorflow as tf  # type: ignore
from multipledispatch import dispatch  # type: ignore


def configure_tf_gpu_mem(max_megabytes=None):
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
        if isinstance(max_megabytes, int):
            tf.config.set_logical_device_configuration(
                gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=max_megabytes)]
            )


configure_tf_gpu_mem()

from nnsmith.abstract.op import AbsOpBase
from nnsmith.gir import GraphIR
from nnsmith.materialize import Model, Oracle
from nnsmith.materialize.tensorflow.forward import ALL_TF_OPS
from nnsmith.materialize.tensorflow.tfnet import TFNet
from nnsmith.util import register_seed_setter

TFNetCallable = Callable[..., Dict[str, tf.Tensor]]


@dispatch(dict)
def randn_from_specs(specs: Dict[str, tf.TensorSpec]) -> Dict[str, tf.Tensor]:
    return {
        name: tf.cast(tf.random.normal(shape=spec.shape), dtype=spec.dtype)
        for name, spec in specs.items()
    }


def np_dict_from_tf(x: Dict[str, tf.Tensor]) -> Dict[str, np.ndarray]:
    return {key: np.array(value.numpy()) for key, value in x.items()}


def tf_dict_from_np(x: Dict[str, np.ndarray]) -> Dict[str, tf.Tensor]:
    return {key: tf.convert_to_tensor(value) for key, value in x.items()}


class EagerModeCtx:
    def __init__(self, eagerly: bool) -> None:
        assert isinstance(
            eagerly, bool
        ), f"argument eagerly should not be {eagerly.__class__}. It must be a boolean."
        self.eagerly = eagerly

    def __enter__(self) -> None:
        self.old_mode = tf.config.functions_run_eagerly()
        tf.config.run_functions_eagerly(self.eagerly)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        tf.config.run_functions_eagerly(self.old_mode)

def tensor_from_numpy(x) : 
    """ 
    some dtypes are not supported by numpy, so directly gen with tf
    """
    if isinstance(x, np.ndarray) : 
        return tf.convert_to_tensor(x)
    else : 
        shape, dtype = x
        absdtype = DType.from_str(dtype)
        if "bfloat" in dtype : 
            ret = tf.random.uniform(shape, -1000_000, 1000_000, dtype=absdtype.tensorflow())
        elif "qint" in dtype :
            rng = 2**(8*absdtype.sizeof()-1)
            minval = tf.constant(-rng, dtype=tf.int64)
            maxval = tf.constant(rng - 1, dtype=tf.int64)
            ret = tf.random.uniform(shape=shape, minval=minval, maxval=maxval, dtype=tf.int64)
            ret = tf.cast(ret, dtype=absdtype.tensorflow())   
        elif "uint" in dtype :
            rng = 2**(8*absdtype.sizeof()-1)
            maxval = tf.constant(rng - 1, dtype=tf.int64)
            ret = tf.random.uniform(shape=shape, minval=0, maxval=maxval, dtype=tf.int64)
            ret = tf.cast(ret, dtype=absdtype.tensorflow())    
        else :
            TF_LOG.error(f"Unknown dtype: {dtype = }")
            raise NotImplementedError(dtype)
        return ret

class TFModel(Model, ABC):  # Don't directly instantiate this class
    """Wrapper class of TFNet (tf.Module)
    It only stores net. Other information like I/O info are temporarily generated.
    Other values like I/O values should be managed by the user.
    """
    package = "tensorflow"
    gen_code = gen_code 
    def __init__(self, ir: GraphIR) -> None:
        super().__init__()
        self.ir = ir
        self.net = TFNet(ir=ir)

    @staticmethod
    def execute_op(inst : "OpInstance") : 
        # tensor_from_numpy = tf.convert_to_tensor 
        is_tensor = lambda x: isinstance(x, tf.Tensor)
        output_info = inst.execute(
            symb_2_value=None,
            tensor_from_numpy=tensor_from_numpy,
            abs_from_dtype=DType.from_tensorflow,
            is_tensor=is_tensor,
            func=eval(inst.name)
        )
        return output_info

    @classmethod
    def from_gir(cls: Type["TFModel"], ir: GraphIR, **kwargs) -> "TFModel":
        return cls(ir)

    @property
    def version(self) -> str:
        return tf.__version__

    @property
    def native_model(self) -> TFNet:
        return self.net

    @property
    def input_like(self) -> Dict[str, AbsTensor]:
        return {ik: self.ir.vars[ik] for ik in self.ir.input_var()}

    @property
    def output_like(self) -> Dict[str, AbsTensor]:
        return {ok: self.ir.vars[ok] for ok in self.ir.leaf_var()}

    @property
    def input_specs(self) -> Dict[str, tf.TensorSpec]:
        ret: Dict[str, tf.TensorSpec] = {}
        for k, aten in self.input_like.items():
            ret[k] = tf.TensorSpec(aten.shape, aten.dtype.tensorflow(), k)
        return ret

    @staticmethod
    def name_suffix() -> str:
        return ""
    
    @staticmethod
    def make_random_input(input_like: Dict[str, tf.Tensor], low=1, high=2) -> Dict[str, tf.Tensor]:
        return {
            name: tf.cast(low + (high - low) * tf.random.uniform(shape=aten.shape, dtype=aten.dtype.tensorflow()), aten.dtype.tensorflow())
            for name, aten in input_like.items()
        }
    
    def make_oracle(
        self, inputs: Dict[str, tf.Tensor | tf.TensorSpec] = None
    ) -> Oracle:
        if inputs is None or isinstance(list(inputs.values())[0], tf.TensorSpec):
            input_dict = self.random_inputs()
        else:
            input_dict = inputs

        # output_dict = self.run_eagerly(input_dict)
        # To pre-check the graph integrity, we use graph mode
        # instead of eager mode.
        concrete_net = self.concrete_net(self.input_specs)
        # model = deepcopy(concrete_net)
        copied_input = deepcopy(input_dict)
        output_dict = concrete_net(**copied_input)

        input_dict = np_dict_from_tf(input_dict)
        output_dict = np_dict_from_tf(output_dict)

        return Oracle(input_dict, output_dict, provider="tf[cpu] eager")

    def dump(self, path: PathLike = "saved_tfmodel") -> None:
        os.makedirs(path, exist_ok=True)
        # gir.pkl
        with open(os.path.join(path, TFModel.gir_name()), "wb") as f:
            pickle.dump(self.ir, f)
        # tfnet
        with self.device:
            concrete_net = self.concrete_net(self.input_specs)
            tf.saved_model.save(
                self.net,
                os.path.join(path, TFModel.tfnet_dir_name()),
                signatures=concrete_net,
            )

    def dump_with_oracle(
        self,
        path: PathLike,
        inputs: Dict[str, tf.Tensor | tf.TensorSpec] = None,
    ) -> None:
        self.dump(path)
        oracle = self.make_oracle(inputs)
        oracle.dump(os.path.join(path, Oracle.name()))

    @classmethod
    def load(cls, path: PathLike) -> "TFModel":
        with open(os.path.join(path, cls.gir_name()), "rb") as f:
            gir: GraphIR = pickle.load(f)
        model = cls(gir)
        model.net = tf.saved_model.load(os.path.join(path, TFModel.tfnet_dir_name()))
        return model

    @staticmethod
    def gir_name() -> str:
        return "gir.pkl"

    @staticmethod
    def in_out_pkl_name():
        return "in_out.pkl"

    @staticmethod
    def tfnet_dir_name():
        return "tfnet"

    def random_inputs(self) -> Dict[str, tf.Tensor]:
        return {
            spec.name: tf.cast(
                tf.random.normal(
                    shape=spec.shape,
                    seed=None,
                ),
                dtype=spec.dtype,
            )
            for spec in self.input_specs.values()
        }

    def concrete_net(
        self, inputs: Dict[str, tf.Tensor | tf.TensorSpec] | None = None
    ) -> TFNetCallable:
        if inputs is None:
            inputs = self.input_specs
        return self.net.__call__.get_concrete_function(**inputs)

    def refine_weights(self) -> None:
        pass

    def run_eagerly(self, inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        with EagerModeCtx(True), self.device:  # disable graph execution
            # TODO some op can only run on GPU (e.g. conv with NCHW)
            results = self.net(**inputs)
        return results

    @staticmethod
    def operators() -> List[Type[AbsOpBase]]:
        return list(ALL_TF_OPS)

    @staticmethod
    def add_seed_setter() -> None:
        register_seed_setter("tensorflow", tf.random.set_seed, overwrite=True)
        register_seed_setter("keras", tf.keras.utils.set_random_seed, overwrite=True)


class TFModelCPU(TFModel):
    @property
    def device(self) -> tf.device:
        return tf.device(tf.config.list_logical_devices("CPU")[0].name)


class TFModelGPU(TFModel):
    @property
    def device(self) -> tf.device:
        gpus = tf.config.list_logical_devices("GPU")
        assert gpus, "No GPU available"
        return tf.device(gpus[3].name)
