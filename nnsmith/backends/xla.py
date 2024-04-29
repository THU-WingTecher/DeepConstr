from typing import Dict
from nnsmith.materialize.tensorflow import tensor_from_numpy
import tensorflow as tf  # type: ignore
from multipledispatch import dispatch

from nnsmith.backends.factory import BackendCallable, BackendFactory
from nnsmith.materialize.tensorflow import (
    EagerModeCtx,
    TFModel,
    np_dict_from_tf,
    tf_dict_from_np,
)

class XLAFactory(BackendFactory):
    def __init__(self, target="cpu", optmax: bool = False):
        super().__init__(target, optmax)

        if self.target == "cpu":
            self.device = tf.device(tf.config.list_logical_devices("CPU")[0].name)
        elif self.target == "cuda":
            self.device = tf.device(tf.config.list_logical_devices("GPU")[3].name)
        else:
            raise ValueError(
                f"Unknown device: {self.target}. Only `cpu` and `cuda` are supported."
            )

    @staticmethod
    def make_random_input(input_like: Dict[str, tf.Tensor], low=1, high=2) -> Dict[str, tf.Tensor]:
        from nnsmith.autoinf.instrument.utils import numpy_random
        return {
            name: tensor_from_numpy(numpy_random(shape=aten.shape, str_dtype=str(aten.dtype)))
            for name, aten in input_like.items()
        }

    @property
    def system_name(self) -> str:
        return "xla"

    @property
    def version(self) -> str:
        return tf.__version__

    @dispatch(TFModel)
    def make_backend(self, model: TFModel) -> BackendCallable:
        with self.device, EagerModeCtx(False):
            compiled = tf.function(jit_compile=True)(model.concrete_net())

        def closure(inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
            with self.device, EagerModeCtx(False):
                result = np_dict_from_tf(compiled(**tf_dict_from_np(inputs)))
            return result

        return closure
