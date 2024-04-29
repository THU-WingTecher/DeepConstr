from copy import deepcopy
import pickle
from os import PathLike
from typing import Dict, List, Tuple, Type
import numpy as np

import torch
from nnsmith.abstract.dtype import AbsTensor, DType

from nnsmith.abstract.op import AbsOpBase
from nnsmith.gir import GraphIR
from nnsmith.logger import TORCH_LOG
from nnsmith.materialize import Model, Oracle
from nnsmith.materialize.torch.forward import ALL_TORCH_OPS
from nnsmith.materialize.torch.input_gen import PracticalHybridSearch
from nnsmith.materialize.torch.symbolnet import SymbolNet
from nnsmith.util import register_seed_setter
from nnsmith.materialize.torch.code_gen import gen_code

def tensor_from_numpy(x) : 
    """ 
    some dtypes are not supported by numpy, so directly gen with tf
    """
    if isinstance(x, np.ndarray) : 
        return torch.from_numpy(x)
    else : 
        shape, dtype = x
        absdtype = DType.from_str(dtype)
        if "bfloat" in dtype : 
            ret = torch.rand(shape, dtype=absdtype.torch())  
        elif "qint" in dtype :
            rng = 2**(8*absdtype.sizeof()-1)
            ret = torch.randint(-rng, rng - 1, size=shape, dtype=absdtype.torch())
        elif "uint" in dtype :
            rng = 2**(8*absdtype.sizeof()-1)
            ret = torch.randint(0, rng - 1, size=shape, dtype=absdtype.torch())
        else :
            TORCH_LOG.error(f"Unknown dtype: {dtype = }")
            raise NotImplementedError(dtype)
        return ret

class TorchModel(Model): 
    package = "torch"
    gen_code = gen_code
    def __init__(self) -> None:
        super().__init__()
        self.torch_model: SymbolNet = None
        self.sat_inputs = None

    @property
    def version(self) -> str:
        return torch.__version__

    @classmethod
    def from_gir(cls: Type["TorchModel"], ir: GraphIR, **kwargs) -> "TorchModel":
        ret = cls()
        ret.torch_model = SymbolNet(ir, **kwargs)
        return ret

    @staticmethod
    def gir_name() -> str:
        return "gir.pkl"

    @staticmethod
    def execute_op(inst : "OpInstance") : 
            
        numpy_from_tensor = lambda x: x.detach().cpu().numpy()
        is_tensor = lambda x: isinstance(x, torch.Tensor)
        output_info = inst.execute(
            symb_2_value=None,
            tensor_from_numpy=tensor_from_numpy,
            abs_from_dtype=DType.from_torch,
            is_tensor=is_tensor,
            func=eval(inst.name)
        )
        return output_info

    def refine_weights(self) -> None:
        self.torch_model.enable_proxy_grad()
        searcher = PracticalHybridSearch(self.torch_model)
        # TODO(@ganler): Can we directly get both inputs and outputs?
        _, inputs = searcher.search(
            max_time_ms=20,
            max_sample=2,
        )
        if inputs:
            self.sat_inputs = inputs
        self.torch_model.disable_proxy_grad()

    def make_oracle(self) -> Oracle:
        with torch.no_grad():
            self.torch_model.eval()
            # fall back to random inputs if no solution is found.
            if self.sat_inputs is None:
                inputs = self.torch_model.get_random_inps()
            else:
                inputs = self.sat_inputs
            # model = deepcopy(self.torch_model)
            # copied_input = deepcopy(inputs)
            outputs: Tuple[torch.Tensor] = self.torch_model.forward(**inputs)

        # numpyify
        input_dict = {k: v.cpu().detach() for k, v in inputs.items()}
        output_dict = {}
        for oname, val in zip(self.output_like.keys(), outputs):
            if val.is_conj():
                output_dict[oname] = val.cpu().detach().resolve_conj()
            else:
                output_dict[oname] = val.cpu().detach()

        return Oracle(input_dict, output_dict, provider="torch[cpu] eager")

    def dump(self, path: PathLike):
        torch.save(self.torch_model.state_dict(), path)
        gir_path = path.replace(
            TorchModel.name_prefix() + TorchModel.name_suffix(),
            TorchModel.gir_name(),
        )
        with open(gir_path, "wb") as f:
            pickle.dump(self.torch_model.ir, f)

    @classmethod
    def load(cls, path: PathLike) -> "TorchModel":
        ret = cls()
        gir_path = path.replace(
            cls.name_prefix() + cls.name_suffix(),
            cls.gir_name(),
        )
        with open(gir_path, "rb") as f:
            ir = pickle.load(f)
        torch_model = SymbolNet(ir)
        torch_model.load_state_dict(torch.load(path), strict=False)
        ret.torch_model = torch_model
        return ret

    @staticmethod
    def name_suffix() -> str:
        return ".pth"

    @property
    def input_like(self) -> Dict[str, AbsTensor]:
        return self.torch_model.input_like

    @property
    def output_like(self) -> Dict[str, AbsTensor]:
        return self.torch_model.output_like

    @property
    def native_model(self) -> SymbolNet:
        return self.torch_model

    @staticmethod
    def operators() -> List[Type[AbsOpBase]]:
        return ALL_TORCH_OPS

    @staticmethod
    def add_seed_setter() -> None:
        register_seed_setter("torch", torch.manual_seed, overwrite=True)
