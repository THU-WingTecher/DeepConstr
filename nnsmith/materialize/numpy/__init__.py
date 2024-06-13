from dataclasses import dataclass
from typing import Callable, Dict, List, cast

from nnsmith.abstract.op import AbsOpBase, Input
from nnsmith.autoinf import AutoInfOpBase
from nnsmith.error import SanityCheck
from nnsmith.gir import GraphIR
from nnsmith.logger import TF_LOG
import numpy as np
import numpy
def forward_fn(op: AutoInfOpBase):
    return op.inst.materialize(eval(op.inst.name), op.attrs)

class NumPyModel():
    """A TensorFlow network whose computation is defined by a GraphIR."""
    package = "numpy"
    def __init__(self, ir: GraphIR) -> None:
        """Build a TensorFlow model from GraphIR

        Args:
            ir (GraphIR): minimal information for constructing a concrete graph.
        """
        super().__init__()
        self.ir: GraphIR = ir
        self.mlist: List[Callable] = []
        self.instructions = []

        for inst in self.ir.insts:
            if not isinstance(inst.iexpr.op, Input):
                op = cast(AbsOpBase, inst.iexpr.op)
                fwd_fn = forward_fn(op)
                SanityCheck.true(fwd_fn is not None, f"Bad impl for {inst.iexpr.op}")
                self.instructions.append(
                (fwd_fn, inst.iexpr.args, inst.retvals(), inst.iexpr.op)
                 )
        self.input_map = {iname: self.ir.vars[iname] for iname in self.ir.input_var()}
        self.output_map = {oname: self.ir.vars[oname] for oname in self.ir.leaf_var()}

    @property
    def input_like(self):
        return self.input_map

    @property
    def output_like(self):
        return self.output_map

    @staticmethod
    def make_random_input(input_like: Dict[str, "np.array"], low=1, high=2) -> Dict[str,  "np.array"]:
        from nnsmith.autoinf.instrument.utils import numpy_random
        return {
            name: numpy_random(shape=aten.shape, str_dtype=str(aten.dtype))
            for name, aten in input_like.items()
        }
    
    @classmethod
    def from_gir(cls) :
        return cls.ir 

    @staticmethod
    def execute_op(inst : "OpInstance") : 
        # tensor_from_numpy = tf.convert_to_tensor 
        output_info = inst.execute(
            symb_2_value=None,
            tensor_from_numpy=lambda x : x,
            abs_from_dtype=lambda x : x,
            is_tensor=lambda x : isinstance(x, np.ndarray),
            func=eval(inst.name)
        )
        return output_info