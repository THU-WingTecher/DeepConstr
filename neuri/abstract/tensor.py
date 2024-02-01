from functools import partial, reduce
from typing import Any, Callable, Dict, List, Union

import z3

from neuri.abstract.arith import *
from neuri.abstract.dtype import DType
from neuri.constrinf.ast2z3 import load_z3_const, TensorZ3
from neuri.error import ConstraintCheck, SanityCheck


class AbsTensor:
    def __init__(self, 
                 shape: List[Union[int, z3.ExprRef]] = [], 
                 dtype: DType = None,
                 possible_dtypes : List[DType] = [],
                 **kwargs):
        assert isinstance(
            shape, (list, tuple)
        ), f"Shape must be a list/tuple, but got {shape}"
        self.shape = list(shape)
        self.rank = len(self.shape)
        assert possible_dtypes or dtype, "Must provide dtype or possible_dtypes"
        self.possible_dtypes : List[DType] = possible_dtypes
        self.dtype = dtype

    @staticmethod
    def from_numpy(x: "np.ndarray") -> "AbsTensor":
        return AbsTensor(list(x.shape), str(x.dtype))
    @staticmethod
    def to_str() -> Any :
        return 'tensor'
    def downcast_rank(self):
        return AbsTensor(shape=[None] * self.ndims, dtype=self.dtype)
    
    def concrete_shape(self, symb_2_value: Dict[str, Any]) -> List[int]:
        return [symb_2_value[s] for s in self.shape]
    
    def set_shape(self, shape: List[Union[int, z3.ExprRef]]):
        self.shape = shape
        if len(self.shape) != self.rank:
            self.rank = len(self.shape)

    def concretize(
        self,
        symb_2_value: Dict[str, Any],
        tensor_from_numpy: Callable = lambda x: x,
        *args,
        **kwargs,
    ):
        from neuri.autoinf.instrument.utils import (
            numpy_random,
        )
        shape = [symb_2_value[s] for s in self.shape]
        return tensor_from_numpy(numpy_random(shape, str(self.dtype)))
    
    def concretize_with_concrete_values(
        self,
        tensor_from_numpy: Callable = lambda x: x,
    ):
        from neuri.autoinf.instrument.utils import (
            numpy_random,
        )
        return tensor_from_numpy(numpy_random(self.shape, str(self.dtype)))
    
    def __hash__(self) -> int:
        return hash((tuple(self.shape), self.dtype))
    
    def __repr__(self) -> str:

        if self.dtype is None :
            return f"AbsTensor<null[{str(len(self.possible_dtypes))}]>{str(self.shape)}"
        else :
            return f"AbsTensor<{self.dtype.short()}>{str(self.shape)}"

    def pretty(self) -> str:
        return f"{self.dtype.short()}{self.shape}"

    def weak_compare(self, other: "AbsTensor") -> bool:
        if self.dtype != other.dtype or self.ndims != other.ndims:
            return False
        for l, r in zip(self.shape, other.shape):
            if isinstance(l, z3.ExprRef) or isinstance(r, z3.ExprRef):
                continue
            if l != r:
                return False
        return True

    def strong_compare(self, other: "AbsTensor") -> bool:
        return self.shape == other.shape and self.dtype == other.dtype

    def __eq__(self, other: "AbsTensor") -> bool:
        return isinstance(other, AbsTensor) and self.strong_compare(other)

    def ge_zero(self):
        ret = []
        for s in self.shape:
            if isinstance(s, z3.ExprRef):
                ret.append(nnsmith_ge(s, 0))
            else:
                ConstraintCheck.ge(s, 0)
        return ret

    def sym_gt_conc_ge_zero(self):
        ret = []
        for s in self.shape:
            if isinstance(s, z3.ExprRef):
                ret.append(nnsmith_gt(s, 0))
            else:
                ConstraintCheck.ge(s, 0)
        return ret

    def gt_zero(self):
        ret = []
        for s in self.shape:
            if isinstance(s, z3.ExprRef):
                ret.append(nnsmith_gt(s, 0))
            else:
                ConstraintCheck.gt(s, 0)
        return ret

    def eq(self, other):
        SanityCheck.eq(self.ndims, other.ndims)
        ret = []
        for i in range(self.ndims):
            if isinstance(self.shape[i], z3.ExprRef) or isinstance(
                other.shape[i], z3.ExprRef
            ):
                ret.append(nnsmith_eq(self.shape[i], other.shape[i]))
            else:
                ConstraintCheck.eq(self.shape[i], other.shape[i])
        return ret

    def torch(self):
        import torch

        return torch.Size(self.shape)
    def consistent_constr(self, other : str) -> List[z3.BoolRef] :
        """ 
        generate constraints that ensure the shape, rank, and dtype as consistent with
        the name of $other tensor
        """
        ## gen z3 var 
        other_obj = self.z3()(other)
        ## rank consistent constr 
        rank_cons = [other_obj.rank == self.ndims]
        ## shape consistent constr 
        shape_cons = [
            other_obj.shape[i] == self.shape[i] for i in range(self.ndims)
        ]
        ## dtype consistent constr
        dtype_cons = [other_obj.dtype == self.dtype.z3_const()]

        return rank_cons + shape_cons + dtype_cons
    
    @classmethod
    def z3(cls) -> "z3.Dtype" :
        z3_load_func = partial(load_z3_const, 
                        var_type=cls.to_str(), 
                        is_array=False)
        
        return z3_load_func

    def constains_symbol(self) -> bool:
        return any(isinstance(s, z3.ExprRef) for s in self.shape)

    def nelement(self):
        if len(self.shape) == 0:  # Scalar
            return 1
        return reduce(lambda x, y: nnsmith_mul(x, y), self.shape, 1)

    def nbytes(self) -> int:
        return self.nelement() * self.dtype.sizeof()

    def deepcopy(self):
        return AbsTensor(shape=list(self.shape), dtype=self.dtype)
    @staticmethod
    def to_iter() :
        from neuri.abstract.dtype import AbsIter
        return AbsIter([AbsTensor])
    @property
    def ndims(self):
        return len(self.shape)

    def is_concrete(self) -> bool:
        return all(isinstance(s, int) for s in self.shape)

    def htype(self):  # High-level type
        return (self.dtype, self.ndims)
