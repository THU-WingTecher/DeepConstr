# Utilities for using and be used by autoinf.
from collections import Counter
from dataclasses import dataclass
from os import PathLike
from typing import Any, Dict, List, Tuple, Type

import numpy as np

from neuri.abstract.dtype import DTYPE_GEN_ALL, DType
from neuri.abstract.op import *
from neuri.abstract.op import __MAX_RANK__
from neuri.abstract.tensor import AbsTensor
from neuri.autoinf.instrument.categorize import gen_inst_with_records
from neuri.autoinf.instrument.op import OpInstance
from neuri.logger import AUTOINF_LOG
from neuri.specloader.rule import gen_rule
from neuri.specloader.utils import load_yaml
from neuri.specloader.materalize import materalize_dtypes

"""
~ Number of input operands:
```
OP_TYPE.n_input()
```
~ Number of output operands:
```
OP_TYPE.n_output()
```
~ Shape transfer rule:
```
op = OP_TYPE()
List of AbsTensor = op.checked_type_transfer(List of AbsTensor(shape=..., dtype=...))
# CHECK: Output shape same.
```
~ Input constraint rule:
```
op = OP_TYPE()
List of predicates = op.checked_requires(List of AbsTensor(shape=..., dtype=...))
# CHECK: All true.
```
"""


@mark_abstract("autoinf")
class AutoInfOpBase(AbsOpBase):
    @property
    def attr_names(self):
        return self.inst.A

    def __init__(self, inst: OpInstance, attrs : Dict[str, Any]):
        # super from self
        self.extra_attrs = {}
        self.attrs = attrs
        self.inst = inst
        self.inp_ranks = [tuple(x.rank for x in inst.input_tensors)]
        self.out_ranks = [tuple(x.rank for x in inst.output_tensors)]
        assert set(attrs.keys()) == set(inst.A), f"{list(attrs.keys())} != {inst.A}"

    def n_input(self) -> int:
        return len(self.inst.input_tensors)

    def n_output(self) -> int:
        return len(self.inst.output_tensors)

    def type_transfer(self, input_shapes: List[AbsTensor]) -> List[AbsTensor]:
        AUTOINF_LOG.debug(f"Input symbols: {self.inst.I}; Attr symbols: {self.inst.A}")
        nnsmith_rule_cands = self.inst.nnsmith_rules()
        if nnsmith_rule_cands:
            op_type = nnsmith_rule_cands[0]
            AUTOINF_LOG.debug(f"{self} uses nnsmith's `type_transfer`: {op_type}")
            # only the inferred shapes are trustworthy.
            ret_shapes = [aten.shape for aten in op_type().type_transfer(input_shapes)]
        else:
            AUTOINF_LOG.debug(
                f"{self} uses raw `type_transfer`: {self.inst.type_transfer_dbg_info}"
            )
            odict_cands = self.inst.type_transfer_expressions(
                self.make_substition(input_shapes)
            )
            odict = {k: v[0] for k, v in odict_cands.items()}
            ret_shapes = [
                [odict[k] for k in oshape.shape] for oshape in self.inst.output_tensors
            ]

        odtypes = self.inst.dtype_i2o[tuple(x.dtype for x in input_shapes)]
        return [
            AbsTensor(shape=shape, dtype=dtype)
            for shape, dtype in zip(ret_shapes, odtypes)
        ]

    def requires(self, input_shapes: List[AbsTensor]):
        AUTOINF_LOG.debug(f"Input symbols: {self.inst.I}; Attr symbols: {self.inst.A}")
        nnsmith_rule_cands = self.inst.nnsmith_rules()
        if nnsmith_rule_cands:
            op_type = nnsmith_rule_cands[0]
            AUTOINF_LOG.debug(f"{self} uses nnsmith's `requires`: {op_type}")
            return op_type().requires(input_shapes)

        AUTOINF_LOG.debug(f"{self} uses raw `requires`: {self.inst.requires_dbg_info}")
        return self.inst.requires_expressions(self.make_substition(input_shapes))

    def deduct_inp_ranks_and_dtype(
        self, out_abs_tensor: List[AbsTensor]
    ) -> List[Tuple[int, DType]]:
        # check shape & type consistency
        assert (
            len(out_abs_tensor) == self.n_output()
        ), f"{len(out_abs_tensor)} != {self.n_output()}"
        idtypes = random.choice(
            self.inst.dtype_o2i[tuple([t.dtype for t in out_abs_tensor])]
        )
        return [(rank, dtype) for rank, dtype in zip(self.inp_ranks[0], idtypes)]

    def make_substition(self, input_shapes: List[AbsTensor]):
        symbol_subst = {}
        # input shape subst
        for inf_ten, smt_ten in zip(self.inst.input_tensors, input_shapes):
            for name, val in zip(inf_ten.shape, smt_ten.shape):
                symbol_subst[name] = val
        # attr subst | update self.attrs
        symbol_subst.update(self.attrs)
        return symbol_subst

    def __str__(self):
        return f"{self.inst.name}[{','.join([f'r{ite.rank}' for ite in self.inst.input_tensors])}]"


@dataclass
class OpConstrFinder:
    """
    name : api_name 
    args :
        names : List[str] := names
        dtypes : List[dtype] := dtypes
        is_pos : List[bool] := whether the arg is positioned??
        values : args_values
    outputs :
        values List[Any](mainly AbsTensor):
    constraints : List[Rule]
    """
    constraints: Dict[str, Any]


BLACKLIST = [
    # PyTorch
    # value randomness
    "torch.rand_like",
    "torch.randn_like",
    "torch.randint_like",
    "torch.Tensor.random_",
    "torch.Tensor.uniform_",
    "torch.empty_like",
    "torch.Tensor.normal_",
    "torch.Tensor.new_empty",
    "torch.Tensor.new_empty_strided",
    "torch.dropout",
    "torch.native_dropout",
    "torch.nn.functional.dropout",
    "torch.nn.functional.dropout1d",
    "torch.nn.functional.dropout2d",
    "torch.nn.functional.dropout3d",
    "torch.nn.functional.feature_alpha_dropout",
    # unlock when preprocessing filters out dynamic output numbers.
    "torch.Tensor.unbind",
    "torch.unbind",
    "torch.Tensor.split",
    "torch.split",
    # some special cases
    "torch.gather",
    "torch.Tensor.resize_as_",  # resize_as_ can't be represented in the JIT at the moment ...
    "torch.Tensor.rename",
    "torch.Tensor.rename_",
    "torch.Tensor.requires_grad_",
    "torch.searchsorted",  # sorter has value constraints but the crash needs to be triggered by a big value.
    "torch.native_batch_norm",  # crash when input constraint is violated.
    "torch.Tensor.sum_to_size",  # some odd dtype transfer
    # TensorFlow
    "tf.raw_ops.Unique",
]


def special_filter(inst) -> bool:
    invoke_str = inst.invoke_str({k: None for k in inst.A})
    if (
        "torch.sort(" in invoke_str and "stable=True" in invoke_str
    ):  # stable sort could lead to crash | https://github.com/pytorch/pytorch/issues/91420
        AUTOINF_LOG.warning(
            f"Skip {invoke_str} because stable sort could lead to crash."
        )
        return True
    return False

def gen_inst_with_records(
    data_dir: str,
):
    """
    yield: record dict
        name : api_name 
        args :
            names : List[str] := names
            dtypes : List[dtype] := dtypes
            is_pos : List[bool] := whether the arg is positioned??
            values : args_values
        outputs :
            values List[Any](mainly AbsTensor):
        constraints : List[Rule]
    """
    for root, _, files in os.walk(data_dir):
        for file in files : 
            record = {}
            path = os.path.join(root, file)
            cfg = load_yaml(path)
            record['name'] = cfg['title']
            record['pass_rate'] = cfg['pass_rate']
            if 'constraints' in cfg.keys() :
                record['args'] = {'name' : [arg_name for arg_name in cfg['constraints'].keys()], 
                                'is_pos' : [False] * len(cfg['constraints'].keys()), 
                                'value' : [None] * len(cfg['constraints'].keys()),
                                'dtype' : [None] * len(cfg['constraints'].keys()),
                                }
                record['rules'] = cfg['rules']
                record['outputs'] = {
                    'value' : [],
                }
                for i_name, name in enumerate(record['args']['name']) :
                    record['args']['dtype'][i_name] = materalize_dtypes(cfg['constraints'][name]['dtype'])   
            
            yield record         

def convert_rule_to_executable(record, rule_cnt) -> List["z3.Exr"] : 

    chosn_dtypes = {} 
    exec_rules = []
    for i_arg, arg_name, in enumerate(record['args']['name']) : ## FIXME -> gen diff constr depend on diff dtype( adding suff_conds is enough)
        if len(record['args']['dtype'][i_arg]) > 0 :
            chosn_dtypes[arg_name] = record['args']['dtype'][i_arg][0]
        else :
            chosn_dtypes[arg_name] = record['args']['dtype'][i_arg]

    for rule in record['rules'] :
        AUTOINF_LOG.debug(f"rule : {rule['txt']}")
        rule = gen_rule(rule['target'],rule['cot'], rule['txt'], {name : dtype for name, dtype in zip(record['args']['name'], record['args']['dtype'])},
                                    ) # multiple dtype list
        
        if rule is None : continue
        rule.ast2z3.set_args_types(chosn_dtypes) # only one dtypes 
        c1 = rule.ast2z3.gen_constraints()
        if c1 is not None :
            exec_rules.append(c1)

    rule_cnt["cnt"] += len(exec_rules)
    AUTOINF_LOG.info(f"{len(exec_rules)} rules are generated")
    return exec_rules

def make_record_finder(
    path: PathLike = None,
    pass_rate: float = 0.8,
    test_pool: List = [],
) -> List[Dict[str, Any]]:

    gen_inst_records = gen_inst_with_records(data_dir=path)

    records = []
    total_rec = 0
    skipped_err = 0
    skipped_blacklist = 0
    skipped_unenough_psrate = 0
    blacklisted = set()
    rule_cnt = {"cnt" : 0}
    if test_pool : AUTOINF_LOG.info(f"testing {test_pool}")
    for record in gen_inst_records:
        total_rec+=1
        if test_pool :
            if record['name'] not in test_pool :
                continue 
        else : # when test_pool is defined; we don't check others
            if record.get('skipped') is not None :
                AUTOINF_LOG.info(f"skipped key --> Skip {record['name']}")
                skipped_err+=1
                continue
            if record.get('pass_rate') is None or record['pass_rate'] < pass_rate :
                AUTOINF_LOG.info(f"low pass_rate[thr:{pass_rate}] {record.get('pass_rate')}")
                skipped_unenough_psrate+=1
                continue
            if record['name'] in BLACKLIST:  # black list
                if record['name'] not in blacklisted:
                    AUTOINF_LOG.warning(f"Blacklist operator {record['name']} found!")
                    blacklisted.add(record['name'])
                skipped_blacklist += 1
                continue

        AUTOINF_LOG.info(f"Loading record name : {record['name']}")
        record['constraints'] = convert_rule_to_executable(record, rule_cnt)
        records.append(record)

    skipped_rec = skipped_err + skipped_unenough_psrate + skipped_blacklist
    AUTOINF_LOG.info(
        f"Got {len(records)} records of {total_rec} records with total {rule_cnt['cnt']} rules."
    )
    AUTOINF_LOG.info(f"Filtered {skipped_rec} records from {total_rec} initial set.")
    AUTOINF_LOG.info(
        f" Skipped : {skipped_err}\nLower_psrate : {skipped_unenough_psrate}\nblack_list : {skipped_blacklist}"
    )

    return records
