# Utilities for using and be used by autoinf.
from collections import Counter
from dataclasses import dataclass
from os import PathLike
from typing import Any, Dict, List, Tuple, Type

from nnsmith.abstract.dtype import DTYPE_GEN_ALL, AbsTensor, DType
from nnsmith.abstract.op import *
from nnsmith.abstract.op import __MAX_RANK__
from nnsmith.abstract.dtype import AbsTensor
from nnsmith.autoinf.instrument.categorize import gen_inst_with_records
from nnsmith.autoinf.instrument.op import OpInstance
from nnsmith.logger import AUTOINF_LOG


def make_reduce_type(axis) -> Type[ReduceBase]:
    class Reduce(ReduceBase):
        def _init_reduce_dim(self, input_shape):
            self.extra_attrs["reduce_dim"] = axis
            return self.extra_attrs["reduce_dim"]

    return Reduce


def make_concat_type(arity, axis) -> Type[Concat]:
    class ConcatVariant(Concat):
        in_dtypes = [tuple(i for _ in range(arity)) for i in DTYPE_GEN_ALL]

        def __init__(self):
            super().__init__(arity)
            self.inp_ranks = [(rank_from(axis + 1))] * arity
            self.out_ranks = [(rank_from(axis + 1))]
            self.extra_attrs["axis"] = axis

        def _init_concat_axis(self, input_shapes: List[AbsTensor]) -> int:
            SanityCheck.gt(input_shapes[0].ndims, axis)
            return axis

    return ConcatVariant


ATTR_FREE_RULES = [
    ElementWiseUnaryOp,
    BcastBinaryOp,
    Where,
    *[make_reduce_type(i) for i in range(__MAX_RANK__)],
    *[
        make_concat_type(arity, axis)
        for arity in range(2, Concat.MAX_ARITY + 1)
        for axis in range(__MAX_RANK__)
    ],
]

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
        self.inst: OpInstance = inst
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
class OpRecordFinder:
    producer: Dict[AbsTensor, List[AbsOpBase]]
    consumer: Dict[AbsTensor, List[AbsOpBase]]
    op2record: Dict[
        OpInstance, List[Tuple[Tuple[AbsTensor], Tuple[AbsTensor], Dict[str, int]]]
    ]
    ihtype2op: Dict[Tuple[int, DType], List[OpInstance]]
    ohtype2op: Dict[Tuple[int, DType], List[OpInstance]]
    dtype_dist: Dict[DType, float]  # instance level.
    rank_dist: Dict[int, float]


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


def make_record_finder(
    gen_inst_records: List[Tuple[OpInstance, List[Tuple[Dict]]]] = None,
    path: PathLike = None,
    max_elem_per_tensor=2**16,
    test_pool: List = [],
):
    if gen_inst_records is None:
        assert path is not None, "Either gen_inst_records or path must be provided."
        gen_inst_records = gen_inst_with_records(data_dir=path, int_policy="fix_dim")

    producer = {}
    consumer = {}
    inst2record = {}
    ihtype2op = {}
    ohtype2op = {}

    total_rec = 0
    skipped_elem = 0
    skipped_err = 0
    skipped_blacklist = 0
    skipped_special = 0

    blacklisted = set()
    valid_apis = set()

    rank_dist_cnt = Counter()
    dtype_dist_cnt = Counter()
    print(f"testing {test_pool}")
    for inst, records in gen_inst_records:
        total_rec += len(records)

        if test_pool and (inst.name not in test_pool or inst.name in ["torch.sin", "tf.cos"]) :
            continue 
        if special_filter(inst): # exclude errotic cases
            skipped_special += len(records)
            continue

        if inst.name in BLACKLIST:  # black list
            if inst.name not in blacklisted:
                AUTOINF_LOG.warning(f"Blacklist operator {inst.name} found!")
                blacklisted.add(inst.name)
            skipped_blacklist += len(records)
            continue

        valid_apis.add(inst.name)

        for record in records:
            try:
                input_abs_tensor = [
                    AbsTensor(shape, DType.from_str(dtype))
                    for shape, dtype in zip(
                        inst.concrete_input_shapes(record[0]), record[2]
                    )
                ]

                if len(input_abs_tensor) == 0:
                    AUTOINF_LOG.error(
                        f"{inst.invoke_str({a: record[0][a] for a in inst.A})} has no INputs."
                    )
                    skipped_err += 1
                    continue

                if any([x.nelement() > max_elem_per_tensor for x in input_abs_tensor]):
                    AUTOINF_LOG.debug(
                        f"Skip {inst.name} <- {input_abs_tensor} for over {max_elem_per_tensor} elements."
                    )
                    skipped_elem += 1
                    continue
            except KeyError:
                AUTOINF_LOG.error(f"{inst.name}: bad subst. {inst.I} -> {record}")
                skipped_err += 1
                continue

            try:
                output_abs_tensor = [
                    AbsTensor(shape, DType.from_str(dtype))
                    for shape, dtype in zip(
                        inst.concrete_output_shapes(record[1]), record[3]
                    )
                ]

                if len(output_abs_tensor) == 0:
                    AUTOINF_LOG.error(
                        f"{inst.invoke_str({a: record[0][a] for a in inst.A})} has no OUTputs."
                    )
                    skipped_err += 1
                    continue

                if any([x.nelement() > max_elem_per_tensor for x in output_abs_tensor]):
                    AUTOINF_LOG.debug(
                        f"Skip {inst.name} -> {output_abs_tensor} for over {max_elem_per_tensor} elements."
                    )
                    skipped_elem += 1
                    continue
            except KeyError:
                AUTOINF_LOG.error(f"{inst.name}: bad subst. {inst.O} -> {record}")
                skipped_err += 1
                continue

            for iten in input_abs_tensor:
                prod_list = producer.setdefault(iten, [])
                if inst not in prod_list:
                    prod_list.append(inst)

            for oten in output_abs_tensor:
                cons_list = consumer.setdefault(oten, [])
                if inst not in cons_list:
                    cons_list.append(inst)

            inst2record.setdefault(inst, []).append(
                (
                    tuple(input_abs_tensor),
                    tuple(output_abs_tensor),
                    {k: record[0][k] for k in inst.A},
                )
            )

        if inst in inst2record:
            rks = set()
            dts = set()
            for it, ot, _ in inst2record[inst]:
                for x in it + ot:
                    rks.add(x.ndims)
                    dts.add(x.dtype)
            rank_dist_cnt.update(rks)
            dtype_dist_cnt.update(dts)

            if not inst.infer_failed():
                # output dtypes -> input dtypes
                inst.dtype_o2i = {}
                inst.dtype_i2o = {}
                for it, ot, _ in inst2record[inst]:
                    isig = tuple([DType(x.dtype) for x in it])
                    osig = tuple([DType(x.dtype) for x in ot])
                    # input dtypes -> output dtypes
                    if isig in inst.dtype_i2o:
                        assert (
                            inst.dtype_i2o[isig] == osig
                        ), f"Non-unique i2o dtype mapping of {inst.name}: {isig} -> {osig} or {inst.dtype_i2o[isig]}"
                    inst.dtype_i2o[isig] = osig  # val is unique.
                    # output dtypes -> List [ input dtypes ]
                    lo2i = inst.dtype_o2i.setdefault(osig, [])
                    if osig not in lo2i:
                        lo2i.append(isig)

                iranks = tuple([x.rank for x in inst.input_tensors])
                for idtypes in inst.dtype_i2o:
                    for dtype, rank in zip(idtypes, iranks):
                        li = ihtype2op.setdefault((dtype, rank), [])
                        if inst not in li:
                            li.append(inst)
                oranks = tuple([x.rank for x in inst.output_tensors])
                for odtypes in inst.dtype_o2i:
                    for dtype, rank in zip(odtypes, oranks):
                        lo = ohtype2op.setdefault((dtype, rank), [])
                        if inst not in lo:
                            lo.append(inst)

    skipped_rec = skipped_elem + skipped_err + skipped_blacklist + skipped_special
    final_rec = total_rec - skipped_rec
    AUTOINF_LOG.info(
        f"Got {final_rec} records of {len(inst2record)} OpInstance of {len(valid_apis)} APIs"
    )
    AUTOINF_LOG.info(f"Filtered {skipped_rec} records from {total_rec} initial set.")
    AUTOINF_LOG.info(
        f"~ {skipped_elem}: over {max_elem_per_tensor} elem.  ~ {skipped_err}: bad subst.  ~ {skipped_blacklist}: blacklisted."
    )

    # counter to distribution
    dtype_total = sum(dtype_dist_cnt.values())
    dtype_dist = {k: v / dtype_total for k, v in dtype_dist_cnt.items()}
    rank_total = sum(rank_dist_cnt.values())
    rank_dist = {k: v / rank_total for k, v in rank_dist_cnt.items()}

    # sort by key
    dtype_dist = {
        k: dtype_dist[k] for k in sorted(dtype_dist.keys(), key=lambda x: x.value)
    }
    rank_dist = {k: rank_dist[k] for k in sorted(rank_dist.keys())}

    AUTOINF_LOG.info(f"Distribution of tensor dtypes (id: op + tensor.dtype):")
    for dt in dtype_dist:
        AUTOINF_LOG.info(f"  {dt}:\t{dtype_dist[dt] * 100:.2f}%")
    AUTOINF_LOG.info(f"Distribution of tensor ranks (id: op + tensor.ndims):")
    for rk in rank_dist:
        AUTOINF_LOG.info(f"  {rk}:\t{rank_dist[rk] * 100:.2f}%")
    return OpRecordFinder(
        producer,
        consumer,
        inst2record,
        ihtype2op,
        ohtype2op,
        dtype_dist=dtype_dist,
        rank_dist=rank_dist,
    )
