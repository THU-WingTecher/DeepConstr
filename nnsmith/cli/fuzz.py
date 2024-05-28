import random
import os
import time
import traceback
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import FunctionType
from typing import Tuple, Type

import hydra
from omegaconf import DictConfig


from nnsmith.backends.factory import BackendFactory
from nnsmith.cli.model_exec import verify_testcase
from nnsmith.error import InternalError
from nnsmith.filter import FILTERS
from nnsmith.gir import GraphIR
from nnsmith.graph_gen import model_gen
from nnsmith.logger import FUZZ_LOG, AUTOINF_LOG
from nnsmith.macro import NNSMITH_BUG_PATTERN_TOKEN
from nnsmith.materialize import Model, TestCase
from nnsmith.narrow_spec import auto_opset
from nnsmith.util import mkdir, parse_timestr, set_seed


# Status.csv
# Each line represents the status of one generation.
# META: seed,stat,tgen,tsmt,tsave,trun,elapsed_s
# stat can be one of:
#  - "ok": no error.
#  - "fail": invalid testcase.
#  - "bug": bug found.
class StatusCollect:
    def __init__(self, root, resume=False, op_usage=False):
        self.root = Path(root)
        stat_fname = self.root / "status.csv"
        self.n_bugs = 0
        self.n_fail_make_test = 0
        self.n_testcases = 0
        self._elapsed_s = 0

        if (
            os.path.exists(self.root) and resume
        ):  # parse n_bugs, n_fail_make_test, n_testcases.
            with open(stat_fname, "r") as f:
                lines = f.readlines()
            for line in lines[1:]:
                if line == "":
                    continue
                tokens = line.split(",")
                stat = tokens[1]
                self._elapsed_s = float(tokens[-1])
                if stat == "bug":
                    self.n_bugs += 1
                elif stat == "fail":
                    self.n_fail_make_test += 1
                elif stat != "ok":
                    raise ValueError(f"Unknown stat: {stat}")
                self.n_testcases += 1
            FUZZ_LOG.info(f"Resuming from {self.n_testcases} testcases.")
        else:
            mkdir(self.root)
            with open(stat_fname, "w") as f:
                f.write("seed,stat,tgen(ms),tsmt(ms),tsave(ms),trun,elapsed(s)\n")

        self.stat_file = open(
            self.root / "status.csv", "a", buffering=1024
        )  # 1KB buffer
        self.tik = None

        self.op_usage = op_usage
        if self.op_usage:
            self.op_used_file = open(self.root / "op_used.txt", "a", buffering=128)
            self.op_rej_file = open(self.root / "op_rej.txt", "a", buffering=128)

    def ok_inst(self, ir: GraphIR):
        if self.op_usage:
            insts = ir.leaf_inst()
            if insts and insts[0].iexpr.op.__class__.__name__ == "AutoInfOpBase":
                self.op_used_file.write(f"{insts[0].iexpr.op.inst.name_index}\n")

    def rej_inst(self, ir: GraphIR):
        if self.op_usage:
            insts = ir.leaf_inst()
            if insts and insts[0].iexpr.op.__class__.__name__ == "AutoInfOpBase":
                self.op_rej_file.write(f"{insts[0].iexpr.op.inst.name_index}\n")
                self.op_rej_file.flush()  # flush every time to avoid losing data.

    @property
    def elapsed_s(self) -> float:
        if self.tik is None:
            self.tik = time.time()
        else:
            tok = time.time()
            self._elapsed_s += tok - self.tik
            self.tik = tok
        return self._elapsed_s

    def record(self, seed, stat, tgen, tsmt=0, trun=0, tsave=0):
        self.n_testcases += 1
        if stat == "bug":
            self.n_bugs += 1
        elif stat == "fail":
            self.n_fail_make_test += 1
        elif stat != "ok":
            raise ValueError(f"Unknown stat: {stat}")
        self.stat_file.write(
            f"{seed},{stat},{tgen},{tsmt},{tsave},{trun},{self.elapsed_s:.3f}\n"
        )
        self.stat_file.flush()
        FUZZ_LOG.info(
            f"tgen={tgen:.1f}ms, tsmt={tsmt:.1f}ms, trun={trun:.1f}ms, tsave={tsave:.1f}ms"
        )

    def get_next_bug_path(self):
        return self.root / f"bug-{NNSMITH_BUG_PATTERN_TOKEN}-{self.n_bugs}"


class FuzzingLoop:
    def __init__(
        self,
        cfg: DictConfig,
    ):
        self.cfg = cfg

        # FIXME(@ganler): well-form the fix or report to TF
        # Dirty fix for TFLite on CUDA-enabled systems.
        # If CUDA is not needed, disable them all.
        cmpwith = cfg["cmp"]["with"]
        if cfg["backend"]["type"] == "tflite" and (
            cmpwith is None or cmpwith["target"] != "cuda"
        ):
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        if (
            cfg["fuzz"]["crash_safe"]
            and cfg["backend"]["type"] == "xla"
            and cfg["backend"]["target"] == "cuda"
        ) or (
            cmpwith is not None
            and cmpwith["type"] == "xla"
            and cmpwith["target"] == "cuda"
        ):
            raise ValueError(
                "Please set `fuzz.crash_safe=false` for XLA on CUDA. "
                "Also see https://github.com/ise-uiuc/nnsmith/blob/main/doc/known-issues.md"
            )

        self.crash_safe = bool(cfg["fuzz"]["crash_safe"])
        self.test_timeout = cfg["fuzz"]["test_timeout"]
        if self.test_timeout is not None:
            if isinstance(self.test_timeout, str):
                self.test_timeout = parse_timestr(self.test_timeout)
            assert isinstance(
                self.test_timeout, int
            ), "`fuzz.test_timeout` must be an integer (or with `s` (default), `m`/`min`, or `h`/`hr`)."

        if not self.crash_safe and self.test_timeout is not None:
            # user enabled test_timeout but not crash_safe.
            FUZZ_LOG.warning(
                "`fuzz.crash_safe` is automatically enabled given `fuzz.test_timeout` is set."
            )

        self.filters = []
        # add filters.
        filter_types = (
            [cfg["filter"]["type"]]
            if isinstance(cfg["filter"]["type"], str)
            else cfg["filter"]["type"]
        )
        if filter_types:
            patches = (
                [cfg["filter"]["patch"]]
                if isinstance(cfg["filter"]["patch"], str)
                else cfg["filter"]["patch"]
            )
            for f in patches:
                assert os.path.isfile(
                    f
                ), "filter.patch must be a list of file locations."
                assert "@filter(" in open(f).read(), f"No filter found in the {f}."
                spec = spec_from_file_location("module.name", f)
                spec.loader.exec_module(module_from_spec(spec))
                FUZZ_LOG.info(f"Imported filter patch: {f}")
            for filter in filter_types:
                filter = str(filter)
                if filter not in FILTERS:
                    raise ValueError(
                        f"Filter {filter} not found. Available filters: {FILTERS.keys()}"
                    )
                fn_or_cls = FILTERS[filter]
                if isinstance(fn_or_cls, Type):
                    self.filters.append(fn_or_cls())
                elif isinstance(fn_or_cls, FunctionType):
                    self.filters.append(fn_or_cls)
                else:
                    raise InternalError(
                        f"Invalid filter type: {fn_or_cls} (aka {filter})"
                    )
                FUZZ_LOG.info(f"Filter enabled: {filter}")

        resume = cfg["fuzz"]["resume"]
        self.status = StatusCollect(
            cfg["fuzz"]["root"],
            resume=resume,
            op_usage=(
                cfg["mgen"]["max_nodes"] == 1 and cfg["mgen"]["method"] == "neuri-i"
            ),
        )

        self.factory = BackendFactory.init(
            cfg["backend"]["type"],
            target=cfg["backend"]["target"],
            optmax=cfg["backend"]["optmax"],
        )

        model_cfg = self.cfg["model"]
        self.ModelType = Model.init(
            model_cfg["type"], backend_target=cfg["backend"]["target"]
        )
        self.ModelType.add_seed_setter()
        self.record_finder = None

        if cfg["mgen"]["test_pool"] : 
            AUTOINF_LOG.info(f"Using test pool: {cfg['mgen']['test_pool']}")

        self.opset = auto_opset(
        self.ModelType, 
        self.factory, 
        vulops=cfg["mgen"]["vulops"],
        test_pool=cfg["mgen"]["test_pool"],
        ) 
        if "deepconstr" in cfg["mgen"]["method"]:
            from deepconstr.gen.record import make_record_finder
            self.record_finder = make_record_finder(
                path=cfg["mgen"]["record_path"],
                pass_rate=cfg["mgen"]["pass_rate"],
                test_pool=cfg["mgen"]["test_pool"],
                filter=cfg["mgen"]["filter"],
            )
            assert len(self.record_finder) > 0, "No record found."
        else :
            from nnsmith.autoinf import make_record_finder
            self.record_finder = make_record_finder(
                path=cfg["mgen"]["record_path"],
                max_elem_per_tensor=cfg["mgen"]["max_elem_per_tensor"],
                test_pool=cfg["mgen"]["test_pool"],
            )
        
        # assert len(self.opset) > 0, "No opset found."
        seed = cfg["fuzz"]["seed"] or random.getrandbits(32)
        set_seed(seed)

        FUZZ_LOG.info(
            f"Test success info supressed -- only showing logs for failed tests"
        )

        # Time budget checking.
        self.timeout_s = self.cfg["fuzz"]["time"]
        if isinstance(self.timeout_s, str):
            self.timeout_s = parse_timestr(self.timeout_s)
        assert isinstance(
            self.timeout_s, int
        ), "`fuzz.time` must be an integer (with `s` (default), `m`/`min`, or `h`/`hr`)."

        self.save_test = cfg["fuzz"]["save_test"]
        if isinstance(self.save_test, str) and not (
            os.path.exists(self.save_test) and resume
        ):  # path of root dir.
            FUZZ_LOG.info(f"Saving all intermediate testcases to {self.save_test}")
            mkdir(self.save_test)

    def make_testcase(self, seed) -> Tuple[TestCase, int]:
        mgen_cfg = self.cfg["mgen"]
        gen = model_gen(
            opset=self.opset,
            record_finder=self.record_finder,
            method=mgen_cfg["method"],
            seed=seed,
            max_elem_per_tensor=mgen_cfg["max_elem_per_tensor"],
            max_nodes=mgen_cfg["max_nodes"],
            timeout_ms=mgen_cfg["timeout_ms"],
            noise=mgen_cfg["noise"],
            allow_zero_length_rate=mgen_cfg["allow_zero_length_rate"],
            allow_zero_rate=mgen_cfg["allow_zero_rate"],
            model=self.ModelType
        )
        ir = gen.make_concrete()

        try:
            model = self.ModelType.from_gir(ir)
            if self.cfg["debug"]["viz"]:
                model.attach_viz(ir)
            # model.refine_weights()  # DType enum error: either random generated or gradient-based.
            oracle = model.make_oracle()
            FUZZ_LOG.info(f"IR: {ir.pretty()}")
        except Exception as e:
            self.status.rej_inst(ir)
            ir.debug()
            raise e

        self.status.ok_inst(ir)
        return TestCase(model, oracle), gen.acc_smt_time_ms

    def validate_and_report(self, testcase: TestCase) -> bool:
        if not verify_testcase(
            self.cfg["cmp"],
            factory=self.factory,
            testcase=testcase,
            output_dir=self.status.get_next_bug_path(),
            filters=self.filters,
            crash_safe=self.crash_safe,
            timeout=self.test_timeout,
        ):
            self.status.n_bugs += 1
            return False
        return True

    def run(self):
        while self.status.elapsed_s < self.timeout_s:
            seed = random.getrandbits(32)
            FUZZ_LOG.debug(f"Making testcase with seed: {seed}")

            stat = {}

            gen_start = time.time()
            try :
                try:
                    testcase, tsmt = self.make_testcase(seed)
                except Exception:
                    FUZZ_LOG.error(
                        f"`make_testcase` failed with seed {seed}. It can be NNSmith or Generator ({self.cfg['model']['type']}) bug."
                    )
                    FUZZ_LOG.error(traceback.format_exc())

                    self.status.record(
                        seed=seed,
                        stat="fail",
                        tgen=round((time.time() - gen_start) * 1000),
                        **stat,
                    )
                    continue
                stat["tgen"] = round((time.time() - gen_start) * 1000)

                test_pass = True
                eval_start = time.time()
                if not self.validate_and_report(testcase):
                    test_pass = False
                    FUZZ_LOG.warning(f"Failed model seed: {seed}")
                stat["trun"] = round((time.time() - eval_start) * 1000)

                if self.save_test:
                    save_start = time.time()
                    testcase_dir = os.path.join(
                        self.save_test, f"{self.status.elapsed_s:.3f}"
                    )
                    mkdir(testcase_dir)
                    tmp, testcase.model.dotstring = testcase.model.dotstring, None
                    testcase.dump(testcase_dir)
                    testcase.model.dotstring = tmp
                    stat["tsave"] = round((time.time() - save_start) * 1000)

                self.status.record(
                    seed=seed,
                    stat="ok" if test_pass else "bug",
                    tsmt=tsmt,
                    **stat,
                )
            except : 
                pass 

        FUZZ_LOG.info(f"Total {self.status.n_testcases} testcases generated.")
        FUZZ_LOG.info(f"Total {self.status.n_bugs} bugs found.")
        FUZZ_LOG.info(f"Total {self.status.n_fail_make_test} failed to make testcases.")

@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):
    FuzzingLoop(cfg).run()


if __name__ == "__main__":
    main()
