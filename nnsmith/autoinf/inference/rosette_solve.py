import argparse
import multiprocessing as mp
import os
import pickle
import time
from subprocess import PIPE, Popen, TimeoutExpired

from nnsmith.autoinf.inference.const import DATA_DIR, GEN_DIR
from nnsmith.autoinf.inference.invocations import inst_augmented_invocations
from nnsmith.logger import AUTOINF_LOG
from nnsmith.autoinf.instrument.categorize import gen_inst_with_records
from nnsmith.autoinf.instrument.op import OpInstance


def gen_rosette_code(invocations: list, o_id: int) -> str:
    argCount = len(invocations[0][0])
    arglist = " ".join([f"s{i}" for i in range(argCount)])

    header = f"\
#lang rosette\n\
(require rosette/lib/synthax)\n\
(define int32? (bitvector 32))\n\
(define (int32 i) (bv i int32?))\n"

    definitions = f"(define-symbolic {arglist} int32?)\n"

    database = f"\
(define (database {arglist})\n\
  (cond\n"
    for (in_list, out_list) in invocations:
        entry = "    [(and"
        for i in range(argCount):
            entry += f" (eq? s{i} (int32 {in_list[i]}))"
        entry += f") => (int32 {out_list[o_id]})]\n"
        database += entry
    database += "    [else (int32 -1)]\n  ))\n"

    checker = f"\
(define (check-equal impl std {arglist})\n\
  (define out (impl {arglist}))\n\
  (define ans (std {arglist}))\n\
  (assert (or (eq? out ans) (eq? (int32 -1) ans)))\n\
)\n"

    grammar = f"\
(define-grammar (ArithExpTree {arglist})\n\
  [expr\n\
   (choose {arglist} (int32 1) (int32 2)\n\
           ((bop) (expr) (expr)))]\n\
  [bop\n\
   (choose bvadd bvsub bvmul bvudiv bvurem)])\n\
(define (myterm {arglist})\n\
  (ArithExpTree {arglist} #:depth 3))\n"

    synthesis = f"\
(define sol\n\
  (synthesize\n\
   #:forall (list {arglist})\n\
   #:guarantee\n\
     (check-equal myterm database {arglist})\n\
  ))\n"

    output = f'(if (sat? sol) (print-forms sol) (println "UNSAT"))'

    return header + definitions + database + checker + grammar + synthesis + output


def solve_inst(inst: OpInstance, dump_dir: str):
    library = "torch" if "torch" in inst.name_index else "tf"
    rosette_dir = os.path.join(dump_dir, f"{library}_rosette")
    augmented_dir = os.path.join(dump_dir, f"{library}_augmented_records")
    invocations = inst_augmented_invocations(inst, "success", augmented_dir)
    if len(invocations) == 0:
        return
    output_rank = len(invocations[0][1])
    time_used = 0
    status = "success"
    time_limit = 100 / output_rank
    rules = []
    filename = f"{inst.name_index}.rkt"
    for o_id in range(output_rank):
        code = gen_rosette_code(invocations, o_id)
        with open(filename, "w") as f:
            f.write(code)
        start_time = time.time()
        proc = Popen(["racket", filename], stdout=PIPE)
        try:
            outs, errs = proc.communicate(timeout=time_limit)
        except TimeoutExpired:
            proc.terminate()
            status = "timeout"
            outs, errs = None, None
        end_time = time.time()
        time_used += end_time - start_time
        outs = outs.decode()
        res = outs.split("\n")[-2]
        if res == "UNSAT":
            status = "unsat"
        else:
            rules.append(res)
        if status != "success":
            break
    inst_summary = {"status": status, "time": time_used, "rules": rules}
    with open(os.path.join(rosette_dir, f"{inst.name_index}.pkl"), "wb") as f:
        pickle.dump(inst_summary, f)
    os.system(f"rm {filename}")
    AUTOINF_LOG.info(f"{inst.name_index} complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_dir", type=str, default=GEN_DIR)
    parser.add_argument("--parallel", type=int, default=16)
    parser.add_argument("--library", nargs="+", default=["torch"])
    args = parser.parse_args()
    for library in args.library:
        record_dir = os.path.join(DATA_DIR, f"{library}_records")
        rosette_dir = os.path.join(args.dump_dir, f"{library}_rosette")
        os.system(f"rm {rosette_dir} -r")
        os.makedirs(rosette_dir)
        p = mp.Pool(args.parallel)
        gen_with_records = gen_inst_with_records(
            data_dir=record_dir, int_policy="fix_dim"
        )
        for i_op, (inst, records) in enumerate(gen_with_records):
            p.apply_async(solve_inst, (inst, args.dump_dir))
            # solve_inst(inst, args.dump_dir)
        p.close()
        p.join()
