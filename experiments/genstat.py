if __name__ == "__main__":
    import argparse
    import os

    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str)
    args = parser.parse_args()

    with open(os.path.join(args.root, "status.csv"), "r") as f:
        n_ok = 0
        n_fail = 0
        n_bug = 0
        ok_tgen = []
        ok_tsmt = []
        ok_tsave = []
        ok_trun = []
        ok_tsum = []
        last_tsum = 0
        for line in f.readlines()[1:]:
            seed, stat, tgen, tsmt, tsave, trun, elapsed = line.split(",")
            tcur = float(elapsed)
            if stat == "ok":
                n_ok += 1
                ok_tgen.append(float(tgen))
                ok_tsmt.append(float(tsmt))
                ok_tsave.append(float(tsave))
                ok_trun.append(float(trun))
                ok_tsum.append(tcur - last_tsum)
            elif stat == "fail":
                n_fail += 1
            elif stat == "bug":
                n_bug += 1
            last_tsum = tcur

    n_valid = n_ok + n_bug
    print(
        f"validity rate = {n_valid}/{n_valid + n_fail} = {100 * n_valid / (n_valid + n_fail):.1f}%"
    )

    # numpify
    ok_tgen = np.array(ok_tgen)
    ok_tsmt = np.array(ok_tsmt)
    ok_tsave = np.array(ok_tsave)
    ok_trun = np.array(ok_trun)
    ok_tsum = np.array(ok_tsum) * 1000

    # compute avg, p50, p90, p99 for each stat
    # and draw a latex table
    def summarize_stat(l: np.ndarray):
        l.sort()
        avg = np.mean(l)
        p50 = l[int(len(l) * 0.5)]
        p90 = l[int(len(l) * 0.9)]
        p99 = l[int(len(l) * 0.99)]
        return avg, p50, p90, p99

    prefix = r"""
\begin{table}[]
\centering
\begin{tabular}{c cccc}
\hline
           & Gen. (SMT)     & Eval.      & Save        & Total  \\
\hline
"""
    template = r"""
Avgerage   & {avg_tgen:.0f} ({avg_tsmt:.0f}) & {avg_trun:.0f} & {avg_tsave:.0f} & {avg_tsum:.0f} \\
P50        & {p50_tgen:.0f} ({p50_tsmt:.0f}) & {p50_trun:.0f} & {p50_tsave:.0f} & {p50_tsum:.0f} \\
P90        & {p90_tgen:.0f} ({p90_tsmt:.0f}) & {p90_trun:.0f} & {p90_tsave:.0f} & {p90_tsum:.0f} \\
P99        & {p99_tgen:.0f} ({p99_tsmt:.0f}) & {p99_trun:.0f} & {p99_tsave:.0f} & {p99_tsum:.0f} \\
Percentage & {pgen:.0f}\%   ({psmt:.0f}\%  ) & {prun:.0f}\%   & {psave:.0f}\% & 100\% \\
"""
    suffix = r"""
\hline
\end{tabular}
\caption{Time breakdown per test-case (millisecond).}
\label{tab:tstat}
\end{table}
    """

    # Perc.   & {avg_tgen / avg_tsum * 100:.0f} & {avg_tsmt / avg_tsum * 100:.0f} & {avg_trun / avg_tsum * 100:.0f} & {avg_tsave / avg_tsum * 100:.0f} & 100 \\
    avg_tgen, p50_tgen, p90_tgen, p99_tgen = summarize_stat(ok_tgen)
    avg_tsmt, p50_tsmt, p90_tsmt, p99_tsmt = summarize_stat(ok_tsmt)
    avg_tsave, p50_tsave, p90_tsave, p99_tsave = summarize_stat(ok_tsave)
    avg_trun, p50_trun, p90_trun, p99_trun = summarize_stat(ok_trun)
    avg_tsum, p50_tsum, p90_tsum, p99_tsum = summarize_stat(ok_tsum)
    print(
        prefix
        + template.format(
            avg_tgen=avg_tgen,
            p50_tgen=p50_tgen,
            p90_tgen=p90_tgen,
            p99_tgen=p99_tgen,
            avg_tsmt=avg_tsmt,
            p50_tsmt=p50_tsmt,
            p90_tsmt=p90_tsmt,
            p99_tsmt=p99_tsmt,
            avg_tsave=avg_tsave,
            p50_tsave=p50_tsave,
            p90_tsave=p90_tsave,
            p99_tsave=p99_tsave,
            avg_trun=avg_trun,
            p50_trun=p50_trun,
            p90_trun=p90_trun,
            p99_trun=p99_trun,
            avg_tsum=avg_tsum,
            p50_tsum=p50_tsum,
            p90_tsum=p90_tsum,
            p99_tsum=p99_tsum,
            pgen=avg_tgen / avg_tsum * 100,
            psmt=avg_tsmt / avg_tsum * 100,
            prun=avg_trun / avg_tsum * 100,
            psave=avg_tsave / avg_tsum * 100,
        )
        + suffix
    )
