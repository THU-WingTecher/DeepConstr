import os

import matplotlib.pyplot as plt
import numpy as np

SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 17

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=MEDIUM_SIZE - 1)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

MIN_FAC = 0.85
MAX_FAC = 1.02

plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{xfrac}")


class Ploter:
    def __init__(self, cov_lim=None, use_pdf=False, one_plot=False, scale=1) -> None:
        # cov / time, cov / iteration, iteration / time
        fig0, axs0 = plt.subplots(1, 1, constrained_layout=True, figsize=(5.7, 3.4))
        fig1, axs1 = plt.subplots(1, 1, constrained_layout=True, figsize=(5, 3))

        self.one_plot = one_plot
        self.fig = [fig0, fig1]
        self.axs = [axs0, axs1]
        self.cov_lim = cov_lim
        self.cov_maxes = []
        self.xspan = 0
        self.use_pdf = use_pdf
        self.scale = scale

    def add(self, data, name=None):
        df = np.array(data)
        df[:, 2] = df[:, 2] / self.scale

        LW = 2
        MARKER_SIZE = 10
        N_MARKER = 8
        MARKERS = ["d", "^", "p", "*", "o", "s"]
        LS = ":"
        COLORS = ["dodgerblue", "violet", "coral", "limegreen", "gold", "lightgray"]

        # make it even over time
        markevery = np.zeros_like(df[:, 0], dtype=bool)
        step = int(df[:, 0].max() / N_MARKER)
        offset = step
        for _ in range(N_MARKER):
            idx = list(map(lambda i: i >= offset, df[:, 0])).index(True)
            markevery[idx] = True
            offset += step

        # markevery = int(len(df) / N_MARKER)
        marker = MARKERS[len(self.cov_maxes) % len(MARKERS)]
        color = COLORS[len(self.cov_maxes) % len(MARKERS)]

        # linestyle=LS, marker=marker, markevery=markevery, markersize=MARKER_SIZE, alpha=ALPHA, lw=LW
        style_kw = {
            "linestyle": LS,
            "marker": marker,
            "markevery": markevery,
            "markersize": MARKER_SIZE,
            "lw": LW,
            "color": color,
            "markeredgecolor": "k",
            "markeredgewidth": 1.5,
        }

        self.axs[0].plot(df[:, 0] / 60, df[:, 2], label=name, **style_kw)  # cov / time
        print(
            f"----> max cov {df[:, 2].max() * self.scale} + max tests {int(df[:, 1].max())}"
        )

        self.axs[1].plot(df[:, 1], df[:, 2], label=name, **style_kw)  # cov / iteration

        self.xspan = max(self.xspan, df[-1, 0])

        self.cov_maxes.append(df[:, 2].max())

    def plot(self, save="cov", cov_lim=None, loc=0, name='not_defined'):
        for ax in self.axs:
            handles, labels = ax.get_legend_handles_labels()
            # rank by self.cov_maxes argsort
            labels = [labels[i] for i in reversed(np.argsort(self.cov_maxes))]
            handles = [handles[i] for i in reversed(np.argsort(self.cov_maxes))]
            ax.legend(handles, labels, loc=loc, ncol=3)

        self.cov_maxes = sorted(self.cov_maxes)

        if len(self.cov_maxes) > 1:
            for rk in range(1, len(self.cov_maxes)):
                best = self.cov_maxes[-1]
                cur = self.cov_maxes[-(rk + 1)]
                print(
                    f"==> Best one is [{best} / {cur}] = **{best / cur:.2f}x** better than the NO. {rk + 1} baseline."
                )

        cov_max = max(self.cov_maxes)
        cov_min = min(self.cov_maxes)

        if cov_lim is not None:
            ann_size = BIGGER_SIZE + 4
            max_cov_unscale = int(cov_max * self.scale)
            self.axs[0].annotate(
                f"$\\sfrac{{{max_cov_unscale // 1000}k_\\mathbf{{best}}}}{{{int(cov_lim / 1000)}k_\\mathbf{{max}}}}$ = \\textbf{{{max_cov_unscale / cov_lim * 100 :.1f}\%}}",
                xy=(0.97, 0.3),
                xycoords="axes fraction",
                ha="right",
                fontsize=ann_size,
                bbox=dict(boxstyle="round", fc="w", alpha=0.6),
            )

        if self.cov_lim is not None:
            self.axs[0].set_ylim(bottom=self.cov_lim, top=cov_max * MAX_FAC)
            self.axs[1].set_ylim(bottom=self.cov_lim, top=cov_max * MAX_FAC)
        else:
            self.axs[0].set_ylim(bottom=cov_min * MIN_FAC, top=cov_max * MAX_FAC)
            self.axs[1].set_ylim(bottom=cov_min * MIN_FAC, top=cov_max * MAX_FAC)

        ylabel = "\# Coverage"
        if self.scale != 1:
            ylabel = f"\# Coverage ({self.scale} branches)"

        self.axs[0].set(xlabel="Time (Minute)")
        # self.axs[0].set(xlabel="Time (Minute)", ylabel=ylabel)
        self.axs[0].grid(alpha=0.5, ls=":")

        self.axs[1].set(xlabel="\# Iteration", ylabel=ylabel)
        self.axs[1].grid(alpha=0.5, ls=":")

        if self.use_pdf:
            self.fig[0].savefig(save + f"{name}-time.pdf")
            self.fig[1].savefig(save + f"{name}-iter.pdf")
        self.fig[0].savefig(save + f"{name}-time.png")
        self.fig[1].savefig(save + f"{name}-iter.png")


def cov_summerize(data, tlimit=None, gen_time=None):
    model_total = 0

    branch_by_time = [[0, 0, 0]]
    final_bf = 0

    for time, value in data.items():
        bf = 0

        n_model = value["n_model"]
        cov = value["merged_cov"]
        model_total += n_model
        if gen_time is not None:
            time -= gen_time[0][:model_total].sum()

        branch_cov = 0
        for fname in cov:
            branch_cov += len(cov[fname]["branches"])
            bf += cov[fname]["bf"]

        branch_by_time.append([time, model_total, branch_cov])

        final_bf = max(final_bf, bf)

        if tlimit is not None and time > tlimit:
            break
    return branch_by_time, final_bf


def plot_one_round(
    folder,
    data,
    fuzz_tags=None,
    target_tag="",
    tlimit=None,
    pdf=False,
    scale=1,
    name=None,
):
    branch_ploter = Ploter(use_pdf=pdf, scale=scale)

    assert fuzz_tags is not None

    # Due to lcov, diff lcov's total cov might be slightly different.
    # We took the max.
    bf = 0

    for idx, (k, v) in enumerate(data.items()):
        branch_by_time, bf_ = cov_summerize(v, tlimit=tlimit)
        branch_ploter.add(data=branch_by_time, name=fuzz_tags[idx])
        bf = max(bf, bf_)

    branch_ploter.plot(save=os.path.join(folder, target_tag + "branch_cov"), cov_lim=bf, name=name)

    plt.close()


if "__main__" == __name__:
    import argparse
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--folders", type=str, nargs="+", help="bug report folder"
    )
    parser.add_argument(
        "-n", "--name", type=str, help="file_name"
    )
    parser.add_argument("--tags", type=str, nargs="+", help="tags")
    parser.add_argument(
        "-o", "--output", type=str, default="results", help="results folder"
    )
    parser.add_argument("-t", "--tlimit", type=int, default=4 * 3600, help="time limit")
    parser.add_argument("--pdf", action="store_true", help="use pdf as well")
    parser.add_argument("--minfac", type=float, default=0.85, help="min factor")
    parser.add_argument("--maxfac", type=float, default=1.02, help="max factor")
    args = parser.parse_args()

    MIN_FAC = args.minfac
    MAX_FAC = args.maxfac

    if args.tags is None:
        args.tags = [os.path.split(f)[-1].split("-")[0] for f in args.folders]
        # args.tags = [os.path.split(os.path.split(f)[-2])[-1]
        #              for f in args.folders]
    else:
        assert len(args.tags) == len(args.folders)

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    data = {}
    gen_time = None
    for f in args.folders:
        with open(os.path.join(f, "merged_cov.pkl"), "rb") as fp:
            data[f] = pickle.load(fp)

    plot_one_round(
        folder=args.output,
        data=data,
        tlimit=args.tlimit,
        scale=1000,
        fuzz_tags=args.tags,
        pdf=args.pdf,
        name=args.name
    )  # no pass
