"""Plot the design studies (ablation_study.json, iterative_study.json) into artifacts/results/plots."""

import json

from _common import parse

from headrouter.reports.report import INK2, SERIES, _style


def main():
    _, cfg, paths = parse(__doc__)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)

    f = paths.results / "ablation_study.json"
    if f.exists():
        d = json.load(open(f))
        tasks = list(d["tasks"])
        fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 3.8), sharey=True)
        for ax, t in zip(axes, tasks):
            r = d["tasks"][t]
            xs = [0] + [l * 100 for l in d["levels"]]
            for i, sc in enumerate(("gradient", "activation", "causal")):
                for comp, ls in (("none", "--"), ("mean", "-")):
                    k = f"{sc}/{comp}"
                    if k in r:
                        ax.plot(xs, [r["dense"]] + r[k], color=SERIES[i], ls=ls, lw=2 if comp == "mean" else 1.3,
                                marker="o", ms=4, label=f"{sc} + {'mean comp.' if comp == 'mean' else 'zero-ablation'}")
            ax.set_title(t)
            ax.set_xlabel("heads pruned (%)")
        axes[0].set_ylabel("dev accuracy")
        axes[0].legend(fontsize=8, loc="lower left")
        fig.suptitle("Design study 1: importance scorer x pruning compensation (one-shot, layer-normalised)", fontsize=11)
        fig.tight_layout()
        fig.savefig(paths.plots / "study_scorer_compensation.png", dpi=150)
        plt.close(fig)
        print("saved study_scorer_compensation.png")

    f = paths.results / "iterative_study.json"
    if f.exists():
        d = json.load(open(f))
        tasks = list(d["tasks"])
        fig, axes = plt.subplots(1, len(tasks), figsize=(4.6 * len(tasks), 3.8), sharey=False)
        axes = [axes] if len(tasks) == 1 else axes
        for ax, t in zip(axes, tasks):
            r = d["tasks"][t]
            xs = [0] + [l * 100 for l in d["levels"]]
            for i, k in enumerate(("oneshot/layernorm", "iterative/layernorm", "oneshot/raw", "iterative/raw")):
                ax.plot(xs, [r["dense"]] + r[k], color=SERIES[i], lw=2, marker="o", ms=4, label=k)
            ax.axhline(r["dense"], color=INK2, lw=1, ls=":")
            ax.set_title(t)
            ax.set_xlabel("heads pruned (%)")
            ax.set_ylabel("dev accuracy")
        axes[0].legend(fontsize=8, loc="lower left")
        fig.suptitle("Design study 2: one-shot vs iterative, layer-normalised vs raw gradient scores (mean comp.)",
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(paths.plots / "study_iterative.png", dpi=150)
        plt.close(fig)
        print("saved study_iterative.png")


if __name__ == "__main__":
    main()
