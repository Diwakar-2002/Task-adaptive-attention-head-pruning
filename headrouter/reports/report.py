"""Plots + Markdown report from benchmark results (all numbers come from benchmark.json)."""

from typing import Dict, List

import numpy as np


# Fixed entity -> colour (validated categorical order, light mode).
CONFIG_COLORS = {"dense": "#2a78d6", "headrouter": "#eb6834", "static": "#1baf7a",
                 "random": "#eda100", "oracle": "#e87ba4"}
CONFIG_LABELS = {"dense": "Dense", "static": "Static pruning", "random": "Random pruning",
                 "oracle": "HeadRouter (oracle task)", "headrouter": "HeadRouter"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(plt):
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": GRID,
        "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "legend.frameon": False, "axes.titleweight": "bold", "axes.titlesize": 11,
    })


def _grouped_bars(ax, tasks, configs, value):
    w = 0.8 / len(configs)
    x = np.arange(len(tasks))
    for i, c in enumerate(configs):
        vals = [value(t, c) for t in tasks]
        ax.bar(x + (i - (len(configs) - 1) / 2) * w, [v if v is not None else 0 for v in vals], w * 0.92,
               color=CONFIG_COLORS[c], label=CONFIG_LABELS[c])
    ax.set_xticks(x, [t for t in tasks])
    ax.grid(axis="x", visible=False)


def make_plots(results: Dict, registry: Dict, importance: np.ndarray, imp_tasks: List[str], out_dir) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    files = []
    tasks = list(registry["tasks"])

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out_dir / name, dpi=150)
        plt.close(fig)
        files.append(name)

    # 1. importance heatmaps with pruned heads marked
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.0 * len(tasks), 5.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, tasks):
        imp = importance[imp_tasks.index(t)]
        ax.imshow(imp, aspect="auto", cmap="Blues", interpolation="nearest")
        pruned = ~np.array(registry["tasks"][t]["keep"], dtype=bool)
        ys, xs = np.nonzero(pruned)
        ax.scatter(xs, ys, marker="x", s=10, color="#e34948", linewidths=0.8)
        ax.set_title(f"{t}\n{registry['tasks'][t]['actual_sparsity']:.0%} pruned")
        ax.set_xlabel("head")
        ax.grid(False)
    axes[0].set_ylabel("layer")
    fig.suptitle("Head importance per task (darker = more important, x = pruned)", fontsize=11)
    save(fig, "importance_heatmaps.png")

    # 1b. task-to-task similarity of importance (Spearman) - is pruning really task-dependent?
    from scipy.stats import spearmanr
    T = len(imp_tasks)
    sim = np.array([[spearmanr(importance[i].ravel(), importance[j].ravel())[0] for j in range(T)] for i in range(T)])
    results["task_similarity"] = {"tasks": imp_tasks, "spearman": sim.round(4).tolist()}
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.imshow(sim, cmap="Blues", vmin=0, vmax=1)
    for i in range(T):
        for j in range(T):
            ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="#fcfcfb" if sim[i, j] > 0.6 else INK)
    ax.set_xticks(range(T), imp_tasks, rotation=30, ha="right")
    ax.set_yticks(range(T), imp_tasks)
    ax.grid(False)
    ax.set_title("Head-importance rank correlation between tasks")
    save(fig, "task_similarity.png")

    # 2. dev score vs sparsity (relative to dense)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, t in enumerate(tasks):
        sw = registry["tasks"][t]["sweep"]
        d = sw[0]["score"] or 1e-9
        xs = [p["sparsity"] * 100 for p in sw]
        ys = [p["score"] / d * 100 for p in sw]
        ax.plot(xs, ys, color=SERIES[i], lw=2, marker="o", ms=5, label=t)
        ax.annotate(t, (xs[-1], ys[-1]), xytext=(5, 0), textcoords="offset points", color=INK2, fontsize=8,
                    va="center")
    b = registry["accuracy_budget"]
    if registry["budget_type"] == "relative":
        ax.axhline(100 - b * 100, color=INK2, ls="--", lw=1)
        ax.text(1, 100 - b * 100, f" budget (-{b:.1%})", color=INK2, fontsize=8, va="bottom")
    ax.set_xlabel("heads pruned (%)")
    ax.set_ylabel("dev score (% of dense)")
    ax.set_title("Accuracy vs sparsity (dev split)")
    ax.legend(loc="lower left")
    save(fig, "accuracy_vs_sparsity.png")

    if results.get("quality"):
        fig, ax = plt.subplots(figsize=(9, 4))
        configs = ["dense", "static", "random", "oracle", "headrouter"]
        _grouped_bars(ax, tasks, configs, lambda t, c: results["quality"][t][c])
        ax.set_ylabel("task score (test split)")
        ax.set_title("Quality by configuration")
        ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
        save(fig, "quality.png")

    if results.get("speed"):
        configs = ["dense", "static", "oracle", "headrouter"]
        for metric, label, name in (("total_ms", "end-to-end latency, median (ms)", "latency.png"),
                                    ("energy_per_token_mj", "GPU energy per token, median (mJ)", "energy.png")):
            if all(results["speed"][t]["dense"].get(metric) is None for t in tasks):
                continue
            fig, ax = plt.subplots(figsize=(9, 4))
            _grouped_bars(ax, tasks, configs,
                          lambda t, c: (results["speed"][t][c][metric] or {}).get("median"))
            ax.set_ylabel(label)
            ax.set_title(f"{label.split(',')[0].capitalize()} - {results['meta']['speed_new_tokens']} new tokens")
            ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
            save(fig, name)

    cs = results.get("context_sweep")
    if cs:
        n = [r["context"] for r in cs["rows"]]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
        a1.plot(n, [r["dense"]["ttft_ms"]["median"] for r in cs["rows"]], color=CONFIG_COLORS["dense"], lw=2,
                marker="o", label="Dense")
        a1.plot(n, [r["pruned"]["ttft_ms"]["median"] for r in cs["rows"]], color=CONFIG_COLORS["headrouter"],
                lw=2, marker="o", label=f"HeadRouter ({cs['task']}, {cs['sparsity']:.0%})")
        a1.set_xlabel("input tokens")
        a1.set_ylabel("prefill / TTFT (ms)")
        a1.set_title("Prefill time vs context length")
        a1.legend()
        for key, lab, col in (("ttft_ms", "prefill", SERIES[0]), ("tpot_ms", "decode / token", SERIES[1])):
            a2.plot(n, [(r["dense"][key]["median"] / r["pruned"][key]["median"] - 1) * 100 for r in cs["rows"]],
                    color=col, lw=2, marker="o", label=lab)
        a2.axhline(0, color=INK2, lw=1)
        a2.set_xlabel("input tokens")
        a2.set_ylabel("speedup vs dense (%)")
        a2.set_title("Speedup grows with context length")
        a2.legend()
        save(fig, "context_sweep.png")

    ss = results.get("sparsity_sweep")
    if ss:
        dense = ss["rows"][0]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
        for ax, key, title in ((a1, "ttft_ms", "Prefill speedup"), (a2, "tpot_ms", "Decode speedup")):
            for i, mode in enumerate(("query", "kv_group")):
                rows = [r for r in ss["rows"] if r["mode"] == mode]
                ax.plot([0] + [r["sparsity"] * 100 for r in rows],
                        [0] + [(dense[key]["median"] / r[key]["median"] - 1) * 100 for r in rows],
                        color=SERIES[i], lw=2, marker="o", label=f"{mode} mode")
            ax.set_xlabel("heads pruned (%)")
            ax.set_ylabel("speedup vs dense (%)")
            ax.set_title(f"{title} at {ss['context']} input tokens")
            ax.legend()
        save(fig, "sparsity_sweep.png")
    return files


def _f(x, fmt="{:.1f}"):
    return "-" if x is None else fmt.format(x)


def write_report(results: Dict, registry: Dict, plots: List[str], path):
    m = results["meta"]
    tasks = list(registry["tasks"])
    L = [f"# HeadRouter benchmark report\n",
         f"Model **{m['model']}** ({m['dtype']}) on **{m['gpu']}**, torch {m['torch']}, "
         f"energy via NVML ({m['energy_mode']}). Generated {m['created']}"
         + (" — **quick mode (small samples)**" if m.get("quick") else "") + ".\n",
         "## Selected masks (dev split)\n",
         "| Task | Metric | Dense | Pruned | Heads pruned |", "|---|---|---|---|---|"]
    for t in tasks:
        e = registry["tasks"][t]
        L.append(f"| {t} | {m['metric_names'][t]} | {e['dense_dev']:.4f} | {e['pruned_dev']:.4f} | "
                 f"{e['actual_sparsity']:.1%} |")
    L.append(f"\nStatic baseline uses one global mask at {results['static_sparsity']:.0%} sparsity. "
             f"Budget: {registry['accuracy_budget']} ({registry['budget_type']}).\n")

    if results.get("quality"):
        L += ["## Quality (test split)\n", "| Task | " + " | ".join(CONFIG_LABELS.values()) + " | Classifier acc |",
              "|---" * 7 + "|"]
        for t in tasks:
            q = results["quality"][t]
            L.append(f"| {t} | " + " | ".join(f"{q[c]:.4f}" for c in CONFIG_LABELS)
                     + f" | {results['routing'][t]['classifier_accuracy']:.2%} |")
    c = results.get("classifier")
    if c:
        L.append(f"\nClassifier (held-out, varied phrasings): accuracy {c['accuracy']:.2%}, "
                 f"latency median {c['latency_ms_median']:.1f} ms (p95 {c['latency_ms_p95']:.1f} ms), "
                 f"{c['routed_fraction']:.0%} of prompts above the {c['confidence_threshold']} confidence threshold.\n")

    if results.get("speed"):
        L += [f"## Speed and energy ({m['speed_new_tokens']} new tokens, median)\n",
              "| Task | Config | Latency ms | p95 ms | TTFT ms | ms/token | tok/s | Energy J | Speedup |",
              "|---|---|---|---|---|---|---|---|---|"]
        for t in tasks:
            s = results["speed"][t]
            d = s["dense"]["total_ms"]["median"]
            for cname, v in s.items():
                g = lambda k: v[k]["median"] if v.get(k) else None  # noqa: E731
                L.append(f"| {t} | {CONFIG_LABELS[cname]} | {_f(g('total_ms'))} | {_f(v['total_ms']['p95'])} | "
                         f"{_f(g('ttft_ms'))} | {_f(g('tpot_ms'), '{:.2f}')} | {_f(g('tokens_per_s'))} | "
                         f"{_f(g('energy_j'), '{:.2f}')} | {d / g('total_ms'):.3f}x |")
    cs = results.get("context_sweep")
    if cs:
        L += [f"\n## Context-length sweep ({cs['task']} mask, {cs['sparsity']:.0%} heads pruned)\n",
              "| Input tokens | Dense TTFT ms | Pruned TTFT ms | Prefill speedup | Dense ms/tok | Pruned ms/tok |",
              "|---|---|---|---|---|---|"]
        for r in cs["rows"]:
            d, p = r["dense"], r["pruned"]
            L.append(f"| {r['context']} | {d['ttft_ms']['median']:.1f} | {p['ttft_ms']['median']:.1f} | "
                     f"{d['ttft_ms']['median'] / p['ttft_ms']['median']:.3f}x | {d['tpot_ms']['median']:.2f} | "
                     f"{p['tpot_ms']['median']:.2f} |")
    L.append("\n## Plots\n")
    L += [f"![{p}](plots/{p})" for p in plots]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def write_outputs(results, registry, importance, imp_tasks, paths):
    plots = make_plots(results, registry, importance, imp_tasks, paths.plots)
    write_report(results, registry, plots, paths.results / "report.md")
