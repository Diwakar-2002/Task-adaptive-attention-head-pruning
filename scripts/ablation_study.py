"""Design ablation: which importance scorer and which pruning compensation keep accuracy best?

Evaluates dev score at several sparsities for every (scorer, compensation) pair on the
classification tasks (cheap single-forward scoring). Results -> artifacts/results/ablation_study.json
"""

import json
import time

import numpy as np
from _common import parse

from headrouter.data.tasks import TASK_INFO, load_examples
from headrouter.evaluation.metrics import evaluate_examples
from headrouter.masks.builder import build_mask, protected_heads
from headrouter.model_utils import load_model
from headrouter.profiling.importance import SCORERS, head_output_means, layer_normalize


def main():
    args, cfg, paths = parse(__doc__, lambda ap: (
        ap.add_argument("--scorers", nargs="*", default=["gradient", "activation", "causal"]),
        ap.add_argument("--levels", nargs="*", type=float, default=[0.1, 0.2, 0.3, 0.4])))
    tasks = [t for t in cfg["tasks"] if TASK_INFO[t]["type"] == "choice"] if not args.tasks else cfg["tasks"]
    model, tok, engine, device = load_model(cfg)
    p, m = cfg["profiling"], cfg["masks"]
    out = {"levels": args.levels, "tasks": {}}
    imps = {}
    for s in args.scorers:
        f = paths.importance(s)
        if f.exists():
            b = np.load(f)
            imps[s] = {t: b["normalized"][list(b["tasks"]).index(t)] for t in tasks if t in list(b["tasks"])}
        else:
            n = p["causal_calib_size"] if s == "causal" else p["calib_size"]
            imps[s] = {t: layer_normalize(SCORERS[s](model, tok, engine, load_examples(t, "train", n, cfg), device,
                                                     p["max_target_tokens"])) for t in tasks}
    for t in tasks:
        calib = load_examples(t, "train", p["calib_size"], cfg)
        means = head_output_means(model, tok, engine, calib, device, p["max_target_tokens"])
        dev = load_examples(t, "dev", m["dev_size"][t], cfg)
        dense = evaluate_examples(model, tok, engine, dev, ["dense"] * len(dev), cfg, device)["score"]
        res = {"dense": dense}
        for s in args.scorers:
            for comp in ("none", "mean"):
                row = []
                for lv in args.levels:
                    keep = build_mask(imps[s][t], lv, m["min_heads_per_layer"])
                    engine.register("_x", keep.tolist(), means if comp == "mean" else None)
                    row.append(evaluate_examples(model, tok, engine, dev, ["_x"] * len(dev), cfg, device)["score"])
                    engine.unregister("_x")
                res[f"{s}/{comp}"] = row
                print(f"[{t}] {s:>10}/{comp:<4} dense={dense:.3f} " + " ".join(f"{lv:.0%}:{v:.3f}" for lv, v in zip(args.levels, row)), flush=True)
        out["tasks"][t] = res
    with open(paths.results / "ablation_study.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
