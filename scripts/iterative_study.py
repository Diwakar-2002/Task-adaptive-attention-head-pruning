"""Design study 2: one-shot vs iterative gradient pruning (both with bias compensation)."""

import json

import numpy as np
from _common import parse

from headrouter.data.tasks import load_examples
from headrouter.evaluation.metrics import evaluate_examples
from headrouter.masks.builder import build_mask, iterative_masks
from headrouter.model_utils import load_model
from headrouter.profiling.importance import gradient_importance, head_output_means, layer_normalize


def main():
    args, cfg, paths = parse(__doc__)
    levels = [0.1, 0.2, 0.3, 0.4, 0.5]
    model, tok, engine, device = load_model(cfg)
    p, m = cfg["profiling"], cfg["masks"]
    out = {}
    for t in cfg["tasks"]:
        calib = load_examples(t, "train", p["calib_size"], cfg)
        means = head_output_means(model, tok, engine, calib, device, p["max_target_tokens"])
        dev = load_examples(t, "dev", m["dev_size"][t], cfg)
        score = lambda var: gradient_importance(model, tok, engine, calib, device, p["max_target_tokens"], var)  # noqa
        ev = lambda keep: (engine.register("_e", keep.tolist(), means),  # noqa
                           evaluate_examples(model, tok, engine, dev, ["_e"] * len(dev), cfg, device)["score"],
                           engine.unregister("_e"))[1]
        dense = evaluate_examples(model, tok, engine, dev, ["dense"] * len(dev), cfg, device)["score"]
        res = {"dense": dense}
        for norm_name, norm in (("layernorm", layer_normalize), ("raw", lambda x: x)):
            base = norm(score("dense"))
            res[f"oneshot/{norm_name}"] = [ev(build_mask(base, s, m["min_heads_per_layer"])) for s in levels]
            it = iterative_masks(score, levels, m["min_heads_per_layer"], None, "query", engine.num_kv,
                                 lambda n, k: engine.register(n, k, means), norm)
            engine.unregister("_iter")
            res[f"iterative/{norm_name}"] = [ev(it[s]) for s in levels]
            for k in (f"oneshot/{norm_name}", f"iterative/{norm_name}"):
                print(f"[{t}] {k:<20} dense={dense:.3f} " + " ".join(f"{s:.0%}:{v:.3f}" for s, v in zip(levels, res[k])), flush=True)
        out[t] = res
    with open(paths.results / "iterative_study.json", "w") as f:
        json.dump({"levels": levels, "tasks": out}, f, indent=1)


if __name__ == "__main__":
    main()
