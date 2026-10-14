"""Step 3 - build constrained per-task masks, pick the sparsity that fits the accuracy budget."""

import time

import numpy as np
from _common import parse

from headrouter.data.tasks import load_examples
from headrouter.evaluation.metrics import evaluate_examples
from headrouter.masks.builder import (build_mask, iterative_masks, load_head_means, protected_heads,
                                      save_registry, static_scores, within_budget)
from headrouter.model_utils import load_model
from headrouter.profiling.importance import gradient_importance, layer_normalize


def main():
    args, cfg, paths = parse(__doc__, lambda ap: ap.add_argument("--scorer", default=None))
    m = cfg["masks"]
    scorer = args.scorer or cfg["profiling"]["scorer"]
    blob = np.load(paths.importance(scorer))
    key = "normalized" if cfg["profiling"].get("layer_normalize", True) else "raw"
    imp_tasks = list(blob["tasks"])
    importance = blob[key]

    model, tok, engine, device = load_model(cfg)
    L, H, KV = engine.num_layers, engine.num_heads, engine.num_kv
    prot = protected_heads(importance, m["protected_top_frac"])
    compensation = m.get("compensation", "none")
    means = load_head_means(paths.head_means) if compensation == "mean" else None
    if compensation == "mean" and means is None:
        raise SystemExit("compensation=mean needs artifacts/importance/head_means.npz - rerun scripts/02_profile.py")
    print(f"scorer={scorer}  mode={m['mode']}  compensation={compensation}  protected heads={int(prot.sum())}  "
          f"budget={m['accuracy_budget']} ({m['budget_type']})")

    def mk(scores, s):
        return build_mask(scores, s, m["min_heads_per_layer"], prot, m["mode"], KV)

    registry = {"model": cfg["model"]["name"], "scorer": scorer, "mode": m["mode"],
                "num_layers": L, "num_heads": H, "num_kv_heads": KV,
                "compensation": compensation, "iterative": bool(m.get("iterative")), "accuracy_budget": m["accuracy_budget"], "budget_type": m["budget_type"],
                "min_heads_per_layer": m["min_heads_per_layer"], "protected": prot.tolist(),
                "created": time.strftime("%Y-%m-%d %H:%M:%S"), "tasks": {}}

    for t in cfg["tasks"]:
        scores = importance[imp_tasks.index(t)]
        dev = load_examples(t, "dev", m["dev_size"][t], cfg)
        t0 = time.time()
        dense = evaluate_examples(model, tok, engine, dev, ["dense"] * len(dev), cfg, device)["score"]
        sweep = [{"sparsity": 0.0, "score": dense, "ok": True}]
        print(f"\n[{t}] dense dev score = {dense:.4f}  (n={len(dev)})")
        hm = None if means is None else means[t]
        if m.get("iterative") and scorer == "gradient":
            calib = load_examples(t, "train", cfg["profiling"]["calib_size"], cfg)
            norm = layer_normalize if cfg["profiling"].get("layer_normalize") else (lambda x: x)
            by_level = iterative_masks(
                lambda var: gradient_importance(model, tok, engine, calib, device,
                                                cfg["profiling"]["max_target_tokens"], var),
                m["sparsity_levels"], m["min_heads_per_layer"], prot, m["mode"], KV,
                lambda n, k: engine.register(n, k, hm), norm)
            engine.unregister("_iter")
        else:
            by_level = {s: mk(scores, s) for s in m["sparsity_levels"]}
        chosen = 0.0
        for s in m["sparsity_levels"]:
            keep = by_level[s]
            engine.register("_sweep", keep.tolist(), hm)
            score = evaluate_examples(model, tok, engine, dev, ["_sweep"] * len(dev), cfg, device)["score"]
            engine.unregister("_sweep")
            ok = within_budget(dense, score, m["accuracy_budget"], m["budget_type"])
            sweep.append({"sparsity": s, "actual_sparsity": float(1 - keep.mean()), "score": score, "ok": ok})
            print(f"   sparsity {s:.0%}: {score:.4f}  {'OK' if ok else 'over budget'}")
            if ok:
                chosen = s
            elif m.get("stop_at_first_failure", True):
                break
        keep = by_level[chosen] if chosen > 0 else mk(scores, 0.0)
        registry["tasks"][t] = {"sparsity": chosen, "actual_sparsity": float(1 - keep.mean()),
                                "dense_dev": dense, "pruned_dev": next(x["score"] for x in sweep
                                                                       if x["sparsity"] == chosen),
                                "keep": keep.tolist(), "sweep": sweep, "seconds": time.time() - t0}
        print(f"   -> chosen sparsity {chosen:.0%}  ({time.time() - t0:.0f}s)")

    # Baselines at matched sparsity.
    chosen = [registry["tasks"][t]["sparsity"] for t in cfg["tasks"]]
    levels = [0.0] + list(m["sparsity_levels"])
    s_static = min(levels, key=lambda x: abs(x - float(np.mean(chosen))))
    ordered = np.stack([importance[imp_tasks.index(t)] for t in cfg["tasks"]])
    registry["static"] = {"sparsity": s_static, "keep": mk(static_scores(ordered), s_static).tolist()}
    rng = np.random.default_rng(cfg["seed"])
    registry["random"] = {t: {"sparsity": s, "keep": mk(rng.random((L, H)), s).tolist()}
                          for t, s in zip(cfg["tasks"], chosen)}
    save_registry(paths.registry, registry)
    print(f"\nstatic baseline sparsity = {s_static:.0%}\nsaved {paths.registry}")


if __name__ == "__main__":
    main()
