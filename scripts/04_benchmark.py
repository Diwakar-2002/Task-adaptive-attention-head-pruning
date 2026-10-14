"""
Step 4 - benchmark Dense vs Static vs Random vs Oracle-task HeadRouter vs HeadRouter.

  quality   task metric on the held-out test split for every configuration
  speed     end-to-end latency, TTFT, time/output token, tokens/s, GPU energy, working memory
            (fixed-length greedy generation, configurations interleaved round-robin)
  context   prefill/decode cost vs input length (512 -> 4k tokens)
  sparsity  speed vs sparsity at a fixed long context (query mode and kv_group mode)
"""

import csv
import json
import platform
import time

import numpy as np
import torch
from _common import parse

from headrouter.data.tasks import TASK_INFO, load_examples
from headrouter.evaluation.metrics import evaluate_examples
from headrouter.evaluation.timers import timed_generate
from headrouter.masks.builder import build_mask, variant_for
from headrouter.model_utils import chat_text
from headrouter.reports.report import write_outputs
from headrouter.runtime import Runtime

CONFIGS = ["dense", "static", "random", "oracle", "headrouter"]
SPEED_CONFIGS = ["dense", "static", "oracle", "headrouter"]   # random has oracle's sparsity -> same speed
METRICS = ["total_ms", "ttft_ms", "tpot_ms", "tokens_per_s", "energy_j", "energy_per_token_mj", "working_mem_mb"]


def summarise(runs):
    out = {}
    for k in METRICS:
        vals = [r[k] for r in runs if r.get(k) is not None]
        out[k] = {"median": float(np.median(vals)), "p95": float(np.percentile(vals, 95)), "n": len(vals)} \
            if vals else None
    return out


def quality(rt, cfg, reg):
    res, routing = {}, {}
    for t in cfg["tasks"]:
        ex = load_examples(t, "test", cfg["benchmark"]["quality_size"][t], cfg)
        routes = [rt.router.route(e["prompt"]) for e in ex]
        plans = {c: [variant_for(reg, c, t)] * len(ex) for c in CONFIGS if c != "headrouter"}
        plans["headrouter"] = [r["variant"] for r in routes]
        cache, res[t] = {}, {}
        t0 = time.time()
        for c in CONFIGS:
            key = tuple(plans[c])
            if key not in cache:
                cache[key] = evaluate_examples(rt.model, rt.tok, rt.engine, ex, plans[c], cfg, rt.device)["score"]
            res[t][c] = cache[key]
        routing[t] = {"classifier_accuracy": float(np.mean([r["task"] == t for r in routes])),
                      "fallback_rate": float(np.mean([r["variant"] == "dense" for r in routes])),
                      "routed_to": {v: plans["headrouter"].count(v) for v in set(plans["headrouter"])}}
        print(f"[quality] {t:<14} " + "  ".join(f"{c}={res[t][c]:.4f}" for c in CONFIGS)
              + f"  cls_acc={routing[t]['classifier_accuracy']:.2f}  ({time.time() - t0:.0f}s)")
    return res, routing


def speed(rt, cfg, reg):
    b = cfg["benchmark"]
    out = {}
    for t in cfg["tasks"]:
        ex = load_examples(t, "test", b["speed_prompts_per_task"], cfg)
        for _ in range(b["warmup_runs"]):
            timed_generate(rt.model, rt.tok, ex[0]["prompt"], rt.device, 8, fixed_length=True)
        runs = {c: [] for c in SPEED_CONFIGS}
        for rep in range(b["speed_repeats"]):
            for i, e in enumerate(ex):
                k = (rep + i) % len(SPEED_CONFIGS)
                for c in SPEED_CONFIGS[k:] + SPEED_CONFIGS[:k]:          # rotate order to cancel drift
                    extra = 0.0
                    if c == "headrouter":
                        r = rt.router.route(e["prompt"])
                        var, extra = r["variant"], r["classifier_ms"]
                    else:
                        var = variant_for(reg, c, t)
                    with rt.engine.use(var):
                        m = timed_generate(rt.model, rt.tok, e["prompt"], rt.device, b["speed_new_tokens"],
                                           fixed_length=True, meter=rt.meter)
                    m["total_ms"] += extra
                    m["ttft_ms"] += extra
                    m["tokens_per_s"] = m["new_tokens"] / (m["total_ms"] / 1000)
                    runs[c].append(m)
        out[t] = {c: summarise(runs[c]) for c in SPEED_CONFIGS}
        d = out[t]["dense"]["total_ms"]["median"]
        print(f"[speed]   {t:<14} " + "  ".join(
            f"{c}={out[t][c]['total_ms']['median']:.0f}ms(x{d / out[t][c]['total_ms']['median']:.3f})"
            for c in SPEED_CONFIGS))
    return out


def long_inputs(rt, cfg, n_tokens):
    """Chat-formatted summarisation prompt of exactly n_tokens tokens built from CNN/DM articles."""
    marker = "<<BODY>>"
    pre, post = chat_text(rt.tok, f"Summarize the following text in 2-3 sentences.\n\n{marker}").split(marker)
    pre_ids = rt.tok(pre, add_special_tokens=False)["input_ids"]
    post_ids = rt.tok(post, add_special_tokens=False)["input_ids"]
    body = " ".join(e["prompt"].split("Article: ", 1)[-1] for e in load_examples("summarization", "test", 60, cfg))
    body_ids = rt.tok(body, add_special_tokens=False)["input_ids"]
    need = n_tokens - len(pre_ids) - len(post_ids)
    assert len(body_ids) >= need, "not enough text for the requested context length"
    return torch.tensor([pre_ids + body_ids[:need] + post_ids], device=rt.device)


def timed_median(rt, ids, variant, repeats, new_tokens=32):
    runs = []
    with rt.engine.use(variant):
        timed_generate(rt.model, rt.tok, "", rt.device, 4, fixed_length=True, input_ids=ids)
        for _ in range(repeats):
            runs.append(timed_generate(rt.model, rt.tok, "", rt.device, new_tokens, fixed_length=True,
                                       input_ids=ids, meter=rt.meter))
    return summarise(runs)


def context_sweep(rt, cfg, reg):
    b = cfg["benchmark"]
    cands = [t for t in cfg["tasks"] if reg["tasks"][t]["sparsity"] > 0]
    if not cands:
        return None
    task = "summarization" if "summarization" in cands else max(cands, key=lambda t: reg["tasks"][t]["sparsity"])
    rows = []
    for n in b["context_lengths"]:
        ids = long_inputs(rt, cfg, n)
        d = timed_median(rt, ids, "dense", b["context_repeats"])
        p = timed_median(rt, ids, task, b["context_repeats"])
        rows.append({"context": n, "dense": d, "pruned": p})
        print(f"[context] {n:>5} tok  TTFT dense={d['ttft_ms']['median']:.1f}ms pruned={p['ttft_ms']['median']:.1f}ms"
              f"  TPOT dense={d['tpot_ms']['median']:.2f} pruned={p['tpot_ms']['median']:.2f}")
    return {"task": task, "sparsity": reg["tasks"][task]["actual_sparsity"], "rows": rows}


def sparsity_sweep(rt, cfg, reg, importance, imp_tasks):
    b, m = cfg["benchmark"], cfg["masks"]
    n = max(b["context_lengths"])
    ids = long_inputs(rt, cfg, n)
    scores = importance[imp_tasks.index("summarization" if "summarization" in imp_tasks else imp_tasks[0])]
    prot = np.array(reg["protected"])
    rows = [{"mode": "dense", "sparsity": 0.0, **timed_median(rt, ids, "dense", b["context_repeats"])}]
    for mode in ("query", "kv_group"):
        for s in m["sparsity_levels"]:
            keep = build_mask(scores, s, m["min_heads_per_layer"], prot, mode, rt.engine.num_kv)
            if keep.all():
                continue
            rt.engine.register("_sw", keep.tolist())
            rows.append({"mode": mode, "sparsity": float(1 - keep.mean()), **timed_median(rt, ids, "_sw", b["context_repeats"])})
            rt.engine.unregister("_sw")
            print(f"[sparsity] {mode:<8} {1 - keep.mean():.0%}  TTFT={rows[-1]['ttft_ms']['median']:.1f}ms  "
                  f"TPOT={rows[-1]['tpot_ms']['median']:.2f}ms")
    return {"context": n, "rows": rows}


def main():
    args, cfg, paths = parse(__doc__, lambda ap: ap.add_argument(
        "--skip", nargs="*", default=[], choices=["quality", "speed", "context", "sparsity"]))
    rt = Runtime(cfg)
    reg = rt.registry
    if reg is None or rt.classifier is None:
        raise SystemExit("Run scripts 01-03 first (classifier + head_masks.json are required).")
    print(f"model={cfg['model']['name']}  gpu={rt.gpu_name()}  energy={rt.meter.mode}")
    rt.warmup(cfg["benchmark"]["warmup_runs"])

    blob = np.load(paths.importance(reg["scorer"]))
    imp_key = "normalized" if cfg["profiling"].get("layer_normalize", True) else "raw"
    results = {
        "meta": {"model": cfg["model"]["name"], "gpu": rt.gpu_name(), "device": rt.device,
                 "dtype": cfg["model"]["dtype"], "torch": torch.__version__, "python": platform.python_version(),
                 "energy_mode": rt.meter.mode, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "quick": args.quick, "speed_new_tokens": cfg["benchmark"]["speed_new_tokens"],
                 "metric_names": {t: TASK_INFO[t]["metric"] for t in cfg["tasks"]}},
        "registry": {t: {k: v for k, v in e.items() if k != "keep"} for t, e in reg["tasks"].items()},
        "static_sparsity": reg["static"]["sparsity"],
        "classifier": json.load(open(paths.classifier_report)) if paths.classifier_report.exists() else None,
    }
    if "quality" not in args.skip:
        results["quality"], results["routing"] = quality(rt, cfg, reg)
    if "speed" not in args.skip:
        results["speed"] = speed(rt, cfg, reg)
    if "context" not in args.skip:
        results["context_sweep"] = context_sweep(rt, cfg, reg)
    if "sparsity" not in args.skip:
        results["sparsity_sweep"] = sparsity_sweep(rt, cfg, reg, blob[imp_key], list(blob["tasks"]))

    write_outputs(results, reg, blob[imp_key], list(blob["tasks"]), paths)
    with open(paths.results / "benchmark.json", "w") as f:
        json.dump(results, f, indent=1)
    with open(paths.results / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "config", "quality", "latency_ms_median", "latency_ms_p95", "ttft_ms", "tpot_ms",
                    "tokens_per_s", "energy_j", "working_mem_mb"])
        for t in cfg["tasks"]:
            for c in CONFIGS:
                q = results.get("quality", {}).get(t, {}).get(c)
                s = results.get("speed", {}).get(t, {}).get(c)
                g = (lambda k: (s[k]["median"] if s and s.get(k) else "")) if s else (lambda k: "")
                w.writerow([t, c, q if q is not None else "", g("total_ms"),
                            s["total_ms"]["p95"] if s else "", g("ttft_ms"), g("tpot_ms"), g("tokens_per_s"),
                            g("energy_j"), g("working_mem_mb")])
    print(f"\nsaved {paths.results}/benchmark.json, results.csv, report.md and plots/")


if __name__ == "__main__":
    main()
