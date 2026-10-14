"""
HeadRouter API server.

Every number returned here is measured at request time (or read from the
artifacts produced by scripts/01-04). Nothing is hard-coded.
"""

import os
import statistics
import sys
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from headrouter.config import ROOT, load_config  # noqa: E402
from headrouter.data.tasks import TASK_INFO, load_examples  # noqa: E402
from headrouter.evaluation.timers import timed_generate  # noqa: E402
from headrouter.runtime import Runtime  # noqa: E402

CFG = load_config(os.environ.get("HEADROUTER_CONFIG"))
app = FastAPI(title="HeadRouter", description="Task-aware dynamic attention-head pruning", version="2.0.0")
RT: Optional[Runtime] = None


@app.on_event("startup")
def startup():
    global RT
    RT = Runtime(CFG)
    print(f"[headrouter] warming up {len(RT.engine.masks)} mask variants ...")
    RT.warmup(1)
    print(f"[headrouter] ready on {RT.gpu_name()} (energy: {RT.meter.mode})")


class ClassifyRequest(BaseModel):
    prompt: str
    force_task: Optional[str] = None


class GenerateRequest(BaseModel):
    prompt: str
    mode: str = Field("dense", pattern="^(dense|headrouter)$")
    force_task: Optional[str] = None
    max_new_tokens: int = Field(128, ge=1, le=1024)
    runs: int = Field(1, ge=1, le=10)


def _median(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


@app.get("/api/status")
def status():
    e, reg = RT.engine, RT.registry
    tasks = {}
    for t in CFG["tasks"]:
        entry = (reg or {}).get("tasks", {}).get(t)
        tasks[t] = {"name": TASK_INFO[t]["name"], "metric": TASK_INFO[t]["metric"],
                    "profile": None if entry is None else {
                        k: entry[k] for k in ("sparsity", "actual_sparsity", "dense_dev", "pruned_dev", "sweep")}}
        if t in e.masks:
            tasks[t]["stats"] = e.stats(t)
            tasks[t]["extra_weight_mb"] = e.variant_bytes(t) / 2**20
    return {
        "model": CFG["model"]["name"], "device": RT.device, "gpu": RT.gpu_name(), "energy_mode": RT.meter.mode,
        "arch": {"layers": e.num_layers, "heads": e.num_heads, "kv_heads": e.num_kv, "head_dim": e.head_dim,
                 "total_heads": e.num_layers * e.num_heads},
        "artifacts": {"classifier": RT.classifier is not None, "registry": reg is not None,
                      "benchmark": (RT.paths.results / "benchmark.json").exists()},
        "registry": None if reg is None else {k: reg[k] for k in ("scorer", "mode", "accuracy_budget",
                                                                   "budget_type", "min_heads_per_layer", "created")},
        "threshold": RT.router.threshold, "tasks": tasks, "warnings": RT.warnings,
    }


@app.get("/api/examples")
def examples():
    out = {}
    for t in CFG["tasks"]:
        try:
            ex = load_examples(t, "test", 3, CFG)
        except Exception:
            continue
        out[t] = [e["prompt"] for e in ex]
    out["general"] = ["Write a haiku about autumn leaves.", "Give me three tips for a job interview."]
    return out


@app.post("/api/classify")
def classify(req: ClassifyRequest):
    return RT.router.route(req.prompt, req.force_task or None)


@app.post("/api/generate")
def generate(req: GenerateRequest):
    if not req.prompt.strip():
        raise HTTPException(400, "empty prompt")
    with RT.lock:
        route = None
        variant = "dense"
        if req.mode == "headrouter":
            route = RT.router.route(req.prompt, req.force_task or None)
            variant = route["variant"]
        runs = []
        with RT.engine.use(variant):
            for _ in range(req.runs):
                runs.append(timed_generate(RT.model, RT.tok, req.prompt, RT.device, req.max_new_tokens,
                                           meter=RT.meter))
    cls_ms = route["classifier_ms"] if route else 0.0
    keys = ["total_ms", "ttft_ms", "tpot_ms", "tokens_per_s", "energy_j", "energy_per_token_mj", "working_mem_mb"]
    metrics = {k: _median([r[k] for r in runs]) for k in keys}
    metrics["e2e_ms"] = metrics["total_ms"] + cls_ms
    metrics["classifier_ms"] = cls_ms
    metrics["new_tokens"] = runs[-1]["new_tokens"]
    metrics["input_tokens"] = runs[-1]["input_tokens"]
    metrics["runs"] = req.runs
    metrics["total_ms_all"] = [r["total_ms"] for r in runs]
    return {"mode": req.mode, "variant": variant, "text": runs[-1]["text"], "metrics": metrics,
            "routing": route, "mask": RT.engine.masks[variant], "mask_stats": RT.engine.stats(variant)}


@app.get("/api/importance")
def importance():
    reg = RT.registry
    if reg is None:
        raise HTTPException(404, "no head_masks.json - run scripts/03_build_masks.py")
    path = RT.paths.importance(reg["scorer"])
    blob = np.load(path)
    key = "normalized" if CFG["profiling"].get("layer_normalize", True) else "raw"
    return {"scorer": reg["scorer"], "normalized": key == "normalized", "tasks": [str(t) for t in blob["tasks"]],
            "importance": blob[key].round(5).tolist(),
            "keep": {t: e["keep"] for t, e in reg["tasks"].items()},
            "static_keep": reg["static"]["keep"], "protected": reg["protected"]}


@app.get("/api/benchmark")
def benchmark():
    import json
    p = RT.paths.results / "benchmark.json"
    if not p.exists():
        raise HTTPException(404, "no benchmark yet - run scripts/04_benchmark.py")
    with open(p) as f:
        return json.load(f)


PLOTS_DIR = os.path.join(CFG["artifacts_dir"], "results", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)
app.mount("/plots", StaticFiles(directory=PLOTS_DIR), name="plots")
app.mount("/", StaticFiles(directory=str(ROOT / "static"), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=CFG["server"]["host"], port=CFG["server"]["port"])
