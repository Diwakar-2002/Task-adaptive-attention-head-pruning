"""
Constrained head-mask construction and the head-mask registry.

Constraints:
  * min_heads_per_layer - no layer is ever reduced below this many query heads
  * protected heads     - heads in the top `protected_top_frac` for EVERY task
  * mode=query          - prune individual query heads (shrinks Q/O and SDPA work)
  * mode=kv_group       - prune whole GQA groups (also shrinks K/V projections + KV cache)
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def rank_percentile(scores: np.ndarray) -> np.ndarray:
    """Map scores to [0, 1] percentiles so tasks with different scales are comparable."""
    flat = scores.reshape(-1)
    ranks = flat.argsort().argsort()
    return (ranks / max(1, flat.size - 1)).reshape(scores.shape)


def protected_heads(importance: np.ndarray, top_frac: float) -> np.ndarray:
    """importance: [T, L, H] -> bool [L, H] heads in the top `top_frac` of every task."""
    if top_frac <= 0:
        return np.zeros(importance.shape[1:], dtype=bool)
    prot = np.ones(importance.shape[1:], dtype=bool)
    for t in range(importance.shape[0]):
        prot &= rank_percentile(importance[t]) >= 1 - top_frac
    return prot


def build_mask(scores: np.ndarray, sparsity: float, min_heads_per_layer: int = 2,
               protected: Optional[np.ndarray] = None, mode: str = "query", num_kv: int = 1) -> np.ndarray:
    """Greedy: prune lowest-importance units first while respecting constraints. Returns keep[L, H]."""
    L, H = scores.shape
    keep = np.ones((L, H), dtype=bool)
    if sparsity <= 0:
        return keep
    if protected is None:
        protected = np.zeros((L, H), dtype=bool)

    if mode == "query":
        target = int(round(sparsity * L * H))
        counts = np.full(L, H)
        pruned = 0
        for flat in np.argsort(scores, axis=None, kind="stable"):
            if pruned >= target:
                break
            l, h = divmod(int(flat), H)
            if protected[l, h] or counts[l] <= min_heads_per_layer:
                continue
            keep[l, h] = False
            counts[l] -= 1
            pruned += 1
        return keep

    if mode == "kv_group":
        g = H // num_kv
        unit = scores.reshape(L, num_kv, g).sum(-1)
        unit_prot = protected.reshape(L, num_kv, g).any(-1)
        min_groups = max(1, math.ceil(min_heads_per_layer / g))
        target = int(round(sparsity * L * num_kv))
        counts = np.full(L, num_kv)
        pruned = 0
        for flat in np.argsort(unit, axis=None, kind="stable"):
            if pruned >= target:
                break
            l, k = divmod(int(flat), num_kv)
            if unit_prot[l, k] or counts[l] <= min_groups:
                continue
            keep[l, k * g:(k + 1) * g] = False
            counts[l] -= 1
            pruned += 1
        return keep

    raise ValueError(f"unknown mask mode '{mode}'")


def static_scores(importance: np.ndarray) -> np.ndarray:
    """Task-agnostic importance for the static-pruning baseline: mean rank percentile."""
    return np.mean([rank_percentile(importance[t]) for t in range(importance.shape[0])], axis=0)


def within_budget(dense: float, pruned: float, budget: float, budget_type: str) -> bool:
    drop = dense - pruned
    if budget_type == "relative":
        return drop <= budget * max(dense, 1e-9)
    return drop <= budget


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def save_registry(path: Path, registry: Dict):
    with open(path, "w") as f:
        json.dump(registry, f, indent=1)


def load_registry(path: Path) -> Optional[Dict]:
    if not Path(path).exists():
        return None
    with open(path) as f:
        return json.load(f)


def registry_variants(registry: Dict) -> Dict[str, List[List[bool]]]:
    """All named masks contained in a registry (task masks + baselines)."""
    out = {}
    for task, entry in registry.get("tasks", {}).items():
        if entry["sparsity"] > 0:
            out[task] = entry["keep"]
    if registry.get("static", {}).get("sparsity", 0) > 0:
        out["static"] = registry["static"]["keep"]
    for task, entry in registry.get("random", {}).items():
        if entry["sparsity"] > 0:
            out[f"random:{task}"] = entry["keep"]
    return out


def load_head_means(path: Path) -> Optional[Dict[str, np.ndarray]]:
    if not Path(path).exists():
        return None
    blob = np.load(path)
    return {k: blob[k] for k in blob.files}


def register_all(engine, registry: Dict, head_means: Optional[Dict[str, np.ndarray]]):
    """Register every registry mask in the engine, with bias compensation when configured."""
    comp = registry.get("compensation", "none") == "mean" and head_means is not None
    for name, keep in registry_variants(registry).items():
        hm = None
        if comp:
            hm = head_means["static"] if name == "static" else head_means[name.split(":")[-1]]
        engine.register(name, keep, hm)


def variant_for(registry: Dict, config: str, task: str) -> str:
    """Mask variant used by a benchmark configuration for a task ('dense' when nothing is pruned)."""
    if config == "dense":
        return "dense"
    if config == "static":
        return "static" if registry["static"]["sparsity"] > 0 else "dense"
    if config == "random":
        return f"random:{task}" if registry["random"][task]["sparsity"] > 0 else "dense"
    if config == "oracle":
        return task if registry["tasks"][task]["sparsity"] > 0 else "dense"
    raise KeyError(config)


def iterative_masks(score_fn, levels, min_heads_per_layer, protected, mode, num_kv, register, normalize):
    """
    Iterative pruning (Michel et al. 2019): prune up to the next sparsity level, then re-score the
    surviving heads with that mask active (and compensated), and continue.

    score_fn(variant_name) -> importance [L, H] for the kept heads of that variant.
    register(name, keep)   -> registers a variant in the engine.
    Returns {level: keep}.
    """
    out = {}
    keep = None
    scores = normalize(score_fn("dense"))
    for s in sorted(levels):
        if keep is not None:
            register("_iter", keep.tolist())
            scores = normalize(score_fn("_iter"))
            scores = np.where(keep, scores, -np.inf)     # already-pruned heads stay pruned
        keep = build_mask(scores, s, min_heads_per_layer, protected, mode, num_kv)
        out[s] = keep
    return out
