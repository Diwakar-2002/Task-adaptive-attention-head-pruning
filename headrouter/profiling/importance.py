"""
Offline head-importance profiling. Three scorers, all producing [num_layers, num_heads]:

gradient   Michel, Levy & Neubig (NeurIPS 2019). A gate g=1 multiplies every head's
           output; importance = E_x |dLoss(x)/dg|, with the loss computed only on the
           gold target tokens. One forward+backward per calibration example.
activation Training-free proxy (SEAP-style): mean L2 norm of the head's output
           times the Frobenius norm of its O-projection columns.
causal     Interventional ablation (the original CausalPrune idea): KL divergence
           between dense and single-head-ablated next-token distributions on the
           target positions. O(L*H) forwards, so it uses a smaller calibration set.
"""

from typing import List

import torch
import torch.nn.functional as F

from ..model_utils import teacher_forced_batch


def _target_positions(labels: torch.Tensor) -> torch.Tensor:
    # logits at position t predict token t+1
    return (labels[0, 1:] != -100).nonzero().squeeze(-1)


def _target_logits(model, ids, pos):
    h = model.model(input_ids=ids).last_hidden_state[0, pos]
    return model.lm_head(h).float()


def gradient_importance(model, tok, engine, examples: List[dict], device: str, max_target_tokens: int,
                        variant: str = "dense"):
    """Gate gradients; with a pruned `variant` active only the kept heads receive a score (others 0)."""
    L, H = engine.num_layers, engine.num_heads
    gates = torch.ones(L, H, device=device, dtype=torch.float32, requires_grad=True)

    def hook(layer_idx, kept_q, out):
        g = gates[layer_idx].index_select(0, kept_q).to(out.dtype)
        return out * g.view(1, 1, -1, 1)

    imp = torch.zeros(L, H, device=device)
    engine.activate(variant)
    engine.set_head_hook(hook)
    try:
        for i, ex in enumerate(examples):
            ids, labels = teacher_forced_batch(tok, ex["prompt"], ex["target"], device, max_target_tokens)
            pos = _target_positions(labels)
            with torch.enable_grad():
                logits = _target_logits(model, ids, pos)
                loss = F.cross_entropy(logits, labels[0, 1:][pos])
                (g,) = torch.autograd.grad(loss, gates)
            imp += g.abs()
    finally:
        engine.set_head_hook(None)
        engine.activate("dense")
    return (imp / max(1, len(examples))).cpu().numpy()


@torch.no_grad()
def activation_importance(model, tok, engine, examples: List[dict], device: str, max_target_tokens: int):
    L, H, d = engine.num_layers, engine.num_heads, engine.head_dim
    acc = torch.zeros(L, H, device=device)

    def hook(layer_idx, kept_q, out):
        acc[layer_idx] += out.float().norm(dim=-1).mean(dim=(0, 1))
        return out

    engine.activate("dense")
    engine.set_head_hook(hook)
    try:
        for ex in examples:
            ids, _ = teacher_forced_batch(tok, ex["prompt"], ex["target"], device, max_target_tokens)
            model.model(input_ids=ids)
    finally:
        engine.set_head_hook(None)
    acc /= max(1, len(examples))
    w_norm = torch.stack([
        layer.o_proj.weight.float().view(-1, H, d).norm(dim=(0, 2)) for layer in engine.layers
    ])
    return (acc * w_norm).cpu().numpy()


@torch.no_grad()
def causal_importance(model, tok, engine, examples: List[dict], device: str, max_target_tokens: int):
    """Per-head KL(dense || ablated). Each forward ablates all H heads of one layer
    in parallel (batch element j has head j removed)."""
    L, H = engine.num_layers, engine.num_heads
    imp = torch.zeros(L, H, device=device)
    state = {"layer": -1}
    ablate = (1 - torch.eye(H, device=device))            # row j: zero head j

    def hook(layer_idx, kept_q, out):
        if layer_idx != state["layer"]:
            return out
        return out * ablate.to(out.dtype)[:, None, :, None]

    engine.activate("dense")
    engine.set_head_hook(hook)
    try:
        for ex in examples:
            ids, labels = teacher_forced_batch(tok, ex["prompt"], ex["target"], device, max_target_tokens)
            pos = _target_positions(labels)
            state["layer"] = -1
            base = F.log_softmax(_target_logits(model, ids, pos), -1)          # (T, V)
            batch = ids.expand(H, -1)
            for l in range(L):
                state["layer"] = l
                h = model.model(input_ids=batch).last_hidden_state[:, pos]       # (H, T, D)
                abl = F.log_softmax(model.lm_head(h).float(), -1)
                kl = (base.exp() * (base - abl)).sum(-1).mean(-1)                # (H,)
                imp[l] += kl
    finally:
        engine.set_head_hook(None)
    return (imp / max(1, len(examples))).cpu().numpy()


@torch.no_grad()
def head_output_means(model, tok, engine, examples: List[dict], device: str, max_target_tokens: int):
    """Mean output vector of every head over all tokens of the calibration set -> [L, H, head_dim]."""
    L, H, d = engine.num_layers, engine.num_heads, engine.head_dim
    sums = torch.zeros(L, H, d, device=device, dtype=torch.float64)
    count = {"n": 0}

    def hook(layer_idx, kept_q, out):
        sums[layer_idx] += out.double().sum(dim=(0, 1))
        if layer_idx == 0:
            count["n"] += out.shape[0] * out.shape[1]
        return out

    engine.activate("dense")
    engine.set_head_hook(hook)
    try:
        for ex in examples:
            ids, _ = teacher_forced_batch(tok, ex["prompt"], ex["target"], device, max_target_tokens)
            model.model(input_ids=ids)
    finally:
        engine.set_head_hook(None)
    return (sums / max(1, count["n"])).float().cpu().numpy()


SCORERS = {"gradient": gradient_importance, "activation": activation_importance, "causal": causal_importance}


def layer_normalize(imp):
    import numpy as np
    norm = np.linalg.norm(imp, axis=-1, keepdims=True)
    return imp / np.maximum(norm, 1e-12)
