"""
Pruned execution engine.

Every attention block of a Llama/Qwen2/Mistral-style decoder is replaced by a
`PrunableAttention` that holds several *weight variants*:

    "dense"      references the original Q/K/V/O weights (no copy)
    "<task>"     physically sliced copies containing ONLY the kept heads

For a variant the Q projection has only `n_kept_q * head_dim` output rows, K/V
only the kept GQA groups, and O only the matching input columns, so the
matrix multiplications, the SDPA kernel and the KV cache are all smaller.
Pruned heads are never computed — this is not zero-masking.

Switching tasks is a dictionary lookup (`engine.activate(name)`): no slicing on
the hot path. Dense and pruned variants run through exactly the same code so
that timing differences come only from the reduced work.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


@dataclass
class Variant:
    q_w: torch.Tensor
    q_b: Optional[torch.Tensor]
    k_w: torch.Tensor
    k_b: Optional[torch.Tensor]
    v_w: torch.Tensor
    v_b: Optional[torch.Tensor]
    o_w: torch.Tensor
    o_b: Optional[torch.Tensor]
    kept_q: torch.Tensor      # original indices of kept query heads
    kv_index: torch.Tensor    # for each kept q head: index into the kept KV heads
    n_q: int
    n_kv: int


class PrunableAttention(nn.Module):
    def __init__(self, attn: nn.Module, layer_idx: int, n_heads: int, n_kv: int, head_dim: int):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj
        self.layer_idx = layer_idx
        self.n_heads, self.n_kv, self.head_dim = n_heads, n_kv, head_dim
        self.group = n_heads // n_kv
        self.scaling = getattr(attn, "scaling", head_dim ** -0.5)
        self.variants: Dict[str, Variant] = {}
        self.active = "dense"
        # Optional callback on per-head outputs (b, s, n_q, d) before O-projection.
        # Used only by offline profiling (gates / activation statistics).
        self.head_hook: Optional[Callable] = None
        self.add_variant("dense", [True] * n_heads)

    @staticmethod
    def _bias(lin: nn.Linear, rows=None):
        if lin.bias is None:
            return None
        return lin.bias if rows is None else lin.bias.index_select(0, rows).contiguous()

    def add_variant(self, name: str, keep: Sequence[bool], head_means: Optional[torch.Tensor] = None):
        """
        Build a physically sliced weight copy keeping heads where keep[h] is True.

        head_means (H, head_dim), optional: mean output of each head on calibration
        data. The pruned heads' mean contribution W_o[:, h] @ mean_h is folded into
        the O-projection bias (FLAP-style bias compensation, An et al. 2024), so the
        layer's output keeps its expected value at zero extra inference cost.
        """
        d, dev = self.head_dim, self.q_proj.weight.device
        kept_q = [h for h, k in enumerate(keep) if k]
        if not kept_q:
            raise ValueError(f"layer {self.layer_idx}: a variant must keep at least one head")
        kept_kv = sorted({h // self.group for h in kept_q})
        kv_index = torch.tensor([kept_kv.index(h // self.group) for h in kept_q], device=dev)
        kept_q_t = torch.tensor(kept_q, device=dev)

        if len(kept_q) == self.n_heads:   # dense: reference, never copy
            v = Variant(self.q_proj.weight, self._bias(self.q_proj), self.k_proj.weight, self._bias(self.k_proj),
                        self.v_proj.weight, self._bias(self.v_proj), self.o_proj.weight, self._bias(self.o_proj),
                        kept_q_t, kv_index, self.n_heads, self.n_kv)
        else:
            q_rows = torch.cat([torch.arange(h * d, (h + 1) * d) for h in kept_q]).to(dev)
            kv_rows = torch.cat([torch.arange(g * d, (g + 1) * d) for g in kept_kv]).to(dev)
            o_b = self._bias(self.o_proj)
            if head_means is not None:
                pruned = [h for h, k in enumerate(keep) if not k]
                p_rows = torch.cat([torch.arange(h * d, (h + 1) * d) for h in pruned]).to(dev)
                w = self.o_proj.weight.index_select(1, p_rows).float()
                comp = w @ head_means[pruned].reshape(-1).to(dev, torch.float32)
                o_b = (comp if o_b is None else o_b.float() + comp).to(self.o_proj.weight.dtype)
            v = Variant(
                self.q_proj.weight.index_select(0, q_rows).contiguous(), self._bias(self.q_proj, q_rows),
                self.k_proj.weight.index_select(0, kv_rows).contiguous(), self._bias(self.k_proj, kv_rows),
                self.v_proj.weight.index_select(0, kv_rows).contiguous(), self._bias(self.v_proj, kv_rows),
                self.o_proj.weight.index_select(1, q_rows).contiguous(), o_b,
                kept_q_t, kv_index, len(kept_q), len(kept_kv))
        self.variants[name] = v

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                past_key_value=None, cache_position=None, **kwargs):
        v = self.variants[self.active]
        b, s, _ = hidden_states.shape
        d = self.head_dim
        q = F.linear(hidden_states, v.q_w, v.q_b).view(b, s, v.n_q, d).transpose(1, 2)
        k = F.linear(hidden_states, v.k_w, v.k_b).view(b, s, v.n_kv, d).transpose(1, 2)
        val = F.linear(hidden_states, v.v_w, v.v_b).view(b, s, v.n_kv, d).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary(q, k, cos, sin)
        if past_key_value is not None:
            k, val = past_key_value.update(k, val, self.layer_idx,
                                           {"sin": sin, "cos": cos, "cache_position": cache_position})
        # GQA: give each kept query head its KV head (same cost as HF's repeat_kv).
        k = k.index_select(1, v.kv_index)
        val = val.index_select(1, v.kv_index)

        mask = attention_mask
        if mask is not None:
            mask = mask[:, :, :, : k.shape[-2]]
        out = F.scaled_dot_product_attention(q.contiguous(), k.contiguous(), val.contiguous(), attn_mask=mask,
                                             dropout_p=0.0, scale=self.scaling,
                                             is_causal=(mask is None and s > 1))
        out = out.transpose(1, 2)                       # (b, s, n_q, d)
        if self.head_hook is not None:
            out = self.head_hook(self.layer_idx, v.kept_q, out)
        out = out.reshape(b, s, v.n_q * d)
        return F.linear(out, v.o_w, v.o_b), None


class HeadRouterEngine:
    """Installs PrunableAttention into a HF causal LM and manages mask variants."""

    def __init__(self, model: nn.Module):
        self.model = model
        cfg = model.config
        self.num_layers = cfg.num_hidden_layers
        self.num_heads = cfg.num_attention_heads
        self.num_kv = getattr(cfg, "num_key_value_heads", self.num_heads)
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.num_heads
        self.layers: List[PrunableAttention] = []
        for i, layer in enumerate(model.model.layers):
            pa = PrunableAttention(layer.self_attn, i, self.num_heads, self.num_kv, self.head_dim)
            layer.self_attn = pa
            self.layers.append(pa)
        self.masks: Dict[str, List[List[bool]]] = {"dense": [[True] * self.num_heads] * self.num_layers}
        self.active = "dense"

    def register(self, name: str, keep: List[List[bool]], head_means=None):
        """keep: [L][H] bools. head_means: optional [L, H, head_dim] for bias compensation."""
        assert len(keep) == self.num_layers and all(len(r) == self.num_heads for r in keep)
        if head_means is not None:
            head_means = torch.as_tensor(head_means)
        for l, (layer, row) in enumerate(zip(self.layers, keep)):
            layer.add_variant(name, [bool(x) for x in row], None if head_means is None else head_means[l])
        self.masks[name] = [[bool(x) for x in r] for r in keep]

    def unregister(self, name: str):
        if name == "dense":
            return
        for layer in self.layers:
            layer.variants.pop(name, None)
        self.masks.pop(name, None)
        if self.active == name:
            self.activate("dense")

    def activate(self, name: str):
        if name not in self.masks:
            raise KeyError(f"unknown mask variant '{name}'")
        for layer in self.layers:
            layer.active = name
        self.active = name

    @contextmanager
    def use(self, name: str):
        prev = self.active
        self.activate(name)
        try:
            yield
        finally:
            self.activate(prev)

    def set_head_hook(self, fn: Optional[Callable]):
        for layer in self.layers:
            layer.head_hook = fn

    def stats(self, name: str) -> dict:
        keep = self.masks[name]
        kept = sum(sum(r) for r in keep)
        total = self.num_layers * self.num_heads
        kv_kept = sum(len({h // (self.num_heads // self.num_kv) for h, k in enumerate(r) if k}) for r in keep)
        return {"kept_heads": kept, "total_heads": total, "pruned_heads": total - kept,
                "sparsity": round(1 - kept / total, 4),
                "kept_kv_heads": kv_kept, "total_kv_heads": self.num_layers * self.num_kv}

    def variant_bytes(self, name: str) -> int:
        if name == "dense":
            return 0
        n = 0
        for layer in self.layers:
            v = layer.variants[name]
            for t in (v.q_w, v.q_b, v.k_w, v.k_b, v.v_w, v.v_b, v.o_w, v.o_b):
                if t is not None:
                    n += t.numel() * t.element_size()
        return n
