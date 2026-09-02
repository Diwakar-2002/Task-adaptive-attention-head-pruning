"""
Head Ablation Hook Infrastructure
Provides zero-overhead, non-destructive interventional masking of specific attention heads.
"""

from typing import Dict, List, Set, Tuple, Optional, Union
import torch
import torch.nn as nn

class HeadAblationHook:
    """
    Manages dynamic forward hooks on Transformer attention layers to ablate (zero-out)
    specific (layer, head) pairs during forward passes.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.ablated_heads: Set[Tuple[int, int]] = set()
        self.num_layers, self.num_heads, self.head_dim = self._detect_model_architecture()

    def _detect_model_architecture(self) -> Tuple[int, int, int]:
        """Detect layer count, head count, and head dimension from Hugging Face model config."""
        config = getattr(self.model, "config", None)
        if config is not None:
            num_layers = getattr(config, "n_layer", getattr(config, "num_hidden_layers", 12))
            num_heads = getattr(config, "n_head", getattr(config, "num_attention_heads", 12))
            hidden_size = getattr(config, "n_embd", getattr(config, "hidden_size", 768))
            head_dim = hidden_size // num_heads
            return num_layers, num_heads, head_dim
        # Default fallback (GPT-2 standard)
        return 12, 12, 64

    def set_ablated_heads(self, ablated_pairs: Set[Tuple[int, int]]):
        """Set active ablated head set and re-install hooks."""
        self.clear_hooks()
        self.ablated_heads = set(ablated_pairs)
        if self.ablated_heads:
            self._install_hooks()

    def add_ablated_head(self, layer_idx: int, head_idx: int):
        """Ablate a single head."""
        self.ablated_heads.add((layer_idx, head_idx))
        self.set_ablated_heads(self.ablated_heads)

    def remove_ablated_head(self, layer_idx: int, head_idx: int):
        """Un-ablate a single head."""
        self.ablated_heads.discard((layer_idx, head_idx))
        self.set_ablated_heads(self.ablated_heads)

    def clear_hooks(self):
        """Remove all active PyTorch hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def _install_hooks(self):
        """Register forward pre-hooks on projection layers to zero out target head outputs."""
        # Organize ablated heads by layer
        layer_to_heads: Dict[int, List[int]] = {}
        for l, h in self.ablated_heads:
            if l not in layer_to_heads:
                layer_to_heads[l] = []
            layer_to_heads[l].append(h)

        # Locate attention modules
        attn_modules = self._find_attn_projection_modules()

        for layer_idx, heads in layer_to_heads.items():
            if layer_idx not in attn_modules:
                continue
            module = attn_modules[layer_idx]
            
            # Create closure for current layer and heads
            def make_pre_hook(l_idx: int, target_heads: List[int]):
                def hook(mod, inputs):
                    if not inputs:
                        return inputs
                    x = inputs[0]
                    # x shape: (batch_size, seq_len, hidden_dim) or similar
                    if x.dim() >= 3 and x.shape[-1] == self.num_heads * self.head_dim:
                        x_clone = x.clone()
                        for h_idx in target_heads:
                            start_idx = h_idx * self.head_dim
                            end_idx = (h_idx + 1) * self.head_dim
                            x_clone[..., start_idx:end_idx] = 0.0
                        return (x_clone,) + inputs[1:]
                    return inputs
                return hook

            h_handle = module.register_forward_pre_hook(make_pre_hook(layer_idx, heads))
            self.hooks.append(h_handle)

    def _find_attn_projection_modules(self) -> Dict[int, nn.Module]:
        """Find the out_proj or c_proj layer for each transformer block."""
        attn_modules: Dict[int, nn.Module] = {}
        
        # Check GPT2 structure: model.transformer.h[i].attn.c_proj
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            for i, block in enumerate(self.model.transformer.h):
                if hasattr(block, "attn") and hasattr(block.attn, "c_proj"):
                    attn_modules[i] = block.attn.c_proj
                elif hasattr(block, "attn"):
                    attn_modules[i] = block.attn
        # Check LLaMA / Generic HF structure: model.model.layers[i].self_attn.o_proj
        elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            for i, block in enumerate(self.model.model.layers):
                if hasattr(block, "self_attn") and hasattr(block.self_attn, "o_proj"):
                    attn_modules[i] = block.self_attn.o_proj
        
        return attn_modules

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clear_hooks()
