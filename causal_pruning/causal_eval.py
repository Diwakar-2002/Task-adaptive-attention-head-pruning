"""
Causal Evaluator Engine
Computes interventional KL divergence, logit distribution shifts, and magnitude saliency matrices.
"""

from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .ablation import HeadAblationHook

class CausalEvaluator:
    """
    Evaluates causal impact of each attention head by measuring KL divergence
    between baseline predictions and ablated predictions across dataset prompts.
    """
    def __init__(self, model: nn.Module, tokenizer: Any, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.hook_manager = HeadAblationHook(self.model)
        self.num_layers = self.hook_manager.num_layers
        self.num_heads = self.hook_manager.num_heads
        self.head_dim = self.hook_manager.head_dim

    def compute_kl_divergence(
        self,
        baseline_logits: torch.Tensor,
        ablated_logits: torch.Tensor,
        temperature: float = 1.0
    ) -> float:
        """
        Compute mean KL Divergence D_KL(P_baseline || P_ablated) over all output token positions.
        logits shape: (batch_size, seq_len, vocab_size)
        """
        with torch.no_grad():
            p = F.softmax(baseline_logits / temperature, dim=-1)
            log_p = F.log_softmax(baseline_logits / temperature, dim=-1)
            log_q = F.log_softmax(ablated_logits / temperature, dim=-1)
            
            # KL(P || Q) = sum(P * (log_p - log_q))
            kl = torch.sum(p * (log_p - log_q), dim=-1)
            return float(kl.mean().item())

    def evaluate_magnitude_saliency(self) -> np.ndarray:
        """
        Compute standard magnitude baseline (L1/L2 norm of out_proj weights for each head).
        Returns matrix of shape (num_layers, num_heads).
        """
        saliency_matrix = np.zeros((self.num_layers, self.num_heads), dtype=np.float32)
        attn_modules = self.hook_manager._find_attn_projection_modules()
        
        for layer_idx in range(self.num_layers):
            if layer_idx in attn_modules:
                mod = attn_modules[layer_idx]
                if hasattr(mod, "weight") and mod.weight is not None:
                    w = mod.weight.detach().cpu()
                    # w shape for Linear in c_proj: (hidden_dim, hidden_dim) or Conv1D (hidden_dim, hidden_dim)
                    # For GPT2 Conv1D: w is (in_features, out_features) = (hidden_dim, hidden_dim)
                    # Input to c_proj: chunk of size head_dim along last dimension
                    for h_idx in range(self.num_heads):
                        start_i = h_idx * self.head_dim
                        end_i = (h_idx + 1) * self.head_dim
                        if w.dim() == 2:
                            if w.shape[0] == self.num_layers * self.num_heads * self.head_dim: # standard Linear
                                head_w = w[start_i:end_i, :]
                            else: # Conv1D
                                head_w = w[start_i:end_i, :]
                            saliency_matrix[layer_idx, h_idx] = float(torch.norm(head_w, p=2).item())
                        else:
                            saliency_matrix[layer_idx, h_idx] = 1.0
                else:
                    saliency_matrix[layer_idx, :] = 1.0
            else:
                saliency_matrix[layer_idx, :] = 1.0
                
        # Normalize to [0, 1] range for visual comparability
        max_v = np.max(saliency_matrix)
        if max_v > 0:
            saliency_matrix /= max_v
        return saliency_matrix

    def evaluate_causal_impact_matrix(
        self,
        prompts: List[str],
        max_new_tokens: int = 16
    ) -> Dict[str, Any]:
        """
        Evaluates the causal impact (KL divergence) of all (layer, head) pairs across sample prompts.
        Returns heatmap matrices, head rankings, and summary statistics.
        """
        self.model.eval()
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        # 1. Baseline unpruned forward pass
        with torch.no_grad():
            baseline_outputs = self.model(**inputs)
            baseline_logits = baseline_outputs.logits

        causal_matrix = np.zeros((self.num_layers, self.num_heads), dtype=np.float32)

        # 2. Interventional Ablation Test for each (layer, head)
        for l in range(self.num_layers):
            for h in range(self.num_heads):
                self.hook_manager.set_ablated_heads({(l, h)})
                with torch.no_grad():
                    ablated_outputs = self.model(**inputs)
                    ablated_logits = ablated_outputs.logits
                
                kl_val = self.compute_kl_divergence(baseline_logits, ablated_logits)
                causal_matrix[l, h] = kl_val

        # Clear hooks after evaluation
        self.hook_manager.clear_hooks()

        # Compute magnitude baseline for comparison
        magnitude_matrix = self.evaluate_magnitude_saliency()

        # Rank heads by causal impact (ascending order = candidates for pruning)
        head_rankings = []
        for l in range(self.num_layers):
            for h in range(self.num_heads):
                head_rankings.append({
                    "layer": l,
                    "head": h,
                    "causal_kl": float(causal_matrix[l, h]),
                    "magnitude_score": float(magnitude_matrix[l, h])
                })
        
        head_rankings.sort(key=lambda x: x["causal_kl"])

        return {
            "causal_heatmap": causal_matrix.tolist(),
            "magnitude_heatmap": magnitude_matrix.tolist(),
            "head_rankings": head_rankings,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "mean_kl": float(np.mean(causal_matrix)),
            "max_kl": float(np.max(causal_matrix)),
            "min_kl": float(np.min(causal_matrix)),
        }
