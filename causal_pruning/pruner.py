"""
Causal Pruner Orchestrator
Executes static causal pruning, magnitude pruning, and dynamic task-aware routing.
"""

from typing import Dict, List, Set, Tuple, Any, Optional
import torch
import torch.nn as nn
from .ablation import HeadAblationHook
from .causal_eval import CausalEvaluator
from .router import TaskAwareRouter

class CausalPruner:
    """
    Unified manager for executing model pruning strategies:
    - Static Causal Pruning
    - Static Magnitude Pruning (Baseline)
    - Dynamic Task-Aware Routing
    """
    def __init__(self, model: nn.Module, tokenizer: Any, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.hook_manager = HeadAblationHook(self.model)
        self.evaluator = CausalEvaluator(self.model, self.tokenizer, self.device)
        self.router = TaskAwareRouter(self.hook_manager.num_layers, self.hook_manager.num_heads)
        self.current_strategy: str = "unpruned"
        self.current_ablated_heads: Set[Tuple[int, int]] = set()

    def restore_unpruned(self):
        """Restore model to 100% unpruned state."""
        self.hook_manager.clear_hooks()
        self.current_strategy = "unpruned"
        self.current_ablated_heads = set()

    def prune_static_causal(
        self,
        prompts: List[str],
        mode: str = "safety_threshold",
        kl_threshold: float = 0.05,
        sparsity_target: float = 0.4
    ) -> Dict[str, Any]:
        """
        Evaluate causal impact and ablate attention heads:
        - mode="safety_threshold": Prunes ALL heads where causal KL divergence <= kl_threshold.
        - mode="target_sparsity": Prunes the bottom (sparsity_target * 100)% least impactful heads.
        """
        eval_results = self.evaluator.evaluate_causal_impact_matrix(prompts)
        rankings = eval_results["head_rankings"] # Sorted in ascending order of causal_kl
        total_heads = len(rankings)
        
        ablated_pairs = set()
        
        if mode == "safety_threshold":
            # Dynamically prune ALL heads that pass the interventional safety threshold (KL <= kl_threshold)
            for item in rankings:
                if item["causal_kl"] <= kl_threshold:
                    ablated_pairs.add((item["layer"], item["head"]))
            # If no heads met the strict threshold, pick at least the single safest head
            if not ablated_pairs and rankings:
                ablated_pairs.add((rankings[0]["layer"], rankings[0]["head"]))
            actual_sparsity = len(ablated_pairs) / float(total_heads)
            strategy_name = f"causal_safety_threshold_(KL<={kl_threshold})"
        else:
            # Target Sparsity Mode
            prune_count = int(total_heads * sparsity_target)
            for i in range(prune_count):
                item = rankings[i]
                ablated_pairs.add((item["layer"], item["head"]))
            actual_sparsity = sparsity_target
            strategy_name = f"static_causal_{int(sparsity_target * 100)}%"

        self.hook_manager.set_ablated_heads(ablated_pairs)
        self.current_strategy = strategy_name
        self.current_ablated_heads = ablated_pairs

        return {
            "strategy": self.current_strategy,
            "mode": mode,
            "kl_threshold": kl_threshold,
            "actual_sparsity_pct": round(actual_sparsity * 100.0, 1),
            "pruned_heads_count": len(ablated_pairs),
            "total_heads": total_heads,
            "ablated_pairs": sorted(list(ablated_pairs)),
            "eval_summary": eval_results
        }

    def prune_static_magnitude(self, sparsity_target: float = 0.4) -> Dict[str, Any]:
        """
        Baseline method: Evaluate L2 weight norm and prune the bottom (sparsity_target * 100)% smallest norm heads.
        """
        magnitude_matrix = self.evaluator.evaluate_magnitude_saliency()
        items = []
        for l in range(self.hook_manager.num_layers):
            for h in range(self.hook_manager.num_heads):
                items.append({
                    "layer": l,
                    "head": h,
                    "magnitude_score": float(magnitude_matrix[l, h])
                })
        
        items.sort(key=lambda x: x["magnitude_score"])
        
        total_heads = len(items)
        prune_count = int(total_heads * sparsity_target)
        
        ablated_pairs = set()
        for i in range(prune_count):
            ablated_pairs.add((items[i]["layer"], items[i]["head"]))

        self.hook_manager.set_ablated_heads(ablated_pairs)
        self.current_strategy = f"static_magnitude_{int(sparsity_target * 100)}%"
        self.current_ablated_heads = ablated_pairs

        return {
            "strategy": self.current_strategy,
            "sparsity_target": sparsity_target,
            "pruned_heads_count": len(ablated_pairs),
            "total_heads": total_heads,
            "ablated_pairs": sorted(list(ablated_pairs))
        }

    def apply_task_routing(self, prompt: str) -> Dict[str, Any]:
        """
        Dynamically route prompt to its task domain and apply its task-specific attention mask.
        """
        routing_info = self.router.route(prompt)
        ablated_pairs = set(routing_info["ablated_pairs"])
        
        self.hook_manager.set_ablated_heads(ablated_pairs)
        self.current_strategy = f"dynamic_task_{routing_info['task_key']}"
        self.current_ablated_heads = ablated_pairs
        
        routing_info["strategy"] = self.current_strategy
        return routing_info
