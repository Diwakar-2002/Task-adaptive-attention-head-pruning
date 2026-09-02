"""
Model Performance Profiler & Benchmarker
Measures generation latency, throughput, memory footprint, FLOPs savings, and accuracy metrics.
"""

from typing import Dict, List, Any, Optional
import time
import torch
import torch.nn as nn
import numpy as np

class ModelProfiler:
    """
    Benchmarks model execution speed, throughput, memory usage, and FLOP reduction
    across different pruning strategies.
    """
    def __init__(self, model: nn.Module, tokenizer: Any, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def measure_generation(
        self,
        prompt: str,
        max_new_tokens: int = 24,
        num_runs: int = 2
    ) -> Dict[str, Any]:
        """
        Measure inference latency (ms/token), total duration, memory, and generated response text.
        """
        self.model.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs.input_ids.shape[1]

        # Warmup run
        with torch.no_grad():
            _ = self.model.generate(**inputs, max_new_tokens=4, do_sample=False)

        latencies = []
        outputs = []
        
        for _ in range(num_runs):
            start_time = time.perf_counter()
            with torch.no_grad():
                out_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id if hasattr(self.tokenizer, "eos_token_id") else None
                )
            end_time = time.perf_counter()
            
            gen_tokens = out_ids.shape[1] - input_len
            total_duration_ms = (end_time - start_time) * 1000.0
            ms_per_token = total_duration_ms / max(1, gen_tokens)
            latencies.append(ms_per_token)
            
            text = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
            outputs.append(text)

        mean_ms_per_token = float(np.mean(latencies))
        gen_text = outputs[-1]

        # Estimate memory usage
        if torch.cuda.is_available() and self.device != "cpu":
            mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            # Memory estimate based on model parameter count + dynamic allocations
            param_count = sum(p.numel() for p in self.model.parameters())
            mem_mb = float((param_count * 4) / (1024 * 1024) + 120.0)

        return {
            "prompt": prompt,
            "generated_text": gen_text,
            "ms_per_token": round(mean_ms_per_token, 2),
            "tokens_per_second": round(1000.0 / max(0.1, mean_ms_per_token), 1),
            "memory_mb": round(mem_mb, 1),
            "generated_token_count": gen_tokens
        }

    def compute_flops_reduction(
        self,
        num_layers: int,
        num_heads: int,
        ablated_head_count: int,
        hidden_dim: int = 768,
        seq_len: int = 128
    ) -> Dict[str, Any]:
        """
        Calculates theoretical FLOPs saving achieved by removing ablated_head_count heads.
        Attention FLOPs per layer: 4 * seq_len * hidden_dim^2 + 2 * seq_len^2 * hidden_dim
        """
        total_heads = num_layers * num_heads
        head_ratio = ablated_head_count / max(1, total_heads)
        
        # Attention layer contribution is ~40% of total Transformer FLOPs
        attn_flop_fraction = 0.42
        total_flops_reduced_pct = head_ratio * attn_flop_fraction * 100.0
        
        return {
            "total_heads": total_heads,
            "pruned_heads": ablated_head_count,
            "active_heads": total_heads - ablated_head_count,
            "pruning_ratio_pct": round(head_ratio * 100.0, 1),
            "attention_flops_reduction_pct": round(head_ratio * 100.0, 1),
            "overall_model_flops_reduction_pct": round(total_flops_reduced_pct, 1),
            "theoretical_speedup": round(1.0 / max(0.01, (1.0 - total_flops_reduced_pct / 100.0)), 2)
        }
