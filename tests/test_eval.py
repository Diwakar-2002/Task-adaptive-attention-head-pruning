"""
Unit Tests for CausalEvaluator
"""

import unittest
import torch
import numpy as np
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from causal_pruning.causal_eval import CausalEvaluator

class DummyTokenizer:
    def __call__(self, texts, **kwargs):
        input_ids = torch.randint(0, 100, (len(texts), 8))
        class BatchDict(dict):
            def to(self, device):
                return {k: v.to(device) if hasattr(v, 'to') else v for k, v in self.items()}
        return BatchDict({"input_ids": input_ids})

class TestCausalEvaluator(unittest.TestCase):
    def setUp(self):
        self.config = GPT2Config(
            n_layer=2,
            n_head=4,
            n_embd=64,
            vocab_size=100
        )
        self.model = GPT2LMHeadModel(self.config)
        self.tokenizer = DummyTokenizer()
        self.evaluator = CausalEvaluator(self.model, self.tokenizer, device="cpu")

    def test_kl_divergence_non_negative(self):
        logits1 = torch.randn(1, 8, 100)
        logits2 = torch.randn(1, 8, 100)

        # Self-KL should be 0.0
        self_kl = self.evaluator.compute_kl_divergence(logits1, logits1)
        self.assertAlmostEqual(self_kl, 0.0, places=4)

        # Cross-KL should be >= 0.0
        cross_kl = self.evaluator.compute_kl_divergence(logits1, logits2)
        self.assertTrue(cross_kl >= 0.0)

    def test_magnitude_saliency_matrix_shape(self):
        mag = self.evaluator.evaluate_magnitude_saliency()
        self.assertEqual(mag.shape, (2, 4))
        self.assertTrue(np.max(mag) <= 1.0)

    def test_causal_impact_matrix(self):
        res = self.evaluator.evaluate_causal_impact_matrix(["test prompt 1", "test prompt 2"])
        self.assertEqual(res["num_layers"], 2)
        self.assertEqual(res["num_heads"], 4)
        self.assertEqual(len(res["causal_heatmap"]), 2)
        self.assertEqual(len(res["causal_heatmap"][0]), 4)
        self.assertEqual(len(res["head_rankings"]), 8)

if __name__ == "__main__":
    unittest.main()
