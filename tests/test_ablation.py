"""
Unit Tests for HeadAblationHook
"""

import unittest
import torch
from transformers import GPT2Config, GPT2LMHeadModel
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from causal_pruning.ablation import HeadAblationHook

class TestHeadAblationHook(unittest.TestCase):
    def setUp(self):
        # Create lightweight toy model config
        self.config = GPT2Config(
            n_layer=2,
            n_head=4,
            n_embd=64,
            vocab_size=100
        )
        self.model = GPT2LMHeadModel(self.config)
        self.model.eval()

    def test_hook_installation_and_clear(self):
        hook_mgr = HeadAblationHook(self.model)
        self.assertEqual(hook_mgr.num_layers, 2)
        self.assertEqual(hook_mgr.num_heads, 4)
        self.assertEqual(hook_mgr.head_dim, 16)

        # Set ablated heads
        hook_mgr.set_ablated_heads({(0, 1), (1, 3)})
        self.assertEqual(len(hook_mgr.hooks), 2)
        self.assertEqual(hook_mgr.ablated_heads, {(0, 1), (1, 3)})

        # Clear hooks
        hook_mgr.clear_hooks()
        self.assertEqual(len(hook_mgr.hooks), 0)

    def test_forward_pass_masking(self):
        hook_mgr = HeadAblationHook(self.model)
        dummy_input = torch.randint(0, 100, (1, 8))

        # Baseline forward pass
        with torch.no_grad():
            base_output = self.model(dummy_input).logits

        # Ablated forward pass
        hook_mgr.set_ablated_heads({(0, 0), (1, 2)})
        with torch.no_grad():
            ablated_output = self.model(dummy_input).logits

        hook_mgr.clear_hooks()
        
        # Verify logits shape remains identical
        self.assertEqual(base_output.shape, ablated_output.shape)
        
        # Verify outputs differ slightly due to ablated heads
        diff = torch.abs(base_output - ablated_output).sum().item()
        self.assertGreaterThan(diff, 0.0)

    def assertGreaterThan(self, a, b):
        self.assertTrue(a > b, f"{a} is not greater than {b}")

if __name__ == "__main__":
    unittest.main()
