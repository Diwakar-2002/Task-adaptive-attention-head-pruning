"""
Unit Tests for TaskAwareRouter
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from causal_pruning.router import TaskAwareRouter

class TestTaskAwareRouter(unittest.TestCase):
    def setUp(self):
        self.router = TaskAwareRouter(num_layers=4, num_heads=4)

    def test_task_classification(self):
        p1 = "Please summarize this document into bullet points"
        self.assertEqual(self.router.classify_prompt(p1), "summarization")

        p2 = "Write a python function to solve array sorting"
        self.assertEqual(self.router.classify_prompt(p2), "logic_math")

        p3 = "The product quality was amazing and extremely positive"
        self.assertEqual(self.router.classify_prompt(p3), "sentiment")

    def test_routing_mask_generation(self):
        res = self.router.route("Summarize the main article")
        self.assertEqual(res["task_key"], "summarization")
        self.assertEqual(res["total_heads"], 16)
        self.assertEqual(len(res["mask"]), 4)
        self.assertEqual(len(res["mask"][0]), 4)
        self.assertTrue(res["active_heads"] <= 16)
        self.assertTrue(res["estimated_speedup"] >= 1.0)

if __name__ == "__main__":
    unittest.main()
