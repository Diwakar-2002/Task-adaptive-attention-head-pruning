"""
Task classifier v1: frozen sentence embedding (all-MiniLM-L6-v2) + logistic regression.

An extra "general" class (Alpaca instructions) catches prompts that are not one
of the profiled tasks; those, and any low-confidence prediction, are routed to
the dense model.
"""

import time
from typing import Dict, List, Optional

import joblib
import numpy as np
import torch


def _clip(text: str, head: int = 150, tail: int = 60) -> str:
    """MiniLM sees 256 tokens; keep the start and the end of long prompts (instructions live there)."""
    words = text.split()
    if len(words) <= head + tail:
        return text
    return " ".join(words[:head]) + " ... " + " ".join(words[-tail:])


class TaskClassifier:
    def __init__(self, embedder: str, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer
        self.embedder_name = embedder
        self.device = device
        self.st = SentenceTransformer(embedder, device=device)
        self.lr = None
        self.labels: List[str] = []

    def embed(self, texts: List[str]) -> np.ndarray:
        return self.st.encode([_clip(t) for t in texts], batch_size=64, normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=False)

    def fit(self, texts: List[str], labels: List[str]):
        from sklearn.linear_model import LogisticRegression
        self.labels = sorted(set(labels))
        y = np.array([self.labels.index(l) for l in labels])
        self.lr = LogisticRegression(max_iter=3000, C=4.0)
        self.lr.fit(self.embed(texts), y)
        return self

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        return self.lr.predict_proba(self.embed(texts))

    def predict(self, text: str) -> Dict:
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        p = self.predict_proba([text])[0]
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        i = int(p.argmax())
        return {"task": self.labels[i], "confidence": float(p[i]),
                "probs": {l: float(v) for l, v in zip(self.labels, p)}, "classifier_ms": ms}

    def save(self, path):
        joblib.dump({"embedder": self.embedder_name, "lr": self.lr, "labels": self.labels}, path)

    @classmethod
    def load(cls, path, device: str = "cpu") -> "TaskClassifier":
        blob = joblib.load(path)
        obj = cls(blob["embedder"], device)
        obj.lr, obj.labels = blob["lr"], blob["labels"]
        return obj


class Router:
    """Classifier + registry -> which mask variant to run."""

    def __init__(self, classifier: Optional[TaskClassifier], task_masks: Dict[str, dict], threshold: float):
        self.classifier = classifier
        self.task_masks = task_masks      # task -> registry entry
        self.threshold = threshold

    def route(self, prompt: str, force_task: Optional[str] = None) -> Dict:
        if force_task:
            pred = {"task": force_task, "confidence": 1.0, "probs": {}, "classifier_ms": 0.0, "forced": True}
        elif self.classifier is None:
            return {"task": None, "confidence": 0.0, "probs": {}, "classifier_ms": 0.0,
                    "variant": "dense", "fallback": "no classifier trained"}
        else:
            pred = self.classifier.predict(prompt)
            pred["forced"] = False
        task = pred["task"]
        if task not in self.task_masks:
            pred.update(variant="dense", fallback=f"'{task}' has no pruning profile")
        elif pred["confidence"] < self.threshold:
            pred.update(variant="dense",
                        fallback=f"confidence {pred['confidence']:.2f} < threshold {self.threshold:.2f}")
        elif self.task_masks[task]["sparsity"] <= 0:
            pred.update(variant="dense", fallback=f"no sparsity level for '{task}' met the accuracy budget")
        else:
            pred.update(variant=task, fallback=None)
        return pred
