"""Configuration loading and artifact paths."""

import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"

# Smaller sizes for a fast end-to-end smoke run (`--quick`).
QUICK_OVERRIDES = {
    "classifier": {"train_per_task": 120, "test_per_task": 40},
    "profiling": {"calib_size": 32, "causal_calib_size": 8},
    "masks": {"dev_size": {"sentiment": 100, "nli": 100, "qa": 100, "math": 24, "summarization": 24}},
    "benchmark": {
        "quality_size": {"sentiment": 100, "nli": 100, "qa": 100, "math": 24, "summarization": 24},
        "speed_prompts_per_task": 2,
        "speed_repeats": 2,
        "context_lengths": [512, 1024, 2048],
        "context_repeats": 3,
    },
}


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Optional[str] = None, quick: bool = False) -> Dict[str, Any]:
    with open(path or DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    if quick:
        cfg = _merge(cfg, QUICK_OVERRIDES)
    art = Path(cfg["artifacts_dir"])
    if not art.is_absolute():
        art = ROOT / art
    cfg["artifacts_dir"] = str(art)
    return cfg


class Paths:
    """All artifact locations, derived from the config."""

    def __init__(self, cfg: Dict[str, Any]):
        self.root = Path(cfg["artifacts_dir"])
        self.data = self.root / "data"
        self.classifier = self.root / "classifier.joblib"
        self.classifier_report = self.root / "classifier_report.json"
        self.importance_dir = self.root / "importance"
        self.registry = self.root / "head_masks.json"
        self.results = self.root / "results"
        self.plots = self.results / "plots"

    @property
    def head_means(self) -> Path:
        return self.importance_dir / "head_means.npz"

    def importance(self, scorer: str) -> Path:
        return self.importance_dir / f"{scorer}.npz"

    def ensure(self):
        for p in (self.root, self.data, self.importance_dir, self.results, self.plots):
            os.makedirs(p, exist_ok=True)
        return self
