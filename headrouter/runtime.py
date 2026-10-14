"""Everything needed at inference time, loaded once: model + engine + registry + router + meter."""

import threading
from typing import Dict, Optional

import torch

from .classifier.embedding import Router, TaskClassifier
from .config import Paths
from .evaluation.energy import EnergyMeter
from .evaluation.timers import timed_generate
from .masks.builder import load_head_means, load_registry, register_all
from .model_utils import load_model


class Runtime:
    def __init__(self, cfg: Dict, load_classifier: bool = True):
        self.cfg = cfg
        self.paths = Paths(cfg).ensure()
        self.model, self.tok, self.engine, self.device = load_model(cfg)
        self.registry = load_registry(self.paths.registry)
        self.warnings = []
        if self.registry and self.registry.get("model") != cfg["model"]["name"]:
            self.warnings.append(f"registry was built for {self.registry.get('model')}, ignoring it")
            self.registry = None
        if self.registry:
            register_all(self.engine, self.registry, load_head_means(self.paths.head_means))
        self.classifier: Optional[TaskClassifier] = None
        if load_classifier and self.paths.classifier.exists():
            self.classifier = TaskClassifier.load(self.paths.classifier, self.device)
        self.router = Router(self.classifier, (self.registry or {}).get("tasks", {}),
                             cfg["classifier"]["confidence_threshold"])
        self.meter = EnergyMeter(self.device)
        self.lock = threading.Lock()

    def gpu_name(self) -> str:
        return torch.cuda.get_device_name(0) if self.device.startswith("cuda") else "CPU"

    def warmup(self, runs: int = 2):
        prompt = "Write one sentence about the ocean."
        for name in list(self.engine.masks):
            with self.engine.use(name):
                for _ in range(runs):
                    timed_generate(self.model, self.tok, prompt, self.device, 16, fixed_length=True)
        if self.classifier is not None:
            self.classifier.predict(prompt)
