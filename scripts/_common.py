import argparse
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from headrouter.config import Paths, load_config  # noqa: E402


def parse(desc: str, extra=None):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    ap.add_argument("--quick", action="store_true", help="small sizes for a fast smoke run")
    ap.add_argument("--tasks", nargs="*", default=None, help="subset of task keys")
    if extra:
        extra(ap)
    args = ap.parse_args()
    cfg = load_config(args.config, args.quick)
    if args.tasks:
        cfg["tasks"] = [t for t in cfg["tasks"] if t in args.tasks]
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    return args, cfg, Paths(cfg).ensure()
