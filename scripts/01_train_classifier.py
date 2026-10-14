"""Step 1 - download/cache task data and train the task classifier."""

import json
import time

import numpy as np
from _common import parse

from headrouter.classifier.embedding import TaskClassifier
from headrouter.data.tasks import load_examples
from headrouter.model_utils import resolve_device


def main():
    args, cfg, paths = parse(__doc__)
    c = cfg["classifier"]
    classes = list(cfg["tasks"]) + (["general"] if c.get("general_class") else [])

    # Pre-fetch every split later steps need, so profiling/benchmarking run offline.
    print("Caching datasets ...")
    for t in cfg["tasks"]:
        load_examples(t, "train", max(cfg["profiling"]["calib_size"], c["train_per_task"]), cfg)
        load_examples(t, "dev", cfg["masks"]["dev_size"][t], cfg)
        load_examples(t, "test", max(cfg["benchmark"]["quality_size"][t], c["test_per_task"]), cfg)

    tr_x, tr_y, te_x, te_y = [], [], [], []
    for t in classes:
        tr = load_examples(t, "train", c["train_per_task"], cfg, random_template=True, seed=cfg["seed"])
        te = load_examples(t, "test", c["test_per_task"], cfg, random_template=True, seed=cfg["seed"] + 1)
        tr_x += [e["prompt"] for e in tr]
        tr_y += [t] * len(tr)
        te_x += [e["prompt"] for e in te]
        te_y += [t] * len(te)

    device = resolve_device(cfg)
    print(f"Training classifier on {len(tr_x)} prompts ({', '.join(classes)}) ...")
    clf = TaskClassifier(c["embedder"], device).fit(tr_x, tr_y)

    proba = clf.predict_proba(te_x)
    pred = [clf.labels[i] for i in proba.argmax(1)]
    conf = proba.max(1)
    acc = float(np.mean([p == y for p, y in zip(pred, te_y)]))
    thr = c["confidence_threshold"]
    routed = conf >= thr
    routed_acc = float(np.mean([p == y for p, y, r in zip(pred, te_y, routed) if r])) if routed.any() else None
    confusion = {y: {p: 0 for p in clf.labels} for y in clf.labels}
    for p, y in zip(pred, te_y):
        confusion[y][p] += 1

    clf.predict(te_x[0])                                     # warm-up
    lat = [clf.predict(x)["classifier_ms"] for x in te_x[:200]]
    report = {
        "classes": clf.labels, "test_size": len(te_x), "accuracy": acc,
        "confidence_threshold": thr, "routed_fraction": float(routed.mean()),
        "accuracy_when_routed": routed_acc, "confusion": confusion,
        "latency_ms_median": float(np.median(lat)), "latency_ms_p95": float(np.percentile(lat, 95)),
        "device": device, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    clf.save(paths.classifier)
    with open(paths.classifier_report, "w") as f:
        json.dump(report, f, indent=1)
    print(f"accuracy={acc:.3f}  routed={routed.mean():.2%}  acc|routed={routed_acc}  "
          f"latency median={report['latency_ms_median']:.1f} ms")
    print(f"saved {paths.classifier}")


if __name__ == "__main__":
    main()
