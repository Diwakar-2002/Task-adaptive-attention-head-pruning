"""Step 2 - offline head-importance profiling for every task."""

import time

import numpy as np
from _common import parse

from headrouter.data.tasks import load_examples
from headrouter.model_utils import load_model
from headrouter.profiling.importance import SCORERS, head_output_means, layer_normalize


def main():
    args, cfg, paths = parse(__doc__, lambda ap: ap.add_argument(
        "--scorer", nargs="*", default=None, choices=list(SCORERS), help="override profiling.scorer"))
    p = cfg["profiling"]
    scorers = args.scorer or [p["scorer"]]
    model, tok, engine, device = load_model(cfg)
    print(f"{cfg['model']['name']}: {engine.num_layers} layers x {engine.num_heads} heads "
          f"({engine.num_kv} KV heads, head_dim {engine.head_dim}) on {device}")

    for scorer in scorers:
        n = p["causal_calib_size"] if scorer == "causal" else p["calib_size"]
        raw = []
        for t in cfg["tasks"]:
            ex = load_examples(t, "train", n, cfg)
            t0 = time.time()
            imp = SCORERS[scorer](model, tok, engine, ex, device, p["max_target_tokens"])
            raw.append(imp)
            print(f"  [{scorer}] {t:<14} {len(ex)} examples  {time.time() - t0:6.1f}s  "
                  f"mean={imp.mean():.3g}  max={imp.max():.3g}")
        raw = np.stack(raw)
        norm = np.stack([layer_normalize(r) for r in raw])
        np.savez(paths.importance(scorer), raw=raw, normalized=norm, tasks=np.array(cfg["tasks"]),
                 scorer=scorer, model=cfg["model"]["name"], calib_size=n)
        print(f"saved {paths.importance(scorer)}")

    # Mean head outputs per task, used for bias compensation of pruned heads.
    means = {}
    for t in cfg["tasks"]:
        means[t] = head_output_means(model, tok, engine, load_examples(t, "train", p["calib_size"], cfg), device,
                                     p["max_target_tokens"])
    means["static"] = np.mean([means[t] for t in cfg["tasks"]], axis=0)
    np.savez(paths.head_means, **means)
    print(f"saved {paths.head_means}")


if __name__ == "__main__":
    main()
