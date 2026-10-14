import numpy as np

from headrouter.masks.builder import build_mask, protected_heads, static_scores, within_budget


def test_query_mode_respects_sparsity_and_min_heads():
    rng = np.random.default_rng(0)
    scores = rng.random((6, 8))
    scores[0] = 0.0                     # layer 0 looks useless; must still keep 2 heads
    keep = build_mask(scores, 0.5, min_heads_per_layer=2)
    assert (~keep).sum() == 24
    assert keep.sum(axis=1).min() >= 2
    assert keep[0].sum() == 2


def test_protected_heads_are_never_pruned():
    imp = np.random.default_rng(1).random((3, 4, 8))
    imp[:, 2, 5] = 10.0                 # top for every task
    prot = protected_heads(imp, 0.1)
    assert prot[2, 5]
    scores = imp[0].copy()
    scores[2, 5] = -1.0                 # this task alone would prune it first
    keep = build_mask(scores, 0.5, 1, prot)
    assert keep[2, 5]


def test_kv_group_mode_prunes_whole_groups():
    scores = np.random.default_rng(2).random((4, 8))
    keep = build_mask(scores, 0.25, min_heads_per_layer=2, mode="kv_group", num_kv=2)
    groups = keep.reshape(4, 2, 4)
    assert np.all(groups.all(-1) | (~groups).all(-1))    # each group fully kept or fully pruned
    assert (~groups.all(-1)).sum() == 2
    assert groups.all(-1).sum(-1).min() >= 1


def test_zero_sparsity_is_dense_and_budget_rules():
    keep = build_mask(np.zeros((2, 4)), 0.0)
    assert keep.all()
    assert within_budget(0.80, 0.79, 0.015, "relative")
    assert not within_budget(0.80, 0.78, 0.015, "relative")
    assert within_budget(0.80, 0.79, 0.015, "absolute")
    assert static_scores(np.random.default_rng(3).random((3, 2, 4))).shape == (2, 4)
