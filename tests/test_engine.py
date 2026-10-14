"""
Correctness of the pruned execution engine on a tiny random Qwen2 model (CPU, float32):
  * dense variant == stock Hugging Face attention
  * physically sliced variant == stock model with the same heads zero-masked
  * incremental decoding with a KV cache matches under pruning (incl. a removed GQA group)
"""

import copy

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from headrouter.engine.pruned_attention import HeadRouterEngine

L, H, KV, D = 3, 8, 2, 16


def tiny_model():
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=128, hidden_size=H * D, intermediate_size=64, num_hidden_layers=L,
                      num_attention_heads=H, num_key_value_heads=KV, max_position_embeddings=128,
                      attn_implementation="sdpa")
    return Qwen2ForCausalLM(cfg).eval()


def zero_mask_hooks(model, keep):
    """Reference implementation: zero the pruned heads' outputs at the O-projection input."""
    handles = []
    for l, layer in enumerate(model.model.layers):
        def pre(mod, inp, l=l):
            x = inp[0].clone().view(*inp[0].shape[:-1], H, D)
            for h in range(H):
                if not keep[l][h]:
                    x[..., h, :] = 0
            return (x.view(*inp[0].shape),)
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(pre))
    return handles


KEEP = [
    [True, False, True, True, False, True, True, True],
    [False, False, False, False, True, True, False, True],    # GQA group 0 removed entirely
    [True] * H,
]


@pytest.fixture
def models():
    ref = tiny_model()
    ours = copy.deepcopy(ref)
    engine = HeadRouterEngine(ours)
    engine.register("task", KEEP)
    return ref, ours, engine


@torch.no_grad()
def test_dense_variant_matches_stock(models):
    ref, ours, engine = models
    ids = torch.randint(0, 128, (2, 12))
    engine.activate("dense")
    torch.testing.assert_close(ours(ids).logits, ref(ids).logits, atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_sliced_variant_matches_zero_masking(models):
    ref, ours, engine = models
    ids = torch.randint(0, 128, (2, 12))
    hs = zero_mask_hooks(ref, KEEP)
    expected = ref(ids).logits
    for h in hs:
        h.remove()
    engine.activate("task")
    torch.testing.assert_close(ours(ids).logits, expected, atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_bias_compensation_matches_mean_replacement():
    """Sliced variant + folded bias == stock model with pruned heads replaced by their mean output."""
    ref = tiny_model()
    ours = copy.deepcopy(ref)
    engine = HeadRouterEngine(ours)
    means = torch.randn(L, H, D)
    engine.register("comp", KEEP, head_means=means)
    hs = []
    for l, layer in enumerate(ref.model.layers):
        def pre(mod, inp, l=l):
            x = inp[0].clone().view(*inp[0].shape[:-1], H, D)
            for h in range(H):
                if not KEEP[l][h]:
                    x[..., h, :] = means[l, h]
            return (x.view(*inp[0].shape),)
        hs.append(layer.self_attn.o_proj.register_forward_pre_hook(pre))
    ids = torch.randint(0, 128, (2, 12))
    expected = ref(ids).logits
    for h in hs:
        h.remove()
    engine.activate("comp")
    torch.testing.assert_close(ours(ids).logits, expected, atol=1e-4, rtol=1e-4)


@torch.no_grad()
def test_weights_are_physically_smaller(models):
    _, _, engine = models
    v = engine.layers[1].variants["task"]
    assert v.q_w.shape[0] == 3 * D and v.o_w.shape[1] == 3 * D
    assert v.k_w.shape[0] == 1 * D and v.v_w.shape[0] == 1 * D      # one KV group left
    assert engine.stats("task")["pruned_heads"] == 2 + 5


@torch.no_grad()
def test_generation_with_cache_and_padding(models):
    ref, ours, engine = models
    ids = torch.randint(1, 128, (2, 10))
    mask = torch.ones_like(ids)
    mask[0, :3] = 0                                                   # left padding
    hs = zero_mask_hooks(ref, KEEP)
    exp = ref.generate(input_ids=ids, attention_mask=mask, max_new_tokens=8, do_sample=False, pad_token_id=0)
    for h in hs:
        h.remove()
    engine.activate("task")
    got = ours.generate(input_ids=ids, attention_mask=mask, max_new_tokens=8, do_sample=False, pad_token_id=0)
    assert torch.equal(got, exp)


def test_variant_switching_is_isolated(models):
    _, _, engine = models
    with engine.use("task"):
        assert all(l.active == "task" for l in engine.layers)
    assert engine.active == "dense"
