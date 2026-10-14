"""Model loading, chat formatting and batched teacher-forced inputs."""

from typing import List, Optional, Tuple

import torch

from .engine.pruned_attention import HeadRouterEngine

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def resolve_device(cfg) -> str:
    dev = cfg["model"].get("device", "cuda")
    if dev == "cuda" and not torch.cuda.is_available():
        print("[headrouter] CUDA not available - falling back to CPU")
        return "cpu"
    return dev


def load_model(cfg, device: Optional[str] = None):
    """Load tokenizer + model and install the HeadRouter engine. Returns (model, tok, engine, device)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or resolve_device(cfg)
    name = cfg["model"]["name"]
    dtype = DTYPES[cfg["model"].get("dtype", "bfloat16")]
    if device == "cpu" and dtype == torch.float16:
        dtype = torch.float32
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, attn_implementation="sdpa")
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    engine = HeadRouterEngine(model)
    return model, tok, engine, device


def chat_text(tok, user_msg: str) -> str:
    return tok.apply_chat_template([{"role": "user", "content": user_msg}], tokenize=False,
                                   add_generation_prompt=True)


def encode_prompts(tok, prompts: List[str], device: str, max_len: Optional[int] = None):
    texts = [chat_text(tok, p) for p in prompts]
    return tok(texts, return_tensors="pt", padding=True, truncation=max_len is not None,
               max_length=max_len, add_special_tokens=False).to(device)


def teacher_forced_batch(tok, prompt: str, target: str, device: str, max_target_tokens: int = 128
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
    """input_ids and labels (-100 on prompt tokens) for loss on the gold target only."""
    p_ids = tok(chat_text(tok, prompt), add_special_tokens=False)["input_ids"]
    t_ids = tok(target, add_special_tokens=False)["input_ids"][:max_target_tokens]
    ids = torch.tensor([p_ids + t_ids], device=device)
    labels = torch.tensor([[-100] * len(p_ids) + t_ids], device=device)
    return ids, labels
