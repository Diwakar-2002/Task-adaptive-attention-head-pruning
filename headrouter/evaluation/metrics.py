"""Task quality metrics and batched evaluation under a given mask variant."""

import re
from typing import Dict, List, Optional, Sequence

import torch

from ..data.tasks import OPTIONS, TASK_INFO
from ..model_utils import encode_prompts

_NUM = r"-?\$?\d[\d,]*(?:\.\d+)?"


def extract_number(text: str) -> Optional[str]:
    m = re.search(r"answer is[:\s]*\**\s*(" + _NUM + ")", text, flags=re.IGNORECASE)
    if m is None:
        nums = re.findall(_NUM, text)
        if not nums:
            return None
        s = nums[-1]
    else:
        s = m.group(1)
    return s.replace(",", "").replace("$", "").rstrip(".")


def math_correct(pred_text: str, gold: str) -> bool:
    p = extract_number(pred_text)
    if p is None:
        return False
    try:
        return abs(float(p) - float(gold)) < 1e-6
    except ValueError:
        return False


_rouge = None


def rouge_l(pred: str, ref: str) -> float:
    global _rouge
    if _rouge is None:
        from rouge_score import rouge_scorer
        _rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return _rouge.score(ref, pred)["rougeL"].fmeasure


def option_token_ids(tok, task: str) -> List[List[int]]:
    """First-token ids for each option (lower-case and Capitalised spellings)."""
    out = []
    for opt in OPTIONS[task]:
        ids = {tok(v, add_special_tokens=False)["input_ids"][0] for v in (opt, opt.capitalize())}
        out.append(sorted(ids))
    flat = [i for ids in out for i in ids]
    assert len(flat) == len(set(flat)), f"option first-tokens collide for {task}"
    return out


@torch.no_grad()
def choice_predict(model, tok, prompts: Sequence[str], task: str, device: str, batch_size: int = 16) -> List[int]:
    opt_ids = option_token_ids(tok, task)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    preds = [0] * len(prompts)
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        enc = encode_prompts(tok, [prompts[i] for i in idx], device)
        logits = model(**enc, logits_to_keep=1).logits[:, -1, :].float()
        logp = torch.log_softmax(logits, dim=-1)
        scores = torch.stack([torch.logsumexp(logp[:, ids], dim=-1) for ids in opt_ids], dim=-1)
        for j, p in zip(idx, scores.argmax(-1).tolist()):
            preds[j] = p
    return preds


def gen_kwargs(tok, max_new_tokens: int) -> dict:
    return dict(max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None, top_k=None,
                pad_token_id=tok.pad_token_id)


@torch.no_grad()
def generate_batch(model, tok, prompts: Sequence[str], device: str, max_new_tokens: int,
                   batch_size: int = 16) -> List[str]:
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    outs = [""] * len(prompts)
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        enc = encode_prompts(tok, [prompts[i] for i in idx], device)
        gen = model.generate(**enc, **gen_kwargs(tok, max_new_tokens))
        texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for j, t in zip(idx, texts):
            outs[j] = t.strip()
    return outs


def max_new_tokens_for(task: str, cfg) -> int:
    g = cfg["generation"]
    return g["math_max_new_tokens"] if task == "math" else g["summarization_max_new_tokens"]


def evaluate_examples(model, tok, engine, examples: List[dict], variants: Sequence[str], cfg,
                      device: str) -> Dict:
    """
    Score examples where example i runs under mask variant variants[i] (all
    examples belong to one task). Returns {"score", "per_example"}.
    """
    task = examples[0]["task"]
    bs = cfg["generation"]["batch_size"]
    per = [0.0] * len(examples)
    for var in sorted(set(variants)):
        idx = [i for i, v in enumerate(variants) if v == var]
        prompts = [examples[i]["prompt"] for i in idx]
        with engine.use(var):
            if TASK_INFO[task]["type"] == "choice":
                preds = choice_predict(model, tok, prompts, task, device, bs)
                for i, p in zip(idx, preds):
                    per[i] = float(p == examples[i]["label"])
            else:
                texts = generate_batch(model, tok, prompts, device, max_new_tokens_for(task, cfg), bs)
                for i, t in zip(idx, texts):
                    ref = examples[i]["reference"]
                    per[i] = float(math_correct(t, ref)) if task == "math" else rouge_l(t, ref)
    return {"score": sum(per) / max(1, len(per)), "per_example": per}
