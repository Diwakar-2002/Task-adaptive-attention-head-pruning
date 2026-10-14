"""Timed generation: TTFT, time per output token, throughput, energy, working memory."""

import time
from typing import Optional

import torch
from transformers.generation.streamers import BaseStreamer

from ..model_utils import encode_prompts
from .energy import EnergyMeter
from .metrics import gen_kwargs


MIN_ENERGY_SECONDS = 0.5


def sync(device: str):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


class TimingStreamer(BaseStreamer):
    """Records a synchronised timestamp each time generate() emits new tokens."""

    def __init__(self, device: str):
        self.device = device
        self.times = []
        self._prompt_seen = False

    def put(self, value):
        if not self._prompt_seen:          # first call carries the prompt
            self._prompt_seen = True
            return
        sync(self.device)
        self.times.append(time.perf_counter())

    def end(self):
        pass


@torch.no_grad()
def timed_generate(model, tok, prompt: str, device: str, max_new_tokens: int,
                   fixed_length: bool = False, meter: Optional[EnergyMeter] = None,
                   input_ids: Optional[torch.Tensor] = None) -> dict:
    """Greedy-generate one prompt and measure it. `fixed_length` forces exactly max_new_tokens."""
    if input_ids is None:
        enc = encode_prompts(tok, [prompt], device)
    else:
        enc = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
    n_in = enc["input_ids"].shape[1]
    kw = gen_kwargs(tok, max_new_tokens)
    if fixed_length:
        kw["min_new_tokens"] = max_new_tokens
    streamer = TimingStreamer(device)

    cuda = device.startswith("cuda")
    sync(device)
    if cuda:
        torch.cuda.reset_peak_memory_stats()
        base_mem = torch.cuda.memory_allocated()
    if meter is not None:
        meter.start()
    t0 = time.perf_counter()
    out = model.generate(**enc, streamer=streamer, **kw)
    sync(device)
    t1 = time.perf_counter()
    energy = meter.stop() if meter is not None else None
    if energy is not None and (t1 - t0) < MIN_ENERGY_SECONDS:
        energy = None          # NVML's counter updates too coarsely to resolve very short runs

    n_new = out.shape[1] - n_in
    ts = streamer.times
    ttft = (ts[0] - t0) if ts else (t1 - t0)
    tpot = (ts[-1] - ts[0]) / (len(ts) - 1) if len(ts) > 1 else None
    total = t1 - t0
    return {
        "text": tok.decode(out[0, n_in:], skip_special_tokens=True).strip(),
        "input_tokens": n_in,
        "new_tokens": int(n_new),
        "total_ms": total * 1000,
        "ttft_ms": ttft * 1000,
        "tpot_ms": tpot * 1000 if tpot is not None else None,
        "tokens_per_s": n_new / total if total > 0 else None,
        "energy_j": energy,
        "energy_per_token_mj": (energy / n_new * 1000) if (energy is not None and n_new) else None,
        "working_mem_mb": ((torch.cuda.max_memory_allocated() - base_mem) / 2**20) if cuda else None,
    }
