"""
Task definitions: dataset sources, prompt templates, gold targets and metric type.

Every example is normalised to a dict:
    task       task key
    prompt     user message (template 0 unless a variant is requested)
    target     gold continuation used for teacher-forced importance scoring
    label      option index (choice tasks) | None
    reference  gold text for generative metrics (math answer number / summary)

Datasets are streamed from the Hugging Face Hub and the rows we need are cached
as JSONL under artifacts/data so later runs are offline and deterministic.
"""

import json
import random
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional

# ---------------------------------------------------------------------------
# Static task metadata
# ---------------------------------------------------------------------------

TASK_INFO = {
    "sentiment": {"name": "Sentiment (SST-2)", "type": "choice", "metric": "accuracy"},
    "nli": {"name": "Natural Language Inference (MNLI)", "type": "choice", "metric": "accuracy"},
    "qa": {"name": "Reading-Comprehension QA (BoolQ)", "type": "choice", "metric": "accuracy"},
    "math": {"name": "Math Reasoning (GSM8K)", "type": "generate", "metric": "exact_match"},
    "summarization": {"name": "Summarization (CNN/DailyMail)", "type": "generate", "metric": "rougeL"},
    "general": {"name": "General instruction (dense fallback)", "type": "none", "metric": None},
}

OPTIONS = {
    "sentiment": ["negative", "positive"],
    "nli": ["entailment", "neutral", "contradiction"],
    "qa": ["no", "yes"],
}

# (hf_path, hf_config, {logical_split: (hf_split, offset)})
SOURCES = {
    "sentiment": ("stanfordnlp/sst2", None,
                  {"train": ("train", 0), "dev": ("validation", 0), "test": ("validation", 300)}),
    "nli": ("nyu-mll/glue", "mnli",
            {"train": ("train", 0), "dev": ("validation_matched", 0), "test": ("validation_matched", 300)}),
    "qa": ("google/boolq", None,
           {"train": ("train", 0), "dev": ("validation", 0), "test": ("validation", 300)}),
    "math": ("openai/gsm8k", "main",
             {"train": ("train", 0), "dev": ("train", 7000), "test": ("test", 0)}),
    "summarization": ("abisee/cnn_dailymail", "3.0.0",
                      {"train": ("train", 0), "dev": ("validation", 0), "test": ("test", 0)}),
    "general": ("tatsu-lab/alpaca", None,
                {"train": ("train", 0), "test": ("train", 20000)}),
}

# ---------------------------------------------------------------------------
# Prompt templates. Template 0 is used for profiling / evaluation; all of them
# are used (randomly) to train the task classifier so it generalises to
# free-form prompts typed in the UI.
# ---------------------------------------------------------------------------

TEMPLATES: Dict[str, List[Callable[[dict], str]]] = {
    "sentiment": [
        lambda r: ("Classify the sentiment of the following movie review as positive or negative.\n\n"
                   f"Review: {r['text']}\n\nAnswer with one word: positive or negative."),
        lambda r: f"Is the following review positive or negative?\n\"{r['text']}\"",
        lambda r: f"What is the sentiment of this sentence: {r['text']}",
        lambda r: f"Review: {r['text']}\nWas the reviewer happy or unhappy with the film?",
        lambda r: f"Tell me whether this opinion is favourable or unfavourable: {r['text']}",
    ],
    "nli": [
        lambda r: (f"Premise: {r['premise']}\nHypothesis: {r['hypothesis']}\n\n"
                   "Does the premise entail the hypothesis? Answer with one word: entailment, neutral, or contradiction."),
        lambda r: (f"Given that \"{r['premise']}\", is it true that \"{r['hypothesis']}\"? "
                   "Answer entailment, neutral or contradiction."),
        lambda r: f"Sentence A: {r['premise']}\nSentence B: {r['hypothesis']}\nDoes A imply B, contradict B, or neither?",
        lambda r: f"Does \"{r['premise']}\" logically follow to \"{r['hypothesis']}\"?",
    ],
    "qa": [
        lambda r: ("Read the passage and answer the question with yes or no.\n\n"
                   f"Passage: {r['passage']}\n\nQuestion: {r['question']}?\n\nAnswer with one word: yes or no."),
        lambda r: f"{r['passage']}\n\nBased on the text above, {r['question']}? (yes/no)",
        lambda r: f"Context: {r['passage']}\nQ: {r['question']}?\nA (yes or no):",
        lambda r: f"Answer the question using the passage. Question: {r['question']}?\nPassage: {r['passage']}",
    ],
    "math": [
        lambda r: ("Solve the following math problem step by step. End your answer with "
                   f"'The answer is <number>.'\n\nProblem: {r['question']}"),
        lambda r: f"{r['question']}",
        lambda r: f"Math word problem: {r['question']}\nShow your working.",
        lambda r: f"Calculate: {r['question']}",
    ],
    "summarization": [
        lambda r: f"Summarize the following news article in 2-3 sentences.\n\nArticle: {r['article']}",
        lambda r: f"{r['article']}\n\nTL;DR:",
        lambda r: f"Give me the key points of this article:\n{r['article']}",
        lambda r: f"Write a short summary of the text below.\n{r['article']}",
    ],
    "general": [
        lambda r: r["instruction"] + (f"\n\n{r['input']}" if r.get("input") else ""),
    ],
}


def _truncate_words(text: str, n: int) -> str:
    words = text.split()
    return text if len(words) <= n else " ".join(words[:n]) + " ..."


def _clean_gsm8k(answer: str):
    rationale, _, final = answer.partition("####")
    rationale = re.sub(r"<<.*?>>", "", rationale).strip()
    final = final.strip().replace(",", "")
    return rationale, final


def _normalise(task: str, row: dict, data_cfg: dict) -> Optional[dict]:
    """Map a raw HF row to our fields; returns None for unusable rows."""
    if task == "sentiment":
        return {"text": row["sentence"].strip(), "label": int(row["label"])}
    if task == "nli":
        if int(row["label"]) < 0:
            return None
        return {"premise": row["premise"], "hypothesis": row["hypothesis"], "label": int(row["label"])}
    if task == "qa":
        q = row["question"].strip().rstrip("?")
        q = q[0].upper() + q[1:]
        return {"question": q, "passage": _truncate_words(row["passage"], data_cfg.get("qa_max_words", 300)),
                "label": int(bool(row["answer"]))}
    if task == "math":
        rationale, final = _clean_gsm8k(row["answer"])
        return {"question": row["question"], "rationale": rationale, "answer": final}
    if task == "summarization":
        return {"article": _truncate_words(row["article"], data_cfg.get("summarization_max_words", 450)),
                "summary": row["highlights"].replace(" .", ".").strip()}
    if task == "general":
        return {"instruction": row["instruction"], "input": row.get("input", "")}
    raise KeyError(task)


def _target(task: str, r: dict) -> Optional[str]:
    if task in OPTIONS:
        return OPTIONS[task][r["label"]]
    if task == "math":
        return f"{r['rationale']}\nThe answer is {r['answer']}."
    if task == "summarization":
        return r["summary"]
    return None


def _fetch_rows(task: str, hf_split: str, n: int, cache_dir: Path, data_cfg: dict) -> List[dict]:
    """Return the first n normalised rows of an HF split, using/extending the JSONL cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{task}__{hf_split}.jsonl"
    rows: List[dict] = []
    if cache.exists():
        with open(cache) as f:
            rows = [json.loads(l) for l in f]
        if len(rows) >= n:
            return rows[:n]

    from datasets import load_dataset

    path, name, _ = SOURCES[task]
    print(f"  downloading {path}{'/' + name if name else ''} [{hf_split}] ({n} rows) ...")
    ds = load_dataset(path, name, split=hf_split, streaming=True)
    rows = []
    for raw in ds:
        r = _normalise(task, raw, data_cfg)
        if r is not None:
            rows.append(r)
        if len(rows) >= n:
            break
    with open(cache, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


def load_examples(task: str, split: str, n: int, cfg: dict, template: int = 0,
                  random_template: bool = False, seed: int = 0) -> List[dict]:
    """Load n examples of a logical split ('train' | 'dev' | 'test') for a task."""
    hf_split, offset = SOURCES[task][2][split]
    cache_dir = Path(cfg["artifacts_dir"]) / "data"
    rows = _fetch_rows(task, hf_split, offset + n, cache_dir, cfg.get("data", {}))[offset:offset + n]
    rng = random.Random(seed)
    out = []
    for r in rows:
        t = rng.randrange(len(TEMPLATES[task])) if random_template else template
        out.append({
            "task": task,
            "prompt": TEMPLATES[task][t](r),
            "target": _target(task, r),
            "label": r.get("label"),
            "reference": r.get("answer") if task == "math" else r.get("summary"),
        })
    return out
