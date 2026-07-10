"""Build a compact instruction-following prompt set for Exp 3a."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import load_dataset


DEFAULT_HF_DATASET = "tatsu-lab/alpaca"
DEFAULT_SPLIT = "train"


def _row_to_prompt(row: dict[str, Any]) -> str | None:
    instruction = (row.get("instruction") or "").strip()
    inp = (row.get("input") or "").strip()
    if not instruction:
        return None
    if inp:
        return f"{instruction}\n\nInput:\n{inp}"
    return instruction


def load_instruction_prompts(
    dataset_name: str = DEFAULT_HF_DATASET,
    split: str = DEFAULT_SPLIT,
    max_prompts: int = 2000,
    seed: int = 42,
) -> list[dict[str, str]]:
    """Return list of {prompt_id, prompt} for GRPO conversational format."""
    ds = load_dataset(dataset_name, split=split)
    ds = ds.shuffle(seed=seed)
    rows: list[dict[str, str]] = []
    for i, row in enumerate(ds):
        if len(rows) >= max_prompts:
            break
        text = _row_to_prompt(row)
        if not text:
            continue
        # keep prompts short enough for free-tier GRPO
        if len(text) > 1200:
            continue
        rows.append({"prompt_id": f"p{i:05d}", "prompt": text})
    return rows


def to_trl_conversational(rows: list[dict[str, str]]) -> list[dict]:
    """TRL GRPO conversational format: prompt is a list of chat messages."""
    out = []
    for r in rows:
        out.append(
            {
                "prompt": [{"role": "user", "content": r["prompt"]}],
                "prompt_id": r["prompt_id"],
            }
        )
    return out


def save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
