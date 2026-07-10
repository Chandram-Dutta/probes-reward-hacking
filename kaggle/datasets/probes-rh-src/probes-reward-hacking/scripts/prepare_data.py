#!/usr/bin/env python3
"""Prepare instruction prompts for Exp 3a."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import ensure_import_path

ensure_import_path()

from probes_rh.data.prep_prompts import (
    load_instruction_prompts,
    save_jsonl,
    to_trl_conversational,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="tatsu-lab/alpaca")
    p.add_argument("--max-prompts", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="data/exp3")
    args = p.parse_args()

    out = Path(args.out_dir)
    rows = load_instruction_prompts(
        dataset_name=args.dataset,
        max_prompts=args.max_prompts,
        seed=args.seed,
    )
    save_jsonl(rows, out / "prompts.jsonl")
    save_jsonl(to_trl_conversational(rows), out / "prompts_trl.jsonl")
    print(f"wrote {len(rows)} prompts to {out}")


if __name__ == "__main__":
    main()
