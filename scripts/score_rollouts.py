#!/usr/bin/env python3
"""Generate rollouts from a checkpoint and score proxy + gold offline.

Gold RM is never used in training.

Dual-GPU (Kaggle T4 x2): by default
  - policy generation on cuda:0
  - gold RM on cuda:1
Falls back to a single device when only one GPU is available.

Writes under /kaggle/working/outputs/... on Kaggle when --out is relative.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import ensure_import_path

ensure_import_path()

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from probes_rh.chat import apply_chat_prompt, strip_think
from probes_rh.paths import exp_dir, on_kaggle, output_root
from probes_rh.rewards.features import extract_surface_features
from probes_rh.rewards.gold import GoldRMConfig, GoldRewardModel
from probes_rh.rewards.proxy import ProxyRewardConfig, score_proxy_batch


def load_prompts(path: Path, limit: int | None) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def pick_devices(force_device: str | None) -> tuple[str, str | None]:
    if force_device is not None:
        return force_device, force_device
    if not torch.cuda.is_available():
        return "cpu", "cpu"
    n = torch.cuda.device_count()
    if n >= 2:
        return "cuda:0", "cuda:1"
    return "cuda:0", "cuda:0"


def resolve_out(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    # Prefer Kaggle working outputs for relative paths
    if on_kaggle() and not str(p).startswith("data/"):
        return output_root() / p if p.parts[0] != "outputs" else Path("/kaggle/working") / p
    return p


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--adapter", default=None, help="LoRA / final checkpoint dir")
    p.add_argument("--prompts", default="data/exp3/prompts.jsonl")
    p.add_argument(
        "--out",
        default=None,
        help="Rollouts jsonl (default: <output_root>/exp3a_grpo/rollouts.jsonl)",
    )
    p.add_argument("--exp-name", default="exp3a_grpo")
    p.add_argument("--limit", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--gold-rm", default="Skywork/Skywork-Reward-V2-Llama-3.2-3B")
    p.add_argument("--skip-gold", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--keep-think",
        action="store_true",
        help="Do not strip <think> blocks from completions",
    )
    args = p.parse_args()

    if args.out is None:
        out_path = exp_dir(args.exp_name) / "rollouts.jsonl"
    else:
        out_path = resolve_out(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    policy_device, gold_device = pick_devices(args.device)
    print(f"policy device={policy_device}  gold device={gold_device}")
    print(f"writing rollouts to {out_path}")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.float16 if str(policy_device).startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(policy_device)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
        model.to(policy_device)
    model.eval()

    gold = None
    if not args.skip_gold:
        gold = GoldRewardModel(
            GoldRMConfig(model_id=args.gold_rm, device=gold_device)
        )

    prompts = load_prompts(Path(args.prompts), args.limit)
    records = []

    for row in tqdm(prompts, desc="generate"):
        prompt = row["prompt"] if isinstance(row["prompt"], str) else row["prompt"]
        if isinstance(prompt, list):
            prompt = prompt[0]["content"]
        text = apply_chat_prompt(tok, prompt, enable_thinking=False)
        enc = tok(text, return_tensors="pt")
        enc = {k: v.to(policy_device) for k, v in enc.items()}
        with torch.inference_mode():
            out_ids = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=0.95,
                pad_token_id=tok.pad_token_id,
            )
        gen = out_ids[0, enc["input_ids"].shape[1] :]
        completion_raw = tok.decode(gen, skip_special_tokens=True)
        completion = (
            completion_raw if args.keep_think else strip_think(completion_raw)
        )
        feats = extract_surface_features(completion)
        proxy = score_proxy_batch(
            [completion],
            ProxyRewardConfig(zscore_within_batch=False),
        )[0]
        records.append(
            {
                "prompt_id": row.get("prompt_id", ""),
                "prompt": prompt,
                "completion": completion,
                "completion_raw": completion_raw,
                "proxy": proxy,
                "features": feats.as_dict(),
            }
        )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if gold is not None:
        scores = gold.score_pairs(
            [r["prompt"] for r in records],
            [r["completion"] for r in records],
        )
        for r, s in zip(records, scores):
            r["gold"] = s
        gold.close()
    else:
        for r in records:
            r["gold"] = None

    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} rollouts to {out_path}")
    if not args.skip_gold:
        import numpy as np

        g = np.array([r["gold"] for r in records], dtype=float)
        pr = np.array([r["proxy"] for r in records], dtype=float)
        print(
            f"proxy mean={pr.mean():.3f}  gold mean={g.mean():.3f}  "
            f"corr(proxy,gold)={np.corrcoef(pr, g)[0,1]:.3f}"
        )


if __name__ == "__main__":
    main()
