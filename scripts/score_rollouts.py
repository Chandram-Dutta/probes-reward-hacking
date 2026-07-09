#!/usr/bin/env python3
"""Generate rollouts from a checkpoint and score proxy + gold offline.

Gold RM is never used in training.

Dual-GPU (Kaggle T4 x2): by default
  - policy generation on cuda:0
  - gold RM on cuda:1 (loaded in parallel, no free/reload)
Falls back to a single device when only one GPU is available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    """Return (policy_device, gold_device). gold_device may equal policy_device."""
    if force_device is not None:
        return force_device, force_device
    if not torch.cuda.is_available():
        return "cpu", "cpu"
    n = torch.cuda.device_count()
    if n >= 2:
        return "cuda:0", "cuda:1"
    return "cuda:0", "cuda:0"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--adapter", default=None, help="LoRA / final checkpoint dir")
    p.add_argument("--prompts", default="data/exp3/prompts.jsonl")
    p.add_argument("--out", default="outputs/exp3a_grpo/rollouts.jsonl")
    p.add_argument("--limit", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--gold-rm", default="Skywork/Skywork-Reward-V2-Llama-3.2-3B")
    p.add_argument("--skip-gold", action="store_true")
    p.add_argument(
        "--device",
        default=None,
        help="Force a single device for both policy and gold (disables dual-GPU split)",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    policy_device, gold_device = pick_devices(args.device)
    print(f"policy device={policy_device}  gold device={gold_device}")

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

    # Load gold RM on the second GPU while policy stays on the first (when dual).
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
        messages = [{"role": "user", "content": prompt}]
        try:
            text = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
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
        completion = tok.decode(gen, skip_special_tokens=True)
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

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
