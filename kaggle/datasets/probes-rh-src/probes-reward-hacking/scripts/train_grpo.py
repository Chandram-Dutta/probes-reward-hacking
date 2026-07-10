#!/usr/bin/env python3
"""Train policy with GRPO on the misspecified proxy (gold is offline only).

Multi-GPU (Kaggle dual T4):
  By default this script re-launches itself with `accelerate launch` when more
  than one CUDA device is visible and we are not already inside a distributed
  worker. Force single-GPU with --single-gpu.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

# path bootstrap lives next to this script (works even without pip install -e .)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bootstrap import ensure_import_path  # noqa: E402

ensure_import_path()


def _already_distributed() -> bool:
    # set by accelerate / torchrun / torch.distributed
    return any(
        os.environ.get(k) is not None
        for k in ("RANK", "LOCAL_RANK", "ACCELERATE_PROCESS_ID", "WORLD_SIZE")
    )


def _cuda_count() -> int:
    try:
        import torch

        return int(torch.cuda.device_count())
    except Exception:
        return 0


def _maybe_relaunch_multi_gpu(argv: list[str]) -> None:
    """Re-exec under accelerate when multiple GPUs are free."""
    if "--single-gpu" in argv:
        return
    if _already_distributed():
        return
    n = _cuda_count()
    if n < 2:
        return

    script = os.path.abspath(__file__)
    # strip our launcher-only flag if present; keep user args
    user_args = [a for a in argv if a != "--single-gpu"]
    cmd = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        f"--num_processes={n}",
        "--num_machines=1",
        "--mixed_precision=fp16",
        "--multi_gpu",
        script,
        *user_args,
        # prevent infinite relaunch inside workers via env already set by accelerate
    ]
    print(f"Launching multi-GPU GRPO on {n} devices:\n  {' '.join(cmd)}", flush=True)
    raise SystemExit(subprocess.call(cmd))


def main() -> None:
    _maybe_relaunch_multi_gpu(sys.argv[1:])

    from probes_rh.train.grpo_proxy import Exp3TrainConfig, default_train_output_dir, train

    p = argparse.ArgumentParser(description="Exp 3a GRPO vs proxy reward")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--data", default="data/exp3/prompts_trl.jsonl")
    p.add_argument(
        "--output-dir",
        default=None,
        help="Default: /kaggle/working/outputs/exp3a_grpo on Kaggle, else outputs/exp3a_grpo",
    )
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--num-generations", type=int, default=4)
    p.add_argument("--max-completion-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--save-steps", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-lora", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--enable-thinking", action="store_true", help="default off for Qwen3")
    p.add_argument("--w-length", type=float, default=1.0)
    p.add_argument("--w-agreement", type=float, default=1.5)
    p.add_argument("--w-confidence", type=float, default=1.0)
    p.add_argument("--w-politeness", type=float, default=0.75)
    p.add_argument(
        "--single-gpu",
        action="store_true",
        help="Do not auto-launch accelerate multi-GPU",
    )
    args = p.parse_args()

    cfg = Exp3TrainConfig(
        model_id=args.model,
        data_path=args.data,
        output_dir=args.output_dir or default_train_output_dir(),
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        save_steps=args.save_steps,
        seed=args.seed,
        use_lora=not args.no_lora,
        bf16=args.bf16,
        fp16=args.fp16 and not args.bf16,
        enable_thinking=args.enable_thinking,
        w_length=args.w_length,
        w_agreement=args.w_agreement,
        w_confidence=args.w_confidence,
        w_politeness=args.w_politeness,
    )
    metrics = train(cfg)
    print("done", metrics)


if __name__ == "__main__":
    main()
