#!/usr/bin/env python3
"""Mid-training curve: score + residual-probe each GRPO checkpoint.

Pipeline:
  1. (optional) train GRPO with intermediate saves (multi-GPU via train_grpo)
  2. for base + each checkpoint: dual-GPU score_rollouts (policy cuda:0, gold cuda:1)
  3. residual probes on each rollout set
  4. write mid_training_curve.json summarizing proxy/gold and AUCs over steps

Designed for Kaggle dual T4. Does not load models itself — shells out so GPU
memory is fully released between stages.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import ensure_import_path

ensure_import_path()

import numpy as np

from probes_rh.paths import on_kaggle, output_root


CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


def run(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None, env=env)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def has_adapter(path: Path) -> bool:
    if not path.is_dir():
        return False
    names = {
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin",
        "pytorch_model.bin",
        "model.safetensors",
    }
    if any((path / n).exists() for n in names):
        return True
    # nested peft layout
    return any(path.rglob("adapter_config.json"))


def discover_eval_points(train_dir: Path, max_steps: int = 100) -> list[dict]:
    """Return ordered eval points: base (step 0) + checkpoints; final if needed."""
    by_step: dict[int, dict] = {
        0: {"step": 0, "name": "base", "adapter": None},
    }

    for p in sorted(train_dir.glob("checkpoint-*")):
        m = CHECKPOINT_RE.match(p.name)
        if not m or not has_adapter(p):
            continue
        step = int(m.group(1))
        by_step[step] = {"step": step, "name": p.name, "adapter": str(p)}

    final = train_dir / "final"
    if has_adapter(final):
        trained_steps = [s for s in by_step if s > 0]
        # Use final when no matching end checkpoint, or as max_steps terminal point
        if max_steps not in by_step:
            by_step[max_steps] = {
                "step": max_steps,
                "name": "final",
                "adapter": str(final),
            }
        elif not trained_steps:
            by_step[max_steps] = {
                "step": max_steps,
                "name": "final",
                "adapter": str(final),
            }

    return [by_step[s] for s in sorted(by_step)]


def load_rollout_summary(path: Path) -> dict:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    if not rows:
        return {}
    proxy = np.array([r["proxy"] for r in rows], dtype=float)
    gold = np.array([r["gold"] for r in rows if r.get("gold") is not None], dtype=float)
    chars = np.array(
        [r.get("features", {}).get("n_chars", len(r.get("completion", ""))) for r in rows],
        dtype=float,
    )
    fluff = np.array(
        [r.get("features", {}).get("fluff_total", 0.0) for r in rows],
        dtype=float,
    )
    out = {
        "n": len(rows),
        "proxy_mean": float(proxy.mean()),
        "proxy_std": float(proxy.std()),
        "mean_chars": float(chars.mean()),
        "mean_fluff": float(fluff.mean()),
    }
    if len(gold) == len(rows) and len(gold) > 1:
        out["gold_mean"] = float(gold.mean())
        out["gold_std"] = float(gold.std())
        out["proxy_gold_corr"] = float(np.corrcoef(proxy, gold)[0, 1])
    return out


def pick_auc(results: list[dict], name: str) -> float | None:
    for r in results:
        if r["name"] == name:
            a = r.get("auc")
            if a is None or a != a:
                return None
            return float(a)
    return None


def best_probe(results: list[dict], prefix: str) -> tuple[str | None, float | None]:
    best_name, best_auc = None, None
    for r in results:
        if not r["name"].startswith(prefix):
            continue
        a = r.get("auc")
        if a is None or a != a:
            continue
        if best_auc is None or a > best_auc:
            best_auc = float(a)
            best_name = r["name"]
    return best_name, best_auc


def summarize_probe(probe_path: Path) -> dict:
    data = json.loads(probe_path.read_text(encoding="utf-8"))
    results = data.get("results", [])
    mean_name, mean_auc = best_probe(results, "mean_completion_")
    last_name, last_auc = best_probe(results, "last_prompt_")
    return {
        "hack_rate": data.get("hack_rate"),
        "n_pos": data.get("n_pos"),
        "proxy_gold_corr": data.get("proxy_gold_corr"),
        "label_mode": data.get("label_mode"),
        "best_mean_probe": mean_name,
        "best_mean_probe_auc": mean_auc,
        "best_last_prompt_probe": last_name,
        "best_last_prompt_probe_auc": last_auc,
        "baseline_proxy_auc": pick_auc(results, "baseline_proxy"),
        "baseline_n_chars_auc": pick_auc(results, "baseline_n_chars"),
        "baseline_all_surface_auc": pick_auc(results, "baseline_all_surface"),
        "shuffled_auc": pick_auc(results, "shuffled_labels"),
    }


def eval_point(
    point: dict,
    *,
    train_dir: Path,
    model: str,
    prompts: str,
    limit: int,
    label_mode: str,
    residual_q: float,
    seed: int,
    force: bool,
    python: str,
    cwd: Path,
    env: dict,
) -> dict:
    step = point["step"]
    name = point["name"]
    adapter = point["adapter"]
    tag = f"step_{step:04d}_{name}"
    point_dir = train_dir / "curve" / tag
    point_dir.mkdir(parents=True, exist_ok=True)

    rollouts = point_dir / "rollouts.jsonl"
    probe_out = point_dir / f"probe_results_{label_mode}.json"
    meta_path = point_dir / "point_summary.json"

    if meta_path.exists() and not force:
        print(f"[skip] {tag} already has summary", flush=True)
        return json.loads(meta_path.read_text(encoding="utf-8"))

    if force or not rollouts.exists():
        cmd = [
            python,
            "scripts/score_rollouts.py",
            "--model",
            model,
            "--prompts",
            prompts,
            "--out",
            str(rollouts),
            "--limit",
            str(limit),
            "--seed",
            str(seed),
        ]
        if adapter:
            cmd.extend(["--adapter", adapter])
        run(cmd, cwd=cwd, env=env)
    else:
        print(f"[skip] rollouts exist for {tag}", flush=True)

    if force or not probe_out.exists():
        cmd = [
            python,
            "scripts/run_probes.py",
            "--rollouts",
            str(rollouts),
            "--model",
            model,
            "--label-mode",
            label_mode,
            "--residual-q",
            str(residual_q),
            "--out",
            str(probe_out),
            "--seed",
            str(seed),
        ]
        if adapter:
            cmd.extend(["--adapter", adapter])
        run(cmd, cwd=cwd, env=env)
    else:
        print(f"[skip] probe results exist for {tag}", flush=True)

    summary = {
        "step": step,
        "name": name,
        "adapter": adapter,
        "tag": tag,
        "rollouts": str(rollouts),
        "probe_results": str(probe_out),
        **load_rollout_summary(rollouts),
        **summarize_probe(probe_out),
    }
    meta_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[{tag}] proxy={summary.get('proxy_mean'):.3f} "
        f"gold={summary.get('gold_mean', float('nan')):.3f} "
        f"best_probe={summary.get('best_mean_probe_auc')} "
        f"surface={summary.get('baseline_all_surface_auc')}",
        flush=True,
    )
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Mid-training residual probe curve")
    p.add_argument(
        "--train-dir",
        default=None,
        help="GRPO output dir (default: outputs/exp3c_midcurve or Kaggle working)",
    )
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--data", default="data/exp3/prompts_trl.jsonl")
    p.add_argument("--prompts", default="data/exp3/prompts.jsonl")
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--save-steps", type=int, default=25)
    p.add_argument("--limit", type=int, default=256, help="Rollouts per checkpoint")
    p.add_argument("--label-mode", default="residual", choices=["residual", "quantile", "residual_high_proxy"])
    p.add_argument("--residual-q", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--force", action="store_true", help="Re-score/probe even if outputs exist")
    p.add_argument("--single-gpu-train", action="store_true")
    args = p.parse_args()

    root = repo_root()
    if args.train_dir:
        train_dir = Path(args.train_dir)
    else:
        train_dir = output_root() / "exp3c_midcurve"
    train_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    src = root / "src"
    if src.is_dir():
        env["PYTHONPATH"] = str(src) + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )

    python = sys.executable

    if not args.skip_train:
        train_cmd = [
            python,
            "scripts/train_grpo.py",
            "--model",
            args.model,
            "--data",
            args.data,
            "--output-dir",
            str(train_dir),
            "--max-steps",
            str(args.max_steps),
            "--save-steps",
            str(args.save_steps),
            "--save-total-limit",
            str(max(10, args.max_steps // max(args.save_steps, 1) + 2)),
            "--seed",
            str(args.seed),
        ]
        if args.single_gpu_train:
            train_cmd.append("--single-gpu")
        run(train_cmd, cwd=root, env=env)
    else:
        print(f"skipping train; using {train_dir}", flush=True)

    points = discover_eval_points(train_dir, max_steps=args.max_steps)
    print("eval points:", json.dumps(points, indent=2), flush=True)
    if len(points) < 2:
        print(
            "WARNING: fewer than 2 eval points — check that save_steps produced "
            "checkpoint-* dirs with adapter weights",
            flush=True,
        )

    curve_points = []
    for pt in points:
        summary = eval_point(
            pt,
            train_dir=train_dir,
            model=args.model,
            prompts=args.prompts,
            limit=args.limit,
            label_mode=args.label_mode,
            residual_q=args.residual_q,
            seed=args.seed,
            force=args.force,
            python=python,
            cwd=root,
            env=env,
        )
        curve_points.append(summary)

    curve = {
        "experiment": "exp3c_mid_training_curve",
        "model": args.model,
        "train_dir": str(train_dir),
        "max_steps": args.max_steps,
        "save_steps": args.save_steps,
        "limit": args.limit,
        "label_mode": args.label_mode,
        "residual_q": args.residual_q,
        "seed": args.seed,
        "on_kaggle": on_kaggle(),
        "points": curve_points,
    }
    out_path = train_dir / "mid_training_curve.json"
    out_path.write_text(json.dumps(curve, indent=2), encoding="utf-8")
    print(f"\n=== mid-training curve ({args.label_mode}) ===", flush=True)
    print(
        f"{'step':>6}  {'name':12}  {'proxy':>7}  {'gold':>7}  "
        f"{'probe':>6}  {'surf':>6}  {'proxyB':>6}  {'hack%':>6}",
        flush=True,
    )
    for s in curve_points:
        print(
            f"{s.get('step', -1):6d}  {str(s.get('name')):12}  "
            f"{s.get('proxy_mean', float('nan')):7.3f}  "
            f"{s.get('gold_mean', float('nan')):7.3f}  "
            f"{(s.get('best_mean_probe_auc') or float('nan')):6.3f}  "
            f"{(s.get('baseline_all_surface_auc') or float('nan')):6.3f}  "
            f"{(s.get('baseline_proxy_auc') or float('nan')):6.3f}  "
            f"{100 * (s.get('hack_rate') or 0):5.1f}%",
            flush=True,
        )
    print(f"wrote {out_path}", flush=True)

    if on_kaggle():
        work = Path("/kaggle/working")
        work_out = work / "mid_training_curve.json"
        work_out.write_text(json.dumps(curve, indent=2), encoding="utf-8")
        print(f"also wrote {work_out}", flush=True)


if __name__ == "__main__":
    main()
