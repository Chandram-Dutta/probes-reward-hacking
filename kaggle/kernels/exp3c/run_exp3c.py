"""Kaggle script kernel: Exp 3c mid-training residual probe curve.

Dataset:
  - chandramdutta/probes-rh-src

Dual T4:
  - train_grpo auto-launches accelerate multi-GPU (both T4s for GRPO)
  - score_rollouts: policy cuda:0, gold RM cuda:1
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Force line-buffered logs on Kaggle
os.environ.setdefault("PYTHONUNBUFFERED", "1")
try:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
except Exception:
    pass

WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
REPO = WORK / "probes-reward-hacking"
TRAIN_DIR = WORK / "outputs" / "exp3c_midcurve"

print("exp3c kernel start", flush=True)


def run(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None, env=env)


def find_repo_root() -> Path:
    candidates = [
        INPUT / "probes-rh-src" / "probes-reward-hacking",
        INPUT / "probes-rh-src",
        INPUT / "datasets" / "chandramdutta" / "probes-rh-src" / "probes-reward-hacking",
        INPUT / "datasets" / "chandramdutta" / "probes-rh-src",
    ]
    for c in candidates:
        if (c / "pyproject.toml").exists() and (c / "scripts").is_dir():
            return c
    for p in INPUT.rglob("pyproject.toml"):
        root = p.parent
        if (root / "scripts" / "run_mid_training_curve.py").exists():
            return root
    raise SystemExit(
        f"could not find source repo under {INPUT}; sample={list(INPUT.rglob('*'))[:40]}"
    )


def log_gpus() -> int:
    print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
    try:
        import torch

        n = torch.cuda.device_count()
        print("torch.cuda.device_count()=", n, flush=True)
        for i in range(n):
            print(f"  gpu{i}:", torch.cuda.get_device_name(i), flush=True)
        return n
    except Exception as e:
        print("torch check failed:", e, flush=True)
        return 0


def main() -> None:
    n_gpu = log_gpus()
    if n_gpu < 2:
        print(
            f"WARNING: expected dual T4 (2 GPUs), got {n_gpu}. "
            "Train will not multi-GPU; scoring may colocate policy+gold.",
            flush=True,
        )

    print("input tree (top):", flush=True)
    if INPUT.exists():
        for p in sorted(INPUT.rglob("*"))[:60]:
            print(" ", p, flush=True)

    src = find_repo_root()
    print("source repo:", src, flush=True)
    if REPO.exists():
        shutil.rmtree(REPO)
    shutil.copytree(src, REPO)

    run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"])
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "-U",
            "peft>=0.14.0",
            "trl>=0.18.0",
            "transformers>=4.51.0",
            "accelerate>=1.0.0",
            "datasets>=3.0.0",
            "scikit-learn>=1.5.0",
            "safetensors",
            "tqdm",
        ]
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    # Prefer src on PYTHONPATH; editable install is best-effort (hatch can miss subpkgs).
    try:
        run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], cwd=REPO, env=env)
    except subprocess.CalledProcessError as e:
        print("editable install failed (continuing with PYTHONPATH):", e, flush=True)

    data_pkg = REPO / "src" / "probes_rh" / "data"
    print("package data dir exists:", data_pkg.is_dir(), data_pkg, flush=True)
    if data_pkg.is_dir():
        print("  contents:", list(data_pkg.iterdir()), flush=True)
    subprocess.check_call(
        [
            sys.executable,
            "-c",
            "import probes_rh, probes_rh.data, probes_rh.data.prep_prompts as p; "
            "print('import ok', probes_rh.__file__, p.__file__)",
        ],
        cwd=str(REPO),
        env=env,
    )

    # data
    run(
        [
            sys.executable,
            "scripts/prepare_data.py",
            "--max-prompts",
            "2000",
            "--out-dir",
            "data/exp3",
        ],
        cwd=REPO,
        env=env,
    )

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    # Full curve: multi-GPU train + dual-device score + residual probes
    # 256 rollouts/ckpt keeps wall time ~2–3h on dual T4.
    run(
        [
            sys.executable,
            "scripts/run_mid_training_curve.py",
            "--train-dir",
            str(TRAIN_DIR),
            "--model",
            "Qwen/Qwen3-0.6B",
            "--max-steps",
            "100",
            "--save-steps",
            "25",
            "--limit",
            "256",
            "--label-mode",
            "residual",
            "--residual-q",
            "0.25",
            "--seed",
            "0",
        ],
        cwd=REPO,
        env=env,
    )

    curve_path = TRAIN_DIR / "mid_training_curve.json"
    if not curve_path.exists():
        raise SystemExit(f"missing curve output: {curve_path}")

    data = json.loads(curve_path.read_text(encoding="utf-8"))
    print("\n==== exp3c mid-training curve ====", flush=True)
    for pt in data.get("points", []):
        print(
            f"  step={pt.get('step')} name={pt.get('name')} "
            f"proxy={pt.get('proxy_mean')} gold={pt.get('gold_mean')} "
            f"probe={pt.get('best_mean_probe_auc')} "
            f"surface={pt.get('baseline_all_surface_auc')} "
            f"hack_rate={pt.get('hack_rate')}",
            flush=True,
        )

    # Promote key artifacts to /kaggle/working root for easy download
    shutil.copy2(curve_path, WORK / "mid_training_curve.json")
    for pt in data.get("points", []):
        tag = pt.get("tag")
        if not tag:
            continue
        src_probe = TRAIN_DIR / "curve" / tag / "probe_results_residual.json"
        if src_probe.exists():
            shutil.copy2(src_probe, WORK / f"probe_{tag}.json")
        src_roll = TRAIN_DIR / "curve" / tag / "rollouts.jsonl"
        if src_roll.exists():
            # keep rollouts only for final / base to save output quota noise
            if pt.get("step") in (0, data["points"][-1].get("step")):
                shutil.copy2(src_roll, WORK / f"rollouts_{tag}.jsonl")

    # list checkpoints kept
    print("train_dir contents:", flush=True)
    for p in sorted(TRAIN_DIR.rglob("*")):
        if p.is_file() and p.suffix in {".json", ".jsonl", ".safetensors"}:
            print(" ", p.relative_to(TRAIN_DIR), flush=True)

    print("DONE exp3c mid-training curve", flush=True)
    print(f"n_gpu_at_start={n_gpu}", flush=True)


if __name__ == "__main__":
    main()
