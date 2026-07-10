"""Kaggle script kernel: Exp 3d residual probe transfer across checkpoints.

Datasets:
  - chandramdutta/probes-rh-src
  - chandramdutta/probes-rh-exp3c-artifacts  (checkpoints + curve rollouts)

No GRPO retrain. Collects activations per checkpoint and builds transfer matrix.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

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

print("exp3d kernel start", flush=True)


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
        if (root / "scripts" / "run_probe_transfer.py").exists():
            return root
    raise SystemExit(f"source repo not found under {INPUT}")


def find_artifacts_root() -> Path:
    """Locate Exp 3c midcurve dir (has curve/ + checkpoint-*)."""
    candidates = [
        INPUT / "probes-rh-exp3c-artifacts" / "exp3c_midcurve",
        INPUT / "probes-rh-exp3c-artifacts",
        INPUT / "datasets" / "chandramdutta" / "probes-rh-exp3c-artifacts" / "exp3c_midcurve",
        INPUT / "datasets" / "chandramdutta" / "probes-rh-exp3c-artifacts",
    ]
    for c in candidates:
        if (c / "curve").is_dir() or any(c.glob("checkpoint-*")):
            return c
        if (c / "outputs" / "exp3c_midcurve" / "curve").is_dir():
            return c / "outputs" / "exp3c_midcurve"
    for p in INPUT.rglob("mid_training_curve.json"):
        return p.parent
    for p in INPUT.rglob("rollouts.jsonl"):
        # .../curve/step_xxx/rollouts.jsonl
        if p.parent.parent.name == "curve":
            return p.parent.parent.parent
    raise SystemExit(
        f"exp3c artifacts not found; sample={list(INPUT.rglob('*'))[:40]}"
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
    src = find_repo_root()
    arts = find_artifacts_root()
    print("source:", src, flush=True)
    print("artifacts:", arts, flush=True)

    if REPO.exists():
        shutil.rmtree(REPO)
    shutil.copytree(src, REPO)

    # Copy artifacts into writable train_dir (npz-style path used by scripts)
    if TRAIN_DIR.exists():
        shutil.rmtree(TRAIN_DIR)
    TRAIN_DIR.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(arts, TRAIN_DIR)
    print("train_dir ready:", TRAIN_DIR, flush=True)
    print("curve points:", sorted((TRAIN_DIR / "curve").glob("step_*")) if (TRAIN_DIR / "curve").exists() else "MISSING", flush=True)

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
            "transformers>=4.51.0",
            "accelerate>=1.0.0",
            "scikit-learn>=1.5.0",
            "safetensors",
            "tqdm",
            "numpy",
        ]
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    try:
        run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], cwd=REPO, env=env)
    except subprocess.CalledProcessError as e:
        print("editable install failed, continuing with PYTHONPATH:", e, flush=True)

    # Prefer cuda:0; second GPU unused for activation-only transfer
    device = "cuda:0" if n_gpu >= 1 else "cpu"
    out_path = WORK / "transfer_results.json"
    run(
        [
            sys.executable,
            "scripts/run_probe_transfer.py",
            "--train-dir",
            str(TRAIN_DIR),
            "--model",
            "Qwen/Qwen3-0.6B",
            "--residual-q",
            "0.25",
            "--layers",
            "mid_late",
            "--seed",
            "0",
            "--device",
            device,
            "--out",
            str(out_path),
        ],
        cwd=REPO,
        env=env,
    )

    if not out_path.exists():
        # script may have written only under train_dir
        alt = TRAIN_DIR / "transfer_results.json"
        if alt.exists():
            shutil.copy2(alt, out_path)
        else:
            raise SystemExit("transfer_results.json missing")

    data = json.loads(out_path.read_text(encoding="utf-8"))
    print("\n==== exp3d transfer summary ====", flush=True)
    steps = data.get("steps", [])
    mat = data.get("probe_auc_matrix", {})
    print("probe AUC matrix:", flush=True)
    print("src\\tgt", *[f"{t:>8}" for t in steps], flush=True)
    for s in steps:
        vals = [f"{mat[str(s)][str(t)]:8.3f}" for t in steps]
        print(f"{s:>7}", *vals, flush=True)
    print("base trajectory:", flush=True)
    for r in data.get("base_probe_trajectory", []):
        print(
            f"  step={r['step']} auc={r['probe_auc']:.3f} "
            f"mean_p={r['mean_proba']:.3f} gold={r['gold_mean']:.3f}",
            flush=True,
        )
    # also copy matrix-friendly json
    shutil.copy2(out_path, TRAIN_DIR / "transfer_results.json")
    print("DONE exp3d", flush=True)
    print(f"n_gpu_at_start={n_gpu}", flush=True)


if __name__ == "__main__":
    main()
