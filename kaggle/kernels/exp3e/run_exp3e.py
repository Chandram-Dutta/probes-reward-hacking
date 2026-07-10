"""Kaggle script kernel: Exp 3e held-out-prompt residual probe transfer.

Datasets:
  - chandramdutta/probes-rh-src
  - chandramdutta/probes-rh-exp3c-artifacts

Fit on prompt group A at source; eval on disjoint B at each target.
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

print("exp3e kernel start", flush=True)


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
        if (root / "scripts" / "run_probe_transfer_heldout.py").exists():
            return root
    raise SystemExit(f"source repo not found under {INPUT}")


def find_artifacts_root() -> Path:
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
        if p.parent.parent.name == "curve":
            return p.parent.parent.parent
    raise SystemExit(f"exp3c artifacts not found; sample={list(INPUT.rglob('*'))[:40]}")


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

    if TRAIN_DIR.exists():
        shutil.rmtree(TRAIN_DIR)
    TRAIN_DIR.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(arts, TRAIN_DIR)
    print("train_dir:", TRAIN_DIR, flush=True)
    print(
        "curve:",
        sorted((TRAIN_DIR / "curve").glob("step_*")) if (TRAIN_DIR / "curve").exists() else "MISSING",
        flush=True,
    )

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
        print("editable install failed, continuing:", e, flush=True)

    device = "cuda:0" if n_gpu >= 1 else "cpu"
    out_path = WORK / "heldout_transfer_results.json"
    run(
        [
            sys.executable,
            "scripts/run_probe_transfer_heldout.py",
            "--train-dir",
            str(TRAIN_DIR),
            "--model",
            "Qwen/Qwen3-0.6B",
            "--residual-q",
            "0.25",
            "--fit-frac",
            "0.5",
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
        alt = TRAIN_DIR / "heldout_transfer_results.json"
        if alt.exists():
            shutil.copy2(alt, out_path)
        else:
            raise SystemExit("heldout_transfer_results.json missing")

    data = json.loads(out_path.read_text(encoding="utf-8"))
    steps = data.get("steps", [])
    print("\n==== exp3e held-out transfer ====", flush=True)
    print(
        f"n_fit_prompts={data.get('n_fit_prompts')} n_eval_prompts={data.get('n_eval_prompts')}",
        flush=True,
    )
    mat = data.get("probe_auc_matrix", {})
    smat = data.get("surface_auc_matrix", {})
    cmat = data.get("carry_residual_auc_matrix", {})
    print("probe AUC:", flush=True)
    print("src\\tgt", *[f"{t:>8}" for t in steps], flush=True)
    for s in steps:
        print(f"{s:>7}", *[f"{mat[str(s)][str(t)]:8.3f}" for t in steps], flush=True)
    print("surface AUC:", flush=True)
    for s in steps:
        print(f"{s:>7}", *[f"{smat[str(s)][str(t)]:8.3f}" for t in steps], flush=True)
    print("carry residual AUC:", flush=True)
    for s in steps:
        print(f"{s:>7}", *[f"{cmat[str(s)][str(t)]:8.3f}" for t in steps], flush=True)

    # Compact JSON without giant per_row for secondary artifact
    compact = {k: v for k, v in data.items() if k != "per_row_scores"}
    compact["n_per_row_scores"] = len(data.get("per_row_scores", []))
    (WORK / "heldout_transfer_summary.json").write_text(
        json.dumps(compact, indent=2), encoding="utf-8"
    )
    shutil.copy2(out_path, TRAIN_DIR / "heldout_transfer_results.json")
    print("DONE exp3e", flush=True)
    print(f"n_gpu_at_start={n_gpu}", flush=True)


if __name__ == "__main__":
    main()
