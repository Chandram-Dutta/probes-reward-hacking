"""Kaggle script kernel: Exp 3b residual probes from Exp 3a checkpoint.

Datasets:
  - chandramdutta/probes-rh-src
  - chandramdutta/probes-rh-exp3a-ckpt

Use dual T4 when available (score_rollouts puts policy on cuda:0, gold on cuda:1).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
OUT = WORK / "outputs" / "exp3a_grpo"
REPO = WORK / "probes-reward-hacking"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)


def find_repo_root() -> Path:
    """Locate uploaded source (layout varies: nested folder vs flat dataset)."""
    # preferred layouts
    candidates = [
        INPUT / "probes-rh-src" / "probes-reward-hacking",
        INPUT / "probes-rh-src",
        INPUT / "datasets" / "chandramdutta" / "probes-rh-src" / "probes-reward-hacking",
        INPUT / "datasets" / "chandramdutta" / "probes-rh-src",
    ]
    for c in candidates:
        if (c / "pyproject.toml").exists() and (c / "scripts").is_dir():
            return c
    # search
    for p in INPUT.rglob("pyproject.toml"):
        root = p.parent
        if (root / "scripts" / "score_rollouts.py").exists():
            return root
    raise SystemExit(
        f"could not find source repo under {INPUT}; tree={list(INPUT.rglob('*'))[:50]}"
    )


def find_adapter() -> Path:
    """Locate LoRA adapter dir (from zip extract or already-unzipped dataset)."""
    # dataset may already contain exp3a_grpo/final/*
    for p in INPUT.rglob("adapter_model.safetensors"):
        return p.parent
    # or a zip to extract
    for z in INPUT.rglob("exp3a_checkpoint.zip"):
        dest = WORK / "outputs"
        dest.mkdir(parents=True, exist_ok=True)
        print("extracting", z, "->", dest, flush=True)
        with zipfile.ZipFile(z, "r") as zf:
            zf.extractall(dest)
        for p in dest.rglob("adapter_model.safetensors"):
            return p.parent
    raise SystemExit("adapter_model.safetensors / exp3a_checkpoint.zip not found in inputs")


def main() -> None:
    print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
    try:
        import torch

        n = torch.cuda.device_count()
        print("torch.cuda.device_count()=", n, flush=True)
        for i in range(n):
            print(f"  gpu{i}:", torch.cuda.get_device_name(i), flush=True)
    except Exception as e:
        print("torch check failed:", e, flush=True)

    print("input tree (top):", flush=True)
    if INPUT.exists():
        for p in sorted(INPUT.rglob("*"))[:80]:
            print(" ", p, flush=True)

    src = find_repo_root()
    print("source repo:", src, flush=True)
    if REPO.exists():
        shutil.rmtree(REPO)
    shutil.copytree(src, REPO)

    # deps
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
    # Prefer PYTHONPATH=src over editable install (avoids incomplete wheel layouts).
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], cwd=REPO)

    # sanity: package data must import
    subprocess.check_call(
        [sys.executable, "-c", "import probes_rh.data.prep_prompts as p; print('import ok', p.__file__)"],
        cwd=str(REPO),
        env=env,
    )

    adapter = find_adapter()
    print("adapter:", adapter, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)

    def run_repo(args: list[str]) -> None:
        print("+", " ".join(args), flush=True)
        subprocess.check_call(args, cwd=str(REPO), env=env)

    run_repo(
        [
            sys.executable,
            "scripts/prepare_data.py",
            "--max-prompts",
            "2000",
            "--out-dir",
            "data/exp3",
        ],
    )

    rollouts = OUT / "rollouts_3b.jsonl"
    run_repo(
        [
            sys.executable,
            "scripts/score_rollouts.py",
            "--model",
            "Qwen/Qwen3-0.6B",
            "--adapter",
            str(adapter),
            "--prompts",
            "data/exp3/prompts.jsonl",
            "--out",
            str(rollouts),
            "--limit",
            "1024",
        ],
    )

    for mode, name in (
        ("residual", "probe_results_residual.json"),
        ("quantile", "probe_results_quantile.json"),
    ):
        outp = OUT / name
        run_repo(
            [
                sys.executable,
                "scripts/run_probes.py",
                "--rollouts",
                str(rollouts),
                "--model",
                "Qwen/Qwen3-0.6B",
                "--adapter",
                str(adapter),
                "--label-mode",
                mode,
                "--out",
                str(outp),
            ],
        )
        data = json.loads(outp.read_text())
        print(f"\n==== {name} mode={mode} n={data.get('n')} hack_rate={data.get('hack_rate')} ====", flush=True)
        rows = sorted(
            data["results"],
            key=lambda x: -(x["auc"] if x["auc"] == x["auc"] else -1),
        )
        for r in rows[:15]:
            print(f"  {r['name']:40s} {r['auc']:.3f}", flush=True)
        shutil.copy2(outp, WORK / name)

    shutil.copy2(rollouts, WORK / "rollouts_3b.jsonl")
    print("DONE exp3b", flush=True)


if __name__ == "__main__":
    main()
