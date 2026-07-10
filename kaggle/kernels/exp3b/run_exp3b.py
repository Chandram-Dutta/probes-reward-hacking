"""Kaggle script kernel: Exp 3b residual probes from Exp 3a checkpoint.

Inputs (Datasets):
  - probes-rh-src          → source tree
  - probes-rh-exp3a-ckpt   → exp3a_checkpoint.zip

Outputs written under /kaggle/working/ for `kaggle kernels output`.
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
SRC_DS = Path("/kaggle/input/probes-rh-src/probes-reward-hacking")
# dataset may store files at root of input
if not SRC_DS.exists():
    # try flat layout
    candidates = list(Path("/kaggle/input/probes-rh-src").rglob("pyproject.toml"))
    if candidates:
        SRC_DS = candidates[0].parent
CKPT_DS = Path("/kaggle/input/probes-rh-exp3a-ckpt")
REPO = WORK / "probes-reward-hacking"
OUT = WORK / "outputs" / "exp3a_grpo"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)


def main() -> None:
    print("cuda devices:", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
    try:
        import torch

        print("torch.cuda.device_count()", torch.cuda.device_count(), flush=True)
    except Exception as e:
        print("torch check failed", e, flush=True)

    if not SRC_DS.exists():
        raise SystemExit(f"missing source dataset at {SRC_DS}; inputs={list(Path('/kaggle/input').iterdir())}")

    if REPO.exists():
        shutil.rmtree(REPO)
    shutil.copytree(SRC_DS, REPO)
    print("repo copied from", SRC_DS, flush=True)

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
        ]
    )
    run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], cwd=REPO)

    # restore checkpoint
    OUT.mkdir(parents=True, exist_ok=True)
    zips = list(CKPT_DS.rglob("exp3a_checkpoint.zip")) if CKPT_DS.exists() else []
    if not zips:
        zips = list(Path("/kaggle/input").rglob("exp3a_checkpoint.zip"))
    if not zips:
        raise SystemExit("exp3a_checkpoint.zip not found in inputs")
    ckpt_zip = zips[0]
    print("using checkpoint", ckpt_zip, flush=True)
    with zipfile.ZipFile(ckpt_zip, "r") as zf:
        zf.extractall(WORK / "outputs")
    adapter = OUT / "final"
    if not (adapter / "adapter_model.safetensors").exists():
        # sometimes nested
        found = list((WORK / "outputs").rglob("adapter_model.safetensors"))
        if not found:
            raise SystemExit("adapter missing after unzip")
        adapter = found[0].parent
    print("adapter:", adapter, flush=True)

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
    )

    rollouts = OUT / "rollouts_3b.jsonl"
    run(
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
        cwd=REPO,
    )

    probe_out = OUT / "probe_results_residual.json"
    run(
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
            "residual",
            "--out",
            str(probe_out),
        ],
        cwd=REPO,
    )

    # also quantile for comparison (no second activation pass would be nicer; re-run is ok)
    probe_q = OUT / "probe_results_quantile.json"
    run(
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
            "quantile",
            "--out",
            str(probe_q),
        ],
        cwd=REPO,
    )

    # summary for logs
    for p in (probe_out, probe_q):
        if p.exists():
            data = json.loads(p.read_text())
            print("\n====", p.name, "====", flush=True)
            print(
                "n=",
                data.get("n"),
                "hack_rate=",
                data.get("hack_rate"),
                "mode=",
                data.get("label_mode"),
                flush=True,
            )
            rows = sorted(
                data["results"],
                key=lambda x: -(x["auc"] if x["auc"] == x["auc"] else -1),
            )
            for r in rows[:12]:
                print(f"  {r['name']:40s} {r['auc']:.3f}", flush=True)

    # light copy to working root for easy download
    for name in (
        "rollouts_3b.jsonl",
        "probe_results_residual.json",
        "probe_results_quantile.json",
    ):
        src = OUT / name
        if src.exists():
            shutil.copy2(src, WORK / name)
            print("published", WORK / name, flush=True)

    print("DONE exp3b", flush=True)


if __name__ == "__main__":
    main()
