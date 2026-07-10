#!/usr/bin/env python3
"""One-shot Kaggle environment fix before training.

Kaggle images often ship an old torchao that breaks recent peft LoRA.
Run once per session:

  !python scripts/kaggle_setup.py
  !python scripts/train_grpo.py ...
"""

from __future__ import annotations

import subprocess
import sys


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def main() -> None:
    py = sys.executable
    # Drop broken torchao; peft only needs it absent or >=0.16
    run([py, "-m", "pip", "uninstall", "-y", "torchao"])
    # Keep peft/trl recent enough for Qwen3 + GRPO
    run(
        [
            py,
            "-m",
            "pip",
            "install",
            "-q",
            "-U",
            "peft>=0.14.0",
            "trl>=0.18.0",
            "transformers>=4.51.0",
            "accelerate>=1.0.0",
        ]
    )
    # install this repo
    run([py, "-m", "pip", "install", "-q", "-e", "."])
    print("kaggle_setup done")


if __name__ == "__main__":
    main()
