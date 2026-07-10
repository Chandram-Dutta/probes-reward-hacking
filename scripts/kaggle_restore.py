#!/usr/bin/env python3
"""Restore experiment outputs after a Kaggle restart.

Typical flow:
  1. Download exp3a_checkpoint.zip / exp3a_results.zip while the session is alive
  2. Re-upload as a Kaggle Dataset or notebook input
  3. Run this script to unpack under /kaggle/working/outputs/

Examples:
  python scripts/kaggle_restore.py --zip /kaggle/input/my-dataset/exp3a_checkpoint.zip
  python scripts/kaggle_restore.py --zip /kaggle/working/exp3a_checkpoint.zip
  python scripts/kaggle_restore.py --from-dir /kaggle/input/my-dataset/exp3a_grpo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import ensure_import_path

ensure_import_path()

from probes_rh.paths import copy_tree, exp_dir, output_root, restore_zip


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--zip", default=None, help="Path to checkpoint or results zip")
    p.add_argument(
        "--from-dir",
        default=None,
        help="Path to an already-extracted exp folder (e.g. .../exp3a_grpo)",
    )
    p.add_argument(
        "--name",
        default="exp3a_grpo",
        help="Experiment folder name under outputs/",
    )
    p.add_argument(
        "--dest",
        default=None,
        help="Override output root (default: /kaggle/working/outputs or ./outputs)",
    )
    args = p.parse_args()

    root = Path(args.dest) if args.dest else output_root()
    root.mkdir(parents=True, exist_ok=True)
    print(f"output root: {root}")

    if args.zip:
        z = Path(args.zip)
        if not z.exists():
            raise SystemExit(f"zip not found: {z}")
        restore_zip(z, root)
        print(f"extracted {z} -> {root}")
    elif args.from_dir:
        src = Path(args.from_dir)
        if not src.exists():
            raise SystemExit(f"dir not found: {src}")
        dest = root / args.name
        copy_tree(src, dest)
        print(f"copied {src} -> {dest}")
    else:
        raise SystemExit("pass --zip or --from-dir")

    # print what we can see
    target = root / args.name
    if target.exists():
        print("contents:")
        for path in sorted(target.rglob("*")):
            if path.is_file():
                print(f"  {path.relative_to(root)}  ({path.stat().st_size} bytes)")
    else:
        # zip may have nested exp3a_grpo already
        print("top-level after restore:")
        for path in sorted(root.iterdir()):
            print(f"  {path}")


if __name__ == "__main__":
    main()
