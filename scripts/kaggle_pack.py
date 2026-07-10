#!/usr/bin/env python3
"""Pack experiment outputs for download / Dataset re-upload.

  python scripts/kaggle_pack.py --name exp3a_grpo
  python scripts/kaggle_pack.py --name exp3a_grpo --light   # json/jsonl only
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import ensure_import_path

ensure_import_path()

from probes_rh.paths import exp_dir, on_kaggle, output_root


LIGHT_NAMES = {
    "train_metrics.json",
    "train_config.json",
    "probe_results.json",
    "probe_results_residual.json",
    "rollouts.jsonl",
    "rollouts_base.jsonl",
    "rollouts_3b.jsonl",
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="exp3a_grpo")
    p.add_argument(
        "--light",
        action="store_true",
        help="Only metrics/rollouts/probes (skip adapter weights)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Zip path (default: /kaggle/working/<name>_results.zip or ./)",
    )
    args = p.parse_args()

    src = exp_dir(args.name)
    if not any(src.iterdir()):
        # maybe empty mkdir; try root/name without forcing create
        src = output_root() / args.name
    if not src.exists():
        raise SystemExit(f"missing experiment dir: {src}")

    if args.out:
        zip_path = Path(args.out)
    elif on_kaggle():
        zip_path = Path("/kaggle/working") / (
            f"{args.name}_light.zip" if args.light else f"{args.name}_pack.zip"
        )
    else:
        zip_path = Path(f"{args.name}_pack.zip")

    n = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in src.rglob("*"):
            if not path.is_file():
                continue
            if args.light and path.name not in LIGHT_NAMES:
                continue
            # skip huge tokenizer dumps in light mode already handled
            if args.light and path.suffix in {".bin", ".safetensors"}:
                continue
            arc = str(path.relative_to(src.parent))
            zf.write(path, arcname=arc)
            n += 1
            print("added", arc)

    print(f"wrote {zip_path} ({n} files, {zip_path.stat().st_size / 1e6:.2f} MB)")
    print("Download this file, then after restart re-upload and run kaggle_restore.py")


if __name__ == "__main__":
    main()
