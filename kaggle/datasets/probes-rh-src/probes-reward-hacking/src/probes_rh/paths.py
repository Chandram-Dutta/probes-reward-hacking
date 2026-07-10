"""Output paths that survive a Kaggle notebook as long as /kaggle/working does.

On Kaggle, always write under /kaggle/working/outputs/... so results are easy
to zip, download, re-upload as a Dataset, and restore after a restart.

After a full session wipe you must re-upload the zip (or attach a Dataset);
nothing in /kaggle/working persists across brand-new sessions forever.
"""

from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path


def on_kaggle() -> bool:
    return Path("/kaggle/working").is_dir()


def output_root() -> Path:
    """Root for all experiment artifacts."""
    env = os.environ.get("PROBES_OUTPUT_ROOT")
    if env:
        return Path(env)
    if on_kaggle():
        return Path("/kaggle/working/outputs")
    return Path("outputs")


def exp_dir(name: str = "exp3a_grpo") -> Path:
    d = output_root() / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def resolve_existing(*candidates: str | Path) -> Path | None:
    """Return the first path that exists (file or dir)."""
    for c in candidates:
        if c is None:
            continue
        p = Path(c)
        if p.exists():
            return p
    return None


def pack_dir(src: Path, zip_path: Path, patterns: list[str] | None = None) -> Path:
    """Zip an experiment folder (optionally only matching names)."""
    src = Path(src)
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        if src.is_file():
            zf.write(src, arcname=src.name)
            return zip_path
        for path in src.rglob("*"):
            if not path.is_file():
                continue
            if patterns is not None and path.name not in patterns:
                # also allow suffix match
                if not any(path.name.endswith(p) or path.name == p for p in patterns):
                    continue
            zf.write(path, arcname=str(path.relative_to(src.parent)))
    return zip_path


def restore_zip(zip_path: Path, dest_parent: Path | None = None) -> Path:
    """Unzip a checkpoint/results archive under output_root (or dest_parent)."""
    zip_path = Path(zip_path)
    dest_parent = Path(dest_parent) if dest_parent else output_root()
    dest_parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_parent)
    return dest_parent


def copy_tree(src: Path, dest: Path) -> Path:
    src, dest = Path(src), Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    return dest
