"""Ensure `probes_rh` (including subpackages) is importable on Kaggle."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_import_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if src.is_dir():
        src_s = str(src)
        # Prefer the repo checkout over any broken/partial site-packages install.
        if src_s in sys.path:
            sys.path.remove(src_s)
        sys.path.insert(0, src_s)

    # Require the data subpackage, not just top-level probes_rh.
    try:
        import probes_rh.data  # noqa: F401
        return
    except ImportError:
        pass

    # Last resort: clear a shadowing probes_rh from sys.modules and retry.
    for key in list(sys.modules):
        if key == "probes_rh" or key.startswith("probes_rh."):
            del sys.modules[key]

    import probes_rh.data  # noqa: F401
