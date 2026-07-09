"""Ensure `probes_rh` is importable without a full editable install.

Kaggle notebooks often run scripts before / without `pip install -e .`.
This adds <repo>/src to sys.path when needed.
"""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_import_path() -> None:
    try:
        import probes_rh  # noqa: F401

        return
    except ImportError:
        pass

    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))

    # second chance
    import probes_rh  # noqa: F401
