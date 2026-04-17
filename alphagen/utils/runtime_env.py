from __future__ import annotations

import os
from pathlib import Path


def configure_project_runtime(project_root: Path) -> None:
    """
    Set stable, writable runtime directories for third-party libraries.

    Matplotlib falls back to temporary cache directories when its default cache path
    is unavailable in the current environment. Pointing MPLCONFIGDIR into the repo
    keeps font/config caches stable across local runs and avoids noisy startup warnings.
    """
    matplotlib_cache_dir = project_root / ".cache" / "matplotlib"
    matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))
