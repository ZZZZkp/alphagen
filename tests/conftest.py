"""Pytest configuration for alphagen tests."""
from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path so that imports work
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
