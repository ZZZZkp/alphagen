"""Make `from alphagen.* import ...` work when pytest is invoked from the repo root.

Without this, `.venv/bin/pytest tests/...` fails to import the project modules
because there is no package install. The plan's documented test commands assume
this is in place; adding it once here lets every test file run as written.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
