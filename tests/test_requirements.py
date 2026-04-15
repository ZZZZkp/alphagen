from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "requirements.txt"


def _package_names() -> set[str]:
    names: set[str] = set()
    for raw_line in REQUIREMENTS.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name = re.split(r"[<>=!~\\[]", line, maxsplit=1)[0]
        names.add(name.strip().lower().replace("_", "-"))
    return names


def test_core_runtime_dependencies_are_declared() -> None:
    packages = _package_names()

    expected = {
        "dataclasses-json",
        "fire",
        "gymnasium",
        "openai",
        "pyqlib",
        "sb3-contrib",
        "stable-baselines3",
        "tokentrim",
    }

    assert expected <= packages


def test_legacy_dependency_pins_are_removed() -> None:
    text = REQUIREMENTS.read_text()

    for legacy_pin in (
        "gym==0.26.2",
        "numpy==1.20.1",
        "pandas==1.2.4",
        "matplotlib==3.3.4",
        "qlib==0.0.2.dev20",
        "openai==1.2.3",
    ):
        assert legacy_pin not in text

    assert "gym==" not in text

