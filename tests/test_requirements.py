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
        "protobuf",
        "pyqlib",
        "sb3-contrib",
        "stable-baselines3",
        "tokentrim",
    }

    assert expected <= packages


def test_numpy2_runtime_floor_is_declared() -> None:
    text = REQUIREMENTS.read_text()

    assert "numpy>=2.0.2,<3.0" in text
    assert "pandas>=2.2,<3.0" in text
    assert "scikit-learn>=1.5,<2.0" in text
    assert "stable_baselines3>=2.8,<3.0" in text
    assert "sb3_contrib>=2.8,<3.0" in text


def test_legacy_dependency_pins_are_removed() -> None:
    text = REQUIREMENTS.read_text()

    for legacy_pin in (
        "gym==0.26.2",
        "numpy==1.20.1",
        "numpy>=1.24,<2.0",
        "pandas==1.2.4",
        "matplotlib==3.3.4",
        "qlib==0.0.2.dev20",
        "openai==1.2.3",
    ):
        assert legacy_pin not in text

    assert "gym==" not in text
