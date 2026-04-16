from __future__ import annotations

from pathlib import Path

import pytest

import scripts.rl as rl


def test_resolve_qlib_data_path_prefers_first_existing_candidate(tmp_path: Path) -> None:
    existing = tmp_path / "cn_data"
    existing.mkdir()
    profile = rl.RLProfile(
        name="test",
        description="test profile",
        qlib_candidates=(str(tmp_path / "missing"), str(existing)),
        default_pool_capacity=10,
        default_steps=rl.DEFAULT_STEPS,
    )

    assert rl.resolve_qlib_data_path(None, profile) == str(existing)


def test_validate_qlib_calendar_rejects_missing_segment_coverage(tmp_path: Path) -> None:
    calendars_dir = tmp_path / "calendars"
    calendars_dir.mkdir()
    (calendars_dir / "day.txt").write_text("2022-01-03\n2022-01-04\n", encoding="utf-8")

    with pytest.raises(ValueError, match="does not cover"):
        rl.validate_qlib_calendar(str(tmp_path), (("2022-01-01", "2022-01-31"),))


def test_local_profile_uses_laptop_friendly_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(rl, "resolve_device", lambda device, profile: "cpu")
    monkeypatch.setattr(rl, "resolve_qlib_data_path", lambda path, profile: "/tmp/qlib")
    monkeypatch.setattr(rl, "validate_qlib_calendar", lambda path, segments: None)

    def fake_run_single_experiment(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(rl, "run_single_experiment", fake_run_single_experiment)

    rl.local(random_seeds=3)

    assert captured["seed"] == 3
    assert captured["pool_capacity"] == rl.PROFILES["local"].default_pool_capacity
    assert captured["steps"] == rl.LOCAL_STEPS[rl.PROFILES["local"].default_pool_capacity]
    assert captured["device"] == "cpu"
    assert captured["qlib_data_path"] == "/tmp/qlib"
