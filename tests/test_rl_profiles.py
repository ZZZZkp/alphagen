from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

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


def test_validate_qlib_calendar_respects_stock_data_padding(tmp_path: Path) -> None:
    calendars_dir = tmp_path / "calendars"
    calendars_dir.mkdir()
    (calendars_dir / "day.txt").write_text(
        "2024-01-02\n2024-01-03\n2024-01-04\n2024-01-05\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="usable range"):
        rl.validate_qlib_calendar(
            str(tmp_path),
            (("2024-01-03", "2024-01-05"),),
            max_backtrack_days=1,
            max_future_days=1,
        )


def test_local_profile_uses_laptop_friendly_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(rl, "resolve_device", lambda device, profile: "cpu")
    monkeypatch.setattr(rl, "resolve_qlib_data_path", lambda path, profile: "/tmp/qlib")
    monkeypatch.setattr(
        rl,
        "validate_qlib_calendar",
        lambda path, segments, max_backtrack_days=0, max_future_days=0: None,
    )

    def fake_run_single_experiment(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(rl, "run_single_experiment", fake_run_single_experiment)

    rl.local(random_seeds=3, qlib_region="us")

    assert captured["seed"] == 3
    assert captured["pool_capacity"] == rl.PROFILES["local"].default_pool_capacity
    assert captured["steps"] == rl.LOCAL_STEPS[rl.PROFILES["local"].default_pool_capacity]
    assert captured["device"] == "cpu"
    assert captured["qlib_data_path"] == "/tmp/qlib"
    assert captured["qlib_region"] == "us"
    assert captured["ppo_n_steps"] == 128
    assert captured["batch_size"] == 64
    assert captured["print_expr"] is False


def test_resolve_tensorboard_log_returns_none_when_tensorboard_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rl, "find_spec", lambda name: None)

    assert rl.resolve_tensorboard_log() is None


def test_run_single_experiment_validates_calendar_for_direct_call(monkeypatch: pytest.MonkeyPatch) -> None:
    class Sentinel(Exception):
        pass

    monkeypatch.setattr(rl, "reseed_everything", lambda seed: None)

    def fail_validation(path, segments, max_backtrack_days=0, max_future_days=0):
        raise Sentinel((path, segments, max_backtrack_days, max_future_days))

    monkeypatch.setattr(rl, "validate_qlib_calendar", fail_validation)

    with pytest.raises(Sentinel) as exc_info:
        rl.run_single_experiment(
            qlib_data_path="/tmp/qlib",
            segments=(("2020-01-01", "2020-12-31"),),
        )

    assert exc_info.value.args[0] == (
        "/tmp/qlib",
        (("2020-01-01", "2020-12-31"),),
        rl.STOCK_DATA_MAX_BACKTRACK_DAYS,
        rl.STOCK_DATA_MAX_FUTURE_DAYS,
    )


def test_main_rejects_batch_size_larger_than_rollout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rl, "resolve_device", lambda device, profile: "cpu")
    monkeypatch.setattr(rl, "resolve_qlib_data_path", lambda path, profile: "/tmp/qlib")
    monkeypatch.setattr(
        rl,
        "validate_qlib_calendar",
        lambda path, segments, max_backtrack_days=0, max_future_days=0: None,
    )

    with pytest.raises(ValueError, match="batch_size"):
        rl.main(profile="local", pool_capacity=10, ppo_n_steps=32, batch_size=64)


def test_status_reads_latest_run_status_file(tmp_path: Path) -> None:
    old_run = tmp_path / "run-old"
    old_run.mkdir()
    (old_run / "status.json").write_text('{"event": "old"}', encoding="utf-8")

    new_run = tmp_path / "run-new"
    new_run.mkdir()
    (new_run / "status.json").write_text('{"event": "new"}', encoding="utf-8")

    payload = rl.status(results_root=str(tmp_path))

    assert payload["event"] in {"old", "new"}


def test_custom_callback_defers_model_checkpoints_until_configured_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback = rl.CustomCallback(
        save_path=str(tmp_path),
        test_calculators=[],
        checkpoint_every_n_rollouts=10,
        model_checkpoint_start_step=50,
    )
    saved_models: list[str] = []

    class DummyPool:
        def to_json_dict(self) -> dict[str, object]:
            return {"exprs": [], "weights": []}

    dummy_pool = DummyPool()
    monkeypatch.setattr(rl.CustomCallback, "pool", property(lambda self: dummy_pool))
    training_env = SimpleNamespace(
        envs=[SimpleNamespace(unwrapped=SimpleNamespace(pool=dummy_pool))]
    )
    callback.model = SimpleNamespace(
        save=lambda path: saved_models.append(path),
        get_env=lambda: training_env,
    )

    callback.num_timesteps = 40
    callback._rollout_count = 10
    callback.save_checkpoint()
    assert (tmp_path / "40_steps_pool.json").exists()
    assert saved_models == []

    callback.num_timesteps = 60
    callback._rollout_count = 20
    callback.save_checkpoint()
    assert (tmp_path / "60_steps_pool.json").exists()
    assert saved_models == [str(tmp_path / "60_steps")]
