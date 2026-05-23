from __future__ import annotations

from alphagen.utils.early_stop import EarlyStopState


def test_disabled_when_patience_is_zero() -> None:
    state = EarlyStopState(patience=0, warmup_steps=0, min_delta=0.0)
    for step in (100, 200, 300):
        assert state.update(step, metric=-1.0) is False
    assert state.stopped is False


def test_records_best_during_warmup_but_does_not_trigger() -> None:
    state = EarlyStopState(patience=3, warmup_steps=20_000, min_delta=1e-3)
    # 5 rollouts of decreasing metric, all inside warmup -> never stops.
    metrics = [0.5, 0.4, 0.3, 0.2, 0.1]
    for i, m in enumerate(metrics):
        step = 2048 * (i + 1)  # all <= 10_240, still in warmup
        assert state.update(step, metric=m) is False
    assert state.best == 0.5
    assert state.best_step == 2048
    assert state.no_improve_rollouts == 4
    assert state.stopped is False


def test_triggers_immediately_after_warmup_if_plateau_predates_it() -> None:
    state = EarlyStopState(patience=3, warmup_steps=20_000, min_delta=1e-3)
    # Best at first rollout, then 3 stagnant rollouts still inside warmup.
    state.update(2_000, 0.5)
    state.update(4_000, 0.4)
    state.update(6_000, 0.3)
    state.update(8_000, 0.2)
    assert state.stopped is False  # still in warmup
    # First post-warmup rollout, still stagnant -> should stop.
    assert state.update(22_000, 0.2) is True
    assert state.stopped is True
    assert "plateau" in state.stop_reason


def test_improvement_resets_counter() -> None:
    state = EarlyStopState(patience=3, warmup_steps=0, min_delta=1e-3)
    state.update(1000, 0.10)
    state.update(2000, 0.10)  # no improvement
    state.update(3000, 0.15)  # improvement, resets
    state.update(4000, 0.15)  # no improvement
    state.update(5000, 0.15)  # no improvement
    assert state.no_improve_rollouts == 2
    assert state.stopped is False
    state.update(6000, 0.15)  # patience reached
    assert state.stopped is True


def test_min_delta_blocks_marginal_improvement() -> None:
    state = EarlyStopState(patience=2, warmup_steps=0, min_delta=1e-2)
    state.update(1000, 0.10)
    state.update(2000, 0.105)  # below min_delta -> not an improvement
    state.update(3000, 0.105)
    assert state.stopped is True


def test_snapshot_returns_serializable_dict() -> None:
    state = EarlyStopState(patience=3, warmup_steps=20_000, min_delta=1e-3)
    state.update(2_000, 0.5)
    snap = state.snapshot()
    assert snap == {
        "patience": 3,
        "warmup_steps": 20_000,
        "min_delta": 1e-3,
        "best": 0.5,
        "best_step": 2_000,
        "no_improve_rollouts": 0,
        "stopped": False,
        "stop_reason": None,
    }
