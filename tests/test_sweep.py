from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.sweep import (
    build_comparison_table,
    config_slug,
    expand_grid,
    is_run_complete,
)


def test_expand_grid_cartesian_product() -> None:
    grid = {"lr": [1e-4, 3e-4], "ent_coef": [0.01, 0.05]}
    seeds = [0, 1]
    combos = expand_grid(grid, seeds)
    assert len(combos) == 2 * 2 * 2  # 8
    # Each combo has the grid keys + seed key.
    for combo in combos:
        assert set(combo.keys()) == {"lr", "ent_coef", "seed"}
    # All combos unique.
    assert len({tuple(sorted(c.items())) for c in combos}) == 8


def test_expand_grid_single_value_param_contributes_factor_one() -> None:
    grid = {"lr": [1e-4, 3e-4, 5e-4], "ic_mut_threshold": [0.99]}
    seeds = [0, 1]
    combos = expand_grid(grid, seeds)
    # 3 * 1 * 2 = 6 — single-value param does NOT multiply runs.
    assert len(combos) == 6
    assert all(c["ic_mut_threshold"] == 0.99 for c in combos)


def test_expand_grid_empty_grid_uses_seeds_only() -> None:
    combos = expand_grid({}, seeds=[0, 1, 2])
    assert len(combos) == 3
    assert all(set(c.keys()) == {"seed"} for c in combos)


def test_config_slug_is_deterministic_and_readable() -> None:
    combo = {"lr": 3e-4, "ent_coef": 0.01, "seed": 7}
    slug1 = config_slug(combo)
    slug2 = config_slug(combo)
    assert slug1 == slug2  # deterministic
    # Readable: contains the parameter names.
    assert "lr" in slug1 and "ent_coef" in slug1 and "seed" in slug1
    # Bounded length: short hash suffix keeps it under ~100 chars.
    assert len(slug1) <= 120


def test_config_slug_differs_for_different_combos() -> None:
    a = config_slug({"lr": 3e-4, "seed": 0})
    b = config_slug({"lr": 3e-4, "seed": 1})
    c = config_slug({"lr": 1e-4, "seed": 0})
    assert len({a, b, c}) == 3


def test_is_run_complete_detects_training_end_marker(tmp_path: Path) -> None:
    run_dir = tmp_path / "fake_run"
    run_dir.mkdir()
    # No status.json yet -> not complete.
    assert is_run_complete(run_dir) is False
    # heartbeat event -> not complete.
    (run_dir / "status.json").write_text(json.dumps({"event": "heartbeat"}))
    assert is_run_complete(run_dir) is False
    # training_end event -> complete.
    (run_dir / "status.json").write_text(json.dumps({"event": "training_end"}))
    assert is_run_complete(run_dir) is True


def test_build_comparison_table_pivots_runs_into_one_row_per_config(tmp_path: Path) -> None:
    sweep_root = tmp_path / "my_sweep"
    sweep_root.mkdir()

    def _make_run(slug: str, lr: float, seed: int, test_rank_icir: float) -> None:
        run_dir = sweep_root / slug
        run_dir.mkdir()
        (run_dir / "run_config.json").write_text(json.dumps({
            "learning_rate": lr,
            "seed": seed,
            "ic_mut_threshold": 0.99,
            "complexity_penalty": 0.0,
        }))
        # Minimal best_segment_metrics.csv with valid + test rows.
        (run_dir / "best_segment_metrics.csv").write_text(
            "split,ic,rank_ic,icir,rank_icir,checkpoint\n"
            f"valid,0.01,0.02,0.5,0.6,1_steps_pool.json\n"
            f"test,0.01,0.02,0.5,{test_rank_icir},1_steps_pool.json\n"
        )

    # Two configs (lr=1e-4 and lr=3e-4), each with two seeds.
    _make_run("lr=1e-4__seed=0__abc", 1e-4, 0, 0.20)
    _make_run("lr=1e-4__seed=1__abd", 1e-4, 1, 0.22)
    _make_run("lr=3e-4__seed=0__abe", 3e-4, 0, 0.30)
    _make_run("lr=3e-4__seed=1__abf", 3e-4, 1, 0.34)

    df = build_comparison_table(
        sweep_root=sweep_root,
        grid_keys=("learning_rate",),
    )
    # Two rows, one per (learning_rate,) config.
    assert len(df) == 2
    # Columns include the grid key and aggregated test_rank_icir mean/std.
    assert "learning_rate" in df.columns
    assert "test_rank_icir_mean" in df.columns
    assert "test_rank_icir_std" in df.columns
    # Sorted by test_rank_icir_mean descending.
    assert df.iloc[0]["learning_rate"] == 3e-4
    assert df.iloc[0]["test_rank_icir_mean"] == pytest.approx(0.32, abs=1e-6)
    assert df.iloc[1]["test_rank_icir_mean"] == pytest.approx(0.21, abs=1e-6)
