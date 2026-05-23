"""Sweep runner for AlphaGen RL experiments.

Iterates a grid of hyperparameter values x seeds, runs `run_single_experiment` for
each (skipping configs that already completed), then aggregates results into a
cross-config comparison table.

Usage:
    .venv/bin/python scripts/sweep.py run \\
        --name my_sweep \\
        --grid '{"learning_rate":[1e-4,3e-4], "ic_mut_threshold":[0.7,0.8,0.99]}' \\
        --seeds '[0,1,2]' \\
        --base '{"profile":"colab","instruments":"csi300","pool_capacity":10}'

Resumable: rerunning the same command skips run dirs that already have a
status.json with event == "training_end".
"""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fire
import pandas as pd


def expand_grid(grid: Mapping[str, Sequence[Any]], seeds: Sequence[int]) -> List[Dict[str, Any]]:
    """Cartesian product of grid values, plus seed as an outer dimension.

    A grid entry with one value contributes a factor of one (it does not multiply
    the run count). The returned dicts always include a 'seed' key.
    """
    keys = list(grid.keys())
    value_lists = [list(grid[k]) for k in keys]
    combos: List[Dict[str, Any]] = []
    if not keys:
        for seed in seeds:
            combos.append({"seed": int(seed)})
        return combos
    for values in itertools.product(*value_lists):
        base = dict(zip(keys, values))
        for seed in seeds:
            combo = dict(base)
            combo["seed"] = int(seed)
            combos.append(combo)
    return combos


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def config_slug(combo: Mapping[str, Any]) -> str:
    """Deterministic, human-readable directory slug for a config.

    Format: 'k1=v1__k2=v2__...__<6-char-hash>'. The hash disambiguates if two
    different combos format to the same human-readable prefix.
    """
    parts = [f"{k}={_format_value(combo[k])}" for k in sorted(combo.keys())]
    body = "__".join(parts)
    digest = hashlib.sha1(json.dumps(combo, sort_keys=True, default=str).encode()).hexdigest()[:6]
    return f"{body}__{digest}"


def is_run_complete(run_dir: Path) -> bool:
    """True iff `run_dir/status.json` exists and reports event == 'training_end'."""
    status_path = run_dir / "status.json"
    if not status_path.exists():
        return False
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("event") == "training_end"


def build_comparison_table(
    sweep_root: Path,
    grid_keys: Sequence[str],
) -> pd.DataFrame:
    """Pivot per-run results into one row per (grid_keys) config.

    Reads `<sweep_root>/<slug>/run_config.json` and
    `<sweep_root>/<slug>/best_segment_metrics.csv` for every run dir under
    `sweep_root` (skipping the `aggregate` subdir). Groups by `grid_keys` and
    reports cross-seed mean/std for valid and test rank_ICIR.
    """
    sweep_root = Path(sweep_root)
    rows: List[Dict[str, Any]] = []
    for run_dir in sorted(sweep_root.iterdir()):
        if not run_dir.is_dir() or run_dir.name == "aggregate":
            continue
        cfg_path = run_dir / "run_config.json"
        best_path = run_dir / "best_segment_metrics.csv"
        if not cfg_path.exists() or not best_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        best = pd.read_csv(best_path)
        row: Dict[str, Any] = {k: cfg.get(k) for k in grid_keys}
        row["seed"] = cfg.get("seed")
        for split in ("valid", "test"):
            sub = best.loc[best["split"] == split]
            if not sub.empty:
                row[f"{split}_rank_icir"] = float(sub.iloc[0]["rank_icir"])
                row[f"{split}_ic"] = float(sub.iloc[0]["ic"])
                row[f"{split}_rank_ic"] = float(sub.iloc[0]["rank_ic"])
        rows.append(row)
    if not rows:
        return pd.DataFrame()

    long_df = pd.DataFrame(rows)
    agg_cols: Dict[str, Tuple[str, str]] = {}
    for metric in ("valid_rank_icir", "test_rank_icir", "valid_ic", "test_ic", "valid_rank_ic", "test_rank_ic"):
        if metric in long_df.columns:
            agg_cols[f"{metric}_mean"] = (metric, "mean")
            agg_cols[f"{metric}_std"] = (metric, "std")
    agg_cols["seeds"] = ("seed", "nunique")
    grouped = (
        long_df.groupby(list(grid_keys), as_index=False, dropna=False)
        .agg(**agg_cols)
        .sort_values("test_rank_icir_mean", ascending=False, na_position="last")
        .reset_index(drop=True)
    )
    return grouped


def _merged_kwargs(base: Mapping[str, Any], combo: Mapping[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base)
    merged.update(combo)
    return merged


def run_sweep(
    name: str,
    grid: Mapping[str, Sequence[Any]],
    seeds: Sequence[int],
    base: Mapping[str, Any],
    sweeps_root: str = "./out/sweeps",
    evaluate: bool = True,
    qlib_data: str = "qlib_data/us_data",
    region: str = "us",
) -> Path:
    """Run an entire sweep and produce the comparison table.

    Args:
        name: sweep name; output goes under `<sweeps_root>/<name>/`.
        grid: dict of param_name -> list of candidate values.
        seeds: list of seed ints.
        base: fixed overrides applied to every run (incl. `profile` for `run_single_experiment`).
        sweeps_root: parent dir for sweep output.
        evaluate: if True, call `evaluate_runs` after the grid completes.
        qlib_data: qlib data dir for the offline evaluator.
        region: qlib region for the offline evaluator.

    Returns the sweep root directory.
    """
    from scripts.rl import (
        get_profile,
        resolve_device,
        resolve_qlib_data_path,
        run_single_experiment,
    )

    sweep_root = Path(sweeps_root) / name
    sweep_root.mkdir(parents=True, exist_ok=True)

    combos = expand_grid(grid, seeds)
    grid_keys = tuple(grid.keys())
    print(f"[sweep] {name}: {len(combos)} configs total")
    print(f"[sweep] grid_keys={grid_keys} seeds={list(seeds)}")

    profile_name = base.get("profile", "default")
    profile = get_profile(profile_name)

    for i, combo in enumerate(combos, 1):
        slug = config_slug(combo)
        run_dir = sweep_root / slug
        if is_run_complete(run_dir):
            print(f"[sweep] [{i}/{len(combos)}] skip (already complete): {slug}")
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        merged = _merged_kwargs(base, combo)
        merged.pop("profile", None)
        # Wire sweep-only output and disable model-zip saves.
        merged["output_dir"] = str(run_dir)
        merged["save_model_checkpoints"] = False
        # Defaults from the profile that run_single_experiment expects.
        merged.setdefault("segments", profile.segments)
        merged.setdefault("ppo_n_steps", profile.default_ppo_n_steps)
        merged.setdefault("batch_size", profile.default_batch_size)
        merged.setdefault("print_expr", profile.print_expr)
        merged.setdefault("qlib_data_path", resolve_qlib_data_path(None, profile))
        merged.setdefault("device", resolve_device(None, profile))
        print(f"[sweep] [{i}/{len(combos)}] run: {slug}")
        run_single_experiment(**merged)

    if evaluate:
        print(f"[sweep] evaluating runs under {sweep_root}")
        from scripts.evaluate_local_runs import US_DELTA_TIMES, US_FEATURES, evaluate_runs

        evaluate_runs(
            runs_dir=sweep_root,
            qlib_data=Path(qlib_data),
            region=region,
            features=US_FEATURES,
            delta_times=US_DELTA_TIMES,
        )
        cmp_df = build_comparison_table(sweep_root, grid_keys)
        cmp_path = sweep_root / "sweep_comparison.csv"
        cmp_df.to_csv(cmp_path, index=False)
        print(f"[sweep] wrote {cmp_path}  ({len(cmp_df)} configs)")
    return sweep_root


def _cli_run(
    name: str,
    grid: str,
    seeds: str,
    base: str = "{}",
    sweeps_root: str = "./out/sweeps",
    evaluate: bool = True,
    qlib_data: str = "qlib_data/us_data",
    region: str = "us",
) -> None:
    """Fire-friendly CLI wrapper: parses JSON strings into Python objects."""
    run_sweep(
        name=name,
        grid=json.loads(grid),
        seeds=json.loads(seeds),
        base=json.loads(base),
        sweeps_root=sweeps_root,
        evaluate=evaluate,
        qlib_data=qlib_data,
        region=region,
    )


if __name__ == "__main__":
    fire.Fire({"run": _cli_run})
