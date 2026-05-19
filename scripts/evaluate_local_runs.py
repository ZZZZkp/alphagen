"""Re-evaluate every checkpoint under notebooks/runs/ against local US Qlib data.

For each run dir, writes:
  - checkpoint_metrics.csv      (one row per checkpoint per split)
  - checkpoint_selection.csv    (per-checkpoint valid metrics, sorted)
  - best_segment_metrics.csv    (best checkpoint by valid_rank_icir, train/valid/test rows)

And recomputes notebooks/runs/aggregate/{seed_summary,best_checkpoint_metrics,best_segment_metrics_all}.csv
across every (run_dir, seed) that was evaluated.

Run with:
  .venv/bin/python scripts/evaluate_local_runs.py \
      --runs_dir notebooks/runs \
      --qlib_data qlib_data/us_data
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch

import alphagen.config as config_module
import alphagen.rl.env.wrapper as env_wrapper_module
import alphagen_qlib.stock_data as stock_data_module
from alphagen.data.expression import Expression, Feature, Ref
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen_qlib.stock_data import FeatureType, StockData, initialize_qlib
from alphagen_qlib.utils import load_alpha_pool_by_path


class CachingCalculator(QLibStockDataCalculator):
    """Wraps QLibStockDataCalculator with a per-expression evaluation cache."""

    def __init__(self, data, target=None):
        super().__init__(data, target)
        self._eval_cache: Dict[str, "torch.Tensor"] = {}

    def evaluate_alpha(self, expr: Expression):
        key = str(expr)
        cached = self._eval_cache.get(key)
        if cached is not None:
            return cached
        value = super().evaluate_alpha(expr)
        self._eval_cache[key] = value
        return value

US_FEATURES = [
    FeatureType.OPEN, FeatureType.CLOSE, FeatureType.HIGH,
    FeatureType.LOW, FeatureType.VOLUME,
]
US_DELTA_TIMES = [1, 5, 10, 20, 40, 60]


def _patch_us_runtime() -> None:
    """Match the notebook DELTA_TIMES so any vocabulary-dependent code paths agree."""
    config_module.DELTA_TIMES = list(US_DELTA_TIMES)
    env_wrapper_module.DELTA_TIMES = list(US_DELTA_TIMES)


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _evaluate_checkpoint(
    pool_path: Path,
    split_calculators: Dict[str, CachingCalculator],
    segments: Dict[str, Tuple[str, str]],
    pool_capacity: int,
    device: torch.device,
) -> pd.DataFrame:
    exprs, weights = load_alpha_pool_by_path(str(pool_path))
    rows = []
    for split_name, (start_time, end_time) in segments.items():
        calculator = split_calculators[split_name]
        data = calculator.data
        if len(exprs) == 0:
            ic_mean = icir = rank_ic_mean = rank_icir = float("nan")
        else:
            ic_mean, icir, rank_ic_mean, rank_icir = calculator.calc_pool_all_ret_with_ir(exprs, weights)
        rows.append({
            "split": split_name,
            "start_time": start_time,
            "end_time": end_time,
            "n_days": int(data.n_days),
            "n_stocks": int(data.n_stocks),
            "ic": float(ic_mean),
            "rank_ic": float(rank_ic_mean),
            "icir": float(icir),
            "rank_icir": float(rank_icir),
            "pool_size": len(exprs),
            "checkpoint": pool_path.name,
        })
    return pd.DataFrame(rows)


def _segments_key(seg_list) -> Tuple[Tuple[str, str], ...]:
    return tuple((s, e) for s, e in seg_list)


def _build_split_calculators(
    seg_tuple: Tuple[Tuple[str, str], ...],
    instrument: str,
    device: torch.device,
) -> Dict[str, CachingCalculator]:
    names = ("train", "valid", "test")
    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -20) / close - 1
    calculators: Dict[str, CachingCalculator] = {}
    for name, (start, end) in zip(names, seg_tuple):
        data = StockData(
            instrument=instrument,
            start_time=start,
            end_time=end,
            device=device,
            features=list(US_FEATURES),
        )
        calculators[name] = CachingCalculator(data, target)
        print(f"  loaded {name}: {start} -> {end}  n_days={data.n_days} n_stocks={data.n_stocks}")
    return calculators


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", default="notebooks/runs")
    parser.add_argument("--qlib_data", default="qlib_data/us_data")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-evaluate even if best_segment_metrics.csv already exists in a run dir.",
    )
    parser.add_argument(
        "--only",
        default="",
        help="Substring filter on run dir name (eval just the matching subset).",
    )
    args = parser.parse_args()

    runs_root = Path(args.runs_dir).resolve()
    qlib_path = Path(args.qlib_data).resolve()
    if not runs_root.exists():
        raise SystemExit(f"runs_dir not found: {runs_root}")
    if not qlib_path.exists():
        raise SystemExit(f"qlib_data not found: {qlib_path}")

    _patch_us_runtime()
    device = _pick_device()
    print(f"Device: {device}")
    print(f"Qlib data: {qlib_path}")
    print(f"Runs root: {runs_root}")

    initialize_qlib(str(qlib_path), region="us")

    run_dirs: List[Path] = sorted(
        d for d in runs_root.iterdir()
        if d.is_dir() and d.name != "aggregate" and any(d.glob("*_steps_pool.json"))
    )
    if args.only:
        run_dirs = [d for d in run_dirs if args.only in d.name]
    print(f"Discovered {len(run_dirs)} run dirs.")

    calculators_cache: Dict[Tuple[Tuple[str, str], ...], Dict[str, CachingCalculator]] = {}
    seed_summary_rows: List[dict] = []
    best_segment_frames: List[pd.DataFrame] = []

    for i, run_dir in enumerate(run_dirs, 1):
        cfg = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        seg_tuple = _segments_key(cfg["segments"])
        seg_dict = {name: seg for name, seg in zip(("train", "valid", "test"), seg_tuple)}
        instrument = cfg.get("instruments", "sp500")
        pool_capacity = int(cfg["pool_capacity"])
        seed = int(cfg["seed"])

        if seg_tuple not in calculators_cache:
            print(f"\n[{i}/{len(run_dirs)}] Building calculators for new segment set:")
            calculators_cache[seg_tuple] = _build_split_calculators(seg_tuple, instrument, device)
        split_calcs = calculators_cache[seg_tuple]

        best_csv = run_dir / "best_segment_metrics.csv"
        if best_csv.exists() and not args.force:
            print(f"[{i}/{len(run_dirs)}] skip (cached): {run_dir.name}")
            best_df = pd.read_csv(best_csv)
            best_segment_frames.append(best_df)
            valid_row = best_df.loc[best_df["split"] == "valid"].iloc[0]
            seed_summary_rows.append({
                "seed": seed,
                "run_dir": run_dir.name,
                "checkpoint": valid_row["checkpoint"],
                "step": int(str(valid_row["checkpoint"]).split("_", 1)[0]),
                "valid_ic": float(valid_row["ic"]),
                "valid_rank_ic": float(valid_row["rank_ic"]),
                "valid_icir": float(valid_row["icir"]),
                "valid_rank_icir": float(valid_row["rank_icir"]),
                "best_checkpoint": valid_row["checkpoint"],
            })
            continue

        ckpt_paths = sorted(
            run_dir.glob("*_steps_pool.json"),
            key=lambda p: int(p.name.split("_", 1)[0]),
        )
        print(f"\n[{i}/{len(run_dirs)}] {run_dir.name} | seed={seed} pool={pool_capacity} ckpts={len(ckpt_paths)}")
        start = time.time()
        ckpt_metrics: List[pd.DataFrame] = []
        ckpt_summary_rows: List[dict] = []
        best_by_name: Dict[str, pd.DataFrame] = {}

        for j, pool_path in enumerate(ckpt_paths, 1):
            t0 = time.time()
            metrics_df = _evaluate_checkpoint(pool_path, split_calcs, seg_dict, pool_capacity, device)
            metrics_df["seed"] = seed
            metrics_df["run_dir"] = run_dir.name
            ckpt_metrics.append(metrics_df)
            best_by_name[pool_path.name] = metrics_df
            valid_row = metrics_df.loc[metrics_df["split"] == "valid"].iloc[0]
            ckpt_summary_rows.append({
                "seed": seed,
                "run_dir": run_dir.name,
                "checkpoint": pool_path.name,
                "step": int(pool_path.name.split("_", 1)[0]),
                "valid_ic": float(valid_row["ic"]),
                "valid_rank_ic": float(valid_row["rank_ic"]),
                "valid_icir": float(valid_row["icir"]),
                "valid_rank_icir": float(valid_row["rank_icir"]),
            })
            elapsed = time.time() - t0
            print(
                f"    [{j}/{len(ckpt_paths)}] {pool_path.name}  "
                f"valid_rank_icir={valid_row['rank_icir']:+.4f}  "
                f"valid_ic={valid_row['ic']:+.4f}  ({elapsed:.1f}s)"
            )

        ckpt_metrics_df = pd.concat(ckpt_metrics, ignore_index=True)
        summary_df = pd.DataFrame(ckpt_summary_rows).sort_values(
            by=["valid_rank_icir", "valid_rank_ic", "valid_ic"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        best_ckpt = summary_df.iloc[0]["checkpoint"]
        best_df = best_by_name[best_ckpt].copy()
        best_df["selected_by"] = "valid_rank_icir"
        best_df["is_best_checkpoint"] = True

        ckpt_metrics_df.to_csv(run_dir / "checkpoint_metrics.csv", index=False)
        summary_df.to_csv(run_dir / "checkpoint_selection.csv", index=False)
        best_df.to_csv(best_csv, index=False)

        best_segment_frames.append(best_df)
        best_row = summary_df.iloc[0]
        seed_summary_rows.append({
            "seed": seed,
            "run_dir": run_dir.name,
            "checkpoint": best_ckpt,
            "step": int(best_row["step"]),
            "valid_ic": float(best_row["valid_ic"]),
            "valid_rank_ic": float(best_row["valid_rank_ic"]),
            "valid_icir": float(best_row["valid_icir"]),
            "valid_rank_icir": float(best_row["valid_rank_icir"]),
            "best_checkpoint": best_ckpt,
        })
        print(f"    -> best by valid_rank_icir: {best_ckpt} ({time.time()-start:.1f}s)")

    seed_summary_df = pd.DataFrame(seed_summary_rows)
    if best_segment_frames:
        best_all_df = pd.concat(best_segment_frames, ignore_index=True)
    else:
        best_all_df = pd.DataFrame()

    if not best_all_df.empty:
        agg = (
            best_all_df.groupby("split", as_index=False)
            .agg(
                seeds=("seed", "nunique"),
                ic_mean=("ic", "mean"), ic_std=("ic", "std"),
                rank_ic_mean=("rank_ic", "mean"), rank_ic_std=("rank_ic", "std"),
                icir_mean=("icir", "mean"), icir_std=("icir", "std"),
                rank_icir_mean=("rank_icir", "mean"), rank_icir_std=("rank_icir", "std"),
                n_days=("n_days", "first"), n_stocks_mean=("n_stocks", "mean"),
            )
        )
    else:
        agg = pd.DataFrame()

    out_dir = runs_root / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_summary_df.to_csv(out_dir / "seed_summary.csv", index=False)
    best_all_df.to_csv(out_dir / "best_segment_metrics_all.csv", index=False)
    agg.to_csv(out_dir / "best_checkpoint_metrics.csv", index=False)

    print("\nWrote:")
    print(f"  {out_dir / 'seed_summary.csv'}")
    print(f"  {out_dir / 'best_segment_metrics_all.csv'}")
    print(f"  {out_dir / 'best_checkpoint_metrics.csv'}")


if __name__ == "__main__":
    main()
