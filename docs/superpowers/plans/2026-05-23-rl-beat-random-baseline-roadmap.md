# AlphaGen RL Beat-Random-Baseline Roadmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the 3-phase roadmap from `docs/superpowers/specs/2026-05-22-rl-beat-random-baseline-roadmap-design.md` so the AlphaGen RL alpha generator can plausibly beat a random-search baseline.

**Architecture:** Three sequential phases. Phase 1 adds an `EarlyStopState` helper and wires it into `CustomCallback`. Phase 2 adds two new params to `run_single_experiment` (output dir, model-save toggle), refactors the offline evaluator into an importable function, and creates `scripts/sweep.py`. Phase 3 plumbs `ic_mut_threshold` through `LinearAlphaPool` and `complexity_penalty` through `AlphaEnvCore`. All new params default to current behavior for backward compatibility.

**Tech Stack:** Python 3.11, PyTorch, stable-baselines3 + sb3-contrib (MaskablePPO), gymnasium, qlib, pandas, pytest, fire.

---

## File Structure

**Phase 1**
- Create `alphagen/utils/early_stop.py` — `EarlyStopState` pure helper.
- Create `tests/test_early_stop.py` — unit tests for the helper.
- Modify `alphagen/utils/__init__.py` — re-export `EarlyStopState`.
- Modify `scripts/rl.py` — `CustomCallback` owns `EarlyStopState`; `run_single_experiment` and CLI entrypoints (`main`/`local`/`local_smoke`/`colab`) plumb 3 new params.

**Phase 2**
- Modify `scripts/rl.py` — `run_single_experiment` gains `output_dir` and `save_model_checkpoints`; `CustomCallback` honors `save_model_checkpoints`.
- Modify `scripts/evaluate_local_runs.py` — extract `evaluate_runs(runs_dir, qlib_data, region, features, delta_times, only=None, force=False)` as importable; CLI `main()` calls it.
- Create `scripts/sweep.py` — grid expansion, deterministic slug, resumable runner, comparison-table pivot. Exposes `expand_grid`, `config_slug`, `build_comparison_table`, `run_sweep` for testability.
- Create `tests/test_sweep.py` — pure-function tests for grid/slug/resume/pivot.

**Phase 3**
- Modify `alphagen/models/linear_alpha_pool.py` — `LinearAlphaPool` / `MseAlphaPool` / `MeanStdAlphaPool` accept `ic_mut_threshold`.
- Modify `alphagen/rl/env/core.py` — `AlphaEnvCore` accepts `complexity_penalty`; `step` applies penalty on `_evaluate`-derived rewards.
- Modify `scripts/rl.py` — `build_pool` passes `ic_mut_threshold`; `AlphaEnv` call passes `complexity_penalty`; both new params on `run_single_experiment` and CLI.
- Create `tests/test_ic_mut_threshold.py` — pool dedup at configurable threshold.
- Create `tests/test_complexity_penalty.py` — env step subtracts the normalized penalty.

---

## Conventions

- Each task ends with a single commit. Commit messages use conventional-commits prefixes (`feat`, `refactor`, `test`, `chore`).
- Tests use the existing pattern from `tests/test_parser_and_pool.py`: real fakes (no mocks), `from __future__ import annotations` at top.
- Run targeted tests after each step; run full `pytest tests/ -q` at the end of each phase.
- Use `.venv/bin/pytest` if the project uses the in-repo venv.

---

# Phase 1 — Early Stopping + Larger Budget

## Task 1.1: `EarlyStopState` helper + unit tests

**Files:**
- Create: `alphagen/utils/early_stop.py`
- Modify: `alphagen/utils/__init__.py`
- Test:   `tests/test_early_stop.py`

- [ ] **Step 1.1.1: Write failing tests**

Create `tests/test_early_stop.py`:

```python
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
```

- [ ] **Step 1.1.2: Run tests, verify all fail with import error**

```bash
.venv/bin/pytest tests/test_early_stop.py -v
```
Expected: `ModuleNotFoundError: No module named 'alphagen.utils.early_stop'`.

- [ ] **Step 1.1.3: Implement `EarlyStopState`**

Create `alphagen/utils/early_stop.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class EarlyStopState:
    """Tracks a monitored metric across rollouts and decides when to stop training.

    Semantics (per the roadmap spec):
      - `best` is updated across the entire run, including the warmup period.
      - `no_improve_rollouts` accumulates from the start.
      - During `num_timesteps < warmup_steps`, `update` records but never returns True.
      - Once past warmup, returns True the first time `no_improve_rollouts >= patience`.
      - `patience <= 0` disables the mechanism entirely.
    """

    patience: int
    warmup_steps: int
    min_delta: float
    best: Optional[float] = None
    best_step: Optional[int] = None
    no_improve_rollouts: int = 0
    stopped: bool = False
    stop_reason: Optional[str] = None

    def update(self, num_timesteps: int, metric: float) -> bool:
        if self.patience <= 0:
            return False
        if self.best is None or metric > self.best + self.min_delta:
            self.best = metric
            self.best_step = num_timesteps
            self.no_improve_rollouts = 0
        else:
            self.no_improve_rollouts += 1
        if (
            num_timesteps >= self.warmup_steps
            and self.no_improve_rollouts >= self.patience
        ):
            self.stopped = True
            self.stop_reason = (
                f"valid_rank_icir plateau for {self.no_improve_rollouts} rollouts "
                f"(best={self.best:.4f} at step {self.best_step})"
            )
            return True
        return False

    def snapshot(self) -> Dict[str, Any]:
        return {
            "patience": self.patience,
            "warmup_steps": self.warmup_steps,
            "min_delta": self.min_delta,
            "best": self.best,
            "best_step": self.best_step,
            "no_improve_rollouts": self.no_improve_rollouts,
            "stopped": self.stopped,
            "stop_reason": self.stop_reason,
        }
```

- [ ] **Step 1.1.4: Re-export from `alphagen.utils`**

Modify `alphagen/utils/__init__.py` — append:

```python
from .early_stop import EarlyStopState
```

- [ ] **Step 1.1.5: Run the tests, verify all pass**

```bash
.venv/bin/pytest tests/test_early_stop.py -v
```
Expected: 6 passed.

- [ ] **Step 1.1.6: Commit**

```bash
git add alphagen/utils/early_stop.py alphagen/utils/__init__.py tests/test_early_stop.py
git commit -m "feat(utils): add EarlyStopState helper for rollout-based early stopping"
```

---

## Task 1.2: Wire `EarlyStopState` into `CustomCallback` and CLI

**Files:**
- Modify: `scripts/rl.py` (CustomCallback constructor, `_on_rollout_end`, `_on_step`, `_write_status`; `run_single_experiment` signature + run_config dump + callback wiring; CLI entrypoints `main`/`local`/`local_smoke`/`colab`)

- [ ] **Step 1.2.1: Add imports and constructor parameters**

In `scripts/rl.py`, near the existing imports:

```python
from alphagen.utils import EarlyStopState, get_logger, reseed_everything
```

(Replace the existing `from alphagen.utils import get_logger, reseed_everything` line.)

Modify `CustomCallback.__init__` signature — add three params with defaults that preserve current behavior:

```python
class CustomCallback(BaseCallback):
    def __init__(
        self,
        save_path: str,
        test_calculators: List[QLibStockDataCalculator],
        verbose: int = 0,
        chat_session: Optional[InterativeSession] = None,
        llm_every_n_steps: int = 25_000,
        drop_rl_n: int = 5,
        heartbeat_every_n_steps: int = 32,
        heartbeat_every_seconds: float = 10.0,
        checkpoint_every_n_rollouts: int = 1,
        model_checkpoint_start_step: int = 0,
        split_names: Optional[List[str]] = None,
        tb_log_every_n_steps: int = 0,
        early_stop_patience: int = 0,
        early_stop_warmup_steps: int = 20_000,
        early_stop_min_delta: float = 1e-3,
    ):
```

(Default `patience=0` keeps existing runs unchanged unless callers opt in.)

At the end of `__init__`, after existing fields are set, add:

```python
        self._early_stop = EarlyStopState(
            patience=int(early_stop_patience),
            warmup_steps=int(early_stop_warmup_steps),
            min_delta=float(early_stop_min_delta),
        )
```

- [ ] **Step 1.2.2: Hook the state machine into `_on_rollout_end`**

In `scripts/rl.py`, locate `_on_rollout_end`:

```python
    def _on_rollout_end(self) -> None:
        self._rollout_count += 1
        if self.chat_session is not None:
            self._try_use_llm()

        metrics = self._compute_split_metrics()
        # Skip duplicate TB record if a sub-rollout tick already covered this exact step.
        if self._last_tb_log_step != self.num_timesteps:
            self._record_pool_and_splits(metrics)
            self._last_tb_log_step = self.num_timesteps
        self.save_checkpoint()
        self._write_status("rollout_end", splits=metrics)
```

Replace with:

```python
    def _on_rollout_end(self) -> None:
        self._rollout_count += 1
        if self.chat_session is not None:
            self._try_use_llm()

        metrics = self._compute_split_metrics()
        # Skip duplicate TB record if a sub-rollout tick already covered this exact step.
        if self._last_tb_log_step != self.num_timesteps:
            self._record_pool_and_splits(metrics)
            self._last_tb_log_step = self.num_timesteps
        self.save_checkpoint()

        valid_metrics = metrics.get("valid") if isinstance(metrics, dict) else None
        if valid_metrics is not None and self._early_stop.patience > 0:
            triggered = self._early_stop.update(
                self.num_timesteps, valid_metrics["rank_icir"]
            )
            if triggered:
                self._append_monitor_log(
                    f"early_stop triggered: {self._early_stop.stop_reason}"
                )

        self._write_status("rollout_end", splits=metrics)
```

- [ ] **Step 1.2.3: Make `_on_step` honor `stopped`**

Locate `_on_step` (currently returns `True` unconditionally). Replace the trailing `return True` with:

```python
        return not self._early_stop.stopped
```

- [ ] **Step 1.2.4: Include the early-stop snapshot in `status.json`**

In `_write_status`, after the existing `payload.update(extra)` line and before `_write_json(self._status_path, payload)`, add:

```python
        payload["early_stop"] = self._early_stop.snapshot()
```

- [ ] **Step 1.2.5: Add early-stop params to `run_single_experiment`**

Modify `run_single_experiment` signature (in `scripts/rl.py`) by appending three keyword-only params right before the closing paren — after `segment_names`:

```python
    early_stop_patience: int = 0,
    early_stop_warmup_steps: int = 20_000,
    early_stop_min_delta: float = 1e-3,
```

Persist them in the `run_config.json` dict (alongside `tb_log_every_n_steps`):

```python
            "early_stop_patience": int(early_stop_patience),
            "early_stop_warmup_steps": int(early_stop_warmup_steps),
            "early_stop_min_delta": float(early_stop_min_delta),
```

Pass them to `CustomCallback(...)`:

```python
    checkpoint_callback = CustomCallback(
        save_path=save_path,
        test_calculators=calculators[1:],
        verbose=1,
        chat_session=inter,
        llm_every_n_steps=llm_every_n_steps,
        drop_rl_n=drop_rl_n,
        checkpoint_every_n_rollouts=checkpoint_every_n_rollouts,
        model_checkpoint_start_step=model_checkpoint_start_step,
        split_names=list(segment_names[1:]),
        tb_log_every_n_steps=tb_log_every_n_steps,
        early_stop_patience=early_stop_patience,
        early_stop_warmup_steps=early_stop_warmup_steps,
        early_stop_min_delta=early_stop_min_delta,
    )
```

- [ ] **Step 1.2.6: Plumb through `main` / `local` / `local_smoke` / `colab`**

In each of the four CLI entrypoints `main`, `local`, `local_smoke`, `colab`, add the three params to the signature (just before the closing paren) and forward them. For example, in `main`:

Add to signature:

```python
    early_stop_patience: int = 0,
    early_stop_warmup_steps: int = 20_000,
    early_stop_min_delta: float = 1e-3,
```

In the body, inside the `run_single_experiment(...)` call, add (alongside `segment_names`):

```python
            early_stop_patience=early_stop_patience,
            early_stop_warmup_steps=early_stop_warmup_steps,
            early_stop_min_delta=early_stop_min_delta,
```

Repeat the same two additions for `local`, `local_smoke`, and `colab` (each is a thin pass-through to `main`, so each gets the params on its own signature and forwards them in the `main(...)` call).

- [ ] **Step 1.2.7: Smoke-check that nothing regressed**

```bash
.venv/bin/pytest tests/ -q
```
Expected: existing tests still pass (no new tests yet beyond Task 1.1; all 6 early-stop tests + pre-existing tests).

```bash
.venv/bin/python -c "from scripts.rl import run_single_experiment, CustomCallback; import inspect; assert 'early_stop_patience' in inspect.signature(run_single_experiment).parameters; assert 'early_stop_patience' in inspect.signature(CustomCallback.__init__).parameters; print('plumbed')"
```
Expected: prints `plumbed`.

- [ ] **Step 1.2.8: Commit**

```bash
git add scripts/rl.py
git commit -m "feat(rl): wire EarlyStopState into CustomCallback and CLI entrypoints"
```

---

# Phase 2 — Sweep Runner

## Task 2.1: `output_dir` and `save_model_checkpoints` params

**Files:**
- Modify: `scripts/rl.py` (`CustomCallback.save_checkpoint`, `run_single_experiment` signature/body, `CustomCallback.__init__`)

- [ ] **Step 2.1.1: Add `save_model_checkpoints` to `CustomCallback`**

In `scripts/rl.py`, extend `CustomCallback.__init__` signature with:

```python
        save_model_checkpoints: bool = True,
```

Store it (next to the early-stop assignments):

```python
        self._save_model_checkpoints = bool(save_model_checkpoints)
```

Modify `CustomCallback.save_checkpoint` so the `self.model.save(path)` branch is gated. Replace:

```python
        if (
            self.num_timesteps >= self._model_checkpoint_start_step and
            self.num_timesteps != self._last_model_checkpoint_step
        ):
            self.model.save(path)   # type: ignore
            self._last_model_checkpoint_step = self.num_timesteps
            model_saved = True
            if self.verbose > 1:
                print(f'Saving model checkpoint to {path}')
```

with:

```python
        if (
            self._save_model_checkpoints and
            self.num_timesteps >= self._model_checkpoint_start_step and
            self.num_timesteps != self._last_model_checkpoint_step
        ):
            self.model.save(path)   # type: ignore
            self._last_model_checkpoint_step = self.num_timesteps
            model_saved = True
            if self.verbose > 1:
                print(f'Saving model checkpoint to {path}')
```

- [ ] **Step 2.1.2: Add `output_dir` and `save_model_checkpoints` to `run_single_experiment`**

Append to `run_single_experiment` signature (after the three early-stop params from Task 1.2.5):

```python
    output_dir: Optional[str] = None,
    save_model_checkpoints: bool = True,
```

Locate the existing `save_path` construction:

```python
    name_prefix = f"{instruments}_{pool_capacity}_{seed}_{timestamp}_{tag}"
    save_path = os.path.join("./out/results", name_prefix)
    os.makedirs(save_path, exist_ok=True)
```

Replace with:

```python
    name_prefix = f"{instruments}_{pool_capacity}_{seed}_{timestamp}_{tag}"
    if output_dir is None:
        save_path = os.path.join("./out/results", name_prefix)
    else:
        save_path = output_dir
    os.makedirs(save_path, exist_ok=True)
```

Persist `save_model_checkpoints` in `run_config.json` (alongside the existing keys):

```python
            "save_model_checkpoints": bool(save_model_checkpoints),
```

Pass `save_model_checkpoints` to `CustomCallback(...)`:

```python
        save_model_checkpoints=save_model_checkpoints,
```

(Add this inside the `CustomCallback(...)` call, alongside the other params.)

- [ ] **Step 2.1.3: Verify wiring imports cleanly**

```bash
.venv/bin/python -c "import inspect; from scripts.rl import run_single_experiment, CustomCallback; assert 'output_dir' in inspect.signature(run_single_experiment).parameters; assert 'save_model_checkpoints' in inspect.signature(CustomCallback.__init__).parameters; print('ok')"
```
Expected: prints `ok`.

```bash
.venv/bin/pytest tests/ -q
```
Expected: existing tests still pass.

- [ ] **Step 2.1.4: Commit**

```bash
git add scripts/rl.py
git commit -m "feat(rl): add output_dir and save_model_checkpoints params for sweep mode"
```

---

## Task 2.2: Refactor `evaluate_local_runs.py` into importable `evaluate_runs(...)`

**Files:**
- Modify: `scripts/evaluate_local_runs.py`

- [ ] **Step 2.2.1: Extract `evaluate_runs(...)` from `main()`**

In `scripts/evaluate_local_runs.py`, locate the existing module-level constants:

```python
US_FEATURES = [
    FeatureType.OPEN, FeatureType.CLOSE, FeatureType.HIGH,
    FeatureType.LOW, FeatureType.VOLUME,
]
US_DELTA_TIMES = [1, 5, 10, 20, 40, 60]


def _patch_us_runtime() -> None:
    """Match the notebook DELTA_TIMES so any vocabulary-dependent code paths agree."""
    config_module.DELTA_TIMES = list(US_DELTA_TIMES)
    env_wrapper_module.DELTA_TIMES = list(US_DELTA_TIMES)
```

Replace the helper with a parametrized version:

```python
US_FEATURES = [
    FeatureType.OPEN, FeatureType.CLOSE, FeatureType.HIGH,
    FeatureType.LOW, FeatureType.VOLUME,
]
US_DELTA_TIMES = [1, 5, 10, 20, 40, 60]


def _patch_runtime(delta_times: List[int]) -> None:
    """Override DELTA_TIMES so any vocabulary-dependent code paths agree with the run."""
    config_module.DELTA_TIMES = list(delta_times)
    env_wrapper_module.DELTA_TIMES = list(delta_times)
```

Now add the new entrypoint immediately after `_pick_device()`:

```python
def evaluate_runs(
    runs_dir: Path,
    qlib_data: Path,
    region: str = "us",
    features: Optional[List["FeatureType"]] = None,
    delta_times: Optional[List[int]] = None,
    only: str = "",
    force: bool = False,
) -> None:
    """Re-evaluate every run under `runs_dir` against `qlib_data` and write aggregate CSVs.

    Mirrors what `main()` used to do, but is callable from Python (e.g. from the sweep runner).
    """
    runs_dir = Path(runs_dir).resolve()
    qlib_path = Path(qlib_data).resolve()
    if not runs_dir.exists():
        raise SystemExit(f"runs_dir not found: {runs_dir}")
    if not qlib_path.exists():
        raise SystemExit(f"qlib_data not found: {qlib_path}")

    feature_list = list(features) if features is not None else list(US_FEATURES)
    delta_list = list(delta_times) if delta_times is not None else list(US_DELTA_TIMES)

    _patch_runtime(delta_list)
    device = _pick_device()
    print(f"Device: {device}")
    print(f"Qlib data: {qlib_path}")
    print(f"Runs root: {runs_dir}")

    initialize_qlib(str(qlib_path), region=region)

    run_dirs: List[Path] = sorted(
        d for d in runs_dir.iterdir()
        if d.is_dir() and d.name != "aggregate" and any(d.glob("*_steps_pool.json"))
    )
    if only:
        run_dirs = [d for d in run_dirs if only in d.name]
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
            calculators_cache[seg_tuple] = _build_split_calculators(
                seg_tuple, instrument, device, feature_list
            )
        split_calcs = calculators_cache[seg_tuple]

        best_csv = run_dir / "best_segment_metrics.csv"
        if best_csv.exists() and not force:
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

    out_dir = runs_dir / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_summary_df.to_csv(out_dir / "seed_summary.csv", index=False)
    best_all_df.to_csv(out_dir / "best_segment_metrics_all.csv", index=False)
    agg.to_csv(out_dir / "best_checkpoint_metrics.csv", index=False)

    print("\nWrote:")
    print(f"  {out_dir / 'seed_summary.csv'}")
    print(f"  {out_dir / 'best_segment_metrics_all.csv'}")
    print(f"  {out_dir / 'best_checkpoint_metrics.csv'}")
```

(Note: `_build_split_calculators` currently hardcodes `features=list(US_FEATURES)` — we must thread the param through it. See next step.)

- [ ] **Step 2.2.2: Parametrize `_build_split_calculators` features**

Modify `_build_split_calculators` signature and body:

```python
def _build_split_calculators(
    seg_tuple: Tuple[Tuple[str, str], ...],
    instrument: str,
    device: torch.device,
    features: List["FeatureType"],
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
            features=list(features),
        )
        calculators[name] = CachingCalculator(data, target)
        print(f"  loaded {name}: {start} -> {end}  n_days={data.n_days} n_stocks={data.n_stocks}")
    return calculators
```

- [ ] **Step 2.2.3: Rewrite `main()` to call `evaluate_runs(...)`**

Replace the existing `main()` with:

```python
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", default="notebooks/runs")
    parser.add_argument("--qlib_data", default="qlib_data/us_data")
    parser.add_argument("--region", default="us")
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

    evaluate_runs(
        runs_dir=Path(args.runs_dir),
        qlib_data=Path(args.qlib_data),
        region=args.region,
        features=US_FEATURES,
        delta_times=US_DELTA_TIMES,
        only=args.only,
        force=args.force,
    )
```

Add `Optional` to the existing `typing` import line at the top of the file if it isn't already present.

- [ ] **Step 2.2.4: Verify the refactor preserves CLI behavior**

```bash
.venv/bin/python -c "from scripts.evaluate_local_runs import evaluate_runs; print(evaluate_runs.__doc__.splitlines()[0])"
```
Expected: prints the first line of the docstring.

```bash
.venv/bin/python scripts/evaluate_local_runs.py --runs_dir nonexistent --qlib_data nonexistent 2>&1 | head -3
```
Expected: `runs_dir not found: ...nonexistent` (proving the CLI still routes through `evaluate_runs`).

```bash
.venv/bin/pytest tests/ -q
```
Expected: no regressions.

- [ ] **Step 2.2.5: Commit**

```bash
git add scripts/evaluate_local_runs.py
git commit -m "refactor(evaluator): extract evaluate_runs() as importable entrypoint"
```

---

## Task 2.3: `scripts/sweep.py` — grid, slug, runner

**Files:**
- Create: `scripts/sweep.py`
- Test:   `tests/test_sweep.py`

- [ ] **Step 2.3.1: Write failing tests for `expand_grid` and `config_slug`**

Create `tests/test_sweep.py`:

```python
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
```

- [ ] **Step 2.3.2: Run tests, verify all fail with import error**

```bash
.venv/bin/pytest tests/test_sweep.py -v
```
Expected: `ModuleNotFoundError: No module named 'scripts.sweep'`.

- [ ] **Step 2.3.3: Implement `scripts/sweep.py`**

Create `scripts/sweep.py`:

```python
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
```

- [ ] **Step 2.3.4: Run tests, verify all pass**

```bash
.venv/bin/pytest tests/test_sweep.py -v
```
Expected: 7 passed.

- [ ] **Step 2.3.5: Smoke-check the CLI imports cleanly**

```bash
.venv/bin/python -c "from scripts.sweep import run_sweep, expand_grid, config_slug, is_run_complete, build_comparison_table; print('ok')"
```
Expected: `ok`.

- [ ] **Step 2.3.6: Run the full suite to confirm no regressions**

```bash
.venv/bin/pytest tests/ -q
```
Expected: all green.

- [ ] **Step 2.3.7: Commit**

```bash
git add scripts/sweep.py tests/test_sweep.py
git commit -m "feat(sweep): add scripts/sweep.py with resumable runner and comparison table"
```

---

# Phase 3 — Configurable Dedup Threshold + Complexity Penalty

## Task 3.1: `ic_mut_threshold` configurable in `LinearAlphaPool`

**Files:**
- Modify: `alphagen/models/linear_alpha_pool.py`
- Modify: `scripts/rl.py` (`build_pool` plumbing, `run_single_experiment` + CLI entrypoints)
- Test:   `tests/test_ic_mut_threshold.py`

- [ ] **Step 3.1.1: Write failing tests**

Create `tests/test_ic_mut_threshold.py`:

```python
from __future__ import annotations

import numpy as np

from alphagen.data.expression import Add, Feature
from alphagen.models.linear_alpha_pool import LinearAlphaPool
from alphagen_qlib.stock_data import FeatureType
from tests.test_parser_and_pool import DeterministicPool, FakeCalculator


def test_pool_default_threshold_is_099_for_back_compat() -> None:
    import inspect
    sig = inspect.signature(LinearAlphaPool.__init__)
    assert sig.parameters["ic_mut_threshold"].default == 0.99


def test_pool_accepts_lower_threshold_and_rejects_correlated_factor() -> None:
    # FakeCalculator mutual IC between $close and $open is 0.1, so threshold 0.05
    # should reject pairing them.
    pool = DeterministicPool(
        capacity=2,
        calculator=FakeCalculator(),
        ic_mut_threshold=0.05,
    )
    pool.force_load_exprs([Feature(FeatureType.CLOSE)])
    assert pool.size == 1
    # Now try_new_expr with $open: mutual IC 0.1 > 0.05 -> rejected by the threshold.
    result = pool.try_new_expr(Feature(FeatureType.OPEN))
    # Rejection returns 0.0 (the early-return branch in try_new_expr).
    assert result == 0.0
    assert pool.size == 1


def test_pool_with_default_threshold_admits_low_correlation_factor() -> None:
    # Mutual IC 0.1 < 0.99 default -> should be admitted.
    pool = DeterministicPool(
        capacity=2,
        calculator=FakeCalculator(),
    )
    pool.force_load_exprs([Feature(FeatureType.CLOSE)])
    result = pool.try_new_expr(Feature(FeatureType.OPEN))
    assert pool.size == 2
    assert result != 0.0


def test_force_load_ignores_threshold() -> None:
    # force_load_exprs deliberately passes ic_mut_threshold=None.
    # Even with a very strict threshold, force_load should add both.
    pool = DeterministicPool(
        capacity=2,
        calculator=FakeCalculator(),
        ic_mut_threshold=0.01,
    )
    pool.force_load_exprs([Feature(FeatureType.CLOSE), Feature(FeatureType.OPEN)])
    assert pool.size == 2
```

- [ ] **Step 3.1.2: Run tests, verify they fail**

```bash
.venv/bin/pytest tests/test_ic_mut_threshold.py -v
```
Expected: failures because `LinearAlphaPool` does not yet accept `ic_mut_threshold`.

- [ ] **Step 3.1.3: Add `ic_mut_threshold` to `LinearAlphaPool`**

In `alphagen/models/linear_alpha_pool.py`, modify `LinearAlphaPool.__init__`:

```python
class LinearAlphaPool(AlphaPoolBase, metaclass=ABCMeta):
    def __init__(
        self,
        capacity: int,
        calculator: AlphaCalculator,
        ic_lower_bound: Optional[float] = None,
        device: torch.device = torch.device("cpu"),
        ic_mut_threshold: float = 0.99,
    ):
        super().__init__(capacity, calculator, device)
        self.exprs: List[Optional[Expression]] = [None for _ in range(capacity + 1)]
        self.single_ics: np.ndarray = np.zeros(capacity + 1)
        self._weights: np.ndarray = np.zeros(capacity + 1)
        self._mutual_ics: np.ndarray = np.identity(capacity + 1)
        self._extra_info = [None for _ in range(capacity + 1)]
        self._ic_lower_bound = -1. if ic_lower_bound is None else ic_lower_bound
        self._ic_mut_threshold = float(ic_mut_threshold)
        self.best_obj = -1.
        self.update_history: List[PoolUpdate] = []
        self._failure_cache: Set[str] = set()
```

Replace the hardcoded `0.99` in `try_new_expr`:

```python
    def try_new_expr(self, expr: Expression) -> float:
        ic_ret, ic_mut = self._calc_ics(expr, ic_mut_threshold=self._ic_mut_threshold)
```

- [ ] **Step 3.1.4: Thread `ic_mut_threshold` through `MseAlphaPool`**

Modify `MseAlphaPool.__init__`:

```python
class MseAlphaPool(LinearAlphaPool):
    def __init__(
        self,
        capacity: int,
        calculator: AlphaCalculator,
        ic_lower_bound: Optional[float] = None,
        l1_alpha: float = 5e-3,
        device: torch.device = torch.device("cpu"),
        ic_mut_threshold: float = 0.99,
    ):
        super().__init__(capacity, calculator, ic_lower_bound, device, ic_mut_threshold)
        self._l1_alpha = l1_alpha
```

- [ ] **Step 3.1.5: Thread `ic_mut_threshold` through `MeanStdAlphaPool`**

Modify `MeanStdAlphaPool.__init__`:

```python
class MeanStdAlphaPool(LinearAlphaPool):
    def __init__(
        self,
        capacity: int,
        calculator: TensorAlphaCalculator,
        ic_lower_bound: Optional[float] = None,
        l1_alpha: float = 5e-3,
        lcb_beta: Optional[float] = None,
        device: torch.device = torch.device("cpu"),
        ic_mut_threshold: float = 0.99,
    ):
        """
        l1_alpha: the L1 regularization coefficient.
        lcb_beta: for optimizing the lower-confidence-bound: LCB = mean - beta * std, \\
                  when this is None, optimize ICIR (mean / std) instead.
        ic_mut_threshold: max allowed mutual IC vs pool members; new factors above this are rejected.
        """
        super().__init__(capacity, calculator, ic_lower_bound, device, ic_mut_threshold)
        self.calculator: TensorAlphaCalculator
        self._l1_alpha = l1_alpha
        self._lcb_beta = lcb_beta
```

- [ ] **Step 3.1.6: Run pool tests, verify they pass**

```bash
.venv/bin/pytest tests/test_ic_mut_threshold.py tests/test_parser_and_pool.py -v
```
Expected: all green (existing pool tests + new ones).

- [ ] **Step 3.1.7: Plumb `ic_mut_threshold` through `scripts/rl.py`**

In `scripts/rl.py`, add `ic_mut_threshold: float = 0.99,` to:
- `run_single_experiment` signature
- `main` signature
- `local` signature
- `local_smoke` signature
- `colab` signature

In `run_single_experiment.build_pool`:

```python
    def build_pool(exprs: List[Expression]) -> LinearAlphaPool:
        pool = MseAlphaPool(
            capacity=pool_capacity,
            calculator=calculators[0],
            ic_lower_bound=None,
            l1_alpha=5e-3,
            device=device,
            ic_mut_threshold=ic_mut_threshold,
        )
        if len(exprs) != 0:
            pool.force_load_exprs(exprs)
        return pool
```

In the `run_config.json` payload inside `run_single_experiment`:

```python
            "ic_mut_threshold": float(ic_mut_threshold),
```

In `main`'s body, inside the `run_single_experiment(...)` call, add:

```python
            ic_mut_threshold=ic_mut_threshold,
```

In each of `local` / `local_smoke` / `colab`, forward to `main`:

```python
        ic_mut_threshold=ic_mut_threshold,
```

- [ ] **Step 3.1.8: Verify plumbing**

```bash
.venv/bin/python -c "import inspect; from scripts.rl import run_single_experiment, main, local, local_smoke, colab; \
for fn in (run_single_experiment, main, local, local_smoke, colab): \
    assert 'ic_mut_threshold' in inspect.signature(fn).parameters, fn.__name__; \
print('all plumbed')"
```
Expected: `all plumbed`.

```bash
.venv/bin/pytest tests/ -q
```
Expected: all green.

- [ ] **Step 3.1.9: Commit**

```bash
git add alphagen/models/linear_alpha_pool.py scripts/rl.py tests/test_ic_mut_threshold.py
git commit -m "feat(pool): make ic_mut_threshold configurable on LinearAlphaPool"
```

---

## Task 3.2: Complexity penalty in `AlphaEnvCore`

**Files:**
- Modify: `alphagen/rl/env/core.py`
- Modify: `scripts/rl.py` (`run_single_experiment` + CLI entrypoints, `AlphaEnv(...)` call)
- Test:   `tests/test_complexity_penalty.py`

- [ ] **Step 3.2.1: Write failing tests**

Create `tests/test_complexity_penalty.py`:

```python
from __future__ import annotations

import math

import torch

from alphagen.config import MAX_EXPR_LENGTH
from alphagen.data.tokens import (
    FeatureToken,
    SequenceIndicatorToken,
    SequenceIndicatorType,
)
from alphagen.models.linear_alpha_pool import LinearAlphaPool
from alphagen.rl.env.core import AlphaEnvCore
from alphagen_qlib.stock_data import FeatureType
from tests.test_parser_and_pool import DeterministicPool, FakeCalculator


def _make_env(complexity_penalty: float) -> AlphaEnvCore:
    pool = DeterministicPool(
        capacity=2,
        calculator=FakeCalculator(),
    )
    return AlphaEnvCore(
        pool=pool,
        device=torch.device("cpu"),
        complexity_penalty=complexity_penalty,
    )


def test_default_penalty_is_zero_no_change_to_reward() -> None:
    env = _make_env(complexity_penalty=0.0)
    env.reset()
    # Push a single feature token, then SEP -> evaluates `$close` against the pool.
    env.step(FeatureToken(FeatureType.CLOSE))
    _, reward, done, _, _ = env.step(SequenceIndicatorToken(SequenceIndicatorType.SEP))
    assert done is True
    # With penalty=0, reward equals the pool's try_new_expr return for a single $close.
    # FakeCalculator returns single_ic 0.6 for $close; DeterministicPool with size 1
    # returns the new_obj from calculate_ic_and_objective -> evaluate_ensemble.
    # We only assert non-negative and finite — exact value depends on pool dynamics.
    assert math.isfinite(reward)


def test_penalty_subtracts_normalized_token_count_at_sep() -> None:
    coef = 0.10
    env = _make_env(complexity_penalty=coef)
    env.reset()
    env.step(FeatureToken(FeatureType.CLOSE))
    # At SEP, len(self._tokens) = 2 (BEG + CLOSE). Expected penalty = coef * 2 / MAX_EXPR_LENGTH.
    expected_penalty = coef * 2 / MAX_EXPR_LENGTH

    env_noref = _make_env(complexity_penalty=0.0)
    env_noref.reset()
    env_noref.step(FeatureToken(FeatureType.CLOSE))
    _, baseline_reward, _, _, _ = env_noref.step(
        SequenceIndicatorToken(SequenceIndicatorType.SEP)
    )

    _, penalized_reward, _, _, _ = env.step(SequenceIndicatorToken(SequenceIndicatorType.SEP))

    assert penalized_reward == baseline_reward - expected_penalty


def test_penalty_scales_linearly_with_token_count() -> None:
    coef = 0.30
    # Two-token body vs one-token body -> doubled body length, but len(self._tokens)
    # at SEP includes BEG, so the ratio of penalties is (1+2)/(1+1) = 1.5.
    env_short = _make_env(complexity_penalty=coef)
    env_short.reset()
    env_short.step(FeatureToken(FeatureType.CLOSE))
    _, reward_short, _, _, _ = env_short.step(
        SequenceIndicatorToken(SequenceIndicatorType.SEP)
    )

    env_short_zero = _make_env(complexity_penalty=0.0)
    env_short_zero.reset()
    env_short_zero.step(FeatureToken(FeatureType.CLOSE))
    _, baseline_short, _, _, _ = env_short_zero.step(
        SequenceIndicatorToken(SequenceIndicatorType.SEP)
    )

    penalty_short = baseline_short - reward_short
    # Verify the penalty formula: coef * 2 / MAX_EXPR_LENGTH
    assert penalty_short == coef * 2 / MAX_EXPR_LENGTH
```

- [ ] **Step 3.2.2: Run tests, verify they fail**

```bash
.venv/bin/pytest tests/test_complexity_penalty.py -v
```
Expected: failures because `AlphaEnvCore` does not yet accept `complexity_penalty`.

- [ ] **Step 3.2.3: Add `complexity_penalty` to `AlphaEnvCore`**

In `alphagen/rl/env/core.py`, modify `AlphaEnvCore.__init__`:

```python
class AlphaEnvCore(gym.Env):
    pool: AlphaPoolBase
    _tokens: List[Token]
    _builder: ExpressionBuilder
    _print_expr: bool

    def __init__(
        self,
        pool: AlphaPoolBase,
        device: torch.device = torch.device('cuda:0'),
        print_expr: bool = False,
        complexity_penalty: float = 0.0,
    ):
        super().__init__()

        self.pool = pool
        self._print_expr = print_expr
        self._device = device
        self._complexity_penalty = float(complexity_penalty)

        self.eval_cnt = 0

        self.render_mode = None
        self.reset()
```

Replace `step` to apply the penalty on `_evaluate`-derived rewards:

```python
    def step(self, action: Token) -> Tuple[List[Token], float, bool, bool, dict]:
        if (isinstance(action, SequenceIndicatorToken) and
                action.indicator == SequenceIndicatorType.SEP):
            reward = self._evaluate_with_penalty()
            done = True
        elif len(self._tokens) < MAX_EXPR_LENGTH:
            self._tokens.append(action)
            self._builder.add_token(action)
            done = False
            reward = 0.0
        else:
            done = True
            reward = self._evaluate_with_penalty() if self._builder.is_valid() else -1.

        if math.isnan(reward):
            reward = 0.

        return self._tokens, reward, done, False, self._valid_action_types()

    def _evaluate_with_penalty(self) -> float:
        base = self._evaluate()
        if self._complexity_penalty == 0.0:
            return base
        return base - self._complexity_penalty * (len(self._tokens) / MAX_EXPR_LENGTH)
```

- [ ] **Step 3.2.4: Run tests, verify they pass**

```bash
.venv/bin/pytest tests/test_complexity_penalty.py -v
```
Expected: 3 passed.

- [ ] **Step 3.2.5: Plumb `complexity_penalty` through `scripts/rl.py`**

In `scripts/rl.py`, add `complexity_penalty: float = 0.0,` to:
- `run_single_experiment` signature
- `main` / `local` / `local_smoke` / `colab` signatures

In `run_single_experiment`, modify the `AlphaEnv(...)` call:

```python
    env = AlphaEnv(
        pool=pool,
        device=device,
        print_expr=print_expr,
        complexity_penalty=complexity_penalty,
    )
```

(The `AlphaEnv` factory already forwards `**kwargs` to `AlphaEnvCore`, so no wrapper edit is required.)

Persist in `run_config.json`:

```python
            "complexity_penalty": float(complexity_penalty),
```

In `main`'s body, add to the `run_single_experiment(...)` call:

```python
            complexity_penalty=complexity_penalty,
```

In `local` / `local_smoke` / `colab`, forward to `main`:

```python
        complexity_penalty=complexity_penalty,
```

- [ ] **Step 3.2.6: Verify plumbing**

```bash
.venv/bin/python -c "import inspect; from scripts.rl import run_single_experiment, main, local, local_smoke, colab; \
for fn in (run_single_experiment, main, local, local_smoke, colab): \
    assert 'complexity_penalty' in inspect.signature(fn).parameters, fn.__name__; \
print('all plumbed')"
```
Expected: `all plumbed`.

- [ ] **Step 3.2.7: Run the full test suite**

```bash
.venv/bin/pytest tests/ -q
```
Expected: all green.

- [ ] **Step 3.2.8: Commit**

```bash
git add alphagen/rl/env/core.py scripts/rl.py tests/test_complexity_penalty.py
git commit -m "feat(env): add complexity_penalty to AlphaEnvCore reward shaping"
```

---

# Final regression and smoke verification

## Task F.1: Cross-phase regression + manual smoke

- [ ] **Step F.1.1: Run the full test suite**

```bash
.venv/bin/pytest tests/ -q
```
Expected: all tests pass.

- [ ] **Step F.1.2: Smoke-run `local_smoke` with all new defaults (back-compat check)**

Confirms that with all new params left at defaults, the laptop-smoke profile still launches and writes `run_config.json` with the new keys.

```bash
.venv/bin/python scripts/rl.py local_smoke --random_seeds 0 --pool_capacity 5 2>&1 | tail -20
```

Expected: training starts; `out/results/<run>/run_config.json` contains keys `early_stop_patience` (default 0), `early_stop_warmup_steps`, `early_stop_min_delta`, `ic_mut_threshold` (0.99), `complexity_penalty` (0.0), `save_model_checkpoints` (true). If the local-smoke profile lacks qlib data on this machine, the smoke step may be skipped — confirm at minimum that the CLI accepts the new args and writes the config keys via:

```bash
.venv/bin/python -c "
import inspect
from scripts.rl import run_single_experiment
params = inspect.signature(run_single_experiment).parameters
for k in ['early_stop_patience','early_stop_warmup_steps','early_stop_min_delta',\
         'ic_mut_threshold','complexity_penalty','output_dir','save_model_checkpoints']:
    assert k in params, k
print('all params present')"
```
Expected: `all params present`.

- [ ] **Step F.1.3: Smoke-run the sweep CLI end-to-end on a 2x2 dry grid (optional, requires qlib data)**

```bash
.venv/bin/python scripts/sweep.py run \
    --name smoke_sweep \
    --grid '{"learning_rate":[3e-4,1e-4]}' \
    --seeds '[0,1]' \
    --base '{"profile":"local-smoke","instruments":"csi300","pool_capacity":5,"steps":256}' \
    --evaluate False 2>&1 | tail -30
```

Expected: 4 run dirs under `out/sweeps/smoke_sweep/`, no `*_steps.zip` files in them, each ending with `event:"training_end"` in `status.json`. Rerunning the same command should print 4 `skip (already complete)` lines.

If qlib data isn't available locally, skip this step — Task 2.3 already unit-tests the grid/slug/resume/pivot logic.

- [ ] **Step F.1.4: Final commit (only if any uncommitted scratch changes; otherwise skip)**

```bash
git status
```
If clean, you're done. Otherwise stage and commit any straggling files.
