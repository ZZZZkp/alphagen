import json
import os
import sys
import time
from importlib.util import find_spec
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alphagen.utils import configure_project_runtime

configure_project_runtime(ROOT)

import fire
import numpy as np
import torch
from openai import OpenAI
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from alphagen.data.expression import *
from alphagen.data.parser import ExpressionParser
from alphagen.models.linear_alpha_pool import LinearAlphaPool, MseAlphaPool
from alphagen.rl.env.core import AlphaEnvCore
from alphagen.rl.env.wrapper import AlphaEnv
from alphagen.rl.policy import LSTMSharedNet
from alphagen.utils import get_logger, reseed_everything
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen_qlib.stock_data import initialize_qlib
from alphagen_llm.client import ChatClient, ChatConfig, OpenAIClient
from alphagen_llm.prompts.interaction import DefaultInteraction, InterativeSession
from alphagen_llm.prompts.system_prompt import EXPLAIN_WITH_TEXT_DESC


DEFAULT_SEGMENTS: Tuple[Tuple[str, str], ...] = (
    ("2012-01-01", "2021-12-31"),
    ("2022-01-01", "2022-06-30"),
    ("2022-07-01", "2022-12-31"),
    ("2023-01-01", "2023-06-30"),
)

LOCAL_SMOKE_SEGMENTS: Tuple[Tuple[str, str], ...] = (
    ("2019-01-01", "2020-12-31"),
    ("2021-01-01", "2021-06-30"),
    ("2021-07-01", "2021-12-31"),
)

DEFAULT_STEPS: Dict[int, int] = {
    10: 200_000,
    20: 250_000,
    50: 300_000,
    100: 350_000,
}

LOCAL_STEPS: Dict[int, int] = {
    10: 20_000,
    20: 35_000,
    50: 50_000,
    100: 75_000,
}

LOCAL_SMOKE_STEPS: Dict[int, int] = {
    5: 64,
    10: 128,
    20: 256,
}

STOCK_DATA_MAX_BACKTRACK_DAYS = 100
STOCK_DATA_MAX_FUTURE_DAYS = 30


@dataclass(frozen=True)
class RLProfile:
    name: str
    description: str
    qlib_candidates: Tuple[str, ...]
    default_pool_capacity: int
    default_steps: Dict[int, int]
    prefer_cuda: bool = False
    prefer_mps: bool = False
    segments: Tuple[Tuple[str, str], ...] = DEFAULT_SEGMENTS
    default_ppo_n_steps: int = 2048
    default_batch_size: int = 128
    print_expr: bool = True


PROFILES: Dict[str, RLProfile] = {
    "default": RLProfile(
        name="default",
        description="Repository defaults with automatic device and data-path selection.",
        qlib_candidates=(
            "~/.qlib/qlib_data/cn_data",
            "~/.qlib/qlib_data/cn_data_2024h1",
        ),
        default_pool_capacity=20,
        default_steps=DEFAULT_STEPS,
    ),
    "local": RLProfile(
        name="local",
        description="Laptop-friendly defaults for local iteration on Apple Silicon or CPU.",
        qlib_candidates=(
            "~/.qlib/qlib_data/cn_data_2024h1",
            "~/.qlib/qlib_data/cn_data",
        ),
        default_pool_capacity=10,
        default_steps=LOCAL_STEPS,
        prefer_mps=True,
        default_ppo_n_steps=128,
        default_batch_size=64,
        print_expr=False,
    ),
    "local-smoke": RLProfile(
        name="local-smoke",
        description="Very small local sanity-check profile for Macs and CPU-only runs.",
        qlib_candidates=(
            "~/.qlib/qlib_data/cn_data_2024h1",
            "~/.qlib/qlib_data/cn_data",
        ),
        default_pool_capacity=5,
        default_steps=LOCAL_SMOKE_STEPS,
        prefer_mps=True,
        segments=LOCAL_SMOKE_SEGMENTS,
        default_ppo_n_steps=64,
        default_batch_size=32,
        print_expr=False,
    ),
    "colab": RLProfile(
        name="colab",
        description="Colab-oriented defaults with CUDA-first device selection and broader path probing.",
        qlib_candidates=(
            "/content/qlib_data/cn_data_2024h1",
            "/content/qlib_data/cn_data",
            "/content/drive/MyDrive/qlib_data/cn_data_2024h1",
            "/content/drive/MyDrive/qlib_data/cn_data",
            "~/.qlib/qlib_data/cn_data_2024h1",
            "~/.qlib/qlib_data/cn_data",
        ),
        default_pool_capacity=20,
        default_steps=DEFAULT_STEPS,
        prefer_cuda=True,
        default_ppo_n_steps=2048,
        default_batch_size=128,
    ),
}


def read_alphagpt_init_pool(seed: int) -> List[Expression]:
    DIR = "./out/llm-tests/interaction"
    parser = build_parser()
    for path in Path(DIR).glob(f"v0_{seed}*"):
        with open(path / "report.json") as f:
            data = json.load(f)
            pool_state = data[-1]["pool_state"]
            return [parser.parse(expr) for expr, _ in pool_state]
    return []


def build_parser() -> ExpressionParser:
    return ExpressionParser(
        Operators,
        ignore_case=True,
        non_positive_time_deltas_allowed=False,
        additional_operator_mapping={
            "Max": [Greater],
            "Min": [Less],
            "Delta": [Sub]
        }
    )


def build_chat_client(log_dir: str) -> ChatClient:
    logger = get_logger("llm", os.path.join(log_dir, "llm.log"))
    return OpenAIClient(
        client=OpenAI(base_url="https://api.ai.cs.ac.cn/v1"),
        config=ChatConfig(
            system_prompt=EXPLAIN_WITH_TEXT_DESC,
            logger=logger
        )
    )


def get_profile(name: str = "default") -> RLProfile:
    if name not in PROFILES:
        supported = ", ".join(sorted(PROFILES))
        raise ValueError(f"Unknown RL profile '{name}'. Supported profiles: {supported}")
    return PROFILES[name]


def list_profiles() -> Dict[str, Dict[str, Union[str, int, List[str]]]]:
    return {
        name: {
            "description": profile.description,
            "default_pool_capacity": profile.default_pool_capacity,
            "default_steps": dict(profile.default_steps),
            "qlib_candidates": list(profile.qlib_candidates),
            "default_ppo_n_steps": profile.default_ppo_n_steps,
            "default_batch_size": profile.default_batch_size,
        }
        for name, profile in PROFILES.items()
    }


def _mps_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def resolve_device(device: Optional[str], profile: RLProfile) -> torch.device:
    if device is not None:
        return torch.device(device)
    if profile.prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda:0")
    if profile.prefer_mps and _mps_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if _mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_qlib_data_path(qlib_data_path: Optional[str], profile: RLProfile) -> str:
    if qlib_data_path is not None:
        return os.path.expanduser(qlib_data_path)
    for candidate in profile.qlib_candidates:
        expanded = os.path.expanduser(candidate)
        if Path(expanded).exists():
            return expanded
    return os.path.expanduser(profile.qlib_candidates[0])


def validate_qlib_calendar(
    qlib_data_path: str,
    segments: Sequence[Tuple[str, str]],
    max_backtrack_days: int = 0,
    max_future_days: int = 0,
) -> None:
    calendar_path = Path(qlib_data_path).expanduser() / "calendars" / "day.txt"
    if not calendar_path.exists():
        return
    dates: List[str] = []
    with open(calendar_path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line == "":
                continue
            dates.append(line)
    if not dates:
        return
    if len(dates) <= max_backtrack_days + max_future_days:
        raise ValueError(
            "The selected Qlib dataset is too short for the configured StockData padding: "
            f"{len(dates)} calendar rows for backtrack={max_backtrack_days}, future={max_future_days}."
        )

    first = dates[0]
    last = dates[-1]
    first_usable = dates[max_backtrack_days]
    last_usable = dates[-1 - max_future_days] if max_future_days > 0 else dates[-1]
    earliest = min(start for start, _ in segments)
    latest = max(end for _, end in segments)
    if first_usable > earliest or last_usable < latest:
        raise ValueError(
            "The selected Qlib dataset does not cover the requested training/test segments once "
            "StockData padding is applied: "
            f"calendar range [{first}, {last}], usable range [{first_usable}, {last_usable}], "
            f"required [{earliest}, {latest}] with backtrack={max_backtrack_days}, "
            f"future={max_future_days}. Pass a different --qlib_data_path or shrink the segments."
        )


def resolve_tensorboard_log(default_path: str = "./out/tensorboard") -> Optional[str]:
    if find_spec("tensorboard") is None:
        return None
    return default_path


def _write_json(path: str, payload: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def latest_run(results_root: str = "./out/results") -> Optional[str]:
    root = Path(results_root)
    if not root.exists():
        return None
    dirs = [p for p in root.iterdir() if p.is_dir()]
    if not dirs:
        return None
    return str(max(dirs, key=lambda p: p.stat().st_mtime))


def status(run_dir: Optional[str] = None, results_root: str = "./out/results") -> Dict[str, object]:
    chosen = latest_run(results_root) if run_dir is None else run_dir
    if chosen is None:
        raise ValueError(f"No run directory found under {results_root}")
    status_path = Path(chosen) / "status.json"
    if not status_path.exists():
        raise ValueError(f"No status.json found in {chosen}")
    with open(status_path, encoding="utf-8") as f:
        payload = json.load(f)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


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
    ):
        super().__init__(verbose)
        self.save_path = save_path
        self.test_calculators = test_calculators
        os.makedirs(self.save_path, exist_ok=True)
        self._status_path = os.path.join(self.save_path, "status.json")
        self._monitor_log_path = os.path.join(self.save_path, "monitor.log")

        self.llm_use_count = 0
        self.last_llm_use = 0
        self.obj_history: List[Tuple[int, float]] = []
        self.llm_every_n_steps = llm_every_n_steps
        self.chat_session = chat_session
        self._drop_rl_n = drop_rl_n
        self._heartbeat_every_n_steps = heartbeat_every_n_steps
        self._heartbeat_every_seconds = heartbeat_every_seconds
        self._last_heartbeat_step = 0
        self._last_heartbeat_time = time.time()
        self._started_at = time.time()
        self._checkpoint_every_n_rollouts = max(1, int(checkpoint_every_n_rollouts))
        self._model_checkpoint_start_step = max(0, int(model_checkpoint_start_step))
        self._rollout_count = 0
        self._last_pool_checkpoint_step = -1
        self._last_model_checkpoint_step = -1

    def _on_step(self) -> bool:
        now = time.time()
        step_delta = self.num_timesteps - self._last_heartbeat_step
        time_delta = now - self._last_heartbeat_time
        if step_delta >= self._heartbeat_every_n_steps or time_delta >= self._heartbeat_every_seconds:
            self._write_status("heartbeat")
            self._last_heartbeat_step = self.num_timesteps
            self._last_heartbeat_time = now
        return True

    def _on_training_start(self) -> None:
        self._write_status("training_start")

    def _on_rollout_end(self) -> None:
        self._rollout_count += 1
        if self.chat_session is not None:
            self._try_use_llm()

        self.logger.record('pool/size', self.pool.size)
        self.logger.record('pool/significant', (np.abs(self.pool.weights[:self.pool.size]) > 1e-4).sum())
        self.logger.record('pool/best_ic_ret', self.pool.best_ic_ret)
        self.logger.record('pool/eval_cnt', self.pool.eval_cnt)
        n_days = sum(calculator.data.n_days for calculator in self.test_calculators)
        ic_test_mean, rank_ic_test_mean = 0., 0.
        for i, test_calculator in enumerate(self.test_calculators, start=1):
            ic_test, rank_ic_test = self.pool.test_ensemble(test_calculator)
            ic_test_mean += ic_test * test_calculator.data.n_days / n_days
            rank_ic_test_mean += rank_ic_test * test_calculator.data.n_days / n_days
            self.logger.record(f'test/ic_{i}', ic_test)
            self.logger.record(f'test/rank_ic_{i}', rank_ic_test)
        self.logger.record(f'test/ic_mean', ic_test_mean)
        self.logger.record(f'test/rank_ic_mean', rank_ic_test_mean)
        self.save_checkpoint()
        self._write_status(
            "rollout_end",
            test_ic_mean=ic_test_mean,
            test_rank_ic_mean=rank_ic_test_mean,
        )

    def _on_training_end(self) -> None:
        self.save_checkpoint(force=True)
        self._write_status("training_end")

    def save_checkpoint(self, force: bool = False) -> None:
        if not force and self._rollout_count % self._checkpoint_every_n_rollouts != 0:
            return
        path = os.path.join(self.save_path, f'{self.num_timesteps}_steps')
        pool_saved = False
        model_saved = False
        if self.num_timesteps != self._last_pool_checkpoint_step:
            with open(f'{path}_pool.json', 'w') as f:
                json.dump(self.pool.to_json_dict(), f)
            self._last_pool_checkpoint_step = self.num_timesteps
            pool_saved = True
        if (
            self.num_timesteps >= self._model_checkpoint_start_step and
            self.num_timesteps != self._last_model_checkpoint_step
        ):
            self.model.save(path)   # type: ignore
            self._last_model_checkpoint_step = self.num_timesteps
            model_saved = True
            if self.verbose > 1:
                print(f'Saving model checkpoint to {path}')
        if pool_saved or model_saved:
            saved_parts = []
            if pool_saved:
                saved_parts.append("pool")
            if model_saved:
                saved_parts.append("model")
            self._append_monitor_log(f"checkpoint saved ({'+'.join(saved_parts)}): {path}")

    def show_pool_state(self):
        state = self.pool.state
        print('---------------------------------------------')
        for i in range(self.pool.size):
            weight = state['weights'][i]
            expr_str = str(state['exprs'][i])
            ic_ret = state['ics_ret'][i]
            print(f'> Alpha #{i}: {weight}, {expr_str}, {ic_ret}')
        print(f'>> Ensemble ic_ret: {state["best_ic_ret"]}')
        print('---------------------------------------------')

    def _try_use_llm(self) -> None:
        n_steps = self.num_timesteps
        if n_steps - self.last_llm_use < self.llm_every_n_steps:
            return
        self.last_llm_use = n_steps
        self.llm_use_count += 1
        
        assert self.chat_session is not None
        self.chat_session.client.reset()
        logger = self.chat_session.logger
        logger.debug(
            f"[Step: {n_steps}] Trying to invoke LLM (#{self.llm_use_count}): "
            f"IC={self.pool.best_ic_ret:.4f}, obj={self.pool.best_ic_ret:.4f}")

        try:
            remain_n = max(0, self.pool.size - self._drop_rl_n)
            remain = self.pool.most_significant_indices(remain_n)
            self.pool.leave_only(remain)
            self.chat_session.update_pool(self.pool)
        except Exception as e:
            logger.warning(f"LLM invocation failed due to {type(e)}: {str(e)}")

    def _append_monitor_log(self, message: str) -> None:
        timestamp = datetime.now().isoformat(timespec="seconds")
        with open(self._monitor_log_path, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")

    def _write_status(self, event: str, **extra: object) -> None:
        payload: Dict[str, object] = {
            "event": event,
            "save_path": self.save_path,
            "num_timesteps": int(self.num_timesteps),
            "pool_size": int(self.pool.size),
            "pool_eval_cnt": int(self.pool.eval_cnt),
            "pool_best_ic_ret": float(self.pool.best_ic_ret),
            "significant_count": int((np.abs(self.pool.weights[:self.pool.size]) > 1e-4).sum()),
            "elapsed_seconds": round(time.time() - self._started_at, 2),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        payload.update(extra)
        _write_json(self._status_path, payload)
        self._append_monitor_log(
            f"{event}: steps={payload['num_timesteps']} pool={payload['pool_size']} "
            f"evals={payload['pool_eval_cnt']} best_ic={payload['pool_best_ic_ret']:.4f}"
        )

    @property
    def pool(self) -> LinearAlphaPool:
        assert(isinstance(self.env_core.pool, LinearAlphaPool))
        return self.env_core.pool

    @property
    def env_core(self) -> AlphaEnvCore:
        return self.training_env.envs[0].unwrapped  # type: ignore


def _build_learning_rate(rate: float, schedule: str):
    schedule_normalized = (schedule or "constant").lower()
    rate_value = float(rate)
    if schedule_normalized == "constant":
        return rate_value
    if schedule_normalized == "linear":
        return lambda progress_remaining: rate_value * progress_remaining
    raise ValueError(
        f"Unknown lr_schedule={schedule!r}. Expected 'constant' or 'linear'."
    )


def run_single_experiment(
    seed: int = 0,
    instruments: str = "csi300",
    pool_capacity: int = 10,
    steps: int = 200_000,
    alphagpt_init: bool = False,
    use_llm: bool = False,
    llm_every_n_steps: int = 25_000,
    drop_rl_n: int = 5,
    llm_replace_n: int = 3,
    qlib_data_path: str = "~/.qlib/qlib_data/cn_data",
    qlib_region: str = "cn",
    device: Optional[torch.device] = None,
    segments: Sequence[Tuple[str, str]] = DEFAULT_SEGMENTS,
    ppo_n_steps: int = 2048,
    batch_size: int = 128,
    print_expr: bool = True,
    checkpoint_every_n_rollouts: int = 1,
    model_checkpoint_start_step: int = 0,
    learning_rate: float = 3e-4,
    lr_schedule: str = "constant",
    ent_coef: float = 0.01,
    clip_range: float = 0.2,
) -> str:
    reseed_everything(seed)
    validate_qlib_calendar(
        qlib_data_path,
        segments,
        max_backtrack_days=STOCK_DATA_MAX_BACKTRACK_DAYS,
        max_future_days=STOCK_DATA_MAX_FUTURE_DAYS,
    )
    initialize_qlib(qlib_data_path, region=qlib_region)

    llm_replace_n = 0 if not use_llm else llm_replace_n
    print(f"""[Main] Starting training process
    Seed: {seed}
    Instruments: {instruments}
    Pool capacity: {pool_capacity}
    Total Iteration Steps: {steps}
    AlphaGPT-Like Init-Only LLM Usage: {alphagpt_init}
    Use LLM: {use_llm}
    Invoke LLM every N steps: {llm_every_n_steps}
    Replace N alphas with LLM: {llm_replace_n}
    Drop N alphas before LLM: {drop_rl_n}
    Qlib data path: {qlib_data_path}
    Qlib region: {qlib_region}
    Device: {device or 'auto'}""")

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    # tag = "rlv2" if llm_add_subexpr == 0 else f"afs{llm_add_subexpr}aar1-5"
    tag = (
        "agpt" if alphagpt_init else
        "rl" if not use_llm else
        f"llm_d{drop_rl_n}")
    name_prefix = f"{instruments}_{pool_capacity}_{seed}_{timestamp}_{tag}"
    save_path = os.path.join("./out/results", name_prefix)
    os.makedirs(save_path, exist_ok=True)
    _write_json(
        os.path.join(save_path, "run_config.json"),
        {
            "seed": seed,
            "instruments": instruments,
            "pool_capacity": pool_capacity,
            "steps": steps,
            "alphagpt_init": alphagpt_init,
            "use_llm": use_llm,
            "llm_every_n_steps": llm_every_n_steps,
            "drop_rl_n": drop_rl_n,
            "llm_replace_n": llm_replace_n,
            "qlib_data_path": qlib_data_path,
            "qlib_region": qlib_region,
            "device": str(device or "auto"),
            "segments": list(segments),
            "ppo_n_steps": ppo_n_steps,
            "batch_size": batch_size,
            "print_expr": print_expr,
            "checkpoint_every_n_rollouts": checkpoint_every_n_rollouts,
            "model_checkpoint_start_step": model_checkpoint_start_step,
            "learning_rate": float(learning_rate),
            "lr_schedule": lr_schedule,
            "ent_coef": float(ent_coef),
            "clip_range": float(clip_range),
        },
    )
    _write_json(
        os.path.join(save_path, "status.json"),
        {
            "event": "created",
            "save_path": save_path,
            "num_timesteps": 0,
            "pool_size": 0,
            "pool_eval_cnt": 0,
            "pool_best_ic_ret": -1.0,
            "significant_count": 0,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        },
    )

    device = device or torch.device("cpu")
    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -20) / close - 1

    def get_dataset(start: str, end: str) -> StockData:
        return StockData(
            instrument=instruments,
            start_time=start,
            end_time=end,
            max_backtrack_days=STOCK_DATA_MAX_BACKTRACK_DAYS,
            max_future_days=STOCK_DATA_MAX_FUTURE_DAYS,
            device=device
        )

    datasets = [get_dataset(*s) for s in segments]
    calculators = [QLibStockDataCalculator(d, target) for d in datasets]

    def build_pool(exprs: List[Expression]) -> LinearAlphaPool:
        pool = MseAlphaPool(
            capacity=pool_capacity,
            calculator=calculators[0],
            ic_lower_bound=None,
            l1_alpha=5e-3,
            device=device
        )
        if len(exprs) != 0:
            pool.force_load_exprs(exprs)
        return pool

    chat, inter, pool = None, None, build_pool([])
    if alphagpt_init:
        pool = build_pool(read_alphagpt_init_pool(seed))
    elif use_llm:
        chat = build_chat_client(save_path)
        inter = DefaultInteraction(
            build_parser(), chat, build_pool,
            calculator_train=calculators[0], calculators_test=calculators[1:],
            replace_k=llm_replace_n, forgetful=True
        )
        pool = inter.run()

    env = AlphaEnv(
        pool=pool,
        device=device,
        print_expr=print_expr
    )
    checkpoint_callback = CustomCallback(
        save_path=save_path,
        test_calculators=calculators[1:],
        verbose=1,
        chat_session=inter,
        llm_every_n_steps=llm_every_n_steps,
        drop_rl_n=drop_rl_n,
        checkpoint_every_n_rollouts=checkpoint_every_n_rollouts,
        model_checkpoint_start_step=model_checkpoint_start_step,
    )
    model = MaskablePPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(
            features_extractor_class=LSTMSharedNet,
            features_extractor_kwargs=dict(
                n_layers=2,
                d_model=128,
                dropout=0.1,
                device=device,
            ),
        ),
        gamma=1.,
        ent_coef=float(ent_coef),
        clip_range=float(clip_range),
        n_steps=ppo_n_steps,
        batch_size=batch_size,
        learning_rate=_build_learning_rate(learning_rate, lr_schedule),
        tensorboard_log=resolve_tensorboard_log(),
        device=device,
        verbose=1,
    )
    model.learn(
        total_timesteps=steps,
        callback=checkpoint_callback,
        tb_log_name=name_prefix,
    )
    return save_path


def main(
    random_seeds: Union[int, Tuple[int]] = 0,
    pool_capacity: Optional[int] = None,
    instruments: str = "csi300",
    alphagpt_init: bool = False,
    use_llm: bool = False,
    drop_rl_n: int = 10,
    steps: Optional[int] = None,
    llm_every_n_steps: int = 25000,
    profile: str = "default",
    device: Optional[str] = None,
    qlib_data_path: Optional[str] = None,
    qlib_region: str = "cn",
    ppo_n_steps: Optional[int] = None,
    batch_size: Optional[int] = None,
    print_expr: Optional[bool] = None,
    checkpoint_every_n_rollouts: int = 1,
    model_checkpoint_start_step: int = 0,
    learning_rate: float = 3e-4,
    lr_schedule: str = "constant",
    ent_coef: float = 0.01,
    clip_range: float = 0.2,
):
    """
    :param random_seeds: Random seeds
    :param pool_capacity: Maximum size of the alpha pool
    :param instruments: Stock subset name
    :param alphagpt_init: Use an alpha set pre-generated by LLM as the initial pool
    :param use_llm: Enable LLM usage
    :param drop_rl_n: Drop n worst alphas before invoke the LLM
    :param steps: Total iteration steps
    :param llm_every_n_steps: Invoke LLM every n steps
    :param profile: Runtime profile name. Supported: default, local, local-smoke, colab
    :param device: Optional PyTorch device string, e.g. cpu, mps, cuda:0
    :param qlib_data_path: Optional Qlib data directory override
    :param qlib_region: Qlib market region, e.g. cn or us
    :param ppo_n_steps: PPO rollout length before each optimization phase
    :param batch_size: PPO minibatch size
    :param print_expr: Whether to print each generated expression
    :param checkpoint_every_n_rollouts: Save checkpoints every n rollout ends
    :param model_checkpoint_start_step: Start saving model weights once this timestep is reached
    :param learning_rate: PPO learning rate (start value when lr_schedule != 'constant')
    :param lr_schedule: 'constant' or 'linear' (decay learning_rate -> 0 over total steps)
    :param ent_coef: PPO entropy coefficient (higher = more exploration)
    :param clip_range: PPO policy-update clip threshold (lower = more conservative updates)
    """
    rl_profile = get_profile(profile)
    selected_pool_capacity = rl_profile.default_pool_capacity if pool_capacity is None else int(pool_capacity)
    selected_steps = rl_profile.default_steps
    resolved_device = resolve_device(device, rl_profile)
    resolved_qlib_data_path = resolve_qlib_data_path(qlib_data_path, rl_profile)
    validate_qlib_calendar(
        resolved_qlib_data_path,
        rl_profile.segments,
        max_backtrack_days=STOCK_DATA_MAX_BACKTRACK_DAYS,
        max_future_days=STOCK_DATA_MAX_FUTURE_DAYS,
    )
    resolved_ppo_n_steps = rl_profile.default_ppo_n_steps if ppo_n_steps is None else int(ppo_n_steps)
    resolved_batch_size = rl_profile.default_batch_size if batch_size is None else int(batch_size)
    resolved_print_expr = rl_profile.print_expr if print_expr is None else bool(print_expr)
    if steps is None and selected_pool_capacity not in selected_steps:
        supported = ", ".join(str(k) for k in sorted(selected_steps))
        raise ValueError(
            f"Pool capacity {selected_pool_capacity} has no default step count in profile "
            f"'{rl_profile.name}'. Supported capacities: {supported}. Pass --steps explicitly."
        )
    if resolved_batch_size > resolved_ppo_n_steps:
        raise ValueError(
            f"batch_size ({resolved_batch_size}) must be <= ppo_n_steps ({resolved_ppo_n_steps}) "
            "for stable local runs. Pass a smaller --batch_size or larger --ppo_n_steps."
        )

    if isinstance(random_seeds, int):
        random_seeds = (random_seeds, )
    for s in random_seeds:
        run_single_experiment(
            seed=s,
            instruments=instruments,
            pool_capacity=selected_pool_capacity,
            steps=selected_steps[int(selected_pool_capacity)] if steps is None else int(steps),
            alphagpt_init=alphagpt_init,
            drop_rl_n=drop_rl_n,
            use_llm=use_llm,
            llm_every_n_steps=llm_every_n_steps,
            qlib_data_path=resolved_qlib_data_path,
            qlib_region=qlib_region,
            device=resolved_device,
            segments=rl_profile.segments,
            ppo_n_steps=resolved_ppo_n_steps,
            batch_size=resolved_batch_size,
            print_expr=resolved_print_expr,
            checkpoint_every_n_rollouts=checkpoint_every_n_rollouts,
            model_checkpoint_start_step=model_checkpoint_start_step,
            learning_rate=learning_rate,
            lr_schedule=lr_schedule,
            ent_coef=ent_coef,
            clip_range=clip_range,
        )


def local(
    random_seeds: Union[int, Tuple[int]] = 0,
    pool_capacity: Optional[int] = None,
    instruments: str = "csi300",
    alphagpt_init: bool = False,
    use_llm: bool = False,
    drop_rl_n: int = 10,
    steps: Optional[int] = None,
    llm_every_n_steps: int = 25000,
    device: Optional[str] = None,
    qlib_data_path: Optional[str] = None,
    qlib_region: str = "cn",
    ppo_n_steps: Optional[int] = None,
    batch_size: Optional[int] = None,
    print_expr: Optional[bool] = None,
    checkpoint_every_n_rollouts: int = 1,
    model_checkpoint_start_step: int = 0,
    learning_rate: float = 3e-4,
    lr_schedule: str = "constant",
    ent_coef: float = 0.01,
    clip_range: float = 0.2,
):
    return main(
        random_seeds=random_seeds,
        pool_capacity=pool_capacity,
        instruments=instruments,
        alphagpt_init=alphagpt_init,
        use_llm=use_llm,
        drop_rl_n=drop_rl_n,
        steps=steps,
        llm_every_n_steps=llm_every_n_steps,
        profile="local",
        device=device,
        qlib_data_path=qlib_data_path,
        qlib_region=qlib_region,
        ppo_n_steps=ppo_n_steps,
        batch_size=batch_size,
        print_expr=print_expr,
        checkpoint_every_n_rollouts=checkpoint_every_n_rollouts,
        model_checkpoint_start_step=model_checkpoint_start_step,
        learning_rate=learning_rate,
        lr_schedule=lr_schedule,
        ent_coef=ent_coef,
        clip_range=clip_range,
    )


def local_smoke(
    random_seeds: Union[int, Tuple[int]] = 0,
    pool_capacity: Optional[int] = None,
    instruments: str = "csi300",
    alphagpt_init: bool = False,
    use_llm: bool = False,
    drop_rl_n: int = 10,
    steps: Optional[int] = None,
    llm_every_n_steps: int = 25000,
    device: Optional[str] = None,
    qlib_data_path: Optional[str] = None,
    qlib_region: str = "cn",
    ppo_n_steps: Optional[int] = None,
    batch_size: Optional[int] = None,
    print_expr: Optional[bool] = None,
    checkpoint_every_n_rollouts: int = 1,
    model_checkpoint_start_step: int = 0,
    learning_rate: float = 3e-4,
    lr_schedule: str = "constant",
    ent_coef: float = 0.01,
    clip_range: float = 0.2,
):
    return main(
        random_seeds=random_seeds,
        pool_capacity=pool_capacity,
        instruments=instruments,
        alphagpt_init=alphagpt_init,
        use_llm=use_llm,
        drop_rl_n=drop_rl_n,
        steps=steps,
        llm_every_n_steps=llm_every_n_steps,
        profile="local-smoke",
        device=device,
        qlib_data_path=qlib_data_path,
        qlib_region=qlib_region,
        ppo_n_steps=ppo_n_steps,
        batch_size=batch_size,
        print_expr=print_expr,
        checkpoint_every_n_rollouts=checkpoint_every_n_rollouts,
        model_checkpoint_start_step=model_checkpoint_start_step,
        learning_rate=learning_rate,
        lr_schedule=lr_schedule,
        ent_coef=ent_coef,
        clip_range=clip_range,
    )


def colab(
    random_seeds: Union[int, Tuple[int]] = 0,
    pool_capacity: Optional[int] = None,
    instruments: str = "csi300",
    alphagpt_init: bool = False,
    use_llm: bool = False,
    drop_rl_n: int = 10,
    steps: Optional[int] = None,
    llm_every_n_steps: int = 25000,
    device: Optional[str] = None,
    qlib_data_path: Optional[str] = None,
    qlib_region: str = "cn",
    ppo_n_steps: Optional[int] = None,
    batch_size: Optional[int] = None,
    print_expr: Optional[bool] = None,
    checkpoint_every_n_rollouts: int = 1,
    model_checkpoint_start_step: int = 0,
    learning_rate: float = 3e-4,
    lr_schedule: str = "constant",
    ent_coef: float = 0.01,
    clip_range: float = 0.2,
):
    return main(
        random_seeds=random_seeds,
        pool_capacity=pool_capacity,
        instruments=instruments,
        alphagpt_init=alphagpt_init,
        use_llm=use_llm,
        drop_rl_n=drop_rl_n,
        steps=steps,
        llm_every_n_steps=llm_every_n_steps,
        profile="colab",
        device=device,
        qlib_data_path=qlib_data_path,
        qlib_region=qlib_region,
        ppo_n_steps=ppo_n_steps,
        batch_size=batch_size,
        print_expr=print_expr,
        checkpoint_every_n_rollouts=checkpoint_every_n_rollouts,
        model_checkpoint_start_step=model_checkpoint_start_step,
        learning_rate=learning_rate,
        lr_schedule=lr_schedule,
        ent_coef=ent_coef,
        clip_range=clip_range,
    )


if __name__ == '__main__':
    fire.Fire(
        {
            "main": main,
            "local": local,
            "local_smoke": local_smoke,
            "colab": colab,
            "profiles": list_profiles,
            "status": status,
        }
    )
