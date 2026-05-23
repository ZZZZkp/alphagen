from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


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

    def snapshot(self) -> dict[str, Any]:
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
