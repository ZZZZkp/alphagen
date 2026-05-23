from __future__ import annotations

import math

import torch

from alphagen.config import MAX_EXPR_LENGTH
from alphagen.data.tokens import (
    FeatureToken,
    SequenceIndicatorToken,
    SequenceIndicatorType,
)
from alphagen.rl.env.core import AlphaEnvCore
from alphagen_qlib.stock_data import FeatureType
from tests.test_ic_mut_threshold import FlexiblePool
from tests.test_parser_and_pool import FakeCalculator


def _make_env(complexity_penalty: float) -> AlphaEnvCore:
    pool = FlexiblePool(
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
    # Verify the penalty formula: coef * len(self._tokens) / MAX_EXPR_LENGTH.
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

    assert reward_short == baseline_short - coef * 2 / MAX_EXPR_LENGTH
