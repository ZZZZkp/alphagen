from __future__ import annotations

import numpy as np
import pytest

from alphagen.data.calculator import AlphaCalculator
from alphagen.data.expression import Add, Feature, Greater
from alphagen.data.parser import ExpressionParsingError
from alphagen.data.tokens import DeltaTimeToken, FeatureToken, OperatorToken
from alphagen.data.tree import ExpressionBuilder
from alphagen.models.linear_alpha_pool import LinearAlphaPool
from alphagen_qlib.stock_data import FeatureType
from scripts.rl import build_parser


class FakeCalculator(AlphaCalculator):
    def __init__(self) -> None:
        self._single = {
            "$close": 0.6,
            "$open": 0.4,
            "Add($close,1.0)": 0.2,
        }
        self._mutual = {
            frozenset({"$close", "$open"}): 0.1,
            frozenset({"$close", "Add($close,1.0)"}): 0.05,
            frozenset({"$open", "Add($close,1.0)"}): 0.05,
        }

    def calc_single_IC_ret(self, expr) -> float:
        return self._single[str(expr)]

    def calc_single_rIC_ret(self, expr) -> float:
        return self._single[str(expr)] / 2

    def calc_mutual_IC(self, expr1, expr2) -> float:
        key = frozenset({str(expr1), str(expr2)})
        if len(key) == 1:
            return 1.0
        return self._mutual[key]

    def calc_pool_IC_ret(self, exprs, weights) -> float:
        return float(sum(self._single[str(expr)] * weight for expr, weight in zip(exprs, weights)))

    def calc_pool_rIC_ret(self, exprs, weights) -> float:
        return self.calc_pool_IC_ret(exprs, weights) / 2

    def calc_pool_all_ret(self, exprs, weights):
        ic = self.calc_pool_IC_ret(exprs, weights)
        return ic, ic / 2


class DeterministicPool(LinearAlphaPool):
    def optimize(self, lr: float = 5e-4, max_steps: int = 10000, tolerance: int = 500) -> np.ndarray:
        if self.size == 2:
            return np.array([0.7, 0.3])
        if self.size == 3:
            return np.array([0.7, 0.2, 0.01])
        raise AssertionError(f"Unexpected pool size: {self.size}")

    def _calc_main_objective(self):
        return None


def test_build_parser_supports_current_aliases() -> None:
    parser = build_parser()

    expr = parser.parse("max(close, open)")

    assert isinstance(expr, Greater)
    assert str(expr) == "Greater($close,$open)"


def test_build_parser_rejects_non_positive_time_deltas() -> None:
    parser = build_parser()

    with pytest.raises(ExpressionParsingError):
        parser.parse("Ref(close,-1d)")


def test_expression_builder_creates_ref_expression_from_rl_tokens() -> None:
    builder = ExpressionBuilder()

    builder.add_token(FeatureToken(FeatureType.CLOSE))
    builder.add_token(DeltaTimeToken(5))
    builder.add_token(OperatorToken(build_parser().parse("Ref(close,5d)").__class__))

    expr = builder.get_tree()
    assert str(expr) == "Ref($close,5d)"
    assert builder.is_valid() is True


def test_linear_alpha_pool_rejects_new_worst_expression_and_caches_failure() -> None:
    pool = DeterministicPool(capacity=2, calculator=FakeCalculator())
    first = Feature(FeatureType.CLOSE)
    second = Feature(FeatureType.OPEN)
    rejected = Add(Feature(FeatureType.CLOSE), 1.0)

    pool.force_load_exprs([first, second])
    best_before = pool.best_obj
    eval_before = pool.eval_cnt

    returned = pool.try_new_expr(rejected)

    assert returned == best_before
    assert pool.eval_cnt == eval_before + 1
    assert pool.size == 2
    assert [str(expr) for expr in pool.exprs[:pool.size]] == ["$close", "$open"]

    returned_again = pool.try_new_expr(rejected)

    assert returned_again == best_before
    assert pool.eval_cnt == eval_before + 1


def test_linear_alpha_pool_test_ensemble_returns_neutral_metrics_for_empty_pool() -> None:
    pool = DeterministicPool(capacity=2, calculator=FakeCalculator())

    assert pool.test_ensemble(pool.calculator) == (0.0, 0.0)
