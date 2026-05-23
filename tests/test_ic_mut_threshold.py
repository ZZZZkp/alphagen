from __future__ import annotations

import numpy as np

from alphagen.data.expression import Add, Feature
from alphagen.models.linear_alpha_pool import LinearAlphaPool
from alphagen_qlib.stock_data import FeatureType
from tests.test_parser_and_pool import DeterministicPool, FakeCalculator


class FlexiblePool(DeterministicPool):
    """DeterministicPool extended to handle size == 1 (single-factor pool)."""

    def optimize(self, lr: float = 5e-4, max_steps: int = 10000, tolerance: int = 500) -> np.ndarray:
        if self.size == 1:
            return np.array([1.0])
        return super().optimize(lr, max_steps, tolerance)


def test_pool_default_threshold_is_099_for_back_compat() -> None:
    import inspect
    sig = inspect.signature(LinearAlphaPool.__init__)
    assert sig.parameters["ic_mut_threshold"].default == 0.99


def test_pool_accepts_lower_threshold_and_rejects_correlated_factor() -> None:
    # FakeCalculator mutual IC between $close and $open is 0.1, so threshold 0.05
    # should reject pairing them.
    pool = FlexiblePool(
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
    pool = FlexiblePool(
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
    pool = FlexiblePool(
        capacity=2,
        calculator=FakeCalculator(),
        ic_mut_threshold=0.01,
    )
    pool.force_load_exprs([Feature(FeatureType.CLOSE), Feature(FeatureType.OPEN)])
    assert pool.size == 2
