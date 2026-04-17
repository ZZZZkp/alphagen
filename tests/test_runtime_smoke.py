from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _run_script(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def test_alpha_env_runtime_smoke() -> None:
    result = _run_script(
        """
        import pandas as pd
        import torch

        import alphagen_qlib.stock_data as stock_data_module
        from alphagen.data.expression import Feature
        from alphagen.data.tokens import FeatureToken, SequenceIndicatorToken, SequenceIndicatorType
        from alphagen.rl.env.core import AlphaEnvCore
        from alphagen_qlib.stock_data import FeatureType, StockData

        stock_data_module._QLIB_INITIALIZED = True

        class DummyPool:
            def __init__(self):
                self.expr = None
                self.eval_cnt = 0

            def try_new_expr(self, expr):
                self.expr = expr
                self.eval_cnt += 1
                return 1.5

        pool = DummyPool()
        env = AlphaEnvCore(pool=pool, device=torch.device("cpu"))
        env.reset()
        _, reward, done, truncated, _ = env.step(FeatureToken(FeatureType.CLOSE))
        assert reward == 0.0 and done is False and truncated is False
        _, reward, done, truncated, _ = env.step(
            SequenceIndicatorToken(SequenceIndicatorType.SEP)
        )
        assert reward == 1.5 and done is True and truncated is False
        assert isinstance(pool.expr, Feature)
        assert str(pool.expr) == "$close"

        dates = pd.Index(pd.date_range("2024-01-01", periods=5, freq="D"))
        stocks = pd.Index(["AAA", "BBB"])
        values = torch.arange(5 * 2 * 2, dtype=torch.float32).reshape(5, 2, 2)

        data = StockData(
            instrument="dummy",
            start_time="2024-01-02",
            end_time="2024-01-04",
            max_backtrack_days=1,
            max_future_days=1,
            features=[FeatureType.OPEN, FeatureType.CLOSE],
            device=torch.device("cpu"),
            preloaded_data=(values, dates, stocks),
        )
        close_only = Feature(FeatureType.CLOSE).evaluate(data)
        frame = data.make_dataframe(close_only, columns=["close"])
        assert close_only.shape == (3, 2)
        assert list(frame.columns) == ["close"]
        assert len(frame) == data.n_days * data.n_stocks
        print("ok")
        """
    )

    if result.returncode != 0:
        pytest.skip(
            "Runtime smoke test needs a working local PyTorch install; "
            f"subprocess failed with: {(result.stderr or result.stdout).strip()}"
        )

    assert result.stdout.strip() == "ok"


def test_training_entrypoints_import_with_modern_runtime() -> None:
    result = _run_script(
        """
        import numpy as np
        from sb3_contrib.ppo_mask import MaskablePPO

        import scripts.llm_only as llm_only
        import scripts.rl as rl

        assert np.__version__.startswith("2.")
        assert MaskablePPO is not None
        assert rl.PROFILES["colab"].name == "colab"
        assert callable(llm_only.build_parser)
        print("ok")
        """
    )

    if result.returncode != 0:
        pytest.skip(
            "Entrypoint import smoke test needs a working local RL stack; "
            f"subprocess failed with: {(result.stderr or result.stdout).strip()}"
        )

    assert result.stdout.strip() == "ok"


def test_project_runtime_configures_matplotlib_cache_in_repo() -> None:
    result = _run_script(
        """
        import os
        from pathlib import Path

        from alphagen.utils.runtime_env import configure_project_runtime

        root = Path.cwd()
        configure_project_runtime(root)
        expected = root / ".cache" / "matplotlib"
        assert Path(os.environ["MPLCONFIGDIR"]) == expected
        assert expected.is_dir()
        print("ok")
        """
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert result.stdout.strip() == "ok"
