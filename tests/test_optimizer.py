"""Optuna 優化模組單元測試。"""

import pytest
import pandas as pd

from src.backtest.optimizer import StrategyOptimizer
from src.data.fetcher import generate_sample_data


@pytest.fixture
def sample_data():
    return generate_sample_data(n_bars=300, freq="15T", seed=42)


@pytest.fixture
def optimizer(sample_data):
    config = {
        "backtest": {"commission": 22, "slippage": 1, "size": 1, "point_value": 10},
        "optimization": {"n_trials": 5, "timeout": 60},
    }
    return StrategyOptimizer(ltf_data=sample_data, config=config)


class TestStrategyOptimizer:
    def test_creation(self, optimizer):
        assert optimizer.ltf_data is not None
        assert optimizer.n_trials == 5

    def test_optimize_returns_params_or_none(self, optimizer):
        result = optimizer.optimize()
        # With only 5 trials it's unlikely to hit threshold, but should not crash
        assert result is None or isinstance(result, dict)

    def test_optimize_with_few_trials(self, sample_data):
        config = {
            "backtest": {"commission": 22, "slippage": 1, "size": 1, "point_value": 10},
            "optimization": {"n_trials": 3, "timeout": 30},
        }
        opt = StrategyOptimizer(ltf_data=sample_data, config=config)
        result = opt.optimize()
        # Should complete without error
        assert result is None or isinstance(result, dict)

    def test_best_result_initially_none(self, optimizer):
        assert optimizer.get_best_result() is None
