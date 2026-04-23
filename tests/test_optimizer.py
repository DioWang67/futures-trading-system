"""Optuna 優化模組單元測試。"""

import optuna
import pytest
import pandas as pd

from src.backtest.engine import BacktestResult
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

    def test_objective_can_limit_optimized_params(self, sample_data, monkeypatch):
        config = {
            "strategy": {
                "swing_lookback": 10,
                "bos_min_move": 35.0,
                "ob_max_age": 19,
                "ob_body_ratio": 0.55,
                "fvg_min_gap": 13.0,
                "fvg_enabled": True,
                "pin_bar_ratio": 0.75,
                "engulf_ratio": 1.4,
                "rr_ratio": 1.25,
                "use_structure_tp": True,
                "sl_buffer": 3.0,
                "pa_confirm": False,
                "adx_filter_enabled": False,
                "atr_filter_enabled": True,
                "atr_period": 14,
                "atr_min_points": 20.0,
                "blocked_entry_hours": [0, 1, 10],
            },
            "optimization": {
                "n_trials": 1,
                "timeout": 10,
                "enabled_params": ["rr_ratio", "sl_buffer"],
                "fixed_params": {"atr_min_points": 15.0},
            },
        }
        opt = StrategyOptimizer(ltf_data=sample_data, config=config)
        captured = {}

        def fake_run(ltf_data, htf_data, params):
            captured["params"] = params
            return BacktestResult(
                total_trades=10,
                win_rate=0.6,
                profit_factor=1.8,
                sharpe_ratio=1.3,
                max_drawdown=0.1,
                avg_rr=1.6,
            )

        monkeypatch.setattr(opt.engine, "run", fake_run)
        trial = optuna.trial.FixedTrial({"rr_ratio": 1.5, "sl_buffer": 2.5})
        score = opt._objective(trial)

        assert score > 0
        assert captured["params"]["rr_ratio"] == 1.5
        assert captured["params"]["sl_buffer"] == 2.5
        assert captured["params"]["swing_lookback"] == 10
        assert captured["params"]["blocked_entry_hours"] == [0, 1, 10]
        assert captured["params"]["atr_min_points"] == 15.0
