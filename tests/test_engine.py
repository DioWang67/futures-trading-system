"""回測引擎單元測試。"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine, BacktestResult, run_backtest
from src.data.fetcher import generate_sample_data


@pytest.fixture
def sample_ltf():
    return generate_sample_data(n_bars=500, freq="15T", seed=42)


@pytest.fixture
def sample_htf():
    return generate_sample_data(n_bars=125, freq="60T", seed=42)


class TestBacktestResult:
    def test_meets_threshold_boundary_is_inclusive(self):
        r = BacktestResult(
            win_rate=0.55, profit_factor=1.5, sharpe_ratio=1.2,
            max_drawdown=0.15, avg_rr=1.5,
        )
        assert r.meets_threshold

    def test_meets_threshold_all_pass(self):
        r = BacktestResult(
            total_trades=50, winning_trades=30, losing_trades=20,
            win_rate=0.60, profit_factor=2.0, sharpe_ratio=1.5,
            max_drawdown=0.10, avg_rr=2.0,
        )
        assert r.meets_threshold

    def test_meets_threshold_fail_wr(self):
        r = BacktestResult(
            win_rate=0.40, profit_factor=2.0, sharpe_ratio=1.5,
            max_drawdown=0.10, avg_rr=2.0,
        )
        assert not r.meets_threshold

    def test_meets_threshold_fail_mdd(self):
        r = BacktestResult(
            win_rate=0.60, profit_factor=2.0, sharpe_ratio=1.5,
            max_drawdown=0.25, avg_rr=2.0,
        )
        assert not r.meets_threshold

    def test_custom_threshold(self):
        r = BacktestResult(
            win_rate=0.50, profit_factor=1.2, sharpe_ratio=1.0,
            max_drawdown=0.10, avg_rr=1.2,
        )
        # 用寬鬆門檻
        assert r.meets_custom_threshold(
            min_win_rate=0.45, min_profit_factor=1.0,
            min_sharpe=0.8, max_mdd=0.20, min_rr=1.0,
        )


class TestBacktestEngine:
    def test_engine_creation(self):
        engine = BacktestEngine()
        assert engine.initial_cash == 1_000_000.0

    def test_engine_run_returns_result(self, sample_ltf):
        engine = BacktestEngine()
        result = engine.run(sample_ltf)
        assert isinstance(result, BacktestResult)

    def test_engine_run_with_params(self, sample_ltf):
        engine = BacktestEngine()
        params = {"swing_lookback": 3, "bos_min_move": 10.0}
        result = engine.run(sample_ltf, strategy_params=params)
        assert isinstance(result, BacktestResult)

    def test_engine_with_htf(self, sample_ltf, sample_htf):
        engine = BacktestEngine()
        result = engine.run(sample_ltf, sample_htf)
        assert isinstance(result, BacktestResult)

    def test_run_backtest_convenience(self, sample_ltf):
        result = run_backtest(sample_ltf)
        assert isinstance(result, BacktestResult)

    def test_result_fields_are_numeric(self, sample_ltf):
        engine = BacktestEngine()
        result = engine.run(sample_ltf)
        assert isinstance(result.win_rate, (int, float))
        assert isinstance(result.profit_factor, (int, float))
        assert isinstance(result.sharpe_ratio, (int, float))
        assert isinstance(result.max_drawdown, (int, float))
        assert isinstance(result.total_trades, int)
