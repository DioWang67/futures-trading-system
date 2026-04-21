"""Tests for Monte Carlo simulation module."""

import numpy as np
import pytest

from src.backtest.monte_carlo import (
    MonteCarloResult,
    monte_carlo_shuffle,
    monte_carlo_bootstrap,
    monte_carlo_noise,
    _compute_max_drawdown_pct,
    _build_equity_curve,
)


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def winning_pnls():
    """Trade PnLs with strong positive expectancy."""
    return [100.0, -50.0, 200.0, -30.0, 150.0, -80.0, 120.0, 60.0, -40.0, 180.0]


@pytest.fixture
def losing_pnls():
    """Trade PnLs that will reliably ruin the account."""
    return [-500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0]


@pytest.fixture
def breakeven_pnls():
    """Trade PnLs near breakeven."""
    return [10.0, -10.0, 10.0, -10.0, 10.0, -10.0]


# ---------------------------------------------------------------------------
# _compute_max_drawdown_pct
# ---------------------------------------------------------------------------

class TestMaxDrawdown:
    def test_no_drawdown(self):
        eq = np.array([100.0, 110.0, 120.0, 130.0])
        assert _compute_max_drawdown_pct(eq) == 0.0

    def test_known_drawdown(self):
        # Peak=200, trough=150 → 25% drawdown
        eq = np.array([100.0, 200.0, 150.0, 180.0])
        assert abs(_compute_max_drawdown_pct(eq) - 0.25) < 1e-9

    def test_full_drawdown(self):
        eq = np.array([100.0, 0.0])
        assert abs(_compute_max_drawdown_pct(eq) - 1.0) < 1e-9

    def test_empty_array(self):
        eq = np.array([])
        assert _compute_max_drawdown_pct(eq) == 0.0

    def test_single_value(self):
        eq = np.array([100.0])
        assert _compute_max_drawdown_pct(eq) == 0.0


# ---------------------------------------------------------------------------
# _build_equity_curve
# ---------------------------------------------------------------------------

class TestBuildEquityCurve:
    def test_cumulative(self):
        pnls = np.array([10.0, -5.0, 20.0])
        eq = _build_equity_curve(pnls, 100.0)
        np.testing.assert_array_almost_equal(eq, [110.0, 105.0, 125.0])

    def test_zero_pnls(self):
        pnls = np.array([0.0, 0.0, 0.0])
        eq = _build_equity_curve(pnls, 50.0)
        np.testing.assert_array_almost_equal(eq, [50.0, 50.0, 50.0])


# ---------------------------------------------------------------------------
# MonteCarloResult.summary()
# ---------------------------------------------------------------------------

class TestMonteCarloResultSummary:
    def test_summary_format(self):
        r = MonteCarloResult(
            n_simulations=1000,
            method="shuffle",
            final_equity_mean=105000.0,
            final_equity_median=104000.0,
            final_equity_5th=90000.0,
            final_equity_95th=120000.0,
            final_equity_std=8000.0,
            max_dd_mean=0.10,
            max_dd_median=0.09,
            max_dd_95th=0.20,
            max_dd_worst=0.30,
            ruin_probability=0.05,
            ruin_threshold=50000.0,
            profit_probability=0.75,
        )
        s = r.summary()
        assert "shuffle" in s
        assert "1,000" in s
        assert "5.0%" in s  # ruin probability
        assert "75.0%" in s  # profit probability


# ---------------------------------------------------------------------------
# monte_carlo_shuffle
# ---------------------------------------------------------------------------

class TestMonteCarloShuffle:
    def test_empty_pnls(self):
        result = monte_carlo_shuffle([], seed=42)
        assert result.method == "shuffle"
        assert result.n_simulations == 0

    def test_basic_run(self, winning_pnls):
        result = monte_carlo_shuffle(winning_pnls, n_simulations=500, seed=42)
        assert result.n_simulations == 500
        assert result.method == "shuffle"
        assert len(result.all_final_equities) == 500
        assert len(result.all_max_drawdowns) == 500
        assert len(result.equity_curves) == 100  # capped at 100

    def test_deterministic_with_seed(self, winning_pnls):
        r1 = monte_carlo_shuffle(winning_pnls, n_simulations=100, seed=123)
        r2 = monte_carlo_shuffle(winning_pnls, n_simulations=100, seed=123)
        assert r1.all_final_equities == r2.all_final_equities

    def test_final_equity_preserves_total(self, winning_pnls):
        """Shuffle preserves total PnL — all final equities should equal original."""
        total_pnl = sum(winning_pnls)
        expected_final = 100_000.0 + total_pnl
        result = monte_carlo_shuffle(winning_pnls, n_simulations=50, seed=42)
        for eq in result.all_final_equities:
            assert abs(eq - expected_final) < 1e-6

    def test_profit_probability(self, winning_pnls):
        result = monte_carlo_shuffle(winning_pnls, n_simulations=100, seed=42)
        # Sum of winning_pnls = 610 > 0, so all sims end above initial
        assert result.profit_probability == 1.0

    def test_ruin_with_losing_pnls(self, losing_pnls):
        result = monte_carlo_shuffle(
            losing_pnls, initial_capital=1000.0,
            n_simulations=100, ruin_threshold_pct=0.5, seed=42,
        )
        # Total loss = -4000, starting from 1000 → always ruins
        assert result.ruin_probability == 1.0


# ---------------------------------------------------------------------------
# monte_carlo_bootstrap
# ---------------------------------------------------------------------------

class TestMonteCarloBootstrap:
    def test_empty_pnls(self):
        result = monte_carlo_bootstrap([], seed=42)
        assert result.method == "bootstrap"
        assert result.n_simulations == 0

    def test_basic_run(self, winning_pnls):
        result = monte_carlo_bootstrap(winning_pnls, n_simulations=200, seed=42)
        assert result.n_simulations == 200
        assert result.method == "bootstrap"
        assert len(result.all_final_equities) == 200

    def test_custom_trade_count(self, winning_pnls):
        result = monte_carlo_bootstrap(
            winning_pnls, n_simulations=50, n_trades_per_sim=20, seed=42,
        )
        # With 20 trades sampled, equity curves should have 20 points
        assert len(result.equity_curves[0]) == 20

    def test_deterministic_with_seed(self, winning_pnls):
        r1 = monte_carlo_bootstrap(winning_pnls, n_simulations=50, seed=99)
        r2 = monte_carlo_bootstrap(winning_pnls, n_simulations=50, seed=99)
        assert r1.all_final_equities == r2.all_final_equities

    def test_statistics_populated(self, winning_pnls):
        result = monte_carlo_bootstrap(winning_pnls, n_simulations=500, seed=42)
        assert result.final_equity_mean > 0
        assert result.final_equity_std > 0
        assert 0 <= result.max_dd_mean <= 1
        assert result.max_dd_95th >= result.max_dd_mean


# ---------------------------------------------------------------------------
# monte_carlo_noise
# ---------------------------------------------------------------------------

class TestMonteCarloNoise:
    def test_empty_pnls(self):
        result = monte_carlo_noise([], seed=42)
        assert result.method == "noise"
        assert result.n_simulations == 0

    def test_basic_run(self, winning_pnls):
        result = monte_carlo_noise(winning_pnls, n_simulations=300, seed=42)
        assert result.n_simulations == 300
        assert result.method == "noise"
        assert len(result.all_final_equities) == 300

    def test_zero_noise_matches_original(self, winning_pnls):
        """With noise_std_pct=0, all simulations should match exactly."""
        result = monte_carlo_noise(
            winning_pnls, n_simulations=10, noise_std_pct=0.0, seed=42,
        )
        expected_final = 100_000.0 + sum(winning_pnls)
        for eq in result.all_final_equities:
            assert abs(eq - expected_final) < 1e-6

    def test_noise_creates_variance(self, winning_pnls):
        result = monte_carlo_noise(
            winning_pnls, n_simulations=200, noise_std_pct=0.2, seed=42,
        )
        assert result.final_equity_std > 0  # should have variance

    def test_deterministic_with_seed(self, winning_pnls):
        r1 = monte_carlo_noise(winning_pnls, n_simulations=50, seed=77)
        r2 = monte_carlo_noise(winning_pnls, n_simulations=50, seed=77)
        assert r1.all_final_equities == r2.all_final_equities

    def test_ruin_never_with_safe_capital(self, breakeven_pnls):
        """High capital + tiny PnLs → never hits ruin threshold."""
        result = monte_carlo_noise(
            breakeven_pnls, initial_capital=1_000_000.0,
            n_simulations=100, noise_std_pct=0.1,
            ruin_threshold_pct=0.5, seed=42,
        )
        assert result.ruin_probability == 0.0


# ---------------------------------------------------------------------------
# Cross-method consistency
# ---------------------------------------------------------------------------

class TestCrossMethod:
    def test_all_methods_return_correct_type(self, winning_pnls):
        for fn in [monte_carlo_shuffle, monte_carlo_bootstrap, monte_carlo_noise]:
            result = fn(winning_pnls, n_simulations=10, seed=42)
            assert isinstance(result, MonteCarloResult)

    def test_percentile_ordering(self, winning_pnls):
        """5th percentile ≤ median ≤ 95th percentile."""
        result = monte_carlo_bootstrap(winning_pnls, n_simulations=500, seed=42)
        assert result.final_equity_5th <= result.final_equity_median
        assert result.final_equity_median <= result.final_equity_95th

    def test_mdd_ordering(self, winning_pnls):
        """MDD mean ≤ 95th ≤ worst."""
        result = monte_carlo_bootstrap(winning_pnls, n_simulations=500, seed=42)
        assert result.max_dd_mean <= result.max_dd_95th + 1e-9
        assert result.max_dd_95th <= result.max_dd_worst + 1e-9
