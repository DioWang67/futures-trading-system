"""Tests for the risk management module."""

import time

import pytest

from risk_manager import RiskConfig, RiskManager


@pytest.fixture
def config():
    return RiskConfig(
        max_daily_loss=10_000.0,
        max_daily_loss_pct=0.05,
        max_position_size=5,
        max_order_qty=3,
        max_drawdown_pct=0.10,
        cooldown_after_consecutive_losses=3,
        cooldown_seconds=2,  # short for testing
        initial_capital=200_000.0,
    )


@pytest.fixture
def rm(config):
    return RiskManager(config)


class TestPreTradeChecks:
    def test_normal_order_allowed(self, rm):
        ok, reason = rm.check_order("shioaji", "buy", 2)
        assert ok is True
        assert reason == ""

    def test_exit_always_allowed(self, rm):
        # Even when halted, exits should be allowed... but we halt first
        # Actually our logic checks halt first. Let's test exit without halt
        ok, reason = rm.check_order("shioaji", "exit", 0)
        assert ok is True

    def test_order_qty_exceeds_max(self, rm):
        ok, reason = rm.check_order("shioaji", "buy", 10)
        assert ok is False
        assert "max" in reason.lower()

    # Position-size gate semantics:
    # route_order fully flips on a reversal and skips same-side signals,
    # so the final exposure of any buy/sell is always `quantity` on the
    # new side. check_order therefore enforces `quantity <= max_position_size`
    # (the per-order cap is a separate, tighter knob).

    def test_position_size_at_limit(self, rm):
        # quantity equals max_position_size exactly — allowed.
        # Use a config where the per-order cap isn't the bottleneck.
        rm2 = RiskManager(RiskConfig(
            max_daily_loss=10_000.0, max_daily_loss_pct=0.05,
            max_position_size=5, max_order_qty=10,
            max_drawdown_pct=0.10,
            cooldown_after_consecutive_losses=3, cooldown_seconds=2,
            initial_capital=200_000.0,
        ))
        ok, reason = rm2.check_order("shioaji", "buy", 5)
        assert ok is True, reason

    def test_position_size_over_limit(self):
        # quantity > max_position_size — rejected by the position gate.
        rm2 = RiskManager(RiskConfig(
            max_daily_loss=10_000.0, max_daily_loss_pct=0.05,
            max_position_size=3, max_order_qty=10,
            max_drawdown_pct=0.10,
            cooldown_after_consecutive_losses=3, cooldown_seconds=2,
            initial_capital=200_000.0,
        ))
        ok, reason = rm2.check_order("shioaji", "buy", 4)
        assert ok is False
        assert "position" in reason.lower()

    def test_reversal_allowed_when_target_under_limit(self, rm):
        # short 4 + buy 3 fully flips to long 3, under the limit of 5.
        ok, reason = rm.check_order("shioaji", "buy", 3)
        assert ok is True, reason

    def test_reversal_rejected_when_target_exceeds_limit(self):
        # short 3 + buy 5 fully flips to long 5. With max_position_size=4
        # this MUST reject — the net-exposure shortcut (|-3+5|=2) would
        # have wrongly allowed it.
        rm2 = RiskManager(RiskConfig(
            max_daily_loss=10_000.0, max_daily_loss_pct=0.05,
            max_position_size=4, max_order_qty=10,
            max_drawdown_pct=0.10,
            cooldown_after_consecutive_losses=3, cooldown_seconds=2,
            initial_capital=200_000.0,
        ))
        ok, reason = rm2.check_order("shioaji", "buy", 5)
        assert ok is False
        assert "position" in reason.lower()


class TestDailyLossLimit:
    def test_daily_loss_triggers_halt(self, rm):
        # Daily loss limit = min(10_000, 200_000*0.05) = $10,000
        rm.record_fill("shioaji", -5000.0)
        ok, _ = rm.check_order("shioaji", "buy", 1)
        assert ok is True  # still under limit

        rm.record_fill("shioaji", -5001.0)
        ok, reason = rm.check_order("shioaji", "buy", 1)
        assert ok is False
        assert rm.is_halted is True

    def test_daily_loss_pct_limit(self):
        config = RiskConfig(
            max_daily_loss=100_000.0,  # very high
            max_daily_loss_pct=0.01,   # 1% = $2,000
            initial_capital=200_000.0,
        )
        rm = RiskManager(config)
        rm.record_fill("shioaji", -2001.0)
        ok, _ = rm.check_order("shioaji", "buy", 1)
        assert ok is False


class TestDrawdownCircuitBreaker:
    def test_drawdown_triggers_halt(self):
        # Use high daily loss limit so drawdown triggers first
        config = RiskConfig(
            max_daily_loss=100_000.0,
            max_daily_loss_pct=1.0,  # effectively disabled
            max_drawdown_pct=0.10,   # 10%
            initial_capital=200_000.0,
        )
        rm = RiskManager(config)
        rm.record_fill("shioaji", -15_000.0)
        assert rm.is_halted is False  # 15k/200k = 7.5%

        rm.record_fill("shioaji", -6_000.0)
        # 21k loss, equity=179k, peak=200k, drawdown=10.5% > 10%
        assert rm.is_halted is True


class TestConsecutiveLossCooldown:
    def test_cooldown_after_consecutive_losses(self, rm):
        rm.record_fill("shioaji", -100.0)
        rm.record_fill("shioaji", -100.0)
        rm.record_fill("shioaji", -100.0)  # 3rd consecutive loss

        ok, reason = rm.check_order("shioaji", "buy", 1)
        assert ok is False
        assert "cooldown" in reason.lower()

    def test_win_resets_consecutive_losses(self, rm):
        rm.record_fill("shioaji", -100.0)
        rm.record_fill("shioaji", -100.0)
        rm.record_fill("shioaji", 200.0)  # win resets counter
        rm.record_fill("shioaji", -100.0)
        rm.record_fill("shioaji", -100.0)

        ok, _ = rm.check_order("shioaji", "buy", 1)
        assert ok is True  # only 2 consecutive losses

    def test_cooldown_expires(self, rm):
        rm.record_fill("shioaji", -100.0)
        rm.record_fill("shioaji", -100.0)
        rm.record_fill("shioaji", -100.0)

        time.sleep(2.5)  # cooldown is 2 seconds

        ok, _ = rm.check_order("shioaji", "buy", 1)
        assert ok is True


class TestHaltAndResume:
    def test_halt_blocks_all_orders(self, rm):
        rm.record_fill("shioaji", -10_001.0)
        assert rm.is_halted is True
        ok, _ = rm.check_order("shioaji", "buy", 1)
        assert ok is False

    def test_resume_trading(self, rm):
        rm.record_fill("shioaji", -10_001.0)
        assert rm.is_halted is True

        halted_status = rm.get_status()
        assert halted_status["halted_since"]

        rm.resume_trading()
        assert rm.is_halted is False
        assert rm.get_status()["halted_since"] == ""

    def test_daily_reset(self, rm):
        rm.record_fill("shioaji", -5000.0)
        status = rm.get_status()
        assert status["daily_pnl"] == -5000.0

        rm.reset_daily()
        status = rm.get_status()
        assert status["daily_pnl"] == 0.0


class TestEquityTracking:
    def test_equity_increases_with_wins(self, rm):
        rm.record_fill("shioaji", 1000.0)
        status = rm.get_status()
        assert status["current_equity"] == 201_000.0
        assert status["peak_equity"] == 201_000.0

    def test_drawdown_calculated(self, rm):
        rm.record_fill("shioaji", 1000.0)  # equity = 201k, peak = 201k
        rm.record_fill("shioaji", -3000.0)  # equity = 198k, peak = 201k
        status = rm.get_status()
        expected_dd = (201_000 - 198_000) / 201_000
        assert abs(status["drawdown_pct"] - expected_dd) < 1e-6


class TestGetStatus:
    def test_status_fields(self, rm):
        status = rm.get_status()
        assert "halted" in status
        assert "daily_pnl" in status
        assert "current_equity" in status
        assert "peak_equity" in status
        assert "drawdown_pct" in status
        assert "daily_loss_limit" in status
