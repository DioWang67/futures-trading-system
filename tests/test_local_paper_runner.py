from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from position_state import PositionState
import src.live.paper_strategy as paper_strategy_module
from src.live.paper_strategy import (
    LocalPaperStrategyRunner,
    ManagedTrade,
    resample_completed_bars,
)


class _FakeBroker:
    def __init__(self):
        self.position = PositionState("paper-test")
        self.calls: list[tuple[str, int, dict]] = []
        self.fetch_recent_calls: list[tuple[str, int]] = []
        self.fetch_incremental_calls: list[tuple[str, object, object, int]] = []
        self.armed_protection: list[tuple[str, str, int, float, float]] = []
        self.disarmed_protection: list[str] = []
        self._initial_bars = pd.DataFrame()
        self._incremental_bars = pd.DataFrame()

    async def place_order(self, action: str, quantity: int, **kwargs) -> dict:
        self.calls.append((action, quantity, kwargs))
        if action == "buy":
            self.position.apply_fill("Buy", quantity, fill_price=100.0)
        elif action == "sell":
            self.position.apply_fill("Sell", quantity, fill_price=100.0)
        elif action == "exit":
            current = self.position.get_position()
            if current.side == "long":
                self.position.apply_fill("Sell", current.quantity, fill_price=100.0)
            elif current.side == "short":
                self.position.apply_fill("Buy", current.quantity, fill_price=100.0)
        return {"status": "ok", "orders": [{"status": "submitted"}]}

    def fetch_recent_bars(self, ticker: str, lookback_days: int) -> pd.DataFrame:
        self.fetch_recent_calls.append((ticker, lookback_days))
        return self._initial_bars.copy()

    def fetch_bars(
        self,
        ticker: str,
        start_date=None,
        end_date=None,
        lookback_days: int = 10,
    ) -> pd.DataFrame:
        self.fetch_incremental_calls.append((ticker, start_date, end_date, lookback_days))
        return self._incremental_bars.copy()

    def arm_protective_exit(
        self,
        ticker: str,
        side: str,
        quantity: int,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
    ) -> None:
        self.armed_protection.append(
            (ticker, side, quantity, stop_loss, take_profit)
        )

    def disarm_protective_exit(self, ticker: str) -> None:
        self.disarmed_protection.append(ticker)


def _state_path(name: str) -> Path:
    path = Path(".tmp") / "test_local_paper_runner" / f"{name}-{uuid4().hex}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@pytest.mark.asyncio
async def test_runner_enters_long_and_persists_state():
    broker = _FakeBroker()
    runner = LocalPaperStrategyRunner(
        broker=broker,
        symbol="MXF",
        quantity=1,
        strategy_params={},
        state_path=_state_path("enter"),
    )

    signal_df = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp("2026-04-21 09:15:00"),
                "close": 100.0,
                "signal": 1,
                "sl": 95.0,
                "tp": 110.0,
            }
        ]
    )

    result = await runner.process_signal_frame(signal_df)

    assert result["status"] == "ok"
    assert result["action"] == "buy"
    assert len(broker.calls) == 1
    assert broker.position.get_position().side == "long"
    assert runner.state.managed_trade is not None
    assert runner.state.managed_trade.status == "active"
    assert runner.state.managed_trade.sl == 95.0
    assert runner.state.managed_trade.tp == 110.0
    assert broker.armed_protection[-1] == ("MXF", "long", 1, 95.0, 110.0)


@pytest.mark.asyncio
async def test_runner_exits_when_take_profit_is_hit():
    broker = _FakeBroker()
    broker.position.update_position("long", 1, entry_price=100.0)
    runner = LocalPaperStrategyRunner(
        broker=broker,
        symbol="MXF",
        quantity=1,
        strategy_params={},
        state_path=_state_path("exit"),
    )
    runner.state.managed_trade = ManagedTrade(
        status="active",
        side="long",
        quantity=1,
        sl=95.0,
        tp=105.0,
        entry_bar_time="2026-04-21T09:15:00",
        submitted_at="2026-04-21T09:15:00+00:00",
    )

    signal_df = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp("2026-04-21 09:30:00"),
                "close": 106.0,
                "signal": 0,
                "sl": 0.0,
                "tp": 0.0,
            }
        ]
    )

    result = await runner.process_signal_frame(signal_df)

    assert result["status"] == "ok"
    assert result["action"] == "exit"
    assert broker.position.get_position().side == "flat"
    assert runner.state.managed_trade is None
    assert broker.disarmed_protection[-1] == "MXF"


@pytest.mark.asyncio
async def test_runner_refuses_unmanaged_broker_position():
    broker = _FakeBroker()
    broker.position.update_position("short", 1, entry_price=100.0)
    runner = LocalPaperStrategyRunner(
        broker=broker,
        symbol="MXF",
        quantity=1,
        strategy_params={},
        state_path=_state_path("unmanaged"),
    )

    signal_df = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp("2026-04-21 09:30:00"),
                "close": 99.0,
                "signal": 1,
                "sl": 95.0,
                "tp": 105.0,
            }
        ]
    )

    result = await runner.process_signal_frame(signal_df)

    assert result["status"] == "error"
    assert "unmanaged position" in result["reason"]
    assert broker.calls == []


def test_resample_completed_bars_drops_partial_tail():
    raw_df = pd.DataFrame(
        [
            {"datetime": pd.Timestamp("2026-04-21 09:01:00"), "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
            {"datetime": pd.Timestamp("2026-04-21 09:02:00"), "open": 2, "high": 2, "low": 2, "close": 2, "volume": 1},
            {"datetime": pd.Timestamp("2026-04-21 09:03:00"), "open": 3, "high": 3, "low": 3, "close": 3, "volume": 1},
        ]
    )

    out = resample_completed_bars(raw_df, "5min")

    assert out.empty


@pytest.mark.asyncio
async def test_runner_bootstraps_history_once_then_refreshes_incrementally(monkeypatch):
    broker = _FakeBroker()
    broker._initial_bars = pd.DataFrame(
        [
            {"datetime": pd.Timestamp("2026-04-21 09:00:00"), "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
            {"datetime": pd.Timestamp("2026-04-21 09:01:00"), "open": 2, "high": 2, "low": 2, "close": 2, "volume": 1},
            {"datetime": pd.Timestamp("2026-04-21 09:02:00"), "open": 3, "high": 3, "low": 3, "close": 3, "volume": 1},
        ]
    )
    broker._incremental_bars = pd.DataFrame(
        [
            {"datetime": pd.Timestamp("2026-04-21 09:01:00"), "open": 2, "high": 2, "low": 2, "close": 2, "volume": 1},
            {"datetime": pd.Timestamp("2026-04-21 09:02:00"), "open": 3, "high": 3, "low": 3, "close": 3, "volume": 1},
            {"datetime": pd.Timestamp("2026-04-21 09:03:00"), "open": 4, "high": 4, "low": 4, "close": 4, "volume": 1},
        ]
    )

    def fake_precompute_signals(ltf_df, htf_df=None, **kwargs):
        out = ltf_df.copy()
        out["signal"] = 0
        out["sl"] = 0.0
        out["tp"] = 0.0
        return out

    monkeypatch.setattr(paper_strategy_module, "precompute_signals", fake_precompute_signals)

    runner = LocalPaperStrategyRunner(
        broker=broker,
        symbol="MXF",
        quantity=1,
        strategy_params={},
        ltf_freq="1min",
        htf_freq="5min",
        state_path=_state_path("refresh"),
    )

    await runner.fetch_signal_frame()
    await runner.fetch_signal_frame()

    assert len(broker.fetch_recent_calls) == 1
    assert len(broker.fetch_incremental_calls) == 1
    assert len(runner._raw_bars_cache) == 4
    assert runner._raw_bars_cache["datetime"].iloc[-1] == pd.Timestamp("2026-04-21 09:03:00")


def test_runner_aligns_to_next_bar_close():
    runner = LocalPaperStrategyRunner(
        broker=_FakeBroker(),
        symbol="MXF",
        quantity=1,
        strategy_params={},
        ltf_freq="15min",
        htf_freq="60min",
        close_grace_seconds=3,
        state_path=_state_path("align"),
    )

    delay = runner.seconds_until_next_cycle(
        now=pd.Timestamp("2026-04-21 10:14:40").to_pydatetime()
    )

    assert delay == pytest.approx(23.0)
