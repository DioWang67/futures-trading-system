from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from loguru import logger

from src.strategy.bt_strategy import precompute_signals


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _normalize_freq(freq: str) -> str:
    if freq.endswith("T"):
        return freq.replace("T", "min")
    return freq


def _freq_to_seconds(freq: str) -> int:
    return max(60, int(pd.Timedelta(_normalize_freq(freq)).total_seconds()))


def resample_completed_bars(raw_df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample 1-minute bars and drop the still-forming tail bar."""
    if raw_df.empty:
        return raw_df.copy()

    rule = _normalize_freq(freq)
    df = raw_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").drop_duplicates(subset=["datetime"])
    last_raw_time = df["datetime"].iloc[-1]
    df = df.set_index("datetime")

    ohlc_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    resampled = df.resample(rule, label="right", closed="right").agg(ohlc_dict)
    resampled.dropna(inplace=True)
    if not resampled.empty and resampled.index[-1] > last_raw_time:
        resampled = resampled.iloc[:-1]
    return resampled.reset_index()


@dataclass
class ManagedTrade:
    status: str
    side: str
    quantity: int
    sl: float
    tp: float
    entry_bar_time: str
    submitted_at: str
    exit_reason: str = ""


@dataclass
class RunnerState:
    last_processed_bar_time: str = ""
    managed_trade: Optional[ManagedTrade] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "RunnerState":
        if not path.exists():
            return cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(
                "[PaperRunner] Failed to load state file {}: {}",
                path, exc,
            )
            return cls()

        trade = raw.get("managed_trade")
        return cls(
            last_processed_bar_time=raw.get("last_processed_bar_time", ""),
            managed_trade=ManagedTrade(**trade) if trade else None,
            metadata=dict(raw.get("metadata") or {}),
            runtime=dict(raw.get("runtime") or {}),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "last_processed_bar_time": self.last_processed_bar_time,
            "managed_trade": (
                asdict(self.managed_trade) if self.managed_trade else None
            ),
            "metadata": self.metadata,
            "runtime": self.runtime,
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )


class LocalPaperStrategyRunner:
    """Drive the local SMC+PA strategy against a paper broker.

    The runner operates on completed bars only:
      - fetch recent 1-minute bars from the broker
      - resample into LTF / HTF bars
      - compute the latest strategy signal
      - place paper orders directly on the broker
      - persist a lightweight runner state so restarts don't re-submit
        the same bar's signal or lose SL/TP context
    """

    def __init__(
        self,
        broker: Any,
        symbol: str,
        quantity: int,
        strategy_params: dict,
        ltf_freq: str = "15min",
        htf_freq: str = "60min",
        lookback_days: int = 10,
        poll_seconds: int = 30,
        state_path: str | Path = ".tmp/local_paper_state.json",
        pending_fill_grace_seconds: int = 30,
        close_grace_seconds: int = 3,
        align_to_bar_close: bool = True,
        incremental_overlap_days: int = 1,
    ) -> None:
        self._broker = broker
        self._symbol = str(symbol).upper()
        self._quantity = max(1, int(quantity))
        self._strategy_params = dict(strategy_params or {})
        self._ltf_freq = _normalize_freq(ltf_freq)
        self._htf_freq = _normalize_freq(htf_freq)
        self._lookback_days = max(1, int(lookback_days))
        self._poll_seconds = max(1, int(poll_seconds))
        self._pending_fill_grace_seconds = max(1, int(pending_fill_grace_seconds))
        self._close_grace_seconds = max(0, int(close_grace_seconds))
        self._align_to_bar_close = bool(align_to_bar_close)
        self._incremental_overlap_days = max(0, int(incremental_overlap_days))
        self._state_path = Path(state_path)
        self._state = RunnerState.load(self._state_path)
        self._raw_bars_cache = pd.DataFrame(
            columns=["datetime", "open", "high", "low", "close", "volume"]
        )
        self._state.metadata.update(
            {
                "symbol": self._symbol,
                "quantity": self._quantity,
                "ltf_freq": self._ltf_freq,
                "htf_freq": self._htf_freq,
                "lookback_days": self._lookback_days,
                "poll_seconds": self._poll_seconds,
                "close_grace_seconds": self._close_grace_seconds,
                "align_to_bar_close": self._align_to_bar_close,
            }
        )
        self._state.runtime.setdefault("runner_started_at", _utc_now_iso())
        self._state.runtime.setdefault("status", "idle")
        self._save_state()

    @property
    def state(self) -> RunnerState:
        return self._state

    def _save_state(self) -> None:
        self._state.save(self._state_path)

    def _capture_broker_runtime(self) -> None:
        runtime = self._state.runtime
        get_protective = getattr(self._broker, "get_protective_exit", None)
        get_watchdog = getattr(self._broker, "get_watchdog_status", None)
        get_quote = getattr(self._broker, "get_quote_status", None)
        runtime["protective_exit"] = (
            get_protective(self._symbol) if callable(get_protective) else None
        )
        runtime["watchdog"] = (
            get_watchdog() if callable(get_watchdog) else {}
        )
        runtime["quote"] = (
            get_quote() if callable(get_quote) else {}
        )

    def _arm_broker_protection(self, trade: ManagedTrade) -> None:
        arm = getattr(self._broker, "arm_protective_exit", None)
        if callable(arm):
            arm(
                self._symbol,
                trade.side,
                trade.quantity,
                stop_loss=trade.sl,
                take_profit=trade.tp,
            )

    def _disarm_broker_protection(self) -> None:
        disarm = getattr(self._broker, "disarm_protective_exit", None)
        if callable(disarm):
            disarm(self._symbol)

    def _build_idempotency_key(self, bar_time: str, action: str, suffix: str = "") -> str:
        parts = ["local", self._symbol, bar_time, action]
        if suffix:
            parts.append(suffix)
        return "-".join(parts)

    def _update_runtime(
        self,
        *,
        status: Optional[str] = None,
        last_cycle_result: Optional[dict[str, Any]] = None,
        next_cycle_eta: Optional[str] = None,
    ) -> None:
        runtime = self._state.runtime
        runtime["last_cycle_at"] = _utc_now_iso()
        if status is not None:
            runtime["status"] = status
        if last_cycle_result is not None:
            runtime["last_cycle_result"] = last_cycle_result
        if next_cycle_eta is not None:
            runtime["next_cycle_eta"] = next_cycle_eta
        self._capture_broker_runtime()
        self._save_state()

    def _trade_age_seconds(self, trade: ManagedTrade) -> float:
        submitted_at = _parse_iso(trade.submitted_at)
        if submitted_at is None:
            return 0.0
        return max(0.0, (datetime.now(timezone.utc) - submitted_at).total_seconds())

    def _sync_managed_trade(self) -> Optional[dict]:
        trade = self._state.managed_trade
        current = self._broker.position.get_position()
        if trade is None:
            if current.side != "flat" and current.quantity > 0:
                reason = (
                    f"broker has unmanaged position {current.side}x{current.quantity}; "
                    "refusing to auto-trade without local strategy state"
                )
                logger.error("[PaperRunner] {}", reason)
                return {"status": "error", "reason": reason}
            return None

        if trade.status == "pending_entry":
            if current.side == trade.side and current.quantity > 0:
                trade.status = "active"
                trade.quantity = current.quantity
                self._arm_broker_protection(trade)
                self._save_state()
                return None
            if self._trade_age_seconds(trade) > self._pending_fill_grace_seconds:
                logger.warning(
                    "[PaperRunner] Entry never materialized on broker; clearing stale pending state"
                )
                self._disarm_broker_protection()
                self._state.managed_trade = None
                self._save_state()
            return None

        if trade.status == "pending_exit":
            if current.side == "flat" or current.quantity == 0:
                self._disarm_broker_protection()
                self._state.managed_trade = None
                self._save_state()
                return None
            if self._trade_age_seconds(trade) > self._pending_fill_grace_seconds:
                logger.warning(
                    "[PaperRunner] Exit still not reflected on broker after {}s; resuming active management",
                    self._pending_fill_grace_seconds,
                )
                trade.status = "active"
                trade.exit_reason = ""
                self._save_state()
            return None

        if trade.status == "active" and (current.side == "flat" or current.quantity == 0):
            logger.warning(
                "[PaperRunner] Broker is flat but local managed trade exists; clearing stale state"
            )
            self._disarm_broker_protection()
            self._state.managed_trade = None
            self._save_state()
        elif trade.status == "active" and current.side == trade.side and current.quantity > 0:
            trade.quantity = current.quantity
            self._arm_broker_protection(trade)
        return None

    def _merge_raw_bars(self, new_bars: pd.DataFrame) -> pd.DataFrame:
        if new_bars.empty and not self._raw_bars_cache.empty:
            return self._raw_bars_cache

        frames = [df for df in (self._raw_bars_cache, new_bars) if not df.empty]
        if not frames:
            return self._raw_bars_cache

        merged = pd.concat(frames, ignore_index=True)
        merged["datetime"] = pd.to_datetime(merged["datetime"])
        merged = (
            merged.sort_values("datetime")
            .drop_duplicates(subset=["datetime"])
            .reset_index(drop=True)
        )
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=self._lookback_days)
        merged = merged[merged["datetime"] >= cutoff].reset_index(drop=True)
        self._raw_bars_cache = merged
        return self._raw_bars_cache

    async def refresh_market_data(self) -> pd.DataFrame:
        """Bootstrap history once, then fetch only a small recent overlap."""
        if self._raw_bars_cache.empty:
            raw_bars = await asyncio.to_thread(
                self._broker.fetch_recent_bars,
                self._symbol,
                self._lookback_days,
            )
            return self._merge_raw_bars(raw_bars)

        last_dt = pd.Timestamp(self._raw_bars_cache["datetime"].iloc[-1]).to_pydatetime()
        start_dt = last_dt - timedelta(days=self._incremental_overlap_days)

        fetch_bars = getattr(self._broker, "fetch_bars", None)
        if callable(fetch_bars):
            raw_bars = await asyncio.to_thread(
                fetch_bars,
                self._symbol,
                start_dt,
                None,
                self._lookback_days,
            )
        else:
            raw_bars = await asyncio.to_thread(
                self._broker.fetch_recent_bars,
                self._symbol,
                max(1, self._incremental_overlap_days + 1),
            )
        return self._merge_raw_bars(raw_bars)

    async def fetch_signal_frame(self) -> pd.DataFrame:
        raw_bars = await self.refresh_market_data()
        ltf_df = resample_completed_bars(raw_bars, self._ltf_freq)
        htf_df = resample_completed_bars(raw_bars, self._htf_freq)
        if ltf_df.empty:
            return ltf_df
        return precompute_signals(
            ltf_df=ltf_df,
            htf_df=htf_df,
            **self._strategy_params,
        )

    async def process_signal_frame(self, signal_df: pd.DataFrame) -> dict:
        sync_error = self._sync_managed_trade()
        if sync_error:
            return sync_error
        if signal_df.empty:
            return {"status": "skipped", "reason": "no completed bars"}

        latest = signal_df.iloc[-1]
        bar_time = pd.Timestamp(latest["datetime"]).isoformat()
        if self._state.last_processed_bar_time == bar_time:
            return {"status": "skipped", "reason": "bar already processed"}

        current = self._broker.position.get_position()
        trade = self._state.managed_trade

        if trade and trade.status == "active" and current.side == trade.side:
            close_price = float(latest["close"])
            exit_reason = ""
            if trade.side == "long":
                if close_price <= trade.sl:
                    exit_reason = "sl"
                elif close_price >= trade.tp:
                    exit_reason = "tp"
            elif trade.side == "short":
                if close_price >= trade.sl:
                    exit_reason = "sl"
                elif close_price <= trade.tp:
                    exit_reason = "tp"

            if exit_reason:
                result = await self._broker.place_order(
                    "exit",
                    current.quantity,
                    ticker=self._symbol,
                    idempotency_key=self._build_idempotency_key(bar_time, "exit", exit_reason),
                )
                if result.get("status") == "ok":
                    if self._broker.position.get_position().side == "flat":
                        self._disarm_broker_protection()
                        self._state.managed_trade = None
                    else:
                        trade.status = "pending_exit"
                        trade.exit_reason = exit_reason
                        trade.submitted_at = _utc_now_iso()
                self._state.last_processed_bar_time = bar_time
                self._save_state()
                return {"status": result.get("status", "error"), "action": "exit", "reason": exit_reason}

        if current.side == "flat" and trade is None:
            signal = int(latest.get("signal", 0) or 0)
            sl = float(latest.get("sl", 0.0) or 0.0)
            tp = float(latest.get("tp", 0.0) or 0.0)
            if signal in (1, -1) and sl > 0 and tp > 0:
                action = "buy" if signal == 1 else "sell"
                side = "long" if signal == 1 else "short"
                result = await self._broker.place_order(
                    action,
                    self._quantity,
                    ticker=self._symbol,
                    sentiment=side,
                    idempotency_key=self._build_idempotency_key(bar_time, action),
                )
                if result.get("status") == "ok":
                    next_trade = ManagedTrade(
                        status="pending_entry",
                        side=side,
                        quantity=self._quantity,
                        sl=sl,
                        tp=tp,
                        entry_bar_time=bar_time,
                        submitted_at=_utc_now_iso(),
                    )
                    if self._broker.position.get_position().side == side:
                        next_trade.status = "active"
                        self._arm_broker_protection(next_trade)
                    self._state.managed_trade = next_trade
                self._state.last_processed_bar_time = bar_time
                self._save_state()
                return {"status": result.get("status", "error"), "action": action}

        self._state.last_processed_bar_time = bar_time
        self._save_state()
        return {"status": "skipped", "reason": "no action"}

    async def cycle(self) -> dict:
        self._update_runtime(status="running")
        signal_df = await self.fetch_signal_frame()
        result = await self.process_signal_frame(signal_df)
        self._update_runtime(status="idle", last_cycle_result=result)
        return result

    def seconds_until_next_cycle(self, now: datetime | None = None) -> float:
        """Wait until just after the next completed LTF bar closes."""
        if not self._align_to_bar_close:
            return float(self._poll_seconds)

        current = now or datetime.now()
        interval_seconds = _freq_to_seconds(self._ltf_freq)
        now_ts = current.timestamp()
        next_close_ts = ((int(now_ts) // interval_seconds) + 1) * interval_seconds
        delay = next_close_ts - now_ts + self._close_grace_seconds
        return max(1.0, float(delay))

    async def run_forever(self) -> None:
        while True:
            try:
                result = await self.cycle()
                logger.info("[PaperRunner] cycle result: {}", result)
            except Exception as exc:
                self._update_runtime(
                    status="error",
                    last_cycle_result={"status": "error", "reason": str(exc)},
                )
                logger.exception("[PaperRunner] cycle failed: {}", exc)
            delay = self.seconds_until_next_cycle()
            next_eta = datetime.now(timezone.utc) + timedelta(seconds=delay)
            self._update_runtime(
                status="sleeping",
                next_cycle_eta=next_eta.isoformat(),
            )
            logger.info("[PaperRunner] next cycle in {:.1f}s", delay)
            await asyncio.sleep(delay)
