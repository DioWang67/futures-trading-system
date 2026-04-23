from __future__ import annotations

import asyncio
import os
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import pandas as pd
import shioaji as sj
from shioaji import error as sj_error
from loguru import logger

from config import settings
from position_state import PositionState
from risk_manager import RiskManager
from trade_store import TradeStore
from brokers.base import RateLimiter, compute_realized_pnl, route_order


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_delivery_date(value: Any) -> date:
    raw = str(value or "").strip()
    if not raw:
        return date.max
    for fmt in ("%Y/%m/%d", "%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return date.max


@dataclass
class ProtectiveExit:
    ticker: str
    side: str
    quantity: int
    stop_loss: float = 0.0
    take_profit: float = 0.0
    armed_at: str = ""
    status: str = "armed"
    trigger_reason: str = ""
    trigger_price: float = 0.0
    trigger_source: str = "tick_last"
    triggered_at: str = ""
    exit_sent: bool = False
    execution_price_type: str = ""
    submit_price: float = 0.0
    limit_price: float = 0.0
    broker_order_id: str = ""
    protective_event_id: int = 0


class ShioajiBroker:
    """Shioaji broker for TAIFEX futures with local protective exits."""

    # Prefix attached to close-only reasons that *this* broker's watchdog
    # raises. The watchdog only clears the flag when the current reason
    # starts with this prefix, so unrelated close-only states (session
    # down, login failure, protective-exit submission failure) persist
    # until the path that set them clears them.
    _WATCHDOG_CLOSE_ONLY_PREFIX = "watchdog: "

    def __init__(
        self,
        risk_manager: Optional[RiskManager] = None,
        trade_store: Optional[TradeStore] = None,
        notifier: Optional[Any] = None,
        futures_symbol: Optional[str] = None,
        point_value: Optional[float] = None,
        enable_quote_monitor: Optional[bool] = None,
    ) -> None:
        self.api: sj.Shioaji | None = None
        self._contracts: dict[str, Any] = {}
        self.position = PositionState("shioaji")
        self._connected = False
        self._risk_manager = risk_manager
        self._trade_store = trade_store
        self._notifier = notifier
        self._simulation = bool(settings.shioaji.simulation)
        self._futures_symbol = str(
            futures_symbol or settings.shioaji.futures_symbol or "TMF"
        ).upper()
        self._point_value = float(point_value or settings.shioaji.point_value or 10.0)
        self._enable_quote_monitor = bool(
            settings.shioaji.enable_quote_monitor
            if enable_quote_monitor is None else enable_quote_monitor
        )
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        self._protective_lock = threading.Lock()
        self._protective_exits: dict[str, ProtectiveExit] = {}
        self._quote_lock = threading.Lock()
        self._last_tick_at: Optional[datetime] = None
        self._last_tick_price: float = 0.0
        self._quote_stale_seconds = int(settings.shioaji.quote_stale_seconds or 15)
        self._watchdog_interval = int(
            settings.shioaji.protective_watchdog_interval or 5
        )
        self._protective_exit_price_type = str(
            settings.shioaji.protective_exit_price_type or "MKT"
        ).upper()
        self._protective_limit_offset_points = float(
            settings.shioaji.protective_limit_offset_points or 0.0
        )
        self._watchdog_last_reason = ""
        self._watchdog_alert_sent = False
        self._rate_limiter = RateLimiter(max_orders_per_second=2.0)
        # Serialises reconnect so a webhook landing mid-refresh either
        # sees the stable pre-refresh api or waits for the fresh one.
        self._reconnect_lock = threading.RLock()

    @property
    def broker_name(self) -> str:
        return "shioaji"

    @property
    def supported_tickers(self) -> set[str]:
        return {self._futures_symbol}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_simulation(self) -> bool:
        return self._simulation

    @property
    def futures_symbol(self) -> str:
        return self._futures_symbol

    @property
    def point_value(self) -> float:
        return self._point_value

    def attach_event_loop(
        self, loop: Optional[asyncio.AbstractEventLoop] = None
    ) -> None:
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
        self._event_loop = loop

    def _prepare_runtime_environment(self) -> None:
        """Isolate Shioaji cache/log files in simulation mode."""
        if not self._simulation:
            return

        runtime_root = Path(settings.shioaji.runtime_dir).expanduser()
        if not runtime_root.is_absolute():
            runtime_root = Path(__file__).resolve().parent.parent / runtime_root
        shioaji_home = runtime_root / ".shioaji"
        shioaji_home.mkdir(parents=True, exist_ok=True)
        os.environ["HOME"] = str(runtime_root)
        os.environ["USERPROFILE"] = str(runtime_root)
        os.environ["SJ_LOG_PATH"] = str(runtime_root / "shioaji-runtime.log")

    def _resolve_contract(self, symbol: str) -> Any:
        contracts = list(self.api.Contracts.Futures[symbol])
        if not contracts:
            raise RuntimeError(f"No Shioaji futures contracts found for {symbol}")

        non_continuous = [c for c in contracts if not c.code.endswith(("R1", "R2"))]
        today = date.today()
        active = [
            c for c in non_continuous
            if _parse_delivery_date(getattr(c, "delivery_date", "")) >= today
        ]
        candidates = active or non_continuous or contracts
        return min(
            candidates,
            key=lambda c: _parse_delivery_date(getattr(c, "delivery_date", "")),
        )

    def _extract_broker_order_id(self, trade: Any) -> str:
        candidates = []
        if isinstance(trade, dict):
            order = trade.get("order", {})
            status = trade.get("status", {})
            candidates.extend(
                [
                    order.get("id"),
                    order.get("seqno"),
                    order.get("ordno"),
                    status.get("id"),
                ]
            )
        else:
            order = getattr(trade, "order", None)
            status = getattr(trade, "status", None)
            for obj in (order, status, trade):
                if obj is None:
                    continue
                for attr in ("id", "seqno", "ordno"):
                    candidates.append(getattr(obj, attr, ""))
        for value in candidates:
            text = str(value or "").strip()
            if text:
                return text
        return ""

    def _extract_callback_order_id(self, msg: dict) -> str:
        order = msg.get("order", {}) or {}
        status = msg.get("status", {}) or {}
        for key in ("id", "seqno", "ordno", "trade_id"):
            value = str(
                order.get(key) or status.get(key) or msg.get(key) or ""
            ).strip()
            if value:
                return value
        return ""

    def _extract_callback_fill(self, msg: dict) -> tuple[str, int, float]:
        order = msg.get("order", {}) or {}
        status = msg.get("status", {}) or {}
        action = str(order.get("action") or msg.get("action") or "").strip()
        deal_quantity = int(
            status.get("deal_quantity", 0)
            or msg.get("quantity", 0)
            or 0
        )
        deal_price = _safe_float(
            status.get("deal_price", 0.0)
            or msg.get("price", 0.0),
            0.0,
        )
        return action, deal_quantity, deal_price

    def _normalize_callback_status(self, msg: dict) -> str:
        status = msg.get("status", {}) or {}
        operation = msg.get("operation", {}) or {}

        status_text = str(
            status.get("status")
            or status.get("state")
            or status.get("order_status")
            or ""
        ).strip().lower()
        op_type = str(operation.get("op_type") or "").strip().lower()
        op_code = str(operation.get("op_code") or "").strip()

        mapping = {
            "pendingsubmit": "pending_submit",
            "presubmitted": "pre_submitted",
            "submitted": "submitted",
            "filling": "partially_filled",
            "filled": "filled",
            "cancelled": "canceled",
            "canceled": "canceled",
            "failed": "rejected",
            "rejected": "rejected",
        }
        if status_text in mapping:
            return mapping[status_text]
        if op_type == "cancel":
            return "canceled"
        if op_type == "new" and op_code == "00":
            return "submitted"
        return ""

    def _register_quote_monitor(self, contract: Any) -> None:
        if not self._enable_quote_monitor:
            return
        try:
            self.api.quote.set_on_tick_fop_v1_callback(self._tick_callback)
            self.api.quote.subscribe(
                contract,
                quote_type=sj.constant.QuoteType.Tick,
                version=sj.constant.QuoteVersion.v1,
            )
            logger.info(
                "[Shioaji] Tick monitor subscribed for {} ({})",
                self._futures_symbol,
                getattr(contract, "code", self._futures_symbol),
            )
        except Exception as exc:
            logger.warning(
                "[Shioaji] Failed to start tick monitor for {}: {}",
                self._futures_symbol,
                exc,
            )

    def _handle_session_down(self) -> None:
        self._connected = False
        logger.error("[Shioaji] Session down detected")
        if self._risk_manager:
            self._risk_manager.set_close_only(
                True, "Shioaji session down; new entries blocked",
            )
        # Persist the event independently of the notifier so we still
        # have an audit trail when Telegram is down or this callback
        # fires with no running event loop attached.
        if self._trade_store:
            try:
                self._trade_store.record_risk_event(
                    "broker_session_down",
                    "shioaji",
                    f"session down for {self._futures_symbol}; switched to close-only",
                )
            except Exception as exc:
                logger.error(
                    "[Shioaji] Failed to record session-down event: {}", exc,
                )
        if self._notifier:
            self._schedule_coroutine(
                self._notifier.send_error(
                    f"Shioaji session down for {self._futures_symbol}; switched to close-only"
                )
            )

    def _schedule_coroutine(self, coro: Any) -> bool:
        loop = self._event_loop
        if loop is None or loop.is_closed():
            logger.error(
                "[Shioaji] No event loop attached; cannot schedule background task"
            )
            return False
        asyncio.run_coroutine_threadsafe(coro, loop)
        return True

    def _tick_callback(self, exchange: Any, tick: Any) -> None:
        code = str(getattr(tick, "code", "") or "").upper()
        if not code:
            return
        last_price = _safe_float(
            getattr(tick, "close", None)
            or getattr(tick, "last_price", None)
            or getattr(tick, "avg_price", None),
            0.0,
        )
        if last_price <= 0:
            return
        with self._quote_lock:
            self._last_tick_at = datetime.now(timezone.utc)
            self._last_tick_price = last_price

        with self._protective_lock:
            protective = self._protective_exits.get(self._futures_symbol)
            if not protective or protective.exit_sent:
                return

            reason = ""
            if protective.side == "long":
                if protective.stop_loss > 0 and last_price <= protective.stop_loss:
                    reason = "stop_loss"
                elif (
                    protective.take_profit > 0
                    and last_price >= protective.take_profit
                ):
                    reason = "take_profit"
            elif protective.side == "short":
                if protective.stop_loss > 0 and last_price >= protective.stop_loss:
                    reason = "stop_loss"
                elif (
                    protective.take_profit > 0
                    and last_price <= protective.take_profit
                ):
                    reason = "take_profit"

            if not reason:
                return

            protective.exit_sent = True
            protective.status = "triggered"
            protective.trigger_reason = reason
            protective.trigger_price = last_price
            protective.trigger_source = "tick_last"
            protective.triggered_at = datetime.now(timezone.utc).isoformat()

        logger.warning(
            "[Shioaji] Protective exit triggered for {} at {:.1f} ({})",
            self._futures_symbol,
            last_price,
            reason,
        )
        if self._notifier:
            self._schedule_coroutine(
                self._notifier.send_protective_event(
                    self.broker_name,
                    self._futures_symbol,
                    protective.side,
                    protective.quantity,
                    reason,
                    "triggered",
                    trigger_price=last_price,
                )
            )
        scheduled = self._schedule_coroutine(
            self._submit_protective_exit(self._futures_symbol, reason, last_price)
        )
        if not scheduled:
            with self._protective_lock:
                protective = self._protective_exits.get(self._futures_symbol)
                if protective:
                    protective.exit_sent = False
                    protective.status = "armed"

    async def _submit_protective_exit(
        self,
        ticker: str,
        reason: str,
        trigger_price: float,
    ) -> None:
        with self._protective_lock:
            protective = self._protective_exits.get(ticker)
            if not protective:
                return
            quantity = protective.quantity
            side = protective.side
            stop_loss = protective.stop_loss
            take_profit = protective.take_profit
            if protective.trigger_price <= 0:
                protective.trigger_price = trigger_price
            if not protective.trigger_reason:
                protective.trigger_reason = reason
            if not protective.triggered_at:
                protective.triggered_at = datetime.now(timezone.utc).isoformat()

        execution_price_type = self._protective_exit_price_type
        submit_price = trigger_price
        limit_price = 0.0
        if execution_price_type == "LMT":
            offset = self._protective_limit_offset_points
            if side == "long":
                limit_price = max(0.0, trigger_price - offset)
            else:
                limit_price = trigger_price + offset
            submit_price = limit_price

        protective_event_id = 0
        if self._trade_store:
            protective_event_id = self._trade_store.record_protective_event(
                self.broker_name,
                ticker,
                side,
                quantity,
                reason,
                stop_loss,
                take_profit,
                trigger_price,
                execution_price_type,
                submit_price=submit_price,
            )
            with self._protective_lock:
                protective = self._protective_exits.get(ticker)
                if protective:
                    protective.protective_event_id = protective_event_id
                    protective.execution_price_type = execution_price_type
                    protective.submit_price = submit_price
                    protective.limit_price = limit_price

        result = await self.place_order(
            "exit",
            quantity,
            ticker=ticker,
            price_type=execution_price_type,
            limit_price=limit_price,
            idempotency_key=f"protect-{ticker}-{reason}-{uuid4().hex[:12]}",
        )
        status = str(result.get("status") or "")
        orders = result.get("orders") or []
        order_result = orders[-1] if orders else {}
        broker_order_id = str(order_result.get("broker_order_id") or "").strip()
        if status == "ok":
            if self._trade_store and protective_event_id:
                self._trade_store.update_protective_event(
                    protective_event_id,
                    status="submitted",
                    submit_price=submit_price,
                    execution_price_type=execution_price_type,
                    broker_order_id=broker_order_id,
                    details=str(result),
                    mark_submitted=True,
                )
            with self._protective_lock:
                protective = self._protective_exits.get(ticker)
                if protective:
                    protective.status = "submitted"
                    protective.execution_price_type = execution_price_type
                    protective.submit_price = submit_price
                    protective.limit_price = limit_price
                    protective.broker_order_id = broker_order_id
            logger.info(
                "[Shioaji] Protective exit submitted for {} at {:.1f} ({}, type={})",
                ticker,
                submit_price,
                reason,
                execution_price_type,
            )
            if self._notifier:
                self._schedule_coroutine(
                    self._notifier.send_protective_event(
                        self.broker_name,
                        ticker,
                        side,
                        quantity,
                        reason,
                        "submitted",
                        trigger_price=trigger_price,
                        submit_price=submit_price,
                        execution_price_type=execution_price_type,
                    )
                )
            return

        logger.error(
            "[Shioaji] Protective exit failed for {}: {}",
            ticker,
            result,
        )
        if self._trade_store and protective_event_id:
            self._trade_store.update_protective_event(
                protective_event_id,
                status="failed",
                submit_price=submit_price,
                execution_price_type=execution_price_type,
                details=str(result),
            )
        if self._risk_manager:
            self._risk_manager.set_close_only(
                True,
                f"protective exit failed for {ticker}: {result.get('reason') or status}",
            )
        if self._notifier:
            self._schedule_coroutine(
                self._notifier.send_protective_event(
                    self.broker_name,
                    ticker,
                    side,
                    quantity,
                    reason,
                    "failed",
                    trigger_price=trigger_price,
                    submit_price=submit_price,
                    execution_price_type=execution_price_type,
                )
            )
        with self._protective_lock:
            protective = self._protective_exits.get(ticker)
            if protective:
                protective.exit_sent = False
                protective.status = "failed"
                protective.trigger_reason = reason
                protective.trigger_price = trigger_price
                protective.execution_price_type = execution_price_type
                protective.submit_price = submit_price
                protective.limit_price = limit_price

    def arm_protective_exit(
        self,
        ticker: str,
        side: str,
        quantity: int,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
    ) -> None:
        ticker = str(ticker or self._futures_symbol).upper()
        quantity = max(0, int(quantity))
        stop_loss = _safe_float(stop_loss, 0.0)
        take_profit = _safe_float(take_profit, 0.0)
        if quantity <= 0 or (stop_loss <= 0 and take_profit <= 0):
            self.disarm_protective_exit(ticker)
            return
        normalized_side = str(side or "").lower()
        with self._protective_lock:
            existing = self._protective_exits.get(ticker)
            if existing and (
                existing.exit_sent or existing.status in {"triggered", "submitted"}
            ):
                existing.side = normalized_side
                existing.quantity = quantity
                existing.stop_loss = stop_loss
                existing.take_profit = take_profit
                log_message = "[Shioaji] Protective exit updated in-flight: {} {} x{} SL={} TP={}"
            else:
                self._protective_exits[ticker] = ProtectiveExit(
                    ticker=ticker,
                    side=normalized_side,
                    quantity=quantity,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    armed_at=datetime.now(timezone.utc).isoformat(),
                )
                log_message = "[Shioaji] Protective exit armed: {} {} x{} SL={} TP={}"
        logger.info(
            log_message,
            ticker,
            normalized_side,
            quantity,
            stop_loss or "-",
            take_profit or "-",
        )

    def _is_simulation_submit_timeout(self, exc: Exception) -> bool:
        if not self._simulation:
            return False
        topic = str(getattr(exc, "topic", "") or "").strip()
        return isinstance(exc, sj_error.TimeoutError) and topic == "api/v1/paper/place_order"

    def disarm_protective_exit(self, ticker: str) -> None:
        ticker = str(ticker or self._futures_symbol).upper()
        with self._protective_lock:
            removed = self._protective_exits.pop(ticker, None)
        if removed:
            logger.info("[Shioaji] Protective exit disarmed for {}", ticker)

    def get_protective_exit(self, ticker: str = "") -> Optional[dict]:
        ticker = str(ticker or self._futures_symbol).upper()
        with self._protective_lock:
            protective = self._protective_exits.get(ticker)
            return None if protective is None else dict(protective.__dict__)

    def get_quote_status(self) -> dict[str, Any]:
        with self._quote_lock:
            last_tick_at = self._last_tick_at
            last_tick_price = self._last_tick_price

        age_seconds = None
        if last_tick_at is not None:
            age_seconds = max(
                0.0,
                (datetime.now(timezone.utc) - last_tick_at).total_seconds(),
            )
        is_stale = age_seconds is None or age_seconds > self._quote_stale_seconds
        return {
            "last_tick_at": last_tick_at.isoformat() if last_tick_at else "",
            "last_tick_price": last_tick_price,
            "age_seconds": age_seconds,
            "stale_after_seconds": self._quote_stale_seconds,
            "is_stale": is_stale,
        }

    async def wait_for_fresh_quote(
        self,
        timeout_seconds: float = 15.0,
        poll_interval: float = 0.25,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + max(0.1, float(timeout_seconds))
        while time.monotonic() < deadline:
            quote = self.get_quote_status()
            if not quote["is_stale"] and _safe_float(quote["last_tick_price"], 0.0) > 0:
                return quote
            await asyncio.sleep(max(0.05, float(poll_interval)))
        return self.get_quote_status()

    async def trigger_protective_exit(
        self,
        ticker: str = "",
        reason: str = "manual_test",
        trigger_price: float = 0.0,
    ) -> dict[str, Any]:
        ticker = str(ticker or self._futures_symbol).upper()
        with self._protective_lock:
            protective = self._protective_exits.get(ticker)
            if protective is None:
                return {
                    "status": "error",
                    "reason": f"no protective exit armed for {ticker}",
                }
            if protective.exit_sent:
                return {
                    "status": "skipped",
                    "reason": f"protective exit already in-flight for {ticker}",
                }
            if trigger_price <= 0:
                trigger_price = max(
                    0.0,
                    _safe_float(self._last_tick_price, 0.0),
                    _safe_float(protective.stop_loss, 0.0),
                    _safe_float(protective.take_profit, 0.0),
                )
            if trigger_price <= 0:
                return {
                    "status": "error",
                    "reason": f"no trigger price available for {ticker}",
                }
            protective.exit_sent = True
            protective.status = "triggered"
            protective.trigger_reason = str(reason or "manual_test")
            protective.trigger_price = trigger_price
            protective.trigger_source = "manual_test"
            protective.triggered_at = datetime.now(timezone.utc).isoformat()
            side = protective.side
            quantity = protective.quantity

        logger.warning(
            "[Shioaji] Manual protective exit triggered for {} at {:.1f} ({})",
            ticker,
            trigger_price,
            reason,
        )
        if self._notifier:
            self._schedule_coroutine(
                self._notifier.send_protective_event(
                    self.broker_name,
                    ticker,
                    side,
                    quantity,
                    str(reason or "manual_test"),
                    "triggered",
                    trigger_price=trigger_price,
                )
            )
        await self._submit_protective_exit(ticker, str(reason or "manual_test"), trigger_price)
        protective_now = self.get_protective_exit(ticker)
        return {
            "status": "ok",
            "ticker": ticker,
            "trigger_price": trigger_price,
            "protective_exit": protective_now,
        }

    def get_watchdog_status(self) -> dict[str, Any]:
        position = self.position.get_position()
        protective = self.get_protective_exit(self._futures_symbol)
        quote = self.get_quote_status()
        return {
            "connected": self._connected,
            "symbol": self._futures_symbol,
            "position_side": position.side,
            "position_qty": position.quantity,
            "protective_exit": protective,
            "quote": quote,
            "close_only": (
                self._risk_manager.is_close_only if self._risk_manager else False
            ),
            "close_only_reason": (
                self._risk_manager.close_only_reason if self._risk_manager else ""
            ),
            "watchdog_reason": self._watchdog_last_reason,
        }

    def evaluate_protective_watchdog(self) -> dict[str, Any]:
        position = self.position.get_position()
        protective = self.get_protective_exit(self._futures_symbol)
        quote = self.get_quote_status()
        previous_reason = self._watchdog_last_reason

        reason = ""
        if position.side == "flat" or position.quantity == 0:
            if protective:
                self.disarm_protective_exit(self._futures_symbol)
            reason = ""
        elif protective is None:
            reason = f"position {position.side}x{position.quantity} has no protective exit"
        elif protective["side"] != position.side:
            reason = (
                f"protective side mismatch: position={position.side}, "
                f"protective={protective['side']}"
            )
        elif int(protective["quantity"]) != int(position.quantity):
            self.arm_protective_exit(
                self._futures_symbol,
                position.side,
                position.quantity,
                stop_loss=protective.get("stop_loss", 0.0) or 0.0,
                take_profit=protective.get("take_profit", 0.0) or 0.0,
            )
            protective = self.get_protective_exit(self._futures_symbol)
        if not reason and position.side != "flat" and quote["is_stale"]:
            reason = (
                f"quote feed stale for {quote['age_seconds'] or 'unknown'}s "
                f"(limit {quote['stale_after_seconds']}s)"
            )

        self._watchdog_last_reason = reason
        if self._risk_manager:
            if reason:
                # Prefix so we can distinguish our own assertions from
                # close-only flags raised by the login / session / protective
                # exit paths. Only watchdog can clear its own flag.
                self._risk_manager.set_close_only(
                    True, self._WATCHDOG_CLOSE_ONLY_PREFIX + reason,
                )
            elif (
                self._risk_manager.is_close_only
                and self._risk_manager.close_only_reason.startswith(
                    self._WATCHDOG_CLOSE_ONLY_PREFIX
                )
            ):
                self._risk_manager.set_close_only(False)

        return self.get_watchdog_status()

    async def run_watchdog(self) -> None:
        while True:
            status = self.evaluate_protective_watchdog()
            reason = str(status.get("watchdog_reason") or "")
            if reason and self._notifier and not self._watchdog_alert_sent:
                await self._notifier.send_risk_alert(
                    f"[shioaji] Protective watchdog: {reason}"
                )
                self._watchdog_alert_sent = True
            elif not reason:
                self._watchdog_alert_sent = False
            await asyncio.sleep(self._watchdog_interval)

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
    def login(self) -> None:
        cfg = settings.shioaji
        with self._reconnect_lock:
            try:
                self._prepare_runtime_environment()
                self.api = sj.Shioaji(simulation=cfg.simulation)
                self.api.login(
                    api_key=cfg.api_key.get_secret_value(),
                    secret_key=cfg.secret_key.get_secret_value(),
                    contracts_timeout=10_000,
                    subscribe_trade=True,
                )

                if not cfg.simulation and cfg.ca_path:
                    self.api.activate_ca(
                        ca_path=cfg.ca_path,
                        ca_passwd=cfg.ca_password.get_secret_value(),
                    )
                elif cfg.simulation:
                    logger.info("[Shioaji] Running in simulation / paper-trading mode")

                contract = self._resolve_contract(self._futures_symbol)
                self._contracts[self._futures_symbol] = contract
                self.api.set_order_callback(self._order_callback)
                self.api.set_session_down_callback(self._handle_session_down)
                self._register_quote_monitor(contract)
                self._sync_position()
                self._connected = True
                if self._risk_manager:
                    self._risk_manager.set_close_only(False)
                logger.info(
                    "[Shioaji] Login successful ({}) symbol={} contract={} point_value={}",
                    "simulation" if cfg.simulation else "live",
                    self._futures_symbol,
                    getattr(contract, "code", self._futures_symbol),
                    self._point_value,
                )
                if self._notifier:
                    self._schedule_coroutine(
                        self._notifier.send_system(
                            "Shioaji ready: "
                            f"{self._futures_symbol} "
                            f"({'simulation' if cfg.simulation else 'live'}) "
                            f"protective={self._protective_exit_price_type}"
                        )
                    )
            except Exception as e:
                logger.error("[Shioaji] Login failed: {}", e)
                self._connected = False
                self.api = None
                self._contracts.clear()
                if self._risk_manager:
                    self._risk_manager.set_close_only(
                        True, f"Shioaji login failed: {e}",
                    )
                raise

    def logout(self) -> None:
        with self._reconnect_lock:
            api = self.api
            if api:
                try:
                    api.logout()
                    logger.info("[Shioaji] Logged out")
                except Exception as e:
                    logger.error("[Shioaji] Logout error: {}", e)
            self.api = None
            self._contracts.clear()
            self._connected = False

    def reconnect(self) -> None:
        logger.info("[Shioaji] Reconnecting (token refresh)...")
        if self._risk_manager:
            self._risk_manager.set_close_only(
                True, "Shioaji reconnect in progress",
            )
        if self._notifier:
            self._schedule_coroutine(
                self._notifier.send_system(
                    f"Shioaji reconnect started for {self._futures_symbol}"
                )
            )
        # Keep logout+login atomic against submits. The lock is reentrant so
        # the nested calls below stay serialized with _submit_order_sync.
        with self._reconnect_lock:
            try:
                self.logout()
            except Exception:
                pass
            self.login()

    # ------------------------------------------------------------------
    # Position sync
    # ------------------------------------------------------------------
    def _sync_position(self) -> None:
        """Sync position from broker on startup (crash recovery)."""
        try:
            positions = self.api.list_positions(self.api.futopt_account)
            for pos in positions:
                code = str(getattr(pos, "code", "") or "")
                if self._futures_symbol not in code:
                    continue
                qty = getattr(pos, "quantity", 0)
                direction = getattr(pos, "direction", "")
                if qty <= 0:
                    continue
                side = "long" if direction == "Buy" else "short"
                raw_price = (
                    getattr(pos, "price", None)
                    or getattr(pos, "avg_price", None)
                    or getattr(pos, "average_price", None)
                    or 0.0
                )
                entry_price = _safe_float(raw_price, 0.0)
                if entry_price <= 0:
                    # Broker didn't return an average cost. Trust the last
                    # persisted snapshot if it matches this side/quantity —
                    # otherwise realized PnL would be silently reported as
                    # zero on the next close, which defeats the daily-loss
                    # risk gate.
                    fallback_price = self._recover_entry_price_from_store(
                        side, qty
                    )
                    if fallback_price > 0:
                        entry_price = fallback_price
                        logger.warning(
                            "[Shioaji] Synced {} x{} without broker avg cost; "
                            "recovered entry_price={} from local snapshot",
                            side, qty, entry_price,
                        )
                    else:
                        logger.error(
                            "[Shioaji] Synced {} x{} without broker avg cost "
                            "and no usable local snapshot; entering close-only "
                            "mode (operator must reconcile PnL manually)",
                            side, qty,
                        )
                        if self._risk_manager:
                            self._risk_manager.set_close_only(
                                True,
                                f"shioaji synced {side}x{qty} with unknown "
                                f"avg cost; PnL reporting is disabled until "
                                f"operator reconciles",
                            )
                self.position.update_position(
                    side, qty, entry_price=entry_price,
                )
                logger.info(
                    "[Shioaji] Synced position from broker: {} x{} @{}",
                    side, qty, entry_price,
                )
                if self._trade_store:
                    self._trade_store.save_position_snapshot(
                        "shioaji", side, qty, entry_price,
                    )
                return
            logger.info(
                "[Shioaji] No existing {} position found on broker",
                self._futures_symbol,
            )
        except Exception as e:
            logger.warning("[Shioaji] Broker position sync failed: {}", e)

        if self._trade_store:
            snapshot = self._trade_store.get_latest_position("shioaji")
            if snapshot and snapshot["side"] != "flat":
                self.position.update_position(
                    snapshot["side"], snapshot["quantity"], snapshot["entry_price"]
                )
                logger.info(
                    "[Shioaji] Restored position from DB: {} x{}",
                    snapshot["side"], snapshot["quantity"],
                )
                return

        logger.info("[Shioaji] Starting with flat position")

    def _recover_entry_price_from_store(self, side: str, qty: int) -> float:
        """Best-effort lookup of the last persisted entry price when the
        broker fails to return an average cost after a reconnect."""
        if not self._trade_store:
            return 0.0
        try:
            snapshot = self._trade_store.get_latest_position("shioaji")
        except Exception as exc:
            logger.warning(
                "[Shioaji] Failed to load position snapshot while "
                "recovering entry price: {}",
                exc,
            )
            return 0.0
        if not snapshot:
            return 0.0
        if snapshot.get("side") != side:
            return 0.0
        if int(snapshot.get("quantity") or 0) != int(qty):
            return 0.0
        price = _safe_float(snapshot.get("entry_price"), 0.0)
        return price if price > 0 else 0.0

    def _order_callback(self, stat: Any, msg: dict) -> None:
        """Callback for order status and fill updates."""
        try:
            logger.info("[Shioaji] Order callback: stat={}, msg={}", stat, msg)
            order = msg.get("order", {}) or {}
            action = str(order.get("action") or "")
            status = msg.get("status", {}) or {}
            broker_order_id = self._extract_callback_order_id(msg)
            normalized_status = self._normalize_callback_status(msg)
            if (
                not normalized_status
                and str(msg.get("action") or "").strip()
                and _safe_float(msg.get("price"), 0.0) > 0
                and int(msg.get("quantity", 0) or 0) > 0
            ):
                normalized_status = "filled"
            error_message = str(
                (msg.get("operation", {}) or {}).get("op_msg")
                or status.get("errmsg")
                or "",
            ).strip()

            if self._trade_store and broker_order_id and normalized_status:
                self._trade_store.update_order_status_by_broker_order_id(
                    "shioaji",
                    broker_order_id,
                    normalized_status,
                    error_message=error_message,
                )
                protective_snapshot = self.get_protective_exit(self._futures_symbol)
                if (
                    protective_snapshot
                    and protective_snapshot.get("protective_event_id")
                    and protective_snapshot.get("broker_order_id") == broker_order_id
                ):
                    self._trade_store.update_protective_event(
                        int(protective_snapshot["protective_event_id"]),
                        status=normalized_status,
                        broker_order_id=broker_order_id,
                        details=error_message or str(msg),
                    )

            action, deal_quantity, deal_price = self._extract_callback_fill(msg)

            if deal_quantity <= 0:
                return

            fill_info = self.position.apply_fill(action, deal_quantity, deal_price)
            realized_pnl = compute_realized_pnl(
                action=action,
                closed_qty=fill_info["closed_qty"],
                fill_price=deal_price,
                prev_side=fill_info["prev_side"],
                prev_entry=fill_info["prev_entry"],
                point_value=self._point_value,
            )
            is_closing = fill_info["closed_qty"] > 0

            protective_snapshot = self.get_protective_exit(self._futures_symbol)
            if (
                self._trade_store
                and protective_snapshot
                and protective_snapshot.get("protective_event_id")
                and (
                    not broker_order_id
                    or protective_snapshot.get("broker_order_id") == broker_order_id
                )
            ):
                trigger_price = _safe_float(
                    protective_snapshot.get("trigger_price"), 0.0
                )
                slippage_points = 0.0
                if trigger_price > 0:
                    if protective_snapshot.get("side") == "long":
                        slippage_points = deal_price - trigger_price
                    else:
                        slippage_points = trigger_price - deal_price
                self._trade_store.update_protective_event(
                    int(protective_snapshot["protective_event_id"]),
                    status="filled",
                    fill_price=deal_price,
                    slippage_points=slippage_points,
                    broker_order_id=broker_order_id,
                    details=str(msg),
                    mark_filled=True,
                )
                with self._protective_lock:
                    protective = self._protective_exits.get(self._futures_symbol)
                    if protective:
                        protective.status = "filled"
                if self._notifier:
                    self._schedule_coroutine(
                        self._notifier.send_protective_event(
                            self.broker_name,
                            self._futures_symbol,
                            str(protective_snapshot.get("side") or ""),
                            int(protective_snapshot.get("quantity") or 0),
                            str(protective_snapshot.get("trigger_reason") or ""),
                            "filled",
                            trigger_price=trigger_price,
                            submit_price=_safe_float(
                                protective_snapshot.get("submit_price"), 0.0
                            ),
                            fill_price=deal_price,
                            slippage_points=slippage_points,
                            execution_price_type=str(
                                protective_snapshot.get("execution_price_type") or ""
                            ),
                        )
                    )

            if self._trade_store:
                order_id = self._trade_store.claim_pending_order(
                    "shioaji",
                    action,
                    fill_qty=deal_quantity,
                    broker_order_id=broker_order_id,
                )
                self._trade_store.record_fill(
                    "shioaji", action, deal_quantity, deal_price,
                    pnl=realized_pnl,
                    order_id=order_id,
                )
                pos = self.position.get_position()
                self._trade_store.save_position_snapshot(
                    "shioaji", pos.side, pos.quantity, pos.entry_price
                )

            if self._risk_manager and is_closing:
                self._risk_manager.record_fill(
                    "shioaji", realized_pnl, is_closing_trade=True,
                )

            pos_now = self.position.get_position()
            if pos_now.side == "flat" or pos_now.quantity == 0:
                self.disarm_protective_exit(self._futures_symbol)

            if self._notifier and self._event_loop and not self._event_loop.is_closed():
                asyncio.run_coroutine_threadsafe(
                    self._notifier.send_fill(
                        "shioaji", action, deal_quantity, deal_price,
                    ),
                    self._event_loop,
                )

        except Exception:
            logger.exception("[Shioaji] Order callback error")

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------
    def _resolve_price_type(
        self,
        price_type: str = "",
    ) -> tuple[str, Any]:
        normalized = str(price_type or "MKT").strip().upper()
        if normalized not in {"MKT", "MKP", "LMT"}:
            normalized = "MKT"
        resolved = getattr(sj.constant.FuturesPriceType, normalized, None)
        if resolved is None:
            # Mismatch between the configured price type and the installed
            # shioaji SDK. MKP in particular was introduced in later SDKs;
            # fall back to MKT loudly rather than silently.
            logger.error(
                "[Shioaji] FuturesPriceType.{} not found in shioaji SDK "
                "({}); falling back to MKT. Upgrade shioaji or choose a "
                "different SHIOAJI_PROTECTIVE_EXIT_PRICE_TYPE.",
                normalized,
                getattr(sj, "__version__", "unknown"),
            )
            normalized = "MKT"
            resolved = sj.constant.FuturesPriceType.MKT
        return normalized, resolved

    def _submit_order_sync(
        self,
        action: str,
        quantity: int,
        ticker: str,
        price_type: str = "",
        limit_price: float = 0.0,
    ) -> dict:
        # Reject early if the broker is mid-reconnect or hasn't finished
        # login yet — otherwise we'd dereference a half-torn api handle.
        normalized_price_type, sj_price_type = self._resolve_price_type(price_type)
        order_price = float(limit_price or 0.0) if normalized_price_type == "LMT" else 0.0

        try:
            # Hold the session lock through contract lookup and place_order so
            # logout/relogin cannot swap out ``self.api`` mid-submit.
            with self._reconnect_lock:
                api = self.api
                if api is None or not self._connected:
                    return {
                        "status": "error",
                        "reason": "shioaji api not ready (reconnecting or disconnected)",
                    }
                contract = self._contracts.get(ticker)
                if contract is None:
                    return {"status": "error", "reason": f"unsupported ticker: {ticker}"}

                sj_action = sj.constant.Action.Buy if action == "Buy" else sj.constant.Action.Sell
                order = api.Order(
                    price=order_price,
                    quantity=quantity,
                    action=sj_action,
                    price_type=sj_price_type,
                    order_type=sj.constant.OrderType.IOC,
                    octype=sj.constant.FuturesOCType.Auto,
                    account=api.futopt_account,
                )
                trade = api.place_order(contract, order)
        except Exception as exc:
            if self._is_simulation_submit_timeout(exc):
                logger.warning(
                    "[Shioaji] Simulation place_order raised after submit; "
                    "treating as submitted and waiting for callback: {}",
                    exc,
                )
                time.sleep(0.5)
                return {
                    "status": "submitted",
                    "action": action,
                    "quantity": quantity,
                    "ticker": ticker,
                    "reason": "simulation submit accepted but API raised",
                    "price_type": normalized_price_type,
                    "limit_price": order_price,
                }
            raise

        broker_order_id = self._extract_broker_order_id(trade)
        logger.info(
            "[Shioaji] Order submitted: {} {} x{} -> {} (broker_order_id={})",
            ticker,
            action,
            quantity,
            trade,
            broker_order_id or "-",
        )
        return {
            "status": "submitted",
            "action": action,
            "quantity": quantity,
            "ticker": ticker,
            "price_type": normalized_price_type,
            "limit_price": order_price,
            "trade": str(trade),
            "broker_order_id": broker_order_id,
        }

    async def _submit_order(
        self,
        action: str,
        quantity: int,
        ticker: str,
        price_type: str = "",
        limit_price: float = 0.0,
    ) -> dict:
        return await asyncio.to_thread(
            self._submit_order_sync,
            action,
            quantity,
            ticker,
            price_type,
            limit_price,
        )

    def fetch_bars(
        self,
        ticker: str = "",
        start_date: Any | None = None,
        end_date: Any | None = None,
        lookback_days: int = 10,
    ) -> pd.DataFrame:
        """Fetch raw 1-minute bars from the current Shioaji session."""
        if not self.api or not self._connected:
            raise RuntimeError("Shioaji broker is not connected")

        ticker = str(ticker or self._futures_symbol).upper()
        contract = self._contracts.get(ticker)
        if contract is None:
            raise RuntimeError(f"unsupported ticker: {ticker}")

        def _coerce_date(value: Any) -> datetime:
            if value is None:
                return datetime.now()
            if isinstance(value, datetime):
                return value
            return pd.Timestamp(value).to_pydatetime()

        end_dt = _coerce_date(end_date)
        if start_date is None:
            start_dt = end_dt - timedelta(days=max(1, int(lookback_days)))
        else:
            start_dt = _coerce_date(start_date)
        if start_dt > end_dt:
            start_dt, end_dt = end_dt, start_dt

        kbars = self.api.kbars(
            contract=contract,
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
        )

        df = pd.DataFrame({**kbars})
        if df.empty:
            return pd.DataFrame(
                columns=["datetime", "open", "high", "low", "close", "volume"]
            )

        df.rename(columns={"ts": "datetime"}, inplace=True)
        df["datetime"] = pd.to_datetime(df["datetime"])
        if "Open" in df.columns:
            df.columns = [c.lower() for c in df.columns]
        df = df[["datetime", "open", "high", "low", "close", "volume"]]
        return (
            df.sort_values("datetime")
            .drop_duplicates(subset=["datetime"])
            .reset_index(drop=True)
        )

    def fetch_recent_bars(
        self,
        ticker: str = "",
        lookback_days: int = 10,
    ) -> pd.DataFrame:
        return self.fetch_bars(
            ticker=ticker or self._futures_symbol,
            lookback_days=lookback_days,
        )

    async def place_order(
        self, action: str, quantity: int, idempotency_key: str = "", **kwargs
    ) -> dict:
        self.attach_event_loop()
        ticker = str(kwargs.get("ticker") or "").upper()
        if not ticker:
            ticker = self._futures_symbol
        if ticker not in self.supported_tickers:
            return {
                "status": "error",
                "broker": self.broker_name,
                "reason": (
                    f"unsupported ticker: {ticker}; "
                    f"Shioaji broker is configured for "
                    f"{self._futures_symbol} "
                    f"(set SHIOAJI_FUTURES_SYMBOL to change)"
                ),
            }
        price_type = str(kwargs.get("price_type") or "").strip().upper()
        limit_price = _safe_float(kwargs.get("limit_price"), 0.0)

        return await route_order(
            broker_name=self.broker_name,
            position=self.position,
            action=action,
            quantity=quantity,
            submit_fn=lambda order_action, order_qty: self._submit_order(
                order_action,
                order_qty,
                ticker,
                price_type=price_type,
                limit_price=limit_price,
            ),
            is_connected=self._connected and self.api is not None,
            risk_manager=self._risk_manager,
            trade_store=self._trade_store,
            idempotency_key=idempotency_key,
            rate_limiter=self._rate_limiter,
        )
