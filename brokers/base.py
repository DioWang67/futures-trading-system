"""
Broker protocol & shared order-routing logic.

Every concrete broker implements ``BrokerProtocol``.  The shared
``route_order`` helper encodes position-aware flip logic so that
ShioajiBroker and RithmicBroker don't duplicate it.
"""

from __future__ import annotations

import asyncio
import time
import threading
from typing import Any, Optional, Protocol, runtime_checkable

from loguru import logger

from position_state import PositionState
from risk_manager import RiskManager
from trade_store import TradeStore


def compute_realized_pnl(
    action: str,
    closed_qty: int,
    fill_price: float,
    prev_side: str,
    prev_entry: float,
    point_value: float,
) -> float:
    """Compute realized PnL (in quote currency) for the closed portion of
    a fill. ``point_value`` is the $ value per 1.0 of price movement per
    contract (e.g. MXF = 50 TWD, MES = 5 USD)."""
    if closed_qty <= 0 or fill_price <= 0 or prev_entry <= 0:
        return 0.0
    if prev_side == "long" and action == "Sell":
        return (fill_price - prev_entry) * closed_qty * point_value
    if prev_side == "short" and action == "Buy":
        return (prev_entry - fill_price) * closed_qty * point_value
    return 0.0


def _submit_ok(result: Any) -> bool:
    """Treat a broker submit result as successful only when it returned a
    dict whose status is an accepted-by-exchange state. Anything else
    (error, rejected, missing status) is a failure for routing purposes."""
    if not isinstance(result, dict):
        return False
    status = str(result.get("status", "")).lower()
    return status in {"submitted", "ok", "accepted", "filled"}


@runtime_checkable
class BrokerProtocol(Protocol):
    """Minimum interface every broker must satisfy."""

    position: PositionState

    async def place_order(self, action: str, quantity: int) -> dict: ...

    @property
    def broker_name(self) -> str: ...

    @property
    def is_connected(self) -> bool: ...


class RateLimiter:
    """Token-bucket rate limiter for order submission."""

    def __init__(self, max_orders_per_second: float = 2.0) -> None:
        self._lock = threading.Lock()
        self._async_lock = asyncio.Lock()
        self._min_interval = 1.0 / max_orders_per_second
        self._last_order_time = 0.0

    def acquire(self) -> bool:
        """Returns True if order is allowed, False if rate-limited."""
        with self._lock:
            now = time.monotonic()
            if now - self._last_order_time < self._min_interval:
                return False
            self._last_order_time = now
            return True

    def wait(self) -> None:
        """Block until rate limit allows next order."""
        with self._lock:
            now = time.monotonic()
            wait_time = self._min_interval - (now - self._last_order_time)
            if wait_time > 0:
                time.sleep(wait_time)
            self._last_order_time = time.monotonic()

    async def wait_async(self) -> None:
        """Async-friendly wait that does not block the event loop."""
        async with self._async_lock:
            with self._lock:
                now = time.monotonic()
                wait_time = self._min_interval - (now - self._last_order_time)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            with self._lock:
                self._last_order_time = time.monotonic()


# Default orders/second when the caller does not supply a rate limiter.
# Kept conservative because both Shioaji and Rithmic have per-account
# gateway throttles that go well beyond this.
DEFAULT_ORDERS_PER_SECOND = 2.0

# How long the reverse-order flow waits for the close leg to actually
# fill (position back to flat) before sending the open leg. Market IOC
# fills typically come back in < 1s; 5s gives comfortable slack while
# still bounding the over-exposure window.
REVERSAL_FILL_TIMEOUT_SECONDS = 5.0


async def route_order(
    broker_name: str,
    position: PositionState,
    action: str,
    quantity: int,
    submit_fn,
    is_connected: bool,
    risk_manager: Optional[RiskManager] = None,
    trade_store: Optional[TradeStore] = None,
    idempotency_key: str = "",
    reversal_fill_timeout: float = REVERSAL_FILL_TIMEOUT_SECONDS,
    rate_limiter: Optional[RateLimiter] = None,
) -> dict:
    """Position-aware order routing shared by all brokers.

    Parameters
    ----------
    submit_fn : async (action: str, quantity: int) -> dict
        Broker-specific function that actually submits an order.
    risk_manager : RiskManager, optional
        Pre-trade risk checks.
    trade_store : TradeStore, optional
        Persist orders for audit trail and crash recovery.
    idempotency_key : str, optional
        Deduplicate webhook retries.
    """
    if not is_connected:
        return {"status": "error", "broker": broker_name, "reason": "not connected"}

    # Each broker owns its throttle so Shioaji and Rithmic don't compete
    # for the same 2 orders/sec budget. Fall back to a local one-shot
    # limiter only if the caller didn't supply one (tests, scripts).
    effective_rate_limiter = rate_limiter or RateLimiter(
        max_orders_per_second=DEFAULT_ORDERS_PER_SECOND
    )

    # Idempotency gate — MUST fire before any broker submit or any wait.
    # A reversal flow can be mid-wait_for_flat for up to several seconds
    # after the close leg has already hit the exchange; any retry that
    # slips in during that window must be treated as a duplicate, not
    # sent as a second close. The reservations table uses a PRIMARY
    # KEY on the idempotency key so this is an atomic claim.
    #
    # Idempotency keys are single-use for every action, including exits.
    # A duplicate flatten is not harmless: if the first close already
    # hit the market, a second close can reopen exposure in the opposite
    # direction. Operators who intentionally want to re-drive a signal
    # after an ambiguous failure must send a fresh key.
    if idempotency_key and trade_store:
        existing = trade_store.check_idempotency(idempotency_key)
        if existing and existing.get("id"):
            logger.warning(
                "[{}] Duplicate order detected (key={}), returning cached result",
                broker_name, idempotency_key,
            )
            return {
                "status": "duplicate",
                "broker": broker_name,
                "original_order_id": existing.get("id"),
            }

        claimed = trade_store.reserve_idempotency(idempotency_key, broker_name)
        if not claimed:
            # Another request won the race between our check and reserve.
            # Re-read so we can include whatever id/metadata is now visible.
            existing = trade_store.check_idempotency(idempotency_key)
            if existing and existing.get("id"):
                logger.warning(
                    "[{}] Duplicate order detected via reservation race "
                    "(key={}), returning duplicate",
                    broker_name, idempotency_key,
                )
                return {
                    "status": "duplicate",
                    "broker": broker_name,
                    "original_order_id": (existing or {}).get("id"),
                }
            # An in-flight reservation exists but no committed order yet.
            # Reject as concurrent rather than silently double-submitting.
            logger.warning(
                "[{}] Concurrent order in flight (key={}), rejecting retry",
                broker_name, idempotency_key,
            )
            return {
                "status": "duplicate",
                "broker": broker_name,
                "original_order_id": None,
            }

    async def _submit_tracked(
        order_action: str,
        order_qty: int,
        idem: str = "",
    ) -> tuple[dict, Optional[int]]:
        """Record the order, submit it, and update its status based on the
        broker response. Returns (result, order_id)."""
        order_id: Optional[int] = None
        if trade_store:
            order_id = trade_store.record_order(
                broker_name, order_action, order_qty,
                idempotency_key=idem,
            )
        await effective_rate_limiter.wait_async()
        try:
            result = await submit_fn(order_action, order_qty)
        except Exception as exc:
            if trade_store and order_id:
                trade_store.update_order_status(
                    order_id, "error", error_message=str(exc)
                )
            raise

        if trade_store and order_id:
            broker_order_id = str(result.get("broker_order_id") or "").strip()
            if broker_order_id:
                trade_store.set_broker_order_id(order_id, broker_order_id)
            if _submit_ok(result):
                trade_store.update_order_status(order_id, "submitted")
            else:
                trade_store.update_order_status(
                    order_id, "error",
                    error_message=str(result.get("reason") or result),
                )
        return result, order_id

    try:
        current = position.get_position()
        results: list[dict] = []

        if action == "exit":
            if current.side == "flat":
                logger.info("[{}] Already flat, ignoring exit signal", broker_name)
                return {"status": "skipped", "broker": broker_name, "reason": "already flat"}

            # Exits always pass the risk gate, but record for auditability.
            if risk_manager:
                ok, reason = risk_manager.check_order(
                    broker_name, action, 0,
                )
                if not ok:
                    logger.warning("[{}] Risk rejected exit: {}", broker_name, reason)
                    return {"status": "risk_rejected", "broker": broker_name, "reason": reason}

            close_action = "Sell" if current.side == "long" else "Buy"
            result, _ = await _submit_tracked(
                close_action, current.quantity, idempotency_key,
            )
            results.append(result)
            if not _submit_ok(result):
                logger.error(
                    "[{}] Exit submit failed: {}", broker_name, result,
                )
                return {
                    "status": "error", "broker": broker_name,
                    "reason": str(result.get("reason") or "submit failed"),
                    "orders": results,
                }

            if trade_store:
                pos = position.get_position()
                trade_store.save_position_snapshot(
                    broker_name, pos.side, pos.quantity, pos.entry_price
                )

        elif action in ("buy", "sell"):
            open_side = "long" if action == "buy" else "short"
            if current.side == open_side:
                logger.warning(
                    "[{}] Already {}, ignoring {} signal",
                    broker_name, open_side, action,
                )
                return {
                    "status": "skipped", "broker": broker_name,
                    "reason": f"already {open_side}",
                }

            # Risk check — pass direction so reversals aren't wrongly rejected.
            if risk_manager:
                ok, reason = risk_manager.check_order(
                    broker_name, action, quantity,
                )
                if not ok:
                    logger.warning(
                        "[{}] Risk rejected {}: {}", broker_name, action, reason,
                    )
                    if trade_store:
                        trade_store.record_risk_event(
                            "order_rejected", broker_name, reason,
                        )
                    return {"status": "risk_rejected", "broker": broker_name, "reason": reason}

            submit_side = "Buy" if action == "buy" else "Sell"
            opposite_side = "short" if action == "buy" else "long"

            # Reversal: close opposite leg first, REQUIRE both the submit
            # to be accepted AND the position to actually become flat
            # before opening the new leg. Without this gate, a close that
            # is accepted but fills slowly / partially would let the open
            # leg fire on top and cause over-exposure.
            if current.side == opposite_side:
                logger.info(
                    "[{}] Closing {} before opening {}",
                    broker_name, opposite_side, open_side,
                )
                close_result, _ = await _submit_tracked(
                    submit_side, current.quantity,
                )
                results.append(close_result)
                if not _submit_ok(close_result):
                    logger.error(
                        "[{}] Reverse close leg rejected, aborting open: {}",
                        broker_name, close_result,
                    )
                    if trade_store:
                        trade_store.record_risk_event(
                            "reverse_close_failed", broker_name, str(close_result),
                        )
                    return {
                        "status": "error", "broker": broker_name,
                        "reason": (
                            f"reverse close failed: "
                            f"{close_result.get('reason') or close_result}"
                        ),
                        "orders": results,
                    }

                # Accepted != filled. Wait for the close to actually flatten
                # the book before sending the open leg.
                flat = await position.wait_for_flat(reversal_fill_timeout)
                if not flat:
                    pos_now = position.get_position()
                    logger.error(
                        "[{}] Close leg not filled within {}s "
                        "(position still {}x{}), aborting open",
                        broker_name, reversal_fill_timeout,
                        pos_now.side, pos_now.quantity,
                    )
                    if trade_store:
                        trade_store.record_risk_event(
                            "reverse_close_timeout", broker_name,
                            f"still {pos_now.side}x{pos_now.quantity} "
                            f"after {reversal_fill_timeout}s",
                        )
                    return {
                        "status": "error", "broker": broker_name,
                        "reason": (
                            f"close leg did not fill within "
                            f"{reversal_fill_timeout}s"
                        ),
                        "orders": results,
                    }

            open_result, _ = await _submit_tracked(
                submit_side, quantity, idempotency_key,
            )
            results.append(open_result)
            if not _submit_ok(open_result):
                logger.error(
                    "[{}] Open leg failed: {}", broker_name, open_result,
                )
                return {
                    "status": "error", "broker": broker_name,
                    "reason": str(open_result.get("reason") or "submit failed"),
                    "orders": results,
                }

            if trade_store:
                pos = position.get_position()
                trade_store.save_position_snapshot(
                    broker_name, pos.side, pos.quantity, pos.entry_price
                )

        else:
            return {"status": "error", "broker": broker_name, "reason": f"unknown action: {action}"}

        return {"status": "ok", "broker": broker_name, "orders": results}

    except Exception as e:
        logger.error("[{}] place_order error: {}", broker_name, e)
        if trade_store:
            trade_store.record_risk_event("order_error", broker_name, str(e))
        return {"status": "error", "broker": broker_name, "reason": str(e)}
