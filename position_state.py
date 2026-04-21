from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass
from typing import Literal

from loguru import logger

Side = Literal["long", "short", "flat"]


@dataclass(frozen=True)
class PositionData:
    """Immutable snapshot of current position."""

    side: Side = "flat"
    quantity: int = 0
    entry_price: float = 0.0
    broker: str = ""


class PositionState:
    """Thread-safe position state manager for a single broker.

    All mutations go through ``apply_fill`` which atomically reads
    current state and writes the new state in a single lock acquisition,
    eliminating the read-then-write race condition.
    """

    def __init__(self, broker_name: str) -> None:
        self._lock = threading.Lock()
        # Condition shares the same underlying lock so fill callbacks can
        # notify waiters (e.g. reversal flows waiting for the close leg).
        self._condition = threading.Condition(self._lock)
        self._position = PositionData(broker=broker_name)
        self._broker_name = broker_name

    @property
    def broker_name(self) -> str:
        return self._broker_name

    def get_position(self) -> PositionData:
        with self._lock:
            return self._position

    def update_position(
        self,
        side: Side,
        quantity: int,
        entry_price: float = 0.0,
    ) -> None:
        with self._condition:
            old = f"{self._position.side}x{self._position.quantity}"
            # Guard against negative quantities
            if quantity < 0:
                logger.error(
                    "[{}] Attempted negative quantity update: {} -> {}x{}",
                    self._broker_name, old, side, quantity,
                )
                quantity = 0
                side = "flat"
            if quantity == 0:
                side = "flat"
            self._position = PositionData(
                side=side,
                quantity=quantity,
                entry_price=entry_price,
                broker=self._broker_name,
            )
            self._condition.notify_all()
            new = f"{side}x{quantity}"
            logger.info("[{}] Position updated: {} -> {}", self._broker_name, old, new)

    def apply_fill(
        self,
        action: str,
        filled_qty: int,
        fill_price: float = 0.0,
    ) -> dict:
        """Atomically apply a fill to current position.

        Returns a result dict so callers can compute realized PnL::

            {
              "closed_qty":   int,    # contracts that reduced/closed the
                                      # prior position (0 for pure opens)
              "opened_qty":   int,    # contracts that added new exposure
              "prev_side":    str,    # position side before this fill
              "prev_entry":   float,  # entry price before this fill
              "new_side":     str,
              "new_qty":      int,
            }
        """
        with self._condition:
            old = self._position
            old_label = f"{old.side}x{old.quantity}"
            prev_side = old.side
            prev_entry = old.entry_price
            closed_qty = 0
            opened_qty = 0

            if action == "Buy":
                if old.side == "short":
                    remaining = old.quantity - filled_qty
                    closed_qty = min(filled_qty, old.quantity)
                    opened_qty = max(0, filled_qty - old.quantity)
                    if remaining <= 0:
                        new_side: Side = "flat" if remaining == 0 else "long"
                        new_qty = abs(remaining)
                    else:
                        new_side = "short"
                        new_qty = remaining
                else:
                    opened_qty = filled_qty
                    new_qty = (old.quantity + filled_qty) if old.side == "long" else filled_qty
                    new_side = "long"
            elif action == "Sell":
                if old.side == "long":
                    remaining = old.quantity - filled_qty
                    closed_qty = min(filled_qty, old.quantity)
                    opened_qty = max(0, filled_qty - old.quantity)
                    if remaining <= 0:
                        new_side = "flat" if remaining == 0 else "short"
                        new_qty = abs(remaining)
                    else:
                        new_side = "long"
                        new_qty = remaining
                else:
                    opened_qty = filled_qty
                    new_qty = (old.quantity + filled_qty) if old.side == "short" else filled_qty
                    new_side = "short"
            else:
                logger.error("[{}] Unknown fill action: {}", self._broker_name, action)
                return {
                    "closed_qty": 0, "opened_qty": 0,
                    "prev_side": prev_side, "prev_entry": prev_entry,
                    "new_side": old.side, "new_qty": old.quantity,
                }

            if new_qty == 0:
                new_side = "flat"

            # Entry price rules:
            #   - flipping (closed + opened in one fill): new entry = fill_price
            #   - partial close (still same side): keep prior entry
            #   - pure open (no prior, or same-side add): average in at fill_price
            if new_side == "flat":
                price = 0.0
            elif new_side != prev_side:
                price = fill_price
            elif opened_qty > 0 and old.quantity > 0 and prev_entry > 0:
                total_qty = old.quantity + opened_qty
                price = (
                    (prev_entry * old.quantity + fill_price * opened_qty)
                    / total_qty
                ) if total_qty else fill_price
            else:
                price = prev_entry or fill_price

            self._position = PositionData(
                side=new_side,
                quantity=new_qty,
                entry_price=price,
                broker=self._broker_name,
            )
            self._condition.notify_all()
            new_label = f"{new_side}x{new_qty}"
            logger.info(
                "[{}] Fill applied: {} {} -> {} (closed={}, opened={})",
                self._broker_name, action, old_label, new_label,
                closed_qty, opened_qty,
            )
            return {
                "closed_qty": closed_qty,
                "opened_qty": opened_qty,
                "prev_side": prev_side,
                "prev_entry": prev_entry,
                "new_side": new_side,
                "new_qty": new_qty,
            }

    def is_flat(self) -> bool:
        with self._lock:
            return self._position.side == "flat" or self._position.quantity == 0

    async def wait_for_flat(self, timeout: float) -> bool:
        """Block until the position becomes flat, or timeout elapses.

        Returns True if flat was reached, False on timeout. Runs the
        blocking wait in a worker thread so the event loop stays free.
        Callers (e.g. reversal flows in ``route_order``) use this to
        confirm the close leg actually reduced exposure before sending
        the open leg — preventing over-exposure when the broker accepts
        the close but has not yet filled it.
        """
        if timeout <= 0:
            return self.is_flat()

        def _wait() -> bool:
            deadline = time.monotonic() + timeout
            with self._condition:
                while (
                    self._position.side != "flat"
                    or self._position.quantity != 0
                ):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    self._condition.wait(timeout=remaining)
                return True

        return await asyncio.to_thread(_wait)
