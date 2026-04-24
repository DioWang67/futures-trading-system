"""Tests for shared broker order-routing logic."""

import asyncio
import threading

import pytest

from position_state import PositionState
from brokers.base import route_order, RateLimiter


class TestRouteOrder:
    @pytest.fixture
    def position(self):
        return PositionState("test")

    @pytest.fixture
    def submitted_orders(self):
        return []

    @pytest.fixture
    def submit_fn(self, submitted_orders, position):
        """Default mock mirrors a real broker: on submit, apply the fill
        back to the shared position so reversal flows (which wait for
        the close leg to flatten the book before opening the reverse
        leg) can proceed."""
        async def _submit(action: str, quantity: int) -> dict:
            order = {"status": "submitted", "action": action, "quantity": quantity}
            submitted_orders.append(order)
            position.apply_fill(action, quantity, fill_price=100.0)
            return order
        return _submit

    @pytest.mark.asyncio
    async def test_buy_from_flat(self, position, submit_fn, submitted_orders):
        result = await route_order("test", position, "buy", 2, submit_fn, True)
        assert result["status"] == "ok"
        assert len(submitted_orders) == 1
        assert submitted_orders[0]["action"] == "Buy"
        assert submitted_orders[0]["quantity"] == 2

    @pytest.mark.asyncio
    async def test_sell_from_flat(self, position, submit_fn, submitted_orders):
        result = await route_order("test", position, "sell", 1, submit_fn, True)
        assert result["status"] == "ok"
        assert len(submitted_orders) == 1
        assert submitted_orders[0]["action"] == "Sell"

    @pytest.mark.asyncio
    async def test_exit_from_long(self, position, submit_fn, submitted_orders):
        position.update_position("long", 3)
        result = await route_order("test", position, "exit", 0, submit_fn, True)
        assert result["status"] == "ok"
        assert submitted_orders[0]["action"] == "Sell"
        assert submitted_orders[0]["quantity"] == 3

    @pytest.mark.asyncio
    async def test_exit_from_short(self, position, submit_fn, submitted_orders):
        position.update_position("short", 2)
        result = await route_order("test", position, "exit", 0, submit_fn, True)
        assert result["status"] == "ok"
        assert submitted_orders[0]["action"] == "Buy"
        assert submitted_orders[0]["quantity"] == 2

    @pytest.mark.asyncio
    async def test_exit_when_flat_skipped(self, position, submit_fn, submitted_orders):
        result = await route_order("test", position, "exit", 0, submit_fn, True)
        assert result["status"] == "skipped"
        assert len(submitted_orders) == 0

    @pytest.mark.asyncio
    async def test_buy_when_already_long_skipped(self, position, submit_fn, submitted_orders):
        position.update_position("long", 1)
        result = await route_order("test", position, "buy", 1, submit_fn, True)
        assert result["status"] == "skipped"
        assert len(submitted_orders) == 0

    @pytest.mark.asyncio
    async def test_sell_when_already_short_skipped(self, position, submit_fn, submitted_orders):
        position.update_position("short", 1)
        result = await route_order("test", position, "sell", 1, submit_fn, True)
        assert result["status"] == "skipped"
        assert len(submitted_orders) == 0

    @pytest.mark.asyncio
    async def test_buy_flips_short(self, position, submit_fn, submitted_orders):
        position.update_position("short", 2)
        result = await route_order("test", position, "buy", 1, submit_fn, True)
        assert result["status"] == "ok"
        assert len(submitted_orders) == 2  # close short + open long
        assert submitted_orders[0]["action"] == "Buy"
        assert submitted_orders[0]["quantity"] == 2  # close
        assert submitted_orders[1]["action"] == "Buy"
        assert submitted_orders[1]["quantity"] == 1  # open

    @pytest.mark.asyncio
    async def test_sell_flips_long(self, position, submit_fn, submitted_orders):
        position.update_position("long", 3)
        result = await route_order("test", position, "sell", 2, submit_fn, True)
        assert result["status"] == "ok"
        assert len(submitted_orders) == 2
        assert submitted_orders[0]["action"] == "Sell"
        assert submitted_orders[0]["quantity"] == 3
        assert submitted_orders[1]["action"] == "Sell"
        assert submitted_orders[1]["quantity"] == 2

    @pytest.mark.asyncio
    async def test_not_connected_returns_error(self, position, submit_fn):
        result = await route_order("test", position, "buy", 1, submit_fn, False)
        assert result["status"] == "error"
        assert "not connected" in result["reason"]

    @pytest.mark.asyncio
    async def test_unknown_action_returns_error(self, position, submit_fn):
        result = await route_order("test", position, "invalid", 1, submit_fn, True)
        assert result["status"] == "error"
        assert "unknown action" in result["reason"]

    @pytest.mark.asyncio
    async def test_submit_exception_caught(self, position):
        async def failing_submit(action, qty):
            raise ConnectionError("broker offline")

        result = await route_order("test", position, "buy", 1, failing_submit, True)
        assert result["status"] == "error"
        assert "broker offline" in result["reason"]

    @pytest.mark.asyncio
    async def test_reversal_close_leg_records_idempotency_key(self, position):
        position.update_position("short", 2)
        recorded_keys: list[str] = []

        async def submit_fn(action: str, quantity: int) -> dict:
            position.apply_fill(action, quantity, fill_price=100.0)
            return {"status": "submitted", "action": action, "quantity": quantity}

        class DummyStore:
            def check_idempotency(self, _key: str):
                return None

            def reserve_idempotency(self, _key: str, _broker: str) -> bool:
                return True

            def record_order(self, _broker: str, _action: str, _qty: int, idempotency_key: str = "", broker_order_id: str = "") -> int:
                recorded_keys.append(idempotency_key)
                return len(recorded_keys)

            def update_order_status(self, *_args, **_kwargs):
                return None

            def set_broker_order_id(self, *_args, **_kwargs):
                return None

            def save_position_snapshot(self, *_args, **_kwargs):
                return None

        result = await route_order(
            "test",
            position,
            "buy",
            1,
            submit_fn,
            True,
            trade_store=DummyStore(),
            idempotency_key="flip-K1",
        )

        assert result["status"] == "ok"
        assert recorded_keys == ["flip-K1", "flip-K1"]


class TestRateLimiter:
    def test_first_acquire_succeeds(self):
        rl = RateLimiter(max_orders_per_second=10.0)
        assert rl.acquire() is True

    def test_rapid_acquire_blocked(self):
        rl = RateLimiter(max_orders_per_second=1.0)
        assert rl.acquire() is True
        assert rl.acquire() is False  # too fast


class TestIdempotencyRaces:
    def test_concurrent_exit_retry_with_same_key_is_duplicate(self):
        position = PositionState("test")
        position.update_position("long", 1, entry_price=100.0)

        class DummyStore:
            def __init__(self):
                self._barrier = threading.Barrier(2)
                self._lock = threading.Lock()
                self._reserved_keys: set[str] = set()
                self._check_calls = 0

            def check_idempotency(self, key: str):
                with self._lock:
                    self._check_calls += 1
                    call_number = self._check_calls
                if call_number <= 2:
                    self._barrier.wait(timeout=5.0)
                return None

            def reserve_idempotency(self, key: str, broker: str) -> bool:
                with self._lock:
                    if key in self._reserved_keys:
                        return False
                    self._reserved_keys.add(key)
                    return True

            def record_order(
                self,
                broker: str,
                action: str,
                quantity: int,
                idempotency_key: str = "",
                broker_order_id: str = "",
            ) -> int:
                return 1

            def update_order_status(self, *args, **kwargs):
                return None

            def set_broker_order_id(self, *args, **kwargs):
                return None

            def save_position_snapshot(self, *args, **kwargs):
                return None

        store = DummyStore()
        submitted: list[tuple[str, int]] = []
        submit_lock = threading.Lock()
        results: list[dict] = []
        errors: list[Exception] = []

        async def submit_fn(action: str, quantity: int) -> dict:
            with submit_lock:
                submitted.append((action, quantity))
            return {"status": "submitted", "action": action, "quantity": quantity}

        def worker() -> None:
            try:
                result = asyncio.run(
                    route_order(
                        "test",
                        position,
                        "exit",
                        0,
                        submit_fn,
                        True,
                        trade_store=store,
                        idempotency_key="exit-K1",
                    )
                )
                results.append(result)
            except Exception as exc:  # pragma: no cover - exercised on regression
                errors.append(exc)

        first = threading.Thread(target=worker)
        second = threading.Thread(target=worker)
        first.start()
        second.start()
        first.join()
        second.join()

        assert errors == []
        assert sorted(result["status"] for result in results) == ["duplicate", "ok"]
        assert submitted == [("Sell", 1)]
