"""Tests for shared broker order-routing logic."""

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


class TestRateLimiter:
    def test_first_acquire_succeeds(self):
        rl = RateLimiter(max_orders_per_second=10.0)
        assert rl.acquire() is True

    def test_rapid_acquire_blocked(self):
        rl = RateLimiter(max_orders_per_second=1.0)
        assert rl.acquire() is True
        assert rl.acquire() is False  # too fast
