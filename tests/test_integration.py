"""
Integration tests — webhook → risk check → order routing → persistence.

Uses mock brokers to test the full pipeline without real API connections.
"""

import asyncio

import pytest

from position_state import PositionState
from risk_manager import RiskConfig, RiskManager
from trade_store import TradeStore
from brokers.base import route_order


@pytest.fixture
def risk_manager():
    return RiskManager(RiskConfig(
        max_daily_loss=5000.0,
        max_daily_loss_pct=0.05,
        max_position_size=3,
        max_order_qty=2,
        max_drawdown_pct=0.20,
        cooldown_after_consecutive_losses=3,
        cooldown_seconds=1,
        initial_capital=100_000.0,
    ))


@pytest.fixture
def trade_store(tmp_path):
    return TradeStore(db_path=tmp_path / "test.db")


@pytest.fixture
def position():
    return PositionState("test")


@pytest.fixture
def submitted_orders():
    return []


@pytest.fixture
def submit_fn(submitted_orders, position):
    """Default mock: succeed AND apply the fill back to the position, the
    way a real broker callback would. Reversal flows rely on the close
    leg actually flattening the book before the open leg fires, so a
    mock that only returns 'submitted' without touching state no longer
    reflects reality."""
    async def _submit(action: str, quantity: int) -> dict:
        order = {"status": "submitted", "action": action, "quantity": quantity}
        submitted_orders.append(order)
        position.apply_fill(action, quantity, fill_price=100.0)
        return order
    return _submit


def make_filling_submit(pos, submitted_orders):
    """Build a submit_fn bound to an arbitrary position (for tests that
    construct their own PositionState rather than the fixture)."""
    async def _submit(action: str, quantity: int) -> dict:
        order = {"status": "submitted", "action": action, "quantity": quantity}
        submitted_orders.append(order)
        pos.apply_fill(action, quantity, fill_price=100.0)
        return order
    return _submit


class TestRiskIntegration:
    """Test route_order with risk manager plugged in."""

    @pytest.mark.asyncio
    async def test_order_within_risk_limits(
        self, position, submit_fn, submitted_orders, risk_manager, trade_store
    ):
        result = await route_order(
            "test", position, "buy", 2, submit_fn, True,
            risk_manager=risk_manager, trade_store=trade_store,
        )
        assert result["status"] == "ok"
        assert len(submitted_orders) == 1

    @pytest.mark.asyncio
    async def test_order_exceeds_qty_limit(
        self, position, submit_fn, submitted_orders, risk_manager, trade_store
    ):
        result = await route_order(
            "test", position, "buy", 5, submit_fn, True,
            risk_manager=risk_manager, trade_store=trade_store,
        )
        assert result["status"] == "risk_rejected"
        assert len(submitted_orders) == 0

    @pytest.mark.asyncio
    async def test_order_after_daily_loss_halt(
        self, position, submit_fn, submitted_orders, risk_manager, trade_store
    ):
        # Trigger daily loss limit
        risk_manager.record_fill("test", -5001.0)
        assert risk_manager.is_halted

        result = await route_order(
            "test", position, "buy", 1, submit_fn, True,
            risk_manager=risk_manager, trade_store=trade_store,
        )
        assert result["status"] == "risk_rejected"
        assert "halted" in result["reason"]

    @pytest.mark.asyncio
    async def test_exit_allowed_when_halted(
        self, position, submit_fn, submitted_orders, risk_manager, trade_store
    ):
        """Exits must still work after the system halts — operators need
        to flatten positions even when the risk manager has tripped."""
        position.update_position("long", 2, entry_price=100.0)
        # Actually halt the system by blowing through the daily loss limit
        risk_manager.record_fill("test", -5001.0)
        assert risk_manager.is_halted

        result = await route_order(
            "test", position, "exit", 0, submit_fn, True,
            risk_manager=risk_manager, trade_store=trade_store,
        )
        assert result["status"] == "ok", result
        assert submitted_orders[0]["action"] == "Sell"

    @pytest.mark.asyncio
    async def test_entry_rejected_when_halted(
        self, position, submit_fn, submitted_orders, risk_manager, trade_store
    ):
        """Counterpart: new-exposure orders must still be blocked on halt."""
        risk_manager.record_fill("test", -5001.0)
        assert risk_manager.is_halted

        result = await route_order(
            "test", position, "buy", 1, submit_fn, True,
            risk_manager=risk_manager, trade_store=trade_store,
        )
        assert result["status"] == "risk_rejected"
        assert len(submitted_orders) == 0

    @pytest.mark.asyncio
    async def test_reversal_not_rejected_by_position_limit(
        self, submitted_orders, risk_manager, trade_store
    ):
        """short 3 + buy 1 is a full flip: final position is long 1.
        The per-order cap (max_order_qty=2) allows qty=1, so risk should
        pass and both legs should run."""
        pos = PositionState("test")
        pos.update_position("short", 3, entry_price=100.0)
        submit = make_filling_submit(pos, submitted_orders)

        result = await route_order(
            "test", pos, "buy", 1, submit, True,
            risk_manager=risk_manager, trade_store=trade_store,
        )
        assert result["status"] == "ok", result
        final = pos.get_position()
        assert final.side == "long"
        assert final.quantity == 1

    @pytest.mark.asyncio
    async def test_reversal_rejected_when_target_exposure_exceeds_limit(
        self, submitted_orders, trade_store
    ):
        """short 3 + buy 5 fully flips to long 5. Under a max_position_size
        of 4 the risk check MUST reject — the previous net-exposure math
        (|-3+5|=2) was too permissive and would have let this through."""
        rm = RiskManager(RiskConfig(
            max_daily_loss=5000.0, max_daily_loss_pct=1.0,
            max_position_size=4, max_order_qty=10,
            max_drawdown_pct=1.0,
            cooldown_after_consecutive_losses=99, cooldown_seconds=1,
            initial_capital=100_000.0,
        ))
        pos = PositionState("test")
        pos.update_position("short", 3, entry_price=100.0)
        submit = make_filling_submit(pos, submitted_orders)

        result = await route_order(
            "test", pos, "buy", 5, submit, True,
            risk_manager=rm, trade_store=trade_store,
        )
        assert result["status"] == "risk_rejected", result
        assert len(submitted_orders) == 0


class TestReversalSafety:
    """The open leg of a reversal must NOT fire until the close leg is
    both accepted AND actually filled."""

    @pytest.mark.asyncio
    async def test_reverse_aborts_when_close_leg_rejected(self, trade_store):
        pos = PositionState("test")
        pos.update_position("short", 2, entry_price=100.0)

        call_log = []

        async def flaky_submit(action, quantity):
            call_log.append((action, quantity))
            # First call is the close leg — reject it at submit time.
            if len(call_log) == 1:
                return {"status": "rejected", "reason": "exchange said no"}
            return {"status": "submitted", "action": action, "quantity": quantity}

        result = await route_order(
            "test", pos, "buy", 1, flaky_submit, True,
            trade_store=trade_store,
        )
        assert result["status"] == "error", result
        assert len(call_log) == 1, call_log

    @pytest.mark.asyncio
    async def test_reversal_retry_during_wait_for_flat_is_duplicate(
        self, trade_store
    ):
        """The scariest idempotency race: the first request has already
        sent the close leg and is now inside wait_for_flat. A webhook
        retry with the SAME idempotency key arrives. It must be seen as
        a duplicate — not allowed to re-submit another close leg — even
        though no orders row with that key exists yet (the key gets
        attached to the open leg, which hasn't fired yet)."""
        pos = PositionState("test")
        pos.update_position("short", 2, entry_price=100.0)

        close_submitted = asyncio.Event()
        let_close_fill = asyncio.Event()
        first_calls: list = []

        async def slow_fill_submit(action, quantity):
            first_calls.append((action, quantity))
            # Signal the retry to start, then block until the test
            # explicitly lets the fill be applied.
            close_submitted.set()
            await let_close_fill.wait()
            pos.apply_fill(action, quantity, fill_price=100.0)
            return {"status": "submitted", "action": action, "quantity": quantity}

        second_calls: list = []

        async def second_submit(action, quantity):
            # The retry must NEVER reach the broker; if it does, this
            # list stays non-empty and the test fails.
            second_calls.append((action, quantity))
            pos.apply_fill(action, quantity, fill_price=100.0)
            return {"status": "submitted", "action": action, "quantity": quantity}

        first = asyncio.create_task(route_order(
            "test", pos, "buy", 1, slow_fill_submit, True,
            trade_store=trade_store,
            idempotency_key="reversal-K1",
            reversal_fill_timeout=3.0,
        ))

        # Wait until the first request is mid-flight (close leg sent,
        # wait_for_flat active), then fire the retry.
        await close_submitted.wait()

        retry = await route_order(
            "test", pos, "buy", 1, second_submit, True,
            trade_store=trade_store,
            idempotency_key="reversal-K1",
            reversal_fill_timeout=3.0,
        )

        # Release the first request so the test can complete.
        let_close_fill.set()
        first_result = await first

        assert retry["status"] == "duplicate", retry
        # Critical: the retry must not have sent any order to the broker.
        assert second_calls == [], second_calls
        # First request still completes normally: close leg filled,
        # then open leg fires.
        assert first_result["status"] == "ok", first_result
        # slow_fill_submit handled the close; route_order's open leg
        # uses the same submit_fn (closing over first_calls).
        assert [c[0] for c in first_calls] == ["Buy", "Buy"]
        assert first_calls[0][1] == 2  # close qty
        assert first_calls[1][1] == 1  # open qty

    @pytest.mark.asyncio
    async def test_reverse_aborts_when_close_leg_accepted_but_not_filled(
        self, trade_store
    ):
        """The scary one: broker accepts the close submit but never sends
        a fill. route_order must time out on wait_for_flat and refuse to
        send the open leg, otherwise we'd end up doubly exposed."""
        pos = PositionState("test")
        pos.update_position("short", 2, entry_price=100.0)

        call_log = []

        async def accept_but_no_fill(action, quantity):
            call_log.append((action, quantity))
            # Accepted, but we intentionally do NOT apply_fill — simulating
            # a broker that has acknowledged the order but not yet filled.
            return {"status": "submitted", "action": action, "quantity": quantity}

        result = await route_order(
            "test", pos, "buy", 1, accept_but_no_fill, True,
            trade_store=trade_store,
            reversal_fill_timeout=0.1,
        )
        assert result["status"] == "error", result
        assert "fill" in result["reason"].lower()
        # Only the close leg was attempted; the open leg must NOT have fired.
        assert len(call_log) == 1, call_log
        # Position unchanged — still short 2 — since no fill ever arrived.
        pos_now = pos.get_position()
        assert pos_now.side == "short"
        assert pos_now.quantity == 2


class TestRiskFeedbackLoop:
    """Integration check: a losing fill should actually move risk state."""

    def test_losing_fill_updates_daily_pnl_and_can_halt(self, risk_manager):
        # record_fill is the feedback loop from broker -> risk manager.
        risk_manager.record_fill("test", -2500.0)
        risk_manager.record_fill("test", -2501.0)
        assert risk_manager.is_halted
        status = risk_manager.get_status()
        assert status["daily_pnl"] <= -5000.0


class TestIdempotencyIntegration:
    """Test idempotency key deduplication in route_order."""

    @pytest.mark.asyncio
    async def test_duplicate_order_rejected(
        self, position, submit_fn, submitted_orders, trade_store
    ):
        # First order succeeds
        result1 = await route_order(
            "test", position, "buy", 1, submit_fn, True,
            trade_store=trade_store, idempotency_key="order-123",
        )
        assert result1["status"] == "ok"

        # Same key → duplicate
        result2 = await route_order(
            "test", position, "buy", 1, submit_fn, True,
            trade_store=trade_store, idempotency_key="order-123",
        )
        assert result2["status"] == "duplicate"
        assert len(submitted_orders) == 1  # only one order submitted

    @pytest.mark.asyncio
    async def test_duplicate_detected_via_reservation_before_any_submit(
        self, position, submitted_orders, trade_store
    ):
        """If another request has reserved the key but not yet produced
        an order row (e.g. it's still mid-flight), a retry must still
        be rejected. This tests the reservation gate in isolation."""
        # Manually reserve the key as if another request had just started.
        assert trade_store.reserve_idempotency("held-key", "test") is True

        async def should_not_fire(action, quantity):
            raise AssertionError("retry reached the broker!")

        result = await route_order(
            "test", position, "buy", 1, should_not_fire, True,
            trade_store=trade_store, idempotency_key="held-key",
        )
        assert result["status"] == "duplicate", result

    @pytest.mark.asyncio
    async def test_different_keys_both_succeed(
        self, position, submit_fn, submitted_orders, trade_store
    ):
        result1 = await route_order(
            "test", position, "buy", 1, submit_fn, True,
            trade_store=trade_store, idempotency_key="order-A",
        )
        assert result1["status"] == "ok"
        # submit_fn applies the fill, so position is now long 1.

        # Different key, but position already long → skipped (not duplicate)
        result2 = await route_order(
            "test", position, "buy", 1, submit_fn, True,
            trade_store=trade_store, idempotency_key="order-B",
        )
        assert result2["status"] == "skipped"  # already long


class TestPersistenceIntegration:
    """Test that route_order persists orders and position snapshots."""

    @pytest.mark.asyncio
    async def test_order_persisted(
        self, position, submit_fn, trade_store
    ):
        await route_order(
            "test", position, "buy", 1, submit_fn, True,
            trade_store=trade_store,
        )
        pending = trade_store.get_pending_orders("test")
        # Order should be recorded (status=submitted)
        assert len(pending) >= 1

    @pytest.mark.asyncio
    async def test_position_snapshot_saved(
        self, position, submit_fn, trade_store
    ):
        await route_order(
            "test", position, "buy", 2, submit_fn, True,
            trade_store=trade_store,
        )
        snapshot = trade_store.get_latest_position("test")
        assert snapshot is not None

    @pytest.mark.asyncio
    async def test_risk_event_on_rejection(
        self, position, submit_fn, risk_manager, trade_store
    ):
        result = await route_order(
            "test", position, "buy", 10, submit_fn, True,
            risk_manager=risk_manager, trade_store=trade_store,
        )
        assert result["status"] == "risk_rejected"
        # Risk event should be recorded in DB
        # (We can't easily query risk_events without exposing more API,
        #  but at least it shouldn't crash)


class TestConcurrentWebhooks:
    """Test that concurrent webhook processing is safe."""

    @pytest.mark.asyncio
    async def test_concurrent_buy_sell(self, trade_store):
        """Simulate concurrent buy and sell signals. With wait_for_flat
        gating the reversal open-leg, some tasks may time out; the
        contract here is 'no crash, consistent position state'."""
        position = PositionState("concurrent")

        async def mock_submit(action: str, quantity: int) -> dict:
            await asyncio.sleep(0.01)
            position.apply_fill(action, quantity, fill_price=100.0)
            return {"status": "submitted", "action": action, "quantity": quantity}

        tasks = []
        for i in range(10):
            action = "buy" if i % 2 == 0 else "sell"
            tasks.append(
                route_order(
                    "concurrent", position, action, 1, mock_submit, True,
                    trade_store=trade_store,
                    idempotency_key=f"concurrent-{i}",
                    reversal_fill_timeout=0.5,
                )
            )

        results = await asyncio.gather(*tasks)

        for r in results:
            assert r["status"] in ("ok", "skipped", "duplicate", "error")

        pos = position.get_position()
        assert pos.side in ("long", "short", "flat")
        assert pos.quantity >= 0
