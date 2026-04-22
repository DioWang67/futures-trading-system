"""Tests for the trade persistence layer."""
from pathlib import Path

import pytest

from trade_store import TradeStore


@pytest.fixture
def store(tmp_path):
    """Create a TradeStore with a temporary database."""
    db_path = tmp_path / "test_trades.db"
    return TradeStore(db_path=db_path)


class TestOrders:
    def test_record_order(self, store):
        order_id = store.record_order("shioaji", "Buy", 2)
        assert order_id > 0

    def test_record_order_with_idempotency(self, store):
        order_id = store.record_order(
            "shioaji", "Buy", 2, idempotency_key="abc123"
        )
        assert order_id > 0

    def test_idempotency_check_found(self, store):
        store.record_order("shioaji", "Buy", 2, idempotency_key="key1")
        existing = store.check_idempotency("key1")
        assert existing is not None
        assert existing["action"] == "Buy"
        assert existing["quantity"] == 2

    def test_idempotency_check_not_found(self, store):
        assert store.check_idempotency("nonexistent") is None

    def test_update_order_status(self, store):
        order_id = store.record_order("shioaji", "Buy", 1)
        store.update_order_status(order_id, "filled")
        pending = store.get_pending_orders()
        assert all(o["id"] != order_id for o in pending)

    def test_set_broker_order_id_and_update_status_by_broker_order_id(self, store):
        order_id = store.record_order("shioaji", "Buy", 1)
        store.set_broker_order_id(order_id, "broker-1")

        matched = store.update_order_status_by_broker_order_id(
            "shioaji", "broker-1", "filled"
        )

        assert matched == order_id
        pending = store.get_pending_orders("shioaji")
        assert all(o["id"] != order_id for o in pending)

    def test_get_pending_orders(self, store):
        store.record_order("shioaji", "Buy", 1)
        store.record_order("rithmic", "Sell", 2)
        pending = store.get_pending_orders()
        assert len(pending) == 2

    def test_get_pending_orders_by_broker(self, store):
        store.record_order("shioaji", "Buy", 1)
        store.record_order("rithmic", "Sell", 2)
        pending = store.get_pending_orders("shioaji")
        assert len(pending) == 1
        assert pending[0]["broker"] == "shioaji"

    def test_claim_pending_order_full_fill(self, store):
        order_id = store.record_order("shioaji", "Buy", 3)
        matched = store.claim_pending_order("shioaji", "Buy", fill_qty=3)
        assert matched == order_id
        pending = store.get_pending_orders("shioaji")
        assert all(o["id"] != order_id for o in pending)

    def test_claim_pending_order_partial_then_final(self, store):
        """Two partial fills for the same order: the first leaves the
        order still open (so a later fill can still be attributed to
        it), the second closes it out."""
        order_id = store.record_order("shioaji", "Buy", 3)

        first = store.claim_pending_order("shioaji", "Buy", fill_qty=1)
        assert first == order_id
        # Still open — must remain claimable.
        still_pending = store.get_pending_orders("shioaji")
        assert any(o["id"] == order_id for o in still_pending), (
            "partially-filled order must still be pending"
        )

        second = store.claim_pending_order("shioaji", "Buy", fill_qty=2)
        assert second == order_id, (
            "subsequent partial fill must attribute to the same order"
        )
        # Now fully filled — no longer pending.
        final_pending = store.get_pending_orders("shioaji")
        assert all(o["id"] != order_id for o in final_pending)

    def test_claim_pending_order_prefers_broker_order_id(self, store):
        first = store.record_order("shioaji", "Buy", 1)
        second = store.record_order("shioaji", "Buy", 1)
        store.set_broker_order_id(second, "broker-2")

        matched = store.claim_pending_order(
            "shioaji", "Buy", fill_qty=1, broker_order_id="broker-2"
        )

        assert matched == second
        pending = store.get_pending_orders("shioaji")
        assert any(o["id"] == first for o in pending)
        assert all(o["id"] != second for o in pending)

    def test_claim_pending_order_no_match(self, store):
        # Different broker / action should not be claimed.
        store.record_order("shioaji", "Buy", 1)
        assert store.claim_pending_order("rithmic", "Buy", fill_qty=1) is None
        assert store.claim_pending_order("shioaji", "Sell", fill_qty=1) is None

    def test_reserve_idempotency_fresh_key(self, store):
        assert store.reserve_idempotency("op-1", "shioaji") is True
        # Same key twice -> the reservations PK blocks it.
        assert store.reserve_idempotency("op-1", "shioaji") is False

    def test_reserve_idempotency_rejects_key_that_only_exists_in_orders(
        self, store
    ):
        """Key was previously used to write a real order row but never
        got a reservation entry (e.g. old code path). A direct caller
        of reserve_idempotency() must still get False — the method's
        contract is 'True iff the key is genuinely unseen'."""
        store.record_order(
            "shioaji", "Buy", 1, idempotency_key="legacy-key",
        )
        assert store.reserve_idempotency("legacy-key", "shioaji") is False

    def test_prune_idempotency_reservations_removes_only_stale_rows(self, store):
        assert store.reserve_idempotency("stale-key", "shioaji") is True
        assert store.reserve_idempotency("fresh-key", "shioaji") is True
        with store._cursor() as cur:
            cur.execute(
                "UPDATE idempotency_reservations SET created_at = ? WHERE key = ?",
                ("2000-01-01T00:00:00+00:00", "stale-key"),
            )

        deleted = store.prune_idempotency_reservations(retention_days=90)

        assert deleted == 1
        assert store.check_idempotency("stale-key") is None
        fresh = store.check_idempotency("fresh-key")
        assert fresh is not None
        assert fresh["source"] == "reservation"


class TestFills:
    def test_record_fill(self, store):
        fill_id = store.record_fill("shioaji", "Buy", 2, 20500.0, pnl=1200.0)
        assert fill_id > 0

    def test_get_recent_fills(self, store):
        store.record_fill("shioaji", "Buy", 1, 20500.0)
        store.record_fill("rithmic", "Sell", 2, 4500.0)
        fills = store.get_recent_fills(10)
        assert len(fills) == 2

    def test_get_today_trades(self, store):
        store.record_fill("shioaji", "Buy", 1, 20500.0, pnl=500.0)
        store.record_fill("shioaji", "Sell", 1, 20600.0, pnl=-200.0)
        trades = store.get_today_trades("shioaji")
        assert len(trades) == 2


class TestPositionSnapshots:
    def test_save_and_get_snapshot(self, store):
        store.save_position_snapshot("shioaji", "long", 3, 20500.0)
        snap = store.get_latest_position("shioaji")
        assert snap is not None
        assert snap["side"] == "long"
        assert snap["quantity"] == 3
        assert snap["entry_price"] == 20500.0

    def test_latest_snapshot(self, store):
        store.save_position_snapshot("shioaji", "long", 2, 20000.0)
        store.save_position_snapshot("shioaji", "short", 1, 20200.0)
        snap = store.get_latest_position("shioaji")
        assert snap["side"] == "short"
        assert snap["quantity"] == 1

    def test_no_snapshot(self, store):
        assert store.get_latest_position("rithmic") is None

    def test_snapshots_per_broker(self, store):
        store.save_position_snapshot("shioaji", "long", 2, 20000.0)
        store.save_position_snapshot("rithmic", "short", 1, 4500.0)
        sj = store.get_latest_position("shioaji")
        rt = store.get_latest_position("rithmic")
        assert sj["side"] == "long"
        assert rt["side"] == "short"


class TestRiskEvents:
    def test_record_risk_event(self, store):
        store.record_risk_event("halt", "shioaji", "daily loss limit")
        # No assertion error = success (just verify no crash)

    def test_record_risk_event_no_broker(self, store):
        store.record_risk_event("system_start", details="bot started")


class TestDailySummary:
    def test_daily_summary(self, store):
        store.record_fill("shioaji", "Buy", 1, 20500.0, pnl=1000.0)
        store.record_fill("shioaji", "Sell", 1, 20600.0, pnl=-500.0)
        store.record_fill("rithmic", "Buy", 2, 4500.0, pnl=800.0)

        summary = store.get_daily_summary()
        assert summary["total_pnl"] == 1300.0
        assert summary["total_trades"] == 3
        assert len(summary["brokers"]) == 2
