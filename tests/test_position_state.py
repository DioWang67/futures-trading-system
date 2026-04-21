"""PositionState unit tests — thread safety & atomic fill logic."""

import threading

import pytest

from position_state import PositionState


class TestPositionState:
    def test_initial_state_is_flat(self):
        ps = PositionState("test")
        assert ps.is_flat()
        pos = ps.get_position()
        assert pos.side == "flat"
        assert pos.quantity == 0

    def test_update_position_long(self):
        ps = PositionState("test")
        ps.update_position("long", 2, 16500.0)
        pos = ps.get_position()
        assert pos.side == "long"
        assert pos.quantity == 2
        assert pos.entry_price == 16500.0

    def test_update_position_flat(self):
        ps = PositionState("test")
        ps.update_position("long", 1)
        ps.update_position("flat", 0)
        assert ps.is_flat()

    def test_negative_quantity_clamped(self):
        ps = PositionState("test")
        ps.update_position("long", -5)
        pos = ps.get_position()
        assert pos.side == "flat"
        assert pos.quantity == 0

    def test_zero_quantity_forces_flat(self):
        ps = PositionState("test")
        ps.update_position("long", 0)
        assert ps.get_position().side == "flat"

    def test_immutable_snapshot(self):
        ps = PositionState("test")
        ps.update_position("long", 1, 100.0)
        snap = ps.get_position()
        ps.update_position("short", 2, 200.0)
        # snapshot should not have changed
        assert snap.side == "long"
        assert snap.quantity == 1


class TestApplyFill:
    def test_buy_from_flat(self):
        ps = PositionState("test")
        ps.apply_fill("Buy", 2, 100.0)
        pos = ps.get_position()
        assert pos.side == "long"
        assert pos.quantity == 2

    def test_sell_from_flat(self):
        ps = PositionState("test")
        ps.apply_fill("Sell", 3, 200.0)
        pos = ps.get_position()
        assert pos.side == "short"
        assert pos.quantity == 3

    def test_buy_closes_short(self):
        ps = PositionState("test")
        ps.update_position("short", 2)
        ps.apply_fill("Buy", 2)
        assert ps.is_flat()

    def test_sell_closes_long(self):
        ps = PositionState("test")
        ps.update_position("long", 3)
        ps.apply_fill("Sell", 3)
        assert ps.is_flat()

    def test_buy_flips_short_to_long(self):
        ps = PositionState("test")
        ps.update_position("short", 1)
        ps.apply_fill("Buy", 3)
        pos = ps.get_position()
        assert pos.side == "long"
        assert pos.quantity == 2

    def test_sell_flips_long_to_short(self):
        ps = PositionState("test")
        ps.update_position("long", 2)
        ps.apply_fill("Sell", 5)
        pos = ps.get_position()
        assert pos.side == "short"
        assert pos.quantity == 3

    def test_buy_adds_to_long(self):
        ps = PositionState("test")
        ps.update_position("long", 2)
        ps.apply_fill("Buy", 3)
        pos = ps.get_position()
        assert pos.side == "long"
        assert pos.quantity == 5

    def test_sell_adds_to_short(self):
        ps = PositionState("test")
        ps.update_position("short", 1)
        ps.apply_fill("Sell", 2)
        pos = ps.get_position()
        assert pos.side == "short"
        assert pos.quantity == 3

    def test_partial_close(self):
        ps = PositionState("test")
        ps.update_position("long", 5)
        ps.apply_fill("Sell", 2)
        pos = ps.get_position()
        assert pos.side == "long"
        assert pos.quantity == 3

    def test_unknown_action_ignored(self):
        ps = PositionState("test")
        ps.update_position("long", 1)
        ps.apply_fill("Unknown", 1)
        # should remain unchanged
        assert ps.get_position().side == "long"


class TestThreadSafety:
    def test_concurrent_fills(self):
        """Apply 1000 fills concurrently and verify final state is consistent."""
        ps = PositionState("test")
        n_buys = 500
        n_sells = 500

        def buy_fills():
            for _ in range(n_buys):
                ps.apply_fill("Buy", 1)

        def sell_fills():
            for _ in range(n_sells):
                ps.apply_fill("Sell", 1)

        t1 = threading.Thread(target=buy_fills)
        t2 = threading.Thread(target=sell_fills)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Net should be flat (500 buys - 500 sells = 0)
        pos = ps.get_position()
        assert pos.quantity == 0
        assert pos.side == "flat"

    def test_concurrent_mixed_operations(self):
        """Mix get_position, update_position, apply_fill concurrently."""
        ps = PositionState("test")
        errors = []

        def reader():
            for _ in range(200):
                pos = ps.get_position()
                if pos.quantity < 0:
                    errors.append("negative quantity")

        def writer():
            for i in range(200):
                if i % 2 == 0:
                    ps.apply_fill("Buy", 1)
                else:
                    ps.apply_fill("Sell", 1)

        threads = [threading.Thread(target=reader) for _ in range(3)]
        threads.append(threading.Thread(target=writer))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
