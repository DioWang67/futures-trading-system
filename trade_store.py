"""
Trade persistence layer — SQLite-based audit trail and crash recovery.

Stores:
  - Every order placed (with idempotency key)
  - Every fill received
  - Position snapshots (for crash recovery)
  - Risk events (halts, cooldowns)
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from loguru import logger

DB_PATH = Path(__file__).resolve().parent / "data" / "trades.db"


class TradeStore:
    """Thread-safe SQLite trade store."""

    def __init__(self, db_path: Optional[Path | str] = None) -> None:
        self._db_path = db_path or DB_PATH
        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
            )
            self._local.conn.row_factory = sqlite3.Row
            try:
                self._local.conn.execute("PRAGMA journal_mode=WAL")
            except sqlite3.OperationalError as exc:
                logger.warning(
                    "[TradeStore] WAL unavailable for {} ({}); falling back to DELETE journal mode",
                    self._db_path,
                    exc,
                )
                self._local.conn.execute("PRAGMA journal_mode=DELETE")
            self._local.conn.execute("PRAGMA busy_timeout=5000")
        return self._local.conn

    @contextmanager
    def _cursor(self):
        conn = self._get_conn()
        cur = conn.cursor()
        try:
            yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _init_db(self) -> None:
        with self._cursor() as cur:
            cur.executescript("""
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    idempotency_key TEXT UNIQUE,
                    broker TEXT NOT NULL,
                    action TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    filled_qty INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    updated_at TEXT,
                    broker_order_id TEXT,
                    error_message TEXT
                );

                CREATE TABLE IF NOT EXISTS fills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id INTEGER REFERENCES orders(id),
                    broker TEXT NOT NULL,
                    action TEXT NOT NULL,
                    filled_qty INTEGER NOT NULL,
                    fill_price REAL NOT NULL DEFAULT 0.0,
                    pnl REAL DEFAULT 0.0,
                    commission REAL DEFAULT 0.0,
                    filled_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS position_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    broker TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    entry_price REAL NOT NULL DEFAULT 0.0,
                    snapshot_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS risk_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    broker TEXT,
                    details TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS protective_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    broker TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    trigger_reason TEXT NOT NULL,
                    stop_loss REAL NOT NULL DEFAULT 0.0,
                    take_profit REAL NOT NULL DEFAULT 0.0,
                    trigger_price REAL NOT NULL DEFAULT 0.0,
                    submit_price REAL NOT NULL DEFAULT 0.0,
                    fill_price REAL NOT NULL DEFAULT 0.0,
                    slippage_points REAL NOT NULL DEFAULT 0.0,
                    execution_price_type TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'triggered',
                    broker_order_id TEXT,
                    details TEXT,
                    triggered_at TEXT NOT NULL,
                    submitted_at TEXT,
                    filled_at TEXT
                );

                -- Idempotency gate: route_order reserves a key here BEFORE
                -- any broker submit so a webhook retry arriving while the
                -- first request is still mid-flight (e.g. waiting for the
                -- reversal close leg to fill) cannot slip past the
                -- idempotency check and duplicate-submit the close leg.
                CREATE TABLE IF NOT EXISTS idempotency_reservations (
                    key TEXT PRIMARY KEY,
                    broker TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_orders_idempotency
                    ON orders(idempotency_key);
                CREATE INDEX IF NOT EXISTS idx_orders_broker_status
                    ON orders(broker, status);
                CREATE INDEX IF NOT EXISTS idx_fills_broker
                    ON fills(broker, filled_at);
                CREATE INDEX IF NOT EXISTS idx_protective_events_broker
                    ON protective_events(broker, triggered_at);
                CREATE INDEX IF NOT EXISTS idx_position_snapshots_broker
                    ON position_snapshots(broker, snapshot_at);
                CREATE INDEX IF NOT EXISTS idx_idempotency_reservations_created_at
                    ON idempotency_reservations(created_at);
            """)
            # Migrate older DBs that pre-date the filled_qty column.
            cur.execute("PRAGMA table_info(orders)")
            cols = {row["name"] for row in cur.fetchall()}
            if "filled_qty" not in cols:
                cur.execute(
                    "ALTER TABLE orders ADD COLUMN filled_qty "
                    "INTEGER NOT NULL DEFAULT 0"
                )
        logger.info("[TradeStore] Database initialized at {}", self._db_path)

    # ------------------------------------------------------------------
    # Idempotency
    # ------------------------------------------------------------------
    def check_idempotency(self, key: str) -> Optional[dict]:
        """Check whether this idempotency key has already been seen.

        Returns:
          - The existing order row (dict with ``id``), if a real order
            was already recorded for this key.
          - A reservation marker ``{"source": "reservation", "key": ...,
            "broker": ..., "created_at": ...}`` (no ``id``) if another
            request is mid-flight but hasn't recorded its orders yet.
          - None if the key is unknown.
        """
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM orders WHERE idempotency_key = ?", (key,)
            )
            row = cur.fetchone()
            if row:
                return dict(row)
            cur.execute(
                "SELECT key, broker, created_at "
                "FROM idempotency_reservations WHERE key = ?",
                (key,),
            )
            row = cur.fetchone()
            if row:
                return {"source": "reservation", **dict(row)}
            return None

    def reserve_idempotency(self, key: str, broker: str) -> bool:
        """Atomically claim ``key`` for processing. Returns True on a
        fresh reservation, False if the key is already reserved OR has
        already produced an order row.

        Both checks run inside a single cursor/transaction so this
        method's contract holds even when called directly (independent
        of any upstream ``check_idempotency()``).

        Reservations are deliberately *not* cleaned up on failure:
        once a key has been observed it is considered used, and any
        retry with the same key should be treated as a duplicate.
        Operators must issue a new key to re-drive a signal that
        errored mid-flight (e.g. reversal close timeout).
        """
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._cursor() as cur:
                # Reject if the key already produced a real order.
                cur.execute(
                    "SELECT 1 FROM orders WHERE idempotency_key = ? LIMIT 1",
                    (key,),
                )
                if cur.fetchone() is not None:
                    return False
                # Otherwise claim the key; PRIMARY KEY collision means
                # another concurrent request just reserved it.
                cur.execute(
                    "INSERT INTO idempotency_reservations "
                    "(key, broker, created_at) VALUES (?, ?, ?)",
                    (key, broker, now),
                )
            return True
        except sqlite3.IntegrityError:
            return False

    def prune_idempotency_reservations(self, retention_days: int = 90) -> int:
        """Delete stale reservation markers older than ``retention_days``."""
        retention_days = int(retention_days)
        if retention_days <= 0:
            return 0
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=retention_days)
        ).isoformat()
        with self._cursor() as cur:
            cur.execute(
                "DELETE FROM idempotency_reservations WHERE created_at < ?",
                (cutoff,),
            )
            deleted = max(int(cur.rowcount or 0), 0)
        return deleted

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------
    def record_order(
        self,
        broker: str,
        action: str,
        quantity: int,
        idempotency_key: str = "",
        broker_order_id: str = "",
    ) -> int:
        """Record a new order in ``pending`` state. Returns the order ID.

        The caller is expected to flip it to ``submitted`` once the
        broker accepts the order, or to ``error`` if submission fails.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO orders
                   (idempotency_key, broker, action, quantity, status,
                    created_at, broker_order_id)
                   VALUES (?, ?, ?, ?, 'pending', ?, ?)""",
                (idempotency_key or None, broker, action, quantity,
                 now, broker_order_id),
            )
            order_id = cur.lastrowid
        logger.debug(
            "[TradeStore] Order recorded: id={} {} {} x{}",
            order_id, broker, action, quantity,
        )
        return order_id

    def update_order_status(
        self, order_id: int, status: str, error_message: str = ""
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._cursor() as cur:
            cur.execute(
                """UPDATE orders SET status = ?, updated_at = ?,
                   error_message = ? WHERE id = ?""",
                (status, now, error_message, order_id),
            )

    def set_broker_order_id(self, order_id: int, broker_order_id: str) -> None:
        if not broker_order_id:
            return
        now = datetime.now(timezone.utc).isoformat()
        with self._cursor() as cur:
            cur.execute(
                """UPDATE orders
                   SET broker_order_id = ?, updated_at = ?
                   WHERE id = ?""",
                (broker_order_id, now, order_id),
            )

    def update_order_status_by_broker_order_id(
        self,
        broker: str,
        broker_order_id: str,
        status: str,
        error_message: str = "",
    ) -> Optional[int]:
        if not broker_order_id:
            return None
        now = datetime.now(timezone.utc).isoformat()
        with self._cursor() as cur:
            cur.execute(
                """SELECT id FROM orders
                   WHERE broker = ? AND broker_order_id = ?
                   ORDER BY id DESC LIMIT 1""",
                (broker, broker_order_id),
            )
            row = cur.fetchone()
            if not row:
                return None
            order_id = int(row["id"])
            cur.execute(
                """UPDATE orders
                   SET status = ?, updated_at = ?, error_message = ?
                   WHERE id = ?""",
                (status, now, error_message, order_id),
            )
            return order_id

    def claim_pending_order(
        self,
        broker: str,
        action: str,
        fill_qty: int = 0,
        broker_order_id: str = "",
    ) -> Optional[int]:
        """Attribute a fill to the best-matching still-open order for
        ``(broker, action)``.

        Matching precedence:
          1. If ``broker_order_id`` is supplied and a row is tagged
             with it, use that row.
          2. Prefer the oldest order whose **remaining** quantity
             equals ``fill_qty`` — this keeps two same-direction orders
             from cross-contaminating when the exchange reports their
             fills out of order.
          3. Fall back to the oldest order that still has remaining
             capacity for ``fill_qty``.
          4. Last resort: oldest open order regardless of remaining qty.

        The order's ``filled_qty`` is advanced by ``fill_qty``; the
        status becomes ``filled`` once cumulative filled_qty reaches
        the original order quantity, otherwise ``partially_filled``.

        Returns the matched order id, or None if nothing is open.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._cursor() as cur:
            row = None
            if broker_order_id:
                cur.execute(
                    """SELECT id, quantity, filled_qty FROM orders
                       WHERE broker = ? AND action = ?
                         AND broker_order_id = ?
                         AND status IN ('submitted', 'partially_filled', 'pending')
                       ORDER BY id DESC LIMIT 1""",
                    (broker, action, broker_order_id),
                )
                row = cur.fetchone()
            if row is None and fill_qty > 0:
                # Prefer an order whose remaining quantity exactly matches
                # this fill — avoids charging a small fill against a
                # larger order while a smaller same-direction order is
                # still outstanding.
                cur.execute(
                    """SELECT id, quantity, filled_qty FROM orders
                       WHERE broker = ? AND action = ?
                         AND status IN ('submitted', 'partially_filled', 'pending')
                         AND (quantity - COALESCE(filled_qty, 0)) = ?
                       ORDER BY id ASC LIMIT 1""",
                    (broker, action, int(fill_qty)),
                )
                row = cur.fetchone()
                if row is None:
                    cur.execute(
                        """SELECT id, quantity, filled_qty FROM orders
                           WHERE broker = ? AND action = ?
                             AND status IN ('submitted', 'partially_filled', 'pending')
                             AND (quantity - COALESCE(filled_qty, 0)) >= ?
                           ORDER BY id ASC LIMIT 1""",
                        (broker, action, int(fill_qty)),
                    )
                    row = cur.fetchone()
            if row is None:
                cur.execute(
                    """SELECT id, quantity, filled_qty FROM orders
                       WHERE broker = ? AND action = ?
                         AND status IN ('submitted', 'partially_filled', 'pending')
                       ORDER BY id ASC LIMIT 1""",
                    (broker, action),
                )
                row = cur.fetchone()
            if not row:
                return None
            order_id = row["id"]
            order_qty = int(row["quantity"])
            prior_filled = int(row["filled_qty"] or 0)
            new_filled = min(order_qty, prior_filled + max(0, int(fill_qty)))
            new_status = "filled" if new_filled >= order_qty else "partially_filled"
            cur.execute(
                """UPDATE orders
                   SET filled_qty = ?, status = ?, updated_at = ?
                   WHERE id = ?""",
                (new_filled, new_status, now, order_id),
            )
            return order_id

    # ------------------------------------------------------------------
    # Fills
    # ------------------------------------------------------------------
    def record_fill(
        self,
        broker: str,
        action: str,
        filled_qty: int,
        fill_price: float = 0.0,
        pnl: float = 0.0,
        commission: float = 0.0,
        order_id: Optional[int] = None,
    ) -> int:
        """Record a fill event. Returns the fill ID."""
        now = datetime.now(timezone.utc).isoformat()
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO fills
                   (order_id, broker, action, filled_qty, fill_price,
                    pnl, commission, filled_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (order_id, broker, action, filled_qty, fill_price,
                 pnl, commission, now),
            )
            fill_id = cur.lastrowid
        logger.debug(
            "[TradeStore] Fill recorded: id={} {} {} x{} @{:.1f} PnL={:.0f}",
            fill_id, broker, action, filled_qty, fill_price, pnl,
        )
        return fill_id

    def record_protective_event(
        self,
        broker: str,
        ticker: str,
        side: str,
        quantity: int,
        trigger_reason: str,
        stop_loss: float,
        take_profit: float,
        trigger_price: float,
        execution_price_type: str,
        submit_price: float = 0.0,
        status: str = "triggered",
        broker_order_id: str = "",
        details: str = "",
    ) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO protective_events
                   (broker, ticker, side, quantity, trigger_reason,
                    stop_loss, take_profit, trigger_price, submit_price,
                    execution_price_type, status, broker_order_id, details,
                    triggered_at, submitted_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    broker,
                    ticker,
                    side,
                    quantity,
                    trigger_reason,
                    float(stop_loss or 0.0),
                    float(take_profit or 0.0),
                    float(trigger_price or 0.0),
                    float(submit_price or 0.0),
                    execution_price_type,
                    status,
                    broker_order_id or None,
                    details,
                    now,
                    now if status in {"submitted", "filled"} else None,
                ),
            )
            return int(cur.lastrowid)

    def update_protective_event(
        self,
        event_id: int,
        *,
        status: Optional[str] = None,
        submit_price: Optional[float] = None,
        fill_price: Optional[float] = None,
        slippage_points: Optional[float] = None,
        execution_price_type: str = "",
        broker_order_id: str = "",
        details: Optional[str] = None,
        mark_submitted: bool = False,
        mark_filled: bool = False,
    ) -> None:
        updates: list[str] = []
        params: list[object] = []

        if status is not None:
            updates.append("status = ?")
            params.append(status)
        if submit_price is not None:
            updates.append("submit_price = ?")
            params.append(float(submit_price))
        if fill_price is not None:
            updates.append("fill_price = ?")
            params.append(float(fill_price))
        if slippage_points is not None:
            updates.append("slippage_points = ?")
            params.append(float(slippage_points))
        if execution_price_type:
            updates.append("execution_price_type = ?")
            params.append(execution_price_type)
        if broker_order_id:
            updates.append("broker_order_id = ?")
            params.append(broker_order_id)
        if details is not None:
            updates.append("details = ?")
            params.append(details)
        if mark_submitted:
            updates.append("submitted_at = ?")
            params.append(datetime.now(timezone.utc).isoformat())
        if mark_filled:
            updates.append("filled_at = ?")
            params.append(datetime.now(timezone.utc).isoformat())

        if not updates:
            return

        params.append(int(event_id))
        with self._cursor() as cur:
            cur.execute(
                f"UPDATE protective_events SET {', '.join(updates)} WHERE id = ?",
                tuple(params),
            )

    def get_recent_protective_events(self, limit: int = 50) -> list[dict]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM protective_events ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            return [dict(row) for row in cur.fetchall()]

    # ------------------------------------------------------------------
    # Position snapshots
    # ------------------------------------------------------------------
    def save_position_snapshot(
        self,
        broker: str,
        side: str,
        quantity: int,
        entry_price: float = 0.0,
    ) -> None:
        """Persist current position for crash recovery."""
        now = datetime.now(timezone.utc).isoformat()
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO position_snapshots
                   (broker, side, quantity, entry_price, snapshot_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (broker, side, quantity, entry_price, now),
            )

    def get_latest_position(self, broker: str) -> Optional[dict]:
        """Get the most recent position snapshot for crash recovery."""
        with self._cursor() as cur:
            cur.execute(
                """SELECT side, quantity, entry_price, snapshot_at
                   FROM position_snapshots
                   WHERE broker = ?
                   ORDER BY id DESC LIMIT 1""",
                (broker,),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    # ------------------------------------------------------------------
    # Risk events
    # ------------------------------------------------------------------
    def record_risk_event(
        self,
        event_type: str,
        broker: str = "",
        details: str = "",
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO risk_events
                   (event_type, broker, details, created_at)
                   VALUES (?, ?, ?, ?)""",
                (event_type, broker, details, now),
            )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def get_today_trades(self, broker: str = "") -> list[dict]:
        """Get all fills from today."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self._cursor() as cur:
            if broker:
                cur.execute(
                    """SELECT * FROM fills
                       WHERE broker = ? AND filled_at >= ?
                       ORDER BY filled_at""",
                    (broker, today),
                )
            else:
                cur.execute(
                    """SELECT * FROM fills WHERE filled_at >= ?
                       ORDER BY filled_at""",
                    (today,),
                )
            return [dict(row) for row in cur.fetchall()]

    def get_pending_orders(self, broker: str = "") -> list[dict]:
        """Get orders that haven't been filled or cancelled."""
        open_states = ("pending", "submitted", "partially_filled")
        placeholders = ",".join("?" * len(open_states))
        with self._cursor() as cur:
            if broker:
                cur.execute(
                    f"""SELECT * FROM orders
                        WHERE broker = ? AND status IN ({placeholders})
                        ORDER BY created_at""",
                    (broker, *open_states),
                )
            else:
                cur.execute(
                    f"""SELECT * FROM orders
                        WHERE status IN ({placeholders})
                        ORDER BY created_at""",
                    open_states,
                )
            return [dict(row) for row in cur.fetchall()]

    def get_recent_fills(self, limit: int = 50) -> list[dict]:
        """Get most recent fills across all brokers."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM fills ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            return [dict(row) for row in cur.fetchall()]

    def get_recent_risk_events(self, limit: int = 20) -> list[dict]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM risk_events ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            return [dict(row) for row in cur.fetchall()]

    def get_daily_summary(self) -> dict:
        """Get today's trading summary."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self._cursor() as cur:
            cur.execute(
                """SELECT broker,
                          COUNT(*) as trade_count,
                          SUM(pnl) as total_pnl,
                          SUM(commission) as total_commission,
                          SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                          SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losses
                   FROM fills
                   WHERE filled_at >= ?
                   GROUP BY broker""",
                (today,),
            )
            rows = [dict(row) for row in cur.fetchall()]
            return {
                "date": today,
                "brokers": rows,
                "total_pnl": sum(r["total_pnl"] or 0 for r in rows),
                "total_trades": sum(r["trade_count"] for r in rows),
            }
