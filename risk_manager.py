"""
Risk Management Module — production-grade guardrails.

Enforces:
  1. Max daily loss (absolute $ and % of capital)
  2. Max position size per broker
  3. Portfolio drawdown circuit breaker
  4. Per-order quantity cap
  5. Cooldown after consecutive losses
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Optional

from loguru import logger


@dataclass
class RiskConfig:
    """Risk parameters — all values configurable via Settings."""

    max_daily_loss: float = 50_000.0          # Max daily loss in $ (absolute)
    max_daily_loss_pct: float = 0.05          # Max daily loss as % of capital (5%)
    max_position_size: int = 10               # Max contracts per broker
    max_order_qty: int = 5                    # Max contracts per single order
    max_drawdown_pct: float = 0.15            # Portfolio drawdown circuit breaker (15%)
    cooldown_after_consecutive_losses: int = 3  # Pause after N consecutive losses
    cooldown_seconds: int = 300               # Cooldown duration (5 min)
    initial_capital: float = 1_000_000.0      # Starting capital for drawdown calc


@dataclass
class DailyPnL:
    """Tracks intraday PnL."""

    date: date = field(default_factory=lambda: date.today())
    realized_pnl: float = 0.0
    trade_count: int = 0
    win_count: int = 0
    loss_count: int = 0
    consecutive_losses: int = 0


class RiskManager:
    """Thread-safe risk manager that gates every order.

    Usage:
        ok, reason = risk_manager.check_order("shioaji", "buy", 2)
        if not ok:
            # reject the order
    """

    def __init__(self, config: Optional[RiskConfig] = None, trade_store: Any = None) -> None:
        self._config = config or RiskConfig()
        if self._config.initial_capital <= 0:
            # Drawdown % and PnL-pct limits both scale off this; a zero
            # or negative value silently disables both circuit breakers.
            raise ValueError(
                "RiskConfig.initial_capital must be > 0; "
                f"got {self._config.initial_capital}"
            )
        self._lock = threading.Lock()
        self._daily: dict[str, DailyPnL] = {}  # broker_name -> DailyPnL
        self._portfolio_daily = DailyPnL()
        self._peak_equity = self._config.initial_capital
        self._current_equity = self._config.initial_capital
        self._halted = False
        self._halt_reason = ""
        self._halted_at = ""
        self._close_only = False
        self._close_only_reason = ""
        self._cooldown_until: dict[str, float] = {}  # broker -> unix timestamp
        self._trade_store = trade_store
        self._restore_halt_state()

    @property
    def config(self) -> RiskConfig:
        return self._config

    @property
    def is_halted(self) -> bool:
        with self._lock:
            return self._halted

    @property
    def halt_reason(self) -> str:
        with self._lock:
            return self._halt_reason

    @property
    def is_close_only(self) -> bool:
        with self._lock:
            return self._close_only

    @property
    def close_only_reason(self) -> str:
        with self._lock:
            return self._close_only_reason

    def _halt(self, reason: str) -> None:
        self._halted = True
        self._halt_reason = reason
        if not self._halted_at:
            self._halted_at = datetime.now(timezone.utc).isoformat()
        if self._trade_store:
            try:
                self._trade_store.record_risk_event(
                    "halt",
                    details=reason,
                )
            except Exception as exc:
                logger.warning("[RiskManager] failed to persist halt event: {}", exc)
        logger.critical("[RiskManager] TRADING HALTED: {}", reason)

    def _restore_halt_state(self) -> None:
        """Restore same-day halt state from persistent risk events."""
        if self._trade_store is None:
            return
        try:
            events = self._trade_store.get_recent_risk_events(limit=200)
        except Exception as exc:
            logger.warning("[RiskManager] failed to read risk events: {}", exc)
            return

        today_utc = datetime.now(timezone.utc).date()
        for event in events:
            event_type = str(event.get("event_type") or "")
            created_at = str(event.get("created_at") or "")
            if event_type in {"resume_trading", "daily_reset"}:
                return
            if event_type != "halt":
                continue
            try:
                created_dt = datetime.fromisoformat(created_at)
                if created_dt.tzinfo is None:
                    created_dt = created_dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            if created_dt.date() != today_utc:
                continue
            self._halted = True
            self._halt_reason = str(event.get("details") or "restored halt")
            self._halted_at = created_dt.isoformat()
            logger.warning(
                "[RiskManager] Restored halted state from risk_events (at={})",
                self._halted_at,
            )
            return

    def set_close_only(self, enabled: bool, reason: str = "") -> None:
        with self._lock:
            self._close_only = bool(enabled)
            self._close_only_reason = reason if enabled else ""
            if enabled:
                logger.warning("[RiskManager] CLOSE-ONLY enabled: {}", reason)
            else:
                logger.info("[RiskManager] CLOSE-ONLY cleared")

    def _get_daily(self, broker_name: str) -> DailyPnL:
        today = date.today()
        if broker_name not in self._daily or self._daily[broker_name].date != today:
            self._daily[broker_name] = DailyPnL(date=today)
        if self._portfolio_daily.date != today:
            self._portfolio_daily = DailyPnL(date=today)
        return self._daily[broker_name]

    def check_order(
        self,
        broker_name: str,
        action: str,
        quantity: int,
    ) -> tuple[bool, str]:
        """Pre-trade risk check. Returns (allowed, reason).

        Parameters
        ----------
        broker_name : str
        action : str — "buy", "sell", "exit"
        quantity : int — requested order size
        """
        with self._lock:
            # 1. Exit orders always allowed — they reduce risk and must still
            #    work while the system is halted so we can flatten positions.
            if action == "exit":
                return True, ""

            if self._close_only:
                reason = self._close_only_reason or "close-only mode active"
                return False, f"close-only mode active: {reason}"

            # 2. Global halt check (blocks new exposure, not exits)
            if self._halted:
                return False, f"trading halted: {self._halt_reason}"

            # 3. Per-order quantity cap
            if quantity > self._config.max_order_qty:
                return False, (
                    f"order qty {quantity} exceeds max {self._config.max_order_qty}"
                )

            # 4. Position size limit — must match route_order's actual
            #    semantics: a buy/sell signal always ends up holding
            #    ``quantity`` contracts on the new side. Same-side signals
            #    are short-circuited as "skipped" upstream and never reach
            #    here; flat / opposite both fully flip to ``quantity``.
            #    Using the signed-net trick (short 3 + buy 5 -> net 2) is
            #    wrong because the real final exposure is long 5.
            new_exposure = quantity
            if new_exposure > self._config.max_position_size:
                return False, (
                    f"position would be {new_exposure}, "
                    f"max allowed {self._config.max_position_size}"
                )

            # 5. Daily loss limit
            daily = self._get_daily(broker_name)
            portfolio = self._portfolio_daily

            abs_limit = self._config.max_daily_loss
            pct_limit = self._config.max_daily_loss_pct * self._config.initial_capital

            effective_limit = min(abs_limit, pct_limit)

            if portfolio.realized_pnl <= -effective_limit:
                self._halt(
                    f"daily loss limit reached: "
                    f"${portfolio.realized_pnl:,.0f} <= -${effective_limit:,.0f}"
                )
                return False, f"daily loss limit reached: ${portfolio.realized_pnl:,.0f}"

            # 6. Drawdown circuit breaker
            drawdown_pct = 0.0
            if self._peak_equity > 0:
                drawdown_pct = (
                    (self._peak_equity - self._current_equity) / self._peak_equity
                )
            if drawdown_pct >= self._config.max_drawdown_pct:
                self._halt(
                    f"drawdown circuit breaker: {drawdown_pct:.1%} >= "
                    f"{self._config.max_drawdown_pct:.1%}"
                )
                return False, f"drawdown {drawdown_pct:.1%} exceeds limit"

            # 7. Cooldown after consecutive losses
            now_ts = datetime.now(timezone.utc).timestamp()
            cooldown_until = self._cooldown_until.get(broker_name, 0)
            if now_ts < cooldown_until:
                remaining = int(cooldown_until - now_ts)
                return False, f"cooldown active ({remaining}s remaining)"

            return True, ""

    def record_fill(
        self,
        broker_name: str,
        pnl: float,
        is_closing_trade: bool = True,
    ) -> None:
        """Record a completed trade's PnL for risk tracking.

        Parameters
        ----------
        pnl : float
            Realized PnL of this fill (0 for opening trades).
        is_closing_trade : bool
            True if this fill closes/reduces a position (PnL is realized).
        """
        with self._lock:
            if not is_closing_trade:
                return

            daily = self._get_daily(broker_name)
            daily.realized_pnl += pnl
            daily.trade_count += 1

            self._portfolio_daily.realized_pnl += pnl
            self._portfolio_daily.trade_count += 1

            # Update equity tracking
            self._current_equity += pnl
            if self._current_equity > self._peak_equity:
                self._peak_equity = self._current_equity

            # Track wins/losses
            if pnl > 0:
                daily.win_count += 1
                daily.consecutive_losses = 0
                self._portfolio_daily.win_count += 1
                self._portfolio_daily.consecutive_losses = 0
            elif pnl < 0:
                daily.loss_count += 1
                daily.consecutive_losses += 1
                self._portfolio_daily.loss_count += 1
                self._portfolio_daily.consecutive_losses += 1

                # Cooldown check
                threshold = self._config.cooldown_after_consecutive_losses
                if daily.consecutive_losses >= threshold:
                    cooldown_end = (
                        datetime.now(timezone.utc).timestamp()
                        + self._config.cooldown_seconds
                    )
                    self._cooldown_until[broker_name] = cooldown_end
                    logger.warning(
                        "[RiskManager] {} consecutive losses on {}, "
                        "cooldown for {}s",
                        daily.consecutive_losses,
                        broker_name,
                        self._config.cooldown_seconds,
                    )

            logger.info(
                "[RiskManager] {} fill recorded: PnL=${:,.0f}, "
                "daily=${:,.0f}, equity=${:,.0f}, drawdown={:.2%}",
                broker_name, pnl,
                self._portfolio_daily.realized_pnl,
                self._current_equity,
                (self._peak_equity - self._current_equity) / self._peak_equity
                if self._peak_equity > 0 else 0,
            )

            # Check if daily loss limit is now breached
            abs_limit = self._config.max_daily_loss
            pct_limit = self._config.max_daily_loss_pct * self._config.initial_capital
            effective_limit = min(abs_limit, pct_limit)
            portfolio = self._portfolio_daily

            if not self._halted and portfolio.realized_pnl <= -effective_limit:
                self._halt(
                    f"daily loss limit reached: "
                    f"${portfolio.realized_pnl:,.0f} <= -${effective_limit:,.0f}"
                )

            # Check drawdown circuit breaker
            drawdown_pct = 0.0
            if self._peak_equity > 0:
                drawdown_pct = (
                    (self._peak_equity - self._current_equity) / self._peak_equity
                )
            if not self._halted and drawdown_pct >= self._config.max_drawdown_pct:
                self._halt(
                    f"drawdown circuit breaker: {drawdown_pct:.1%} >= "
                    f"{self._config.max_drawdown_pct:.1%}"
                )

    def reset_daily(self) -> None:
        """Reset daily counters (called at start of trading day)."""
        with self._lock:
            self._daily.clear()
            self._portfolio_daily = DailyPnL()
            # Start a fresh daily drawdown baseline from current equity.
            self._peak_equity = self._current_equity
            if self._trade_store:
                try:
                    self._trade_store.record_risk_event(
                        "daily_reset",
                        details="daily counters reset",
                    )
                except Exception as exc:
                    logger.warning("[RiskManager] failed to persist daily_reset event: {}", exc)
            logger.info("[RiskManager] Daily counters reset")

    def resume_trading(self) -> None:
        """Manually resume after a halt (requires human intervention)."""
        with self._lock:
            self._halted = False
            self._halt_reason = ""
            self._halted_at = ""
            self._close_only = False
            self._close_only_reason = ""
            self._cooldown_until.clear()
            if self._trade_store:
                try:
                    self._trade_store.record_risk_event(
                        "resume_trading",
                        details="trading resumed by operator",
                    )
                except Exception as exc:
                    logger.warning("[RiskManager] failed to persist resume_trading event: {}", exc)
            logger.info("[RiskManager] Trading resumed by operator")

    def get_status(self) -> dict:
        """Return current risk status for monitoring."""
        with self._lock:
            portfolio = self._portfolio_daily
            drawdown_pct = 0.0
            if self._peak_equity > 0:
                drawdown_pct = (
                    (self._peak_equity - self._current_equity) / self._peak_equity
                )
            return {
                "halted": self._halted,
                "halt_reason": self._halt_reason,
                "halted_since": self._halted_at,
                "close_only": self._close_only,
                "close_only_reason": self._close_only_reason,
                "daily_pnl": portfolio.realized_pnl,
                "daily_trades": portfolio.trade_count,
                "daily_wins": portfolio.win_count,
                "daily_losses": portfolio.loss_count,
                "consecutive_losses": portfolio.consecutive_losses,
                "current_equity": self._current_equity,
                "peak_equity": self._peak_equity,
                "drawdown_pct": drawdown_pct,
                "max_drawdown_limit": self._config.max_drawdown_pct,
                "daily_loss_limit": min(
                    self._config.max_daily_loss,
                    self._config.max_daily_loss_pct * self._config.initial_capital,
                ),
            }
