"""
Bar-by-bar backtesting engine.
Replicates Pine Script's strategy.entry / strategy.exit behaviour including:
- Position tracking (flat / long / short)
- Signal distance / repeated-signal filtering
- Three exit modes: fixed TP, trailing, partial (split)
- EOD forced close
- Slippage & commission
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from backtest.signals import is_in_session


# ============================================================================
# Trade record
# ============================================================================

@dataclass
class Trade:
    entry_bar: int = 0
    entry_time: Optional[pd.Timestamp] = None
    entry_price: float = 0.0
    direction: str = ""          # "long" | "short"
    quantity: int = 0
    exit_bar: int = 0
    exit_time: Optional[pd.Timestamp] = None
    exit_price: float = 0.0
    exit_reason: str = ""
    pnl: float = 0.0
    commission: float = 0.0
    net_pnl: float = 0.0


# ============================================================================
# Engine
# ============================================================================

class BacktestEngine:
    """Bar-by-bar simulation matching Pine Script strategy behaviour."""

    def __init__(self, df: pd.DataFrame, cfg):
        self.df = df
        self.cfg = cfg

        # Position state
        self.position_size: int = 0       # >0 long, <0 short, 0 flat
        self.entry_price: float = 0.0

        # Locked exit levels (set on entry fill)
        self.locked_atr: float = 0.0
        self.locked_sl: float = 0.0
        self.locked_tp: float = 0.0
        self.locked_tp1: float = 0.0      # for partial exit
        self.exit_placed: bool = False
        self.tp1_hit: bool = False
        self.remaining_qty: int = 0       # for partial exit tracking

        # Trailing stop state
        self.trail_active: bool = False
        self.trail_stop: float = 0.0

        # Signal state (mirrors Pine)
        self.last_raw_signal_bar: int = -cfg.min_signal_distance - 1
        self.last_signal_bar: int = -cfg.min_signal_distance - 1
        self.last_signal: str = "Neutral"
        self.last_exit_bar: int = -cfg.reentry_cooldown - 1
        self.last_exit_dir: str = "None"

        # Results
        self.trades: list[Trade] = []
        self.equity_curve: list[float] = []
        self.capital: float = cfg.initial_capital
        self._current_trade: Optional[Trade] = None

        # Structure state (HH/HL/LH/LL)
        self.sh1: float = np.nan
        self.sh2: float = np.nan
        self.sl1: float = np.nan
        self.sl2: float = np.nan

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> list[Trade]:
        df = self.df
        cfg = self.cfg
        n = len(df)

        for i in range(cfg.struct_lookback + 1, n):
            row = df.iloc[i]
            prev_row = df.iloc[i - 1]
            bar_idx = i

            o = row["open"]
            h = row["high"]
            l = row["low"]
            c = row["close"]
            atr_val = row["atr"]

            # Convert index to NY time for session check
            ts = df.index[i]
            if hasattr(ts, "tz") and ts.tz is not None:
                ts_ny = ts.tz_convert("America/New_York")
            else:
                ts_ny = ts

            in_session = (not cfg.use_time_filter) or \
                         is_in_session(ts_ny, cfg.trading_session_start, cfg.trading_session_end)

            prev_position_size = self.position_size

            # ── Update structure pivots ──
            self._update_structure(df, i, cfg)

            # ── Check exit conditions (before entry, same bar) ──
            if self.position_size != 0:
                self._process_exits(i, row, in_session)

            # ── Entry logic (only when flat and in session) ──
            if self.position_size == 0 and in_session:
                self._process_entries(i, row, bar_idx, ts)

            # ── Track equity ──
            unrealized = 0.0
            if self.position_size > 0:
                unrealized = (c - self.entry_price) * abs(self.position_size) * cfg.point_value
            elif self.position_size < 0:
                unrealized = (self.entry_price - c) * abs(self.position_size) * cfg.point_value
            self.equity_curve.append(self.capital + unrealized)

            # ── Detect position close (for last_exit tracking) ──
            if prev_position_size != 0 and self.position_size == 0:
                self.last_exit_bar = bar_idx
                self.last_exit_dir = "Long" if prev_position_size > 0 else "Short"

        return self.trades

    # ------------------------------------------------------------------
    # Structure pivot tracking (HH/HL/LH/LL)
    # ------------------------------------------------------------------
    def _update_structure(self, df: pd.DataFrame, i: int, cfg):
        ph = df.iloc[i]["struct_ph"]
        pl = df.iloc[i]["struct_pl"]
        if not np.isnan(ph):
            self.sh2 = self.sh1
            self.sh1 = ph
        if not np.isnan(pl):
            self.sl2 = self.sl1
            self.sl1 = pl

    # ------------------------------------------------------------------
    # Entry logic
    # ------------------------------------------------------------------
    def _process_entries(self, i: int, row, bar_idx: int, ts):
        cfg = self.cfg

        smart_buy = bool(row.get("smart_buy", False))
        smart_sell = bool(row.get("smart_sell", False))
        sl_precheck = bool(row.get("sl_precheck_ok", True))

        # ── Cooldown after exit ──
        cooldown_ok_long = (not cfg.avoid_exit_reentry) or \
                           (self.last_exit_dir != "Long") or \
                           (bar_idx - self.last_exit_bar > cfg.reentry_cooldown)
        cooldown_ok_short = (not cfg.avoid_exit_reentry) or \
                            (self.last_exit_dir != "Short") or \
                            (bar_idx - self.last_exit_bar > cfg.reentry_cooldown)

        # ── Repeated signal filter ──
        buy_allowed = (not cfg.restrict_repeated_signals) or (self.last_signal != "Buy")
        sell_allowed = (not cfg.restrict_repeated_signals) or (self.last_signal != "Sell")

        # ── Signal distance ──
        dist_ok = (bar_idx - self.last_raw_signal_bar) >= cfg.min_signal_distance

        # ── Gainz conditions ──
        gainz_buy = smart_buy and cooldown_ok_long and dist_ok and buy_allowed and sl_precheck
        gainz_sell = smart_sell and cooldown_ok_short and dist_ok and sell_allowed and sl_precheck and (not gainz_buy)

        if gainz_buy or gainz_sell:
            self.last_raw_signal_bar = bar_idx

        # ── Trade mode filter ──
        if gainz_buy and cfg.trade_mode == "只做空":
            gainz_buy = False
        if gainz_sell and cfg.trade_mode == "只做多":
            gainz_sell = False

        # ── Execute entry ──
        if gainz_buy:
            self._enter_long(i, row, bar_idx, ts)
        elif gainz_sell:
            self._enter_short(i, row, bar_idx, ts)

    def _enter_long(self, i: int, row, bar_idx: int, ts):
        cfg = self.cfg
        # Fill at next bar's open + slippage (realistic execution)
        if i + 1 < len(self.df):
            fill_price = self.df.iloc[i + 1]["open"]
        else:
            fill_price = row["close"]
        fill_price += cfg.slippage_points * cfg.tick_size

        atr_val = row["atr"]
        qty = cfg.default_qty

        self.position_size = qty
        self.entry_price = fill_price
        self.locked_atr = atr_val
        self.locked_sl = fill_price - atr_val * cfg.sl_atr
        self.locked_tp = fill_price + atr_val * cfg.sl_atr * cfg.tp_ratio
        self.locked_tp1 = fill_price + atr_val * cfg.sl_atr * cfg.tp1_ratio
        self.exit_placed = False
        self.tp1_hit = False
        self.remaining_qty = qty
        self.trail_active = False
        self.trail_stop = 0.0

        self.last_signal = "Buy"
        self.last_signal_bar = bar_idx

        self._current_trade = Trade(
            entry_bar=bar_idx,
            entry_time=ts,
            entry_price=fill_price,
            direction="long",
            quantity=qty,
        )

    def _enter_short(self, i: int, row, bar_idx: int, ts):
        cfg = self.cfg
        # Fill at next bar's open - slippage (realistic execution)
        if i + 1 < len(self.df):
            fill_price = self.df.iloc[i + 1]["open"]
        else:
            fill_price = row["close"]
        fill_price -= cfg.slippage_points * cfg.tick_size

        atr_val = row["atr"]
        qty = cfg.default_qty

        self.position_size = -qty
        self.entry_price = fill_price
        self.locked_atr = atr_val
        self.locked_sl = fill_price + atr_val * cfg.sl_atr
        self.locked_tp = fill_price - atr_val * cfg.sl_atr * cfg.tp_ratio
        self.locked_tp1 = fill_price - atr_val * cfg.sl_atr * cfg.tp1_ratio
        self.exit_placed = False
        self.tp1_hit = False
        self.remaining_qty = qty
        self.trail_active = False
        self.trail_stop = 0.0

        self.last_signal = "Sell"
        self.last_signal_bar = bar_idx

        self._current_trade = Trade(
            entry_bar=bar_idx,
            entry_time=ts,
            entry_price=fill_price,
            direction="short",
            quantity=qty,
        )

    # ------------------------------------------------------------------
    # Exit logic
    # ------------------------------------------------------------------
    def _process_exits(self, i: int, row, in_session: bool):
        cfg = self.cfg
        h = row["high"]
        l = row["low"]
        o = row["open"]
        c = row["close"]

        # ── EOD forced close ──
        if not in_session:
            self._close_position(i, c, "EOD Force Close")
            return

        is_long = self.position_size > 0

        if cfg.exit_mode == "固定止盈 (SL+TP)":
            self._exit_fixed_tp(i, row, is_long)
        elif cfg.exit_mode == "移動止盈 (SL+Trailing)":
            self._exit_trailing(i, row, is_long)
        elif cfg.exit_mode == "分批出場 (1:1 + 2.5R)":
            self._exit_partial(i, row, is_long)

    def _exit_fixed_tp(self, i: int, row, is_long: bool):
        """Fixed SL + TP exit mode."""
        h, l = row["high"], row["low"]

        if is_long:
            # Check SL first (worst case)
            if l <= self.locked_sl:
                self._close_position(i, self.locked_sl, "SL Hit")
            elif h >= self.locked_tp:
                self._close_position(i, self.locked_tp, "TP Hit")
        else:
            if h >= self.locked_sl:
                self._close_position(i, self.locked_sl, "SL Hit")
            elif l <= self.locked_tp:
                self._close_position(i, self.locked_tp, "TP Hit")

    def _exit_trailing(self, i: int, row, is_long: bool):
        """SL + Trailing Stop exit mode."""
        cfg = self.cfg
        h, l = row["high"], row["low"]

        trail_activation = self.locked_atr * cfg.sl_atr * cfg.trail_mult
        trail_offset = self.locked_atr * cfg.trail_offset_mult

        if is_long:
            # Check SL
            if l <= self.locked_sl:
                self._close_position(i, self.locked_sl, "SL Hit")
                return

            # Check trailing activation
            profit_points = h - self.entry_price
            if profit_points >= trail_activation:
                new_trail = h - trail_offset
                if not self.trail_active or new_trail > self.trail_stop:
                    self.trail_stop = new_trail
                    self.trail_active = True

            # Check trailing stop hit
            if self.trail_active and l <= self.trail_stop:
                exit_price = max(self.trail_stop, self.locked_sl)
                self._close_position(i, exit_price, "Trail Stop")
        else:
            if h >= self.locked_sl:
                self._close_position(i, self.locked_sl, "SL Hit")
                return

            profit_points = self.entry_price - l
            if profit_points >= trail_activation:
                new_trail = l + trail_offset
                if not self.trail_active or new_trail < self.trail_stop:
                    self.trail_stop = new_trail
                    self.trail_active = True

            if self.trail_active and h >= self.trail_stop:
                exit_price = min(self.trail_stop, self.locked_sl)
                self._close_position(i, exit_price, "Trail Stop")

    def _exit_partial(self, i: int, row, is_long: bool):
        """分批出場: 50% at TP1 (1R), remainder at TP2 with BE stop."""
        cfg = self.cfg
        h, l = row["high"], row["low"]

        if is_long:
            # SL check
            if l <= self.locked_sl:
                self._close_position(i, self.locked_sl, "SL Hit")
                return

            # TP1 check (partial)
            if not self.tp1_hit and h >= self.locked_tp1:
                self.tp1_hit = True
                # Close half
                partial_qty = self.remaining_qty // 2
                if partial_qty > 0:
                    self._partial_close(i, self.locked_tp1, partial_qty, "TP1 Partial")
                    # Move SL to breakeven
                    self.locked_sl = self.entry_price

            # TP2 check (full)
            if h >= self.locked_tp:
                self._close_position(i, self.locked_tp, "TP2 Hit")
        else:
            if h >= self.locked_sl:
                self._close_position(i, self.locked_sl, "SL Hit")
                return

            if not self.tp1_hit and l <= self.locked_tp1:
                self.tp1_hit = True
                partial_qty = self.remaining_qty // 2
                if partial_qty > 0:
                    self._partial_close(i, self.locked_tp1, partial_qty, "TP1 Partial")
                    self.locked_sl = self.entry_price

            if l <= self.locked_tp:
                self._close_position(i, self.locked_tp, "TP2 Hit")

    # ------------------------------------------------------------------
    # Close helpers
    # ------------------------------------------------------------------
    def _close_position(self, i: int, exit_price: float, reason: str):
        """Fully close the current position."""
        cfg = self.cfg

        if self._current_trade is None:
            self.position_size = 0
            return

        qty = abs(self.position_size)
        direction = self._current_trade.direction

        # Apply slippage on exit
        if direction == "long":
            exit_price -= cfg.slippage_points * cfg.tick_size
            pnl = (exit_price - self.entry_price) * qty * cfg.point_value
        else:
            exit_price += cfg.slippage_points * cfg.tick_size
            pnl = (self.entry_price - exit_price) * qty * cfg.point_value

        commission = cfg.commission_per_contract * qty * 2  # round trip

        self._current_trade.exit_bar = i
        self._current_trade.exit_time = self.df.index[i]
        self._current_trade.exit_price = exit_price
        self._current_trade.exit_reason = reason
        self._current_trade.pnl = pnl
        self._current_trade.commission = commission
        self._current_trade.net_pnl = pnl - commission

        self.capital += self._current_trade.net_pnl
        self.trades.append(self._current_trade)

        # Reset state
        self.position_size = 0
        self.entry_price = 0.0
        self.exit_placed = False
        self.tp1_hit = False
        self.trail_active = False
        self.trail_stop = 0.0
        self._current_trade = None
        self.remaining_qty = 0

    def _partial_close(self, i: int, exit_price: float, qty: int, reason: str):
        """Partially close — record as separate trade, reduce remaining_qty."""
        cfg = self.cfg
        direction = self._current_trade.direction

        if direction == "long":
            adj_price = exit_price - cfg.slippage_points * cfg.tick_size
            pnl = (adj_price - self.entry_price) * qty * cfg.point_value
        else:
            adj_price = exit_price + cfg.slippage_points * cfg.tick_size
            pnl = (self.entry_price - adj_price) * qty * cfg.point_value

        commission = cfg.commission_per_contract * qty * 2

        partial_trade = Trade(
            entry_bar=self._current_trade.entry_bar,
            entry_time=self._current_trade.entry_time,
            entry_price=self.entry_price,
            direction=direction,
            quantity=qty,
            exit_bar=i,
            exit_time=self.df.index[i],
            exit_price=adj_price,
            exit_reason=reason,
            pnl=pnl,
            commission=commission,
            net_pnl=pnl - commission,
        )

        self.capital += partial_trade.net_pnl
        self.trades.append(partial_trade)

        self.remaining_qty -= qty
        if self.position_size > 0:
            self.position_size = self.remaining_qty
        else:
            self.position_size = -self.remaining_qty

        # Update current trade quantity
        self._current_trade.quantity = self.remaining_qty
