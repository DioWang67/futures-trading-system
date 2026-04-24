"""
Backtrader 策略：SMC + PA 多時間框架交易系統

性能架構：
  1. 在 Backtrader 外先用 pandas/numpy 向量化計算所有 SMC/PA 訊號
  2. 把訊號 merge 進 DataFrame 作為額外欄位
  3. Backtrader 策略只做簡單的查表 + 倉位管理
"""

from __future__ import annotations

import backtrader as bt
import numpy as np
import pandas as pd
from typing import Optional

from src.strategy.smc import (
    SMCAnalyzer, Trend, StructureType, OrderBlock, FVG,
    detect_swing_points, detect_structure_breaks, detect_order_blocks, detect_fvg,
    update_ob_validity, check_fvg_filled,
)
from src.strategy.pa import PAAnalyzer, PASignal, detect_pin_bar, detect_engulfing


# ================================================================
# 預計算：在 Backtrader 之外完成所有 SMC/PA 分析
# ================================================================

def _compute_adx(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """計算 ADX, +DI, -DI（純 numpy，無外部依賴）。

    Returns (adx, plus_di, minus_di)，長度與輸入相同，前 period*2 根為 NaN。
    """
    n = len(highs)
    adx = np.full(n, np.nan)
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)

    if n < period * 2 + 1:
        return adx, plus_di, minus_di

    # True Range, +DM, -DM
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)

    for i in range(1, n):
        hl = highs[i] - lows[i]
        hpc = abs(highs[i] - closes[i - 1])
        lpc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, hpc, lpc)

        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        plus_dm[i] = up if up > down and up > 0 else 0
        minus_dm[i] = down if down > up and down > 0 else 0

    # Wilder smoothing（初始值用 SMA，後續用 EMA-like）
    atr = np.zeros(n)
    sm_plus = np.zeros(n)
    sm_minus = np.zeros(n)

    atr[period] = np.sum(tr[1:period + 1])
    sm_plus[period] = np.sum(plus_dm[1:period + 1])
    sm_minus[period] = np.sum(minus_dm[1:period + 1])

    for i in range(period + 1, n):
        atr[i] = atr[i - 1] - atr[i - 1] / period + tr[i]
        sm_plus[i] = sm_plus[i - 1] - sm_plus[i - 1] / period + plus_dm[i]
        sm_minus[i] = sm_minus[i - 1] - sm_minus[i - 1] / period + minus_dm[i]

    # DI
    for i in range(period, n):
        if atr[i] > 0:
            plus_di[i] = 100 * sm_plus[i] / atr[i]
            minus_di[i] = 100 * sm_minus[i] / atr[i]
        else:
            plus_di[i] = 0
            minus_di[i] = 0

    # DX → ADX
    dx = np.zeros(n)
    for i in range(period, n):
        denom = plus_di[i] + minus_di[i]
        dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / denom if denom > 0 else 0

    # ADX = Wilder smoothing of DX
    start = period * 2
    if start < n:
        adx[start] = np.mean(dx[period:start + 1])
        for i in range(start + 1, n):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return adx, plus_di, minus_di


def _compute_atr(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Compute Wilder ATR in points."""
    n = len(highs)
    atr = np.full(n, np.nan)
    if n <= period:
        return atr

    tr = np.zeros(n)
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hpc = abs(highs[i] - closes[i - 1])
        lpc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, hpc, lpc)

    atr[period] = np.mean(tr[1:period + 1])
    for i in range(period + 1, n):
        atr[i] = ((atr[i - 1] * (period - 1)) + tr[i]) / period

    return atr


def _is_entry_hour_allowed(hour: int, blocked_entry_hours: Optional[list[int]] = None) -> bool:
    """Return True when the entry hour is tradable."""
    if blocked_entry_hours is None:
        return True
    return hour not in set(blocked_entry_hours)


def precompute_signals(
    ltf_df: pd.DataFrame,
    htf_df: Optional[pd.DataFrame] = None,
    swing_lookback: int = 5,
    bos_min_move: float = 15.0,
    ob_max_age: int = 20,
    ob_body_ratio: float = 0.4,
    fvg_min_gap: float = 5.0,
    fvg_enabled: bool = True,
    pin_bar_ratio: float = 0.6,
    engulf_ratio: float = 1.0,
    sl_buffer: float = 2.0,
    rr_ratio: float = 1.5,
    use_structure_tp: bool = True,
    pa_confirm: bool = True,
    adx_period: int = 14,
    adx_threshold: float = 20.0,
    adx_filter_enabled: bool = True,
    atr_filter_enabled: bool = False,
    atr_period: int = 14,
    atr_min_points: float = 0.0,
    blocked_entry_hours: Optional[list[int]] = None,
) -> pd.DataFrame:
    """預計算所有訊號，回傳帶有 signal 欄位的 LTF DataFrame。

    新增欄位：
    - htf_trend: 1=bullish, -1=bearish, 0=neutral
    - signal: 1=買進, -1=賣出, 0=無
    - sl: 停損價
    - tp: 停利價
    """
    ltf = ltf_df.copy()
    opens = ltf["open"].values.astype(float)
    highs = ltf["high"].values.astype(float)
    lows = ltf["low"].values.astype(float)
    closes = ltf["close"].values.astype(float)
    n = len(ltf)
    if "datetime" in ltf.columns:
        ltf_dt_values = pd.to_datetime(ltf["datetime"]).to_numpy()
        ltf_hour_series = pd.to_datetime(ltf["datetime"])
    elif isinstance(ltf.index, pd.DatetimeIndex):
        ltf_dt_values = ltf.index.to_numpy()
        ltf_hour_series = pd.Series(ltf.index, index=ltf.index)
    else:
        ltf_dt_values = ltf.index.to_numpy()
        ltf_hour_series = None

    entry_hours: Optional[np.ndarray] = None
    if blocked_entry_hours is not None and ltf_hour_series is not None:
        entry_hours = ltf_hour_series.dt.hour.to_numpy(dtype=int)

    # ---- ADX 趨勢強度過濾 ----
    adx_arr, plus_di_arr, minus_di_arr = _compute_adx(highs, lows, closes, adx_period)
    atr_arr = _compute_atr(highs, lows, closes, atr_period)

    # ---- HTF 趨勢 ----
    htf_trend_arr = np.zeros(n, dtype=int)

    if htf_df is not None and len(htf_df) > swing_lookback * 2 + 1:
        h_highs = htf_df["high"].values.astype(float)
        h_lows = htf_df["low"].values.astype(float)
        h_closes = htf_df["close"].values.astype(float)
        h_dts = pd.to_datetime(htf_df["datetime"]).values if "datetime" in htf_df.columns else htf_df.index.values

        htf_swings = detect_swing_points(h_highs, h_lows, swing_lookback)
        htf_breaks = detect_structure_breaks(htf_swings, h_highs, h_lows, h_closes, bos_min_move)

        # 將 HTF 趨勢映射到 LTF 時間軸
        current_trend = 0
        htf_break_idx = 0

        for i in range(n):
            # 更新趨勢：找到 <= 當前 LTF 時間的最新 HTF break
            while htf_break_idx < len(htf_breaks):
                brk = htf_breaks[htf_break_idx]
                if brk.index < len(h_dts):
                    brk_time = h_dts[brk.index]
                    if brk_time <= ltf_dt_values[i]:
                        current_trend = 1 if brk.direction == Trend.BULLISH else -1
                        htf_break_idx += 1
                    else:
                        break
                else:
                    break
            htf_trend_arr[i] = current_trend
    else:
        # 用 LTF 自身大窗口模擬 HTF
        big_lb = swing_lookback * 4
        if n > big_lb * 2 + 1:
            swings = detect_swing_points(highs, lows, big_lb)
            breaks = detect_structure_breaks(swings, highs, lows, closes, bos_min_move)
            current_trend = 0
            brk_ptr = 0
            for i in range(n):
                while brk_ptr < len(breaks) and breaks[brk_ptr].index <= i:
                    current_trend = 1 if breaks[brk_ptr].direction == Trend.BULLISH else -1
                    brk_ptr += 1
                htf_trend_arr[i] = current_trend

    # ---- LTF Order Blocks ----
    ltf_swings = detect_swing_points(highs, lows, swing_lookback)
    ltf_breaks = detect_structure_breaks(ltf_swings, highs, lows, closes, bos_min_move)
    all_obs = detect_order_blocks(opens, highs, lows, closes, ltf_breaks, ob_body_ratio, ob_max_age)

    # ---- LTF FVG ----
    all_fvgs = detect_fvg(highs, lows, closes, opens, fvg_min_gap) if fvg_enabled else []

    # ---- PA 形態 ----
    pin_bars = detect_pin_bar(opens, highs, lows, closes, pin_bar_ratio)
    engulfings = detect_engulfing(opens, highs, lows, closes, engulf_ratio)

    bullish_pa_set = set()
    bearish_pa_set = set()
    for p in pin_bars:
        if p.signal == PASignal.BULLISH_PIN_BAR:
            bullish_pa_set.add(p.index)
        elif p.signal == PASignal.BEARISH_PIN_BAR:
            bearish_pa_set.add(p.index)
    for p in engulfings:
        if p.signal == PASignal.BULLISH_ENGULFING:
            bullish_pa_set.add(p.index)
        elif p.signal == PASignal.BEARISH_ENGULFING:
            bearish_pa_set.add(p.index)

    # ---- 預計算 OB 失效 bar（向量化） ----
    # 對每個 OB 找出它被穿過的第一根 bar index
    ob_invalid_at = {}
    for ob in all_obs:
        start = ob.start_index + 1
        end = min(ob.start_index + ob_max_age + 1, n)
        if start >= end:
            ob_invalid_at[id(ob)] = start
            continue
        if ob.direction == Trend.BULLISH:
            breach = np.where(closes[start:end] < ob.low)[0]
        else:
            breach = np.where(closes[start:end] > ob.high)[0]
        ob_invalid_at[id(ob)] = (start + breach[0]) if len(breach) > 0 else end

    # ---- 預計算 FVG 填充 bar ----
    fvg_filled_at = {}
    for fvg in all_fvgs:
        start = fvg.index + 1
        if start >= n:
            fvg_filled_at[id(fvg)] = n
            continue
        if fvg.direction == Trend.BULLISH:
            fill = np.where(lows[start:] <= fvg.bottom)[0]
        else:
            fill = np.where(highs[start:] >= fvg.top)[0]
        fvg_filled_at[id(fvg)] = (start + fill[0]) if len(fill) > 0 else n

    # ---- 產生交易訊號 ----
    signal_arr = np.zeros(n, dtype=int)
    sl_arr = np.full(n, np.nan)
    tp_arr = np.full(n, np.nan)

    # 追蹤持倉狀態（避免在已持倉時產生訊號）
    in_position = False

    for i in range(1, n):
        if in_position:
            # 檢查是否觸發停損停利（簡化：用上一個訊號的 sl/tp）
            prev_sig_idx = np.where(signal_arr[:i] != 0)[0]
            if len(prev_sig_idx) > 0:
                last_sig_i = prev_sig_idx[-1]
                last_sl = sl_arr[last_sig_i]
                last_tp = tp_arr[last_sig_i]
                last_dir = signal_arr[last_sig_i]
                if last_dir == 1:  # LONG
                    if closes[i] <= last_sl or closes[i] >= last_tp:
                        in_position = False
                elif last_dir == -1:  # SHORT
                    if closes[i] >= last_sl or closes[i] <= last_tp:
                        in_position = False
            continue

        if htf_trend_arr[i] == 0:
            continue

        # ADX 過濾：震盪市不進場
        if adx_filter_enabled and (np.isnan(adx_arr[i]) or adx_arr[i] < adx_threshold):
            continue
        if atr_filter_enabled and (np.isnan(atr_arr[i]) or atr_arr[i] < atr_min_points):
            continue
        if (
            blocked_entry_hours is not None
            and entry_hours is not None
            and not _is_entry_hour_allowed(int(entry_hours[i]), blocked_entry_hours)
        ):
            continue

        trend = Trend.BULLISH if htf_trend_arr[i] == 1 else Trend.BEARISH

        for ob in all_obs:
            if ob.start_index >= i:
                continue
            if i - ob.start_index > ob_max_age:
                continue
            if ob.direction != trend:
                continue

            # 快速失效檢查（O(1)）
            if i >= ob_invalid_at[id(ob)]:
                continue

            # 價格回測到 OB 區間
            if ob.direction == Trend.BULLISH:
                if not (lows[i] <= ob.top and closes[i] >= ob.bottom):
                    continue
            else:
                if not (highs[i] >= ob.bottom and closes[i] <= ob.top):
                    continue

            # FVG 過濾（O(F) 但只查未填充的）
            if fvg_enabled:
                has_fvg = any(
                    fvg.direction == ob.direction
                    and fvg.index < i
                    and fvg_filled_at[id(fvg)] > i
                    for fvg in all_fvgs
                )
                if not has_fvg:
                    continue

            # PA 確認
            if pa_confirm:
                if ob.direction == Trend.BULLISH and i not in bullish_pa_set:
                    continue
                if ob.direction == Trend.BEARISH and i not in bearish_pa_set:
                    continue

            # ---- 產生訊號 ----
            if ob.direction == Trend.BULLISH:
                sl = ob.bottom - sl_buffer
                risk = closes[i] - sl
                if risk <= 0:
                    continue
                tp = closes[i] + risk * rr_ratio
                if use_structure_tp:
                    htf_levels = _get_structure_levels_at(ltf_breaks, i)
                    higher = [lv for lv in htf_levels if lv > closes[i]]
                    if higher:
                        tp = max(tp, min(higher))  # 取較遠者，保護 R:R
                signal_arr[i] = 1
            else:
                sl = ob.top + sl_buffer
                risk = sl - closes[i]
                if risk <= 0:
                    continue
                tp = closes[i] - risk * rr_ratio
                if use_structure_tp:
                    htf_levels = _get_structure_levels_at(ltf_breaks, i)
                    lower = [lv for lv in htf_levels if lv < closes[i]]
                    if lower:
                        tp = min(tp, max(lower))  # 取較遠者，保護 R:R
                signal_arr[i] = -1

            sl_arr[i] = sl
            tp_arr[i] = tp
            in_position = True
            break  # 一次一筆

    ltf["htf_trend"] = htf_trend_arr
    ltf["signal"] = signal_arr
    ltf["sl"] = sl_arr
    ltf["tp"] = tp_arr

    return ltf


def _get_structure_levels_at(breaks, current_idx):
    """取得到 current_idx 為止的所有結構突破價位。"""
    return [b.level for b in breaks if b.index <= current_idx]


# ================================================================
# Backtrader Data Feed（含預計算訊號欄位）
# ================================================================

class SMCPAData(bt.feeds.PandasData):
    """帶預計算訊號的 data feed。"""
    lines = ("htf_trend", "signal", "sl", "tp",)
    params = (
        ("datetime", None),
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "volume"),
        ("openinterest", -1),
        ("htf_trend", "htf_trend"),
        ("signal", "signal"),
        ("sl", "sl"),
        ("tp", "tp"),
    )


class SMCPAStrategy(bt.Strategy):
    """SMC + PA 策略（輕量版）。

    所有 SMC/PA 分析已在外部預計算完成。
    策略只負責：讀取訊號 → 下單 → 停損停利。
    """

    params = dict(
        # metadata-only：實際策略邏輯由 precompute_signals() 先行計算，
        # Backtrader next() 不會直接讀取這些參數。
        swing_lookback=5,
        bos_min_move=15.0,
        ob_max_age=20,
        ob_body_ratio=0.4,
        fvg_min_gap=5.0,
        fvg_enabled=True,
        pin_bar_ratio=0.6,
        engulf_ratio=1.0,
        rr_ratio=1.5,
        use_structure_tp=True,
        sl_buffer=2.0,
        pa_confirm=True,
        atr_filter_enabled=False,
        atr_period=14,
        atr_min_points=0.0,
    )

    def __init__(self):
        self.trade_log: list[dict] = []
        self.entry_price: Optional[float] = None
        self.sl_price: Optional[float] = None
        self.tp_price: Optional[float] = None
        self.entry_bar: int = 0

    def next(self):
        data = self.datas[0]
        current_close = data.close[0]

        # ---- 倉位管理 ----
        if self.position:
            if self.position.size > 0:
                if current_close <= self.sl_price:
                    self.close()
                    self._log_trade("LONG", "SL", current_close)
                elif current_close >= self.tp_price:
                    self.close()
                    self._log_trade("LONG", "TP", current_close)
            elif self.position.size < 0:
                if current_close >= self.sl_price:
                    self.close()
                    self._log_trade("SHORT", "SL", current_close)
                elif current_close <= self.tp_price:
                    self.close()
                    self._log_trade("SHORT", "TP", current_close)
            return

        # ---- 讀取預計算訊號 ----
        sig = data.signal[0]
        if sig == 0:
            return

        sl = data.sl[0]
        tp = data.tp[0]
        if np.isnan(sl) or np.isnan(tp):
            return

        if sig == 1:
            self.buy()
            self.entry_price = current_close
            self.sl_price = sl
            self.tp_price = tp
            self.entry_bar = len(data) - 1
        elif sig == -1:
            self.sell()
            self.entry_price = current_close
            self.sl_price = sl
            self.tp_price = tp
            self.entry_bar = len(data) - 1

    def _log_trade(self, direction: str, exit_reason: str, exit_price: float):
        if self.entry_price is None:
            return
        pnl = (exit_price - self.entry_price) if direction == "LONG" else (self.entry_price - exit_price)
        risk = abs(self.entry_price - self.sl_price) if self.sl_price else 1
        rr = pnl / risk if risk > 0 else 0

        self.trade_log.append({
            "direction": direction,
            "entry_price": self.entry_price,
            "exit_price": exit_price,
            "sl": self.sl_price,
            "tp": self.tp_price,
            "pnl": pnl,
            "rr": rr,
            "exit_reason": exit_reason,
            "entry_bar": self.entry_bar,
            "exit_bar": len(self.datas[0]) - 1,
        })

    def notify_trade(self, trade):
        pass

    def stop(self):
        pass
