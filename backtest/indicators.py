"""
Technical indicator calculations — replicates Pine Script built-ins.
All functions operate on pandas DataFrames/Series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================================
# Basic indicators
# ============================================================================

def atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Average True Range — matches ta.atr()."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    # Pine uses RMA (Wilder's smoothing) for ATR
    return _rma(tr, period)


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average — matches ta.ema()."""
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average — matches ta.sma()."""
    return series.rolling(period).mean()


def rsi(series: pd.Series, period: int) -> pd.Series:
    """RSI — matches ta.rsi() using RMA smoothing."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = _rma(gain, period)
    avg_loss = _rma(loss, period)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothed moving average (RMA) — used by Pine internally."""
    alpha = 1.0 / period
    return series.ewm(alpha=alpha, adjust=False).mean()


# ============================================================================
# VWAP (daily reset)
# ============================================================================

def vwap_daily(df: pd.DataFrame) -> pd.Series:
    """VWAP with daily reset — matches ta.vwap(hlc3).

    Requires df to have a DatetimeIndex (timezone-aware preferred).
    """
    hlc3 = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].replace(0, np.nan)

    # Detect day boundaries
    dates = df.index.date
    day_change = pd.Series(dates, index=df.index) != pd.Series(dates, index=df.index).shift(1)

    cum_vol = vol.copy()
    cum_pv = (hlc3 * vol).copy()

    # Group by trading day and cumsum
    day_groups = day_change.cumsum()
    cum_vol = vol.groupby(day_groups).cumsum()
    cum_pv = (hlc3 * vol).groupby(day_groups).cumsum()

    result = cum_pv / cum_vol
    return result.ffill()


# ============================================================================
# Pivot points
# ============================================================================

def pivot_high(high: pd.Series, left: int, right: int) -> pd.Series:
    """Detect pivot highs — matches ta.pivothigh(high, left, right).

    Returns the pivot value at the pivot bar (shifted back by `right` bars),
    NaN elsewhere. The pivot is confirmed `right` bars after the actual peak.
    """
    result = pd.Series(np.nan, index=high.index)
    vals = high.values
    n = len(vals)
    for i in range(left, n - right):
        val = vals[i]
        is_pivot = True
        for j in range(i - left, i):
            if vals[j] > val:
                is_pivot = False
                break
        if is_pivot:
            for j in range(i + 1, i + right + 1):
                if vals[j] > val:
                    is_pivot = False
                    break
        if is_pivot:
            # Pine reports pivot at bar_index = i, but it's only known at i + right
            result.iloc[i + right] = val
    return result


def pivot_low(low: pd.Series, left: int, right: int) -> pd.Series:
    """Detect pivot lows — matches ta.pivotlow(low, left, right)."""
    result = pd.Series(np.nan, index=low.index)
    vals = low.values
    n = len(vals)
    for i in range(left, n - right):
        val = vals[i]
        is_pivot = True
        for j in range(i - left, i):
            if vals[j] < val:
                is_pivot = False
                break
        if is_pivot:
            for j in range(i + 1, i + right + 1):
                if vals[j] < val:
                    is_pivot = False
                    break
        if is_pivot:
            result.iloc[i + right] = val
    return result


# ============================================================================
# Multi-timeframe resampling
# ============================================================================

def resample_ohlcv(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Resample base-timeframe OHLCV to a higher timeframe.

    Args:
        df: DataFrame with OHLCV columns and DatetimeIndex.
        tf: Target timeframe string, e.g. "5min", "15min", "30min", "1h", "4h".

    Returns:
        Resampled OHLCV DataFrame aligned back to the original index
        (forward-filled, no lookahead).
    """
    resampled = df.resample(tf, label="right", closed="right").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    return resampled


def htf_indicators(df: pd.DataFrame, tf: str, ema_period: int = 20) -> pd.DataFrame:
    """Compute EMA and VWAP on a higher timeframe then align back to base bars.

    Returns a DataFrame with columns: htf_ema, htf_vwap, aligned to df.index
    with no lookahead (shift forward by 1 HTF bar).
    """
    resampled = resample_ohlcv(df, tf)

    htf_ema = ema(resampled["close"], ema_period)
    htf_vwap = _htf_vwap(resampled)

    # Align to base timeframe: reindex + ffill (no lookahead)
    # shift(1) on resampled ensures we only see completed HTF bars
    htf_ema_shifted = htf_ema.shift(1)
    htf_vwap_shifted = htf_vwap.shift(1)

    # pandas 2.1 deprecated the ``method=`` kwarg on reindex; do the fill
    # explicitly so we keep working on 2.2+.
    result = pd.DataFrame(index=df.index)
    result["htf_ema"] = htf_ema_shifted.reindex(df.index).ffill()
    result["htf_vwap"] = htf_vwap_shifted.reindex(df.index).ffill()
    return result


def _htf_vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP for resampled HTF data (daily reset)."""
    hlc3 = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].replace(0, np.nan)

    dates = df.index.date
    day_change = pd.Series(dates, index=df.index) != pd.Series(dates, index=df.index).shift(1)
    day_groups = day_change.cumsum()

    cum_vol = vol.groupby(day_groups).cumsum()
    cum_pv = (hlc3 * vol).groupby(day_groups).cumsum()
    result = cum_pv / cum_vol
    return result.ffill()


# ============================================================================
# Compute all indicators at once
# ============================================================================

TF_MAP = {
    "5M": "5min",
    "15M": "15min",
    "30M": "30min",
    "1H": "1h",
    "4H": "4h",
}


def compute_all(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Compute every indicator needed by the strategy.

    Args:
        df: Base-timeframe OHLCV DataFrame (DatetimeIndex, tz-aware).
        cfg: StrategyConfig instance.

    Returns:
        df with all indicator columns appended.
    """
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

    # ── ATR ──
    df["atr"] = atr(df, cfg.atr_period_sl)

    # ── Short EMA (9) for slope ──
    df["ema9"] = ema(c, 9)

    # ── 5M EMA & VWAP (for pullback POI) ──
    htf5 = htf_indicators(df, "5min", 20)
    df["ema5M"] = htf5["htf_ema"]
    df["vwap5M"] = htf5["htf_vwap"]

    # ── Multi-timeframe trend indicators ──
    for tf_label in ["15M", "30M", "1H", "4H"]:
        tf_str = TF_MAP[tf_label]
        htf_df = htf_indicators(df, tf_str, 20)
        df[f"ema_{tf_label}"] = htf_df["htf_ema"]
        df[f"vwap_{tf_label}"] = htf_df["htf_vwap"]

    # ── Trend scoring per HTF ──
    for tf_label in ["15M", "30M", "1H", "4H"]:
        ema_col = f"ema_{tf_label}"
        vwap_col = f"vwap_{tf_label}"
        bull = (c > df[ema_col]).astype(int) + (c > df[vwap_col]).astype(int)
        bear = (c < df[ema_col]).astype(int) + (c < df[vwap_col]).astype(int)
        df[f"trend_{tf_label}"] = np.where(bull >= 1, 1, np.where(bear >= 1, -1, 0))

    # ── Volume ──
    df["vol_avg"] = sma(v, cfg.volume_long_period)
    df["vol_ok"] = (v > df["vol_avg"] * cfg.vol_multiplier) | \
                   (v.shift(1) > df["vol_avg"].shift(1) * cfg.vol_multiplier)

    # ── Range position ──
    df["recent_hi"] = h.rolling(cfg.range_period).max()
    df["recent_lo"] = l.rolling(cfg.range_period).min()
    r_size = df["recent_hi"] - df["recent_lo"]
    df["range_ratio"] = np.where(r_size > 0, (c - df["recent_lo"]) / r_size, 0.5)

    # ── EMA slope ──
    df["ema_slope_up"] = df["ema9"] >= df["ema9"].shift(1)
    df["ema_slope_down"] = df["ema9"] <= df["ema9"].shift(1)

    # ── Demand / Supply zones ──
    df["in_demand_zone"] = (df["range_ratio"] <= cfg.range_threshold) & \
                           (~df["ema_slope_down"] | (c > df["ema9"]))
    df["in_supply_zone"] = (df["range_ratio"] >= (1.0 - cfg.range_threshold)) & \
                           (~df["ema_slope_up"] | (c < df["ema9"]))

    # ── Swing hi/lo for structure ──
    df["swing_hi"] = h.rolling(cfg.struct_lookback).max()
    df["swing_lo"] = l.rolling(cfg.struct_lookback).min()

    # ── Near support / resistance / trend line ──
    atr_col = df["atr"]
    df["near_support"] = (c - df["swing_lo"]) <= atr_col * cfg.near_structure_atr
    df["near_resistance"] = (df["swing_hi"] - c) <= atr_col * cfg.near_structure_atr

    htf_ema_col = f"ema_{cfg.higher_tf}"
    htf_vwap_col = f"vwap_{cfg.higher_tf}"
    df["near_trend_line"] = ((c - df[htf_ema_col]).abs() <= atr_col * cfg.near_ema_atr) | \
                            ((c - df[htf_vwap_col]).abs() <= atr_col * cfg.near_ema_atr)

    # ── Location OK ──
    df["location_ok_long"] = df["near_support"] | df["near_trend_line"] | df["in_demand_zone"]
    df["location_ok_short"] = df["near_resistance"] | df["near_trend_line"] | df["in_supply_zone"]

    # ── Pivot points (for sweep) ──
    plen = cfg.pivot_length
    df["pivot_high"] = pivot_high(h, plen, plen)
    df["pivot_low"] = pivot_low(l, plen, plen)

    # Forward-fill last known pivots
    df["last_high"] = df["pivot_high"].ffill()
    df["last_low"] = df["pivot_low"].ffill()

    # ── Structure pivots (for HH/HL/LH/LL) ──
    slen = cfg.struct_swing_len
    df["struct_ph"] = pivot_high(h, slen, slen)
    df["struct_pl"] = pivot_low(l, slen, slen)

    # ── Candlestick components ──
    df["body"] = (c - o).abs()
    df["lower_wick"] = pd.concat([c, o], axis=1).min(axis=1) - l
    df["upper_wick"] = h - pd.concat([c, o], axis=1).max(axis=1)
    df["total_range"] = h - l
    df["mid_bar"] = (h + l) / 2.0
    df["prev_body"] = df["body"].shift(1)

    # ── Volatility checks ──
    df["avg_range_30"] = sma(df["total_range"], 30)
    df["recent_avg_range_5"] = sma(df["total_range"], 5)
    df["is_candle_size_ok"] = df["total_range"] >= atr_col * cfg.min_candle_atr
    df["is_env_volatile"] = df["recent_avg_range_5"] >= atr_col * cfg.env_atr_ratio
    df["is_valid_volatility"] = (df["total_range"] >= df["avg_range_30"] * 0.8) & \
                                 df["is_candle_size_ok"] & df["is_env_volatile"]

    # ── Pin bar patterns ──
    df["is_bullish_pin"] = (
        (df["lower_wick"] > df["body"] * cfg.pin_wick_ratio) &
        (df["lower_wick"] > df["upper_wick"]) &
        (df["body"] < df["total_range"] * 0.45) &
        (c > df["mid_bar"]) &
        (c >= o) &
        df["is_valid_volatility"]
    )
    df["is_bearish_pin"] = (
        (df["upper_wick"] > df["body"] * cfg.pin_wick_ratio) &
        (df["upper_wick"] > df["lower_wick"]) &
        (df["body"] < df["total_range"] * 0.45) &
        (c < df["mid_bar"]) &
        (c <= o) &
        df["is_valid_volatility"]
    )

    # ── Engulfing patterns ──
    prev_c = c.shift(1)
    prev_o = o.shift(1)
    if cfg.use_engulfing:
        df["is_bullish_engulf"] = (
            (c > o) &
            (prev_c < prev_o) &
            (c > prev_o) &
            (o < prev_c) &
            (df["body"] > df["prev_body"] * cfg.engulf_body_ratio) &
            (df["body"] >= atr_col * cfg.min_body_atr) &
            df["is_valid_volatility"]
        )
        df["is_bearish_engulf"] = (
            (c < o) &
            (prev_c > prev_o) &
            (c < prev_o) &
            (o > prev_c) &
            (df["body"] > df["prev_body"] * cfg.engulf_body_ratio) &
            (df["body"] >= atr_col * cfg.min_body_atr) &
            df["is_valid_volatility"]
        )
    else:
        df["is_bullish_engulf"] = False
        df["is_bearish_engulf"] = False

    df["bullish_pattern"] = df["is_bullish_pin"] | df["is_bullish_engulf"]
    df["bearish_pattern"] = df["is_bearish_pin"] | df["is_bearish_engulf"]

    # ── Sweep logic ──
    df["sweep_low"] = (l < df["last_low"]) & (c > df["last_low"]) & (c > df["mid_bar"]) & (c >= o)
    df["sweep_high"] = (h > df["last_high"]) & (c < df["last_high"]) & (c < df["mid_bar"]) & (c <= o)

    # ── Pullback POI ──
    df["pullback_poi_long"] = (l <= df["ema5M"]) | (l <= df["vwap5M"])
    df["pullback_poi_short"] = (h >= df["ema5M"]) | (h >= df["vwap5M"])

    # ── Momentum block ──
    prev_candle_body = df["prev_body"]
    df["prev_is_big_bear"] = (c.shift(1) < o.shift(1)) & (prev_candle_body >= atr_col * cfg.momentum_block_atr)
    df["prev_is_big_bull"] = (c.shift(1) > o.shift(1)) & (prev_candle_body >= atr_col * cfg.momentum_block_atr)
    df["no_adverse_momentum_long"] = ~df["prev_is_big_bear"]
    df["no_adverse_momentum_short"] = ~df["prev_is_big_bull"]

    # ── Trigger conditions ──
    df["trigger_long"] = (
        df["bullish_pattern"] &
        (df["sweep_low"] | df["pullback_poi_long"]) &
        ~df["in_supply_zone"] &
        df["location_ok_long"] &
        df["no_adverse_momentum_long"]
    )
    df["trigger_short"] = (
        df["bearish_pattern"] &
        (df["sweep_high"] | df["pullback_poi_short"]) &
        ~df["in_demand_zone"] &
        df["location_ok_short"] &
        df["no_adverse_momentum_short"]
    )

    # ── HTF trend filters ──
    htf_trend = df[f"trend_{cfg.higher_tf}"]
    df["bullish_trend_ok"] = ~pd.Series(cfg.use_trend_filter, index=df.index) | (htf_trend >= 1)
    df["bearish_trend_ok"] = ~pd.Series(cfg.use_trend_filter, index=df.index) | (htf_trend <= -1)

    # Override: simpler boolean
    if cfg.use_trend_filter:
        df["bullish_trend_ok"] = htf_trend >= 1
        df["bearish_trend_ok"] = htf_trend <= -1
    else:
        df["bullish_trend_ok"] = True
        df["bearish_trend_ok"] = True

    # ── Dual HTF ──
    if cfg.use_dual_htf:
        df["dual_bull_ok"] = (df["trend_1H"] >= 1) & (df["trend_4H"] >= 1)
        df["dual_bear_ok"] = (df["trend_1H"] <= -1) & (df["trend_4H"] <= -1)
    else:
        df["dual_bull_ok"] = True
        df["dual_bear_ok"] = True

    # ── RSI (disabled by default) ──
    if cfg.use_rsi_filter:
        df["rsi"] = rsi(c, cfg.rsi_period)
        df["rsi_long_ok"] = df["rsi"] <= cfg.rsi_long_max
        df["rsi_short_ok"] = df["rsi"] >= cfg.rsi_short_min
    else:
        df["rsi_long_ok"] = True
        df["rsi_short_ok"] = True

    # ── SL precheck ──
    sl_distance = atr_col * cfg.sl_atr
    if cfg.max_sl_points > 0:
        df["sl_precheck_ok"] = sl_distance <= cfg.max_sl_points
    else:
        df["sl_precheck_ok"] = True

    return df
