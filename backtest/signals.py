"""
Signal scoring and entry condition evaluation.
Mirrors Pine Script sections 8 (scoring) and 10 (confirmation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_scores(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Compute long/short scores and raw signal candidates.

    This is the vectorized pre-computation step. The final entry decisions
    (which depend on position state, last_signal, etc.) are handled bar-by-bar
    in the engine.
    """

    # ── Long score ──
    score_long = pd.Series(0, index=df.index, dtype=int)
    score_long += np.where(df["in_demand_zone"], 2, 0)
    score_long += np.where(df["near_support"], 1, 0)
    score_long += np.where(df["near_trend_line"], 1, 0)
    score_long += np.where(df["bullish_trend_ok"], 2, 0)
    score_long += np.where(df["dual_bull_ok"], 1, 0)
    score_long += np.where(df["vol_ok"], 1, 0)
    score_long += np.where(df["rsi_long_ok"], 1, 0)
    # structure_ok_long: since use_structure_filter=False, always +1
    if not cfg.use_structure_filter:
        score_long += 1
    df["score_long"] = score_long

    # ── Short score ──
    score_short = pd.Series(0, index=df.index, dtype=int)
    score_short += np.where(df["in_supply_zone"], 2, 0)
    score_short += np.where(df["near_resistance"], 1, 0)
    score_short += np.where(df["near_trend_line"], 1, 0)
    score_short += np.where(df["bearish_trend_ok"], 2, 0)
    score_short += np.where(df["dual_bear_ok"], 1, 0)
    score_short += np.where(df["vol_ok"], 1, 0)
    score_short += np.where(df["rsi_short_ok"], 1, 0)
    if not cfg.use_structure_filter:
        score_short += 1
    df["score_short"] = score_short

    # ── Smart signals (trigger + score >= threshold) ──
    df["smart_buy"] = df["trigger_long"] & (df["score_long"] >= cfg.score_threshold)
    df["smart_sell"] = df["trigger_short"] & (df["score_short"] >= cfg.score_threshold)

    return df


def is_in_session(ts: pd.Timestamp, start: str, end: str) -> bool:
    """Check if a timestamp (must be in America/New_York) is within the trading session."""
    from datetime import time as dtime

    sh, sm = map(int, start.split(":"))
    eh, em = map(int, end.split(":"))
    t = ts.time()
    return dtime(sh, sm) <= t <= dtime(eh, em)
