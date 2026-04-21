"""
Data quality checks — gap detection, staleness, anomalies.

Usage:
    issues = check_data_quality(df, freq="15min")
    if issues:
        for issue in issues:
            logger.warning(f"Data quality: {issue}")
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def check_data_quality(
    df: pd.DataFrame,
    freq: str = "15min",
    max_gap_bars: int = 5,
    max_zero_volume_pct: float = 0.05,
) -> list[str]:
    """Run data quality checks and return a list of issues found.

    Parameters
    ----------
    df : DataFrame with datetime, open, high, low, close, volume columns
    freq : expected bar frequency (e.g., "15min", "60min")
    max_gap_bars : flag gaps larger than this many bars
    max_zero_volume_pct : flag if zero-volume bars exceed this percentage
    """
    issues: list[str] = []

    if df.empty:
        issues.append("DataFrame is empty")
        return issues

    # Ensure datetime column
    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"])
    elif isinstance(df.index, pd.DatetimeIndex):
        dt = df.index
    else:
        issues.append("No datetime column or DatetimeIndex found")
        return issues

    n_rows = len(df)

    # 1. Gap detection
    if n_rows > 1:
        dt_series = dt if isinstance(dt, pd.Series) else dt.to_series()
        diffs = dt_series.diff().dropna()
        expected_delta = pd.Timedelta(freq)
        large_gaps = diffs[diffs > expected_delta * max_gap_bars]
        if len(large_gaps) > 0:
            for idx, gap in large_gaps.items():
                gap_bars = gap / expected_delta
                issues.append(
                    f"Gap detected: {gap_bars:.0f} bars at {idx} "
                    f"(gap={gap})"
                )

    # 2. Zero-volume bars
    if "volume" in df.columns:
        zero_vol = (df["volume"] == 0).sum()
        zero_pct = zero_vol / n_rows
        if zero_pct > max_zero_volume_pct:
            issues.append(
                f"High zero-volume bar rate: {zero_vol}/{n_rows} "
                f"({zero_pct:.1%} > {max_zero_volume_pct:.1%})"
            )

    # 3. OHLC consistency
    if all(c in df.columns for c in ["open", "high", "low", "close"]):
        bad_high = (df["high"] < df["open"]).sum() + (df["high"] < df["close"]).sum()
        bad_low = (df["low"] > df["open"]).sum() + (df["low"] > df["close"]).sum()
        if bad_high > 0:
            issues.append(f"OHLC inconsistency: {bad_high} bars where high < open or close")
        if bad_low > 0:
            issues.append(f"OHLC inconsistency: {bad_low} bars where low > open or close")

    # 4. Duplicate timestamps
    if "datetime" in df.columns:
        dupes = df["datetime"].duplicated().sum()
    else:
        dupes = dt.duplicated().sum()
    if dupes > 0:
        issues.append(f"Duplicate timestamps: {dupes}")

    # 5. NaN check
    ohlcv = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    nan_count = df[ohlcv].isna().sum().sum()
    if nan_count > 0:
        issues.append(f"NaN values found: {nan_count} across OHLCV columns")

    # 6. Price anomalies (>10% jump between consecutive bars)
    if "close" in df.columns and n_rows > 1:
        pct_change = df["close"].pct_change().abs()
        spikes = pct_change[pct_change > 0.10].dropna()
        if len(spikes) > 0:
            issues.append(
                f"Price spikes >10%: {len(spikes)} bars "
                f"(max={spikes.max():.1%})"
            )

    return issues


def check_cache_freshness(
    cache_path: Path,
    max_age_hours: float = 24.0,
) -> Optional[str]:
    """Check if a cache file is stale.

    Returns an issue string if stale, None if fresh.
    """
    if not cache_path.exists():
        return f"Cache file not found: {cache_path}"

    mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
    age = datetime.now() - mtime
    max_age = timedelta(hours=max_age_hours)

    if age > max_age:
        return (
            f"Cache stale: {cache_path.name} is {age.total_seconds()/3600:.1f}h old "
            f"(max {max_age_hours}h)"
        )
    return None
