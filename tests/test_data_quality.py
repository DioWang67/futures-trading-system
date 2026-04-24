"""Tests for data quality checks."""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src.data.quality import check_data_quality, check_cache_freshness


@pytest.fixture
def good_df():
    """Well-formed 15min OHLCV data."""
    dates = pd.date_range("2024-01-02 09:00", periods=20, freq="15min")
    return pd.DataFrame({
        "datetime": dates,
        "open": range(100, 120),
        "high": range(101, 121),
        "low": range(99, 119),
        "close": range(100, 120),
        "volume": [1000] * 20,
    })


class TestDataQuality:
    def test_good_data_no_issues(self, good_df):
        issues = check_data_quality(good_df, freq="15min")
        assert len(issues) == 0

    def test_empty_df(self):
        issues = check_data_quality(pd.DataFrame())
        assert any("empty" in i.lower() for i in issues)

    def test_gap_detection(self):
        dates = pd.date_range("2024-01-02 09:00", periods=10, freq="15min")
        # Insert a 2-hour gap
        dates = dates.delete(range(3, 8))
        df = pd.DataFrame({
            "datetime": dates,
            "open": range(len(dates)),
            "high": [x + 1 for x in range(len(dates))],
            "low": [x - 1 for x in range(len(dates))],
            "close": range(len(dates)),
            "volume": [100] * len(dates),
        })
        issues = check_data_quality(df, freq="15min", max_gap_bars=3)
        assert any("gap" in i.lower() for i in issues)

    def test_weekend_gap_not_reported(self):
        dates = pd.to_datetime([
            "2024-01-05 13:45",  # Friday
            "2024-01-08 08:45",  # Monday
            "2024-01-08 09:00",
        ])
        df = pd.DataFrame({
            "datetime": dates,
            "open": [100, 101, 102],
            "high": [101, 102, 103],
            "low": [99, 100, 101],
            "close": [100, 101, 102],
            "volume": [100, 100, 100],
        })
        issues = check_data_quality(df, freq="15min", max_gap_bars=3)
        assert not any("gap detected" in i.lower() for i in issues)

    def test_zero_volume_warning(self):
        dates = pd.date_range("2024-01-02 09:00", periods=20, freq="15min")
        df = pd.DataFrame({
            "datetime": dates,
            "open": range(20),
            "high": range(1, 21),
            "low": range(20),
            "close": range(20),
            "volume": [0] * 20,  # all zero
        })
        issues = check_data_quality(df, freq="15min", max_zero_volume_pct=0.05)
        assert any("zero-volume" in i.lower() for i in issues)

    def test_ohlc_inconsistency(self):
        df = pd.DataFrame({
            "datetime": pd.date_range("2024-01-02 09:00", periods=3, freq="15min"),
            "open": [100, 200, 300],
            "high": [90, 210, 310],  # first bar: high < open
            "low": [95, 190, 290],
            "close": [98, 205, 305],
            "volume": [100, 100, 100],
        })
        issues = check_data_quality(df, freq="15min")
        assert any("ohlc" in i.lower() for i in issues)

    def test_duplicate_timestamps(self):
        dates = pd.to_datetime(["2024-01-02 09:00"] * 3)
        df = pd.DataFrame({
            "datetime": dates,
            "open": [100, 101, 102],
            "high": [105, 106, 107],
            "low": [95, 96, 97],
            "close": [103, 104, 105],
            "volume": [100, 100, 100],
        })
        issues = check_data_quality(df, freq="15min")
        assert any("duplicate" in i.lower() for i in issues)

    def test_nan_values(self, good_df):
        good_df.loc[5, "close"] = None
        issues = check_data_quality(good_df, freq="15min")
        assert any("nan" in i.lower() for i in issues)

    def test_price_spike(self):
        df = pd.DataFrame({
            "datetime": pd.date_range("2024-01-02", periods=5, freq="15min"),
            "open": [100, 100, 100, 200, 100],   # 100% spike
            "high": [105, 105, 105, 205, 105],
            "low": [95, 95, 95, 195, 95],
            "close": [100, 100, 100, 200, 100],
            "volume": [100] * 5,
        })
        issues = check_data_quality(df, freq="15min")
        assert any("spike" in i.lower() for i in issues)


class TestCacheFreshness:
    def test_fresh_cache(self, tmp_path):
        cache = tmp_path / "test.parquet"
        cache.touch()  # just created = fresh
        result = check_cache_freshness(cache, max_age_hours=1.0)
        assert result is None  # no issue

    def test_missing_cache(self, tmp_path):
        result = check_cache_freshness(tmp_path / "missing.parquet")
        assert result is not None
        assert "not found" in result.lower()

    def test_stale_cache(self, tmp_path):
        import os
        cache = tmp_path / "old.parquet"
        cache.touch()
        # Set mtime to 48 hours ago
        old_time = datetime.now().timestamp() - 48 * 3600
        os.utime(cache, (old_time, old_time))
        result = check_cache_freshness(cache, max_age_hours=24.0)
        assert result is not None
        assert "stale" in result.lower()
