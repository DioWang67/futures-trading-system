"""資料擷取模組單元測試。"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.data.fetcher import (
    generate_sample_data,
    get_cache_path,
    load_config,
)


class TestGenerateSampleData:
    def test_returns_dataframe(self):
        df = generate_sample_data(n_bars=100)
        assert isinstance(df, pd.DataFrame)

    def test_correct_columns(self):
        df = generate_sample_data(n_bars=100)
        expected = {"datetime", "open", "high", "low", "close", "volume"}
        assert set(df.columns) == expected

    def test_correct_length(self):
        df = generate_sample_data(n_bars=200)
        assert len(df) == 200

    def test_high_gte_open_close(self):
        df = generate_sample_data(n_bars=500, seed=99)
        assert (df["high"] >= df["open"]).all()
        assert (df["high"] >= df["close"]).all()

    def test_low_lte_open_close(self):
        df = generate_sample_data(n_bars=500, seed=99)
        assert (df["low"] <= df["open"]).all()
        assert (df["low"] <= df["close"]).all()

    def test_volume_positive(self):
        df = generate_sample_data(n_bars=100)
        assert (df["volume"] > 0).all()

    def test_deterministic_with_seed(self):
        df1 = generate_sample_data(n_bars=50, seed=42)
        df2 = generate_sample_data(n_bars=50, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds(self):
        df1 = generate_sample_data(n_bars=50, seed=1)
        df2 = generate_sample_data(n_bars=50, seed=2)
        assert not df1["close"].equals(df2["close"])

    def test_freq_15t(self):
        df = generate_sample_data(n_bars=100, freq="15T")
        assert len(df) == 100

    def test_freq_60t(self):
        df = generate_sample_data(n_bars=100, freq="60T")
        assert len(df) == 100


class TestCachePath:
    def test_returns_path(self):
        path = get_cache_path("MXF", "60min")
        assert isinstance(path, Path)
        assert "MXF_60min" in str(path)
        assert str(path).endswith(".parquet")


class TestLoadConfig:
    def test_load_default_config(self):
        config = load_config()
        assert "shioaji" in config
        assert "strategy" in config
        assert "backtest" in config

    def test_config_has_required_keys(self):
        config = load_config()
        assert "api_key" in config["shioaji"]
        assert "commission" in config["backtest"]
        assert "swing_lookback" in config["strategy"]
