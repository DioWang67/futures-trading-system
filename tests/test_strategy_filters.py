import pandas as pd

from src.data.fetcher import generate_sample_data
from src.strategy.bt_strategy import _compute_atr, _is_entry_hour_allowed, precompute_signals


def test_compute_atr_returns_expected_shape():
    highs = pd.Series([10, 12, 13, 14, 15, 16], dtype=float).to_numpy()
    lows = pd.Series([9, 10, 11, 12, 13, 14], dtype=float).to_numpy()
    closes = pd.Series([9.5, 11.0, 12.5, 13.0, 14.0, 15.0], dtype=float).to_numpy()

    atr = _compute_atr(highs, lows, closes, period=3)

    assert len(atr) == len(highs)
    assert pd.isna(atr[0])
    assert not pd.isna(atr[-1])


def test_entry_hour_allowed_respects_blocklist():
    assert _is_entry_hour_allowed(10) is True
    assert _is_entry_hour_allowed(10, [9, 11]) is True
    assert _is_entry_hour_allowed(10, [10, 11]) is False


def test_precompute_signals_atr_filter_can_block_all_entries():
    ltf = generate_sample_data(n_bars=400, freq="15T", seed=42)
    htf = generate_sample_data(n_bars=100, freq="60T", seed=42)

    result = precompute_signals(
        ltf,
        htf,
        atr_filter_enabled=True,
        atr_period=14,
        atr_min_points=1_000_000.0,
    )

    assert int((result["signal"] != 0).sum()) == 0


def test_precompute_signals_blocked_hours_can_block_all_entries():
    ltf = generate_sample_data(n_bars=400, freq="15T", seed=42)
    htf = generate_sample_data(n_bars=100, freq="60T", seed=42)

    result = precompute_signals(
        ltf,
        htf,
        blocked_entry_hours=list(range(24)),
    )

    assert int((result["signal"] != 0).sum()) == 0


def test_precompute_signals_non_datetime_index_does_not_crash():
    ltf = generate_sample_data(n_bars=120, freq="15T", seed=7).drop(columns=["datetime"])
    htf = generate_sample_data(n_bars=30, freq="60T", seed=7)

    result = precompute_signals(
        ltf,
        htf,
        blocked_entry_hours=list(range(24)),
    )

    assert "signal" in result.columns
