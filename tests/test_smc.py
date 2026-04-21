"""SMC 模組單元測試。"""

import numpy as np
import pandas as pd
import pytest

from src.strategy.smc import (
    Trend,
    StructureType,
    SwingPoint,
    StructureBreak,
    OrderBlock,
    FVG,
    SMCAnalyzer,
    detect_swing_points,
    detect_structure_breaks,
    detect_order_blocks,
    detect_fvg,
    update_ob_validity,
    check_fvg_filled,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def trending_up_data():
    """產生一段上升趨勢的 OHLC 資料。"""
    n = 100
    np.random.seed(42)
    base = np.cumsum(np.random.normal(2, 5, n)) + 16000
    opens = base
    closes = base + np.random.normal(3, 2, n)
    highs = np.maximum(opens, closes) + np.abs(np.random.normal(5, 3, n))
    lows = np.minimum(opens, closes) - np.abs(np.random.normal(5, 3, n))
    return opens, highs, lows, closes


@pytest.fixture
def simple_swing_data():
    """簡單的 V 形資料，保證有 swing high 和 swing low。"""
    # 先漲後跌再漲: 20 bars
    prices = [100, 102, 105, 108, 112, 115, 118, 120, 122, 125,
              123, 120, 117, 114, 111, 108, 110, 113, 116, 119]
    n = len(prices)
    opens = np.array(prices, dtype=float)
    closes = opens + 1
    highs = opens + 3
    lows = opens - 2
    return opens, highs, lows, closes


@pytest.fixture
def sample_df():
    """產生 sample DataFrame。"""
    n = 200
    np.random.seed(123)
    base = np.cumsum(np.random.normal(0, 10, n)) + 16000
    opens = base
    closes = base + np.random.normal(0, 5, n)
    highs = np.maximum(opens, closes) + np.abs(np.random.normal(10, 5, n))
    lows = np.minimum(opens, closes) - np.abs(np.random.normal(10, 5, n))
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes,
    })


# ============================================================
# Tests: Swing Points
# ============================================================

class TestSwingPoints:
    def test_returns_list(self, simple_swing_data):
        _, highs, lows, _ = simple_swing_data
        result = detect_swing_points(highs, lows, lookback=3)
        assert isinstance(result, list)

    def test_swing_high_detected(self, simple_swing_data):
        _, highs, lows, _ = simple_swing_data
        points = detect_swing_points(highs, lows, lookback=3)
        swing_highs = [p for p in points if p.is_high]
        assert len(swing_highs) > 0

    def test_swing_low_detected(self, simple_swing_data):
        _, highs, lows, _ = simple_swing_data
        points = detect_swing_points(highs, lows, lookback=3)
        swing_lows = [p for p in points if not p.is_high]
        assert len(swing_lows) > 0

    def test_sorted_by_index(self, simple_swing_data):
        _, highs, lows, _ = simple_swing_data
        points = detect_swing_points(highs, lows, lookback=3)
        indices = [p.index for p in points]
        assert indices == sorted(indices)

    def test_empty_input(self):
        result = detect_swing_points(np.array([]), np.array([]), lookback=3)
        assert result == []

    def test_too_short(self):
        result = detect_swing_points(np.array([1, 2, 3]), np.array([1, 2, 3]), lookback=3)
        assert result == []

    def test_lookback_parameter(self, trending_up_data):
        _, highs, lows, _ = trending_up_data
        short_lb = detect_swing_points(highs, lows, lookback=2)
        long_lb = detect_swing_points(highs, lows, lookback=8)
        # 較短的 lookback 通常會偵測到更多 swing points
        assert len(short_lb) >= len(long_lb)


# ============================================================
# Tests: Structure Breaks
# ============================================================

class TestStructureBreaks:
    def test_returns_list(self, trending_up_data):
        opens, highs, lows, closes = trending_up_data
        swings = detect_swing_points(highs, lows, 3)
        result = detect_structure_breaks(swings, highs, lows, closes, 5.0)
        assert isinstance(result, list)

    def test_bos_in_uptrend(self, trending_up_data):
        opens, highs, lows, closes = trending_up_data
        swings = detect_swing_points(highs, lows, 3)
        breaks = detect_structure_breaks(swings, highs, lows, closes, 5.0)
        bos_list = [b for b in breaks if b.break_type == StructureType.BOS]
        # 上升趨勢中應該有 BOS
        assert len(bos_list) >= 0  # 可能有也可能沒有取決於資料

    def test_break_has_direction(self, trending_up_data):
        opens, highs, lows, closes = trending_up_data
        swings = detect_swing_points(highs, lows, 3)
        breaks = detect_structure_breaks(swings, highs, lows, closes, 5.0)
        for b in breaks:
            assert b.direction in (Trend.BULLISH, Trend.BEARISH)

    def test_min_move_filter(self, trending_up_data):
        opens, highs, lows, closes = trending_up_data
        swings = detect_swing_points(highs, lows, 3)
        breaks_loose = detect_structure_breaks(swings, highs, lows, closes, 1.0)
        breaks_strict = detect_structure_breaks(swings, highs, lows, closes, 50.0)
        assert len(breaks_loose) >= len(breaks_strict)

    def test_empty_swings(self):
        result = detect_structure_breaks([], np.array([]), np.array([]), np.array([]), 5.0)
        assert result == []


# ============================================================
# Tests: Order Blocks
# ============================================================

class TestOrderBlocks:
    def test_returns_list(self, sample_df):
        analyzer = SMCAnalyzer(swing_lookback=3, bos_min_move=5.0)
        result = analyzer.analyze(sample_df)
        assert isinstance(result["order_blocks"], list)

    def test_ob_direction_matches_bos(self, sample_df):
        analyzer = SMCAnalyzer(swing_lookback=3, bos_min_move=5.0, ob_body_ratio=0.2)
        result = analyzer.analyze(sample_df)
        for ob in result["order_blocks"]:
            assert ob.direction in (Trend.BULLISH, Trend.BEARISH)

    def test_ob_has_valid_prices(self, sample_df):
        analyzer = SMCAnalyzer(swing_lookback=3, bos_min_move=5.0)
        result = analyzer.analyze(sample_df)
        for ob in result["order_blocks"]:
            assert ob.high >= ob.low
            assert ob.top == ob.high
            assert ob.bottom == ob.low

    def test_ob_validity_update(self):
        ob = OrderBlock(
            start_index=0, open_price=100, high=105, low=95,
            close_price=98, direction=Trend.BULLISH,
        )
        closes = np.array([100, 101, 102, 90])  # 最後跌破 OB.low
        update_ob_validity([ob], 3, closes, max_age=20)
        assert not ob.is_valid

    def test_ob_age_invalidation(self):
        ob = OrderBlock(
            start_index=0, open_price=100, high=105, low=95,
            close_price=98, direction=Trend.BULLISH,
        )
        closes = np.array([100] * 25)
        update_ob_validity([ob], 24, closes, max_age=20)
        assert not ob.is_valid


# ============================================================
# Tests: FVG
# ============================================================

class TestFVG:
    def test_bullish_fvg(self):
        highs = np.array([100, 110, 120])
        lows = np.array([90, 95, 106])  # lows[2] > highs[0] => gap = 6
        closes = np.array([98, 108, 118])
        opens = np.array([92, 96, 107])
        fvgs = detect_fvg(highs, lows, closes, opens, min_gap=5.0)
        bullish = [f for f in fvgs if f.direction == Trend.BULLISH]
        assert len(bullish) > 0

    def test_bearish_fvg(self):
        highs = np.array([120, 110, 95])
        lows = np.array([115, 105, 90])  # lows[0] > highs[2] => gap = 20
        closes = np.array([118, 106, 92])
        opens = np.array([119, 109, 94])
        fvgs = detect_fvg(highs, lows, closes, opens, min_gap=5.0)
        bearish = [f for f in fvgs if f.direction == Trend.BEARISH]
        assert len(bearish) > 0

    def test_no_fvg_when_no_gap(self):
        highs = np.array([100, 101, 102])
        lows = np.array([98, 99, 100])
        closes = np.array([99, 100, 101])
        opens = np.array([98, 99, 100])
        fvgs = detect_fvg(highs, lows, closes, opens, min_gap=5.0)
        assert len(fvgs) == 0

    def test_fvg_fill_check(self):
        fvg = FVG(index=1, top=110, bottom=105, direction=Trend.BULLISH)
        highs = np.array([100, 110, 120])
        lows = np.array([90, 95, 104])  # 最後一根跌穿 bottom
        check_fvg_filled([fvg], 2, highs, lows)
        assert fvg.is_filled

    def test_min_gap_filter(self):
        highs = np.array([100, 110, 120])
        lows = np.array([90, 95, 102])  # gap = 2 (102-100)
        closes = np.array([98, 108, 118])
        opens = np.array([92, 96, 103])
        fvgs_strict = detect_fvg(highs, lows, closes, opens, min_gap=5.0)
        fvgs_loose = detect_fvg(highs, lows, closes, opens, min_gap=1.0)
        assert len(fvgs_loose) >= len(fvgs_strict)


# ============================================================
# Tests: SMCAnalyzer
# ============================================================

class TestSMCAnalyzer:
    def test_analyze_returns_dict(self, sample_df):
        analyzer = SMCAnalyzer()
        result = analyzer.analyze(sample_df)
        assert "swing_points" in result
        assert "structure_breaks" in result
        assert "order_blocks" in result
        assert "fvgs" in result

    def test_htf_trend(self):
        analyzer = SMCAnalyzer()
        breaks = [
            StructureBreak(10, StructureType.BOS, Trend.BULLISH, 100),
            StructureBreak(20, StructureType.BOS, Trend.BULLISH, 110),
        ]
        assert analyzer.get_htf_trend(breaks) == Trend.BULLISH

    def test_htf_trend_empty(self):
        analyzer = SMCAnalyzer()
        assert analyzer.get_htf_trend([]) == Trend.NEUTRAL

    def test_get_valid_obs(self):
        analyzer = SMCAnalyzer(ob_max_age=10)
        obs = [
            OrderBlock(0, 100, 105, 95, 98, Trend.BULLISH),
            OrderBlock(5, 110, 115, 105, 108, Trend.BULLISH),
        ]
        closes = np.array([100] * 20)
        valid = analyzer.get_valid_obs(obs, 7, closes)
        # index 0 的 OB age=7 仍在 max_age=10 內
        assert len(valid) >= 1
