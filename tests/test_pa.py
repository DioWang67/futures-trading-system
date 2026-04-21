"""PA 模組單元測試。"""

import numpy as np
import pytest

from src.strategy.pa import (
    PASignal,
    PAPattern,
    PAAnalyzer,
    detect_pin_bar,
    detect_engulfing,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def bullish_pin_bar_data():
    """Bullish pin bar: 長下影線。"""
    opens = np.array([105.0])
    highs = np.array([107.0])
    lows = np.array([90.0])   # 長下影線
    closes = np.array([106.0])
    return opens, highs, lows, closes


@pytest.fixture
def bearish_pin_bar_data():
    """Bearish pin bar: 長上影線。"""
    opens = np.array([95.0])
    highs = np.array([110.0])  # 長上影線
    lows = np.array([93.0])
    closes = np.array([94.0])
    return opens, highs, lows, closes


@pytest.fixture
def bullish_engulfing_data():
    """Bullish engulfing: 前跌後漲，後包前。"""
    opens = np.array([105.0, 97.0])   # bar0: 看跌, bar1: 看漲
    highs = np.array([106.0, 108.0])
    lows = np.array([98.0, 96.0])
    closes = np.array([99.0, 107.0])  # bar1.close >= bar0.open, bar1.open <= bar0.close
    return opens, highs, lows, closes


@pytest.fixture
def bearish_engulfing_data():
    """Bearish engulfing: 前漲後跌，後包前。"""
    opens = np.array([95.0, 106.0])   # bar0: 看漲, bar1: 看跌
    highs = np.array([103.0, 107.0])
    lows = np.array([94.0, 93.0])
    closes = np.array([102.0, 94.0])  # bar1.open >= bar0.close, bar1.close <= bar0.open
    return opens, highs, lows, closes


# ============================================================
# Tests: Pin Bar
# ============================================================

class TestPinBar:
    def test_bullish_pin_bar_detected(self, bullish_pin_bar_data):
        opens, highs, lows, closes = bullish_pin_bar_data
        patterns = detect_pin_bar(opens, highs, lows, closes, ratio=0.6)
        bullish = [p for p in patterns if p.signal == PASignal.BULLISH_PIN_BAR]
        assert len(bullish) == 1
        assert bullish[0].index == 0

    def test_bearish_pin_bar_detected(self, bearish_pin_bar_data):
        opens, highs, lows, closes = bearish_pin_bar_data
        patterns = detect_pin_bar(opens, highs, lows, closes, ratio=0.6)
        bearish = [p for p in patterns if p.signal == PASignal.BEARISH_PIN_BAR]
        assert len(bearish) == 1

    def test_no_pin_bar_on_doji(self):
        opens = np.array([100.0])
        highs = np.array([105.0])
        lows = np.array([95.0])
        closes = np.array([100.0])  # 十字線，上下影等長
        patterns = detect_pin_bar(opens, highs, lows, closes, ratio=0.6)
        # 上影線 = 下影線 = 50%，都不到 60%
        assert len(patterns) == 0

    def test_no_pin_bar_on_big_body(self):
        opens = np.array([95.0])
        highs = np.array([106.0])
        lows = np.array([94.0])
        closes = np.array([105.0])  # 大實體
        patterns = detect_pin_bar(opens, highs, lows, closes, ratio=0.6)
        assert len(patterns) == 0

    def test_strength_value(self, bullish_pin_bar_data):
        opens, highs, lows, closes = bullish_pin_bar_data
        patterns = detect_pin_bar(opens, highs, lows, closes, ratio=0.6)
        for p in patterns:
            assert 0 <= p.strength <= 1.0

    def test_empty_input(self):
        result = detect_pin_bar(np.array([]), np.array([]), np.array([]), np.array([]))
        assert result == []

    def test_zero_range_bar(self):
        opens = np.array([100.0])
        highs = np.array([100.0])
        lows = np.array([100.0])
        closes = np.array([100.0])
        result = detect_pin_bar(opens, highs, lows, closes)
        assert result == []


# ============================================================
# Tests: Engulfing
# ============================================================

class TestEngulfing:
    def test_bullish_engulfing_detected(self, bullish_engulfing_data):
        opens, highs, lows, closes = bullish_engulfing_data
        patterns = detect_engulfing(opens, highs, lows, closes, engulf_ratio=1.0)
        bullish = [p for p in patterns if p.signal == PASignal.BULLISH_ENGULFING]
        assert len(bullish) == 1
        assert bullish[0].index == 1

    def test_bearish_engulfing_detected(self, bearish_engulfing_data):
        opens, highs, lows, closes = bearish_engulfing_data
        patterns = detect_engulfing(opens, highs, lows, closes, engulf_ratio=1.0)
        bearish = [p for p in patterns if p.signal == PASignal.BEARISH_ENGULFING]
        assert len(bearish) == 1

    def test_no_engulfing_same_direction(self):
        opens = np.array([100.0, 102.0])  # 兩根都看漲
        highs = np.array([105.0, 108.0])
        lows = np.array([99.0, 101.0])
        closes = np.array([104.0, 107.0])
        patterns = detect_engulfing(opens, highs, lows, closes)
        assert len(patterns) == 0

    def test_no_engulfing_small_body(self):
        opens = np.array([105.0, 104.0])  # bar0: 看跌, bar1: 看漲 但小
        highs = np.array([106.0, 105.0])
        lows = np.array([103.0, 103.0])
        closes = np.array([104.0, 104.5])  # bar1 body=0.5 < bar0 body=1
        patterns = detect_engulfing(opens, highs, lows, closes, engulf_ratio=1.0)
        assert len(patterns) == 0

    def test_single_bar(self):
        opens = np.array([100.0])
        highs = np.array([105.0])
        lows = np.array([95.0])
        closes = np.array([103.0])
        result = detect_engulfing(opens, highs, lows, closes)
        assert result == []

    def test_strength_bounded(self, bullish_engulfing_data):
        opens, highs, lows, closes = bullish_engulfing_data
        patterns = detect_engulfing(opens, highs, lows, closes)
        for p in patterns:
            assert 0 <= p.strength <= 1.0


# ============================================================
# Tests: PAAnalyzer
# ============================================================

class TestPAAnalyzer:
    def test_analyze_returns_sorted(self, bullish_engulfing_data):
        opens, highs, lows, closes = bullish_engulfing_data
        analyzer = PAAnalyzer()
        patterns = analyzer.analyze(opens, highs, lows, closes)
        indices = [p.index for p in patterns]
        assert indices == sorted(indices)

    def test_has_bullish_signal(self, bullish_pin_bar_data):
        opens, highs, lows, closes = bullish_pin_bar_data
        analyzer = PAAnalyzer(pin_bar_ratio=0.6)
        patterns = analyzer.analyze(opens, highs, lows, closes)
        assert analyzer.has_bullish_signal(patterns, 0)

    def test_has_bearish_signal(self, bearish_pin_bar_data):
        opens, highs, lows, closes = bearish_pin_bar_data
        analyzer = PAAnalyzer(pin_bar_ratio=0.6)
        patterns = analyzer.analyze(opens, highs, lows, closes)
        assert analyzer.has_bearish_signal(patterns, 0)

    def test_no_signal_at_wrong_index(self, bullish_pin_bar_data):
        opens, highs, lows, closes = bullish_pin_bar_data
        analyzer = PAAnalyzer()
        patterns = analyzer.analyze(opens, highs, lows, closes)
        assert not analyzer.has_bullish_signal(patterns, 99)
