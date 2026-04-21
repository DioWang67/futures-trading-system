"""
Price Action (PA) K 棒形態辨識模組

偵測：
- Pin Bar（長影線 K 棒）
- Engulfing（吞噬 K 棒）
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from enum import Enum


class PASignal(Enum):
    BULLISH_PIN_BAR = "bullish_pin_bar"
    BEARISH_PIN_BAR = "bearish_pin_bar"
    BULLISH_ENGULFING = "bullish_engulfing"
    BEARISH_ENGULFING = "bearish_engulfing"


@dataclass
class PAPattern:
    index: int
    signal: PASignal
    strength: float  # 0~1, 越高越強


def detect_pin_bar(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    ratio: float = 0.6,
) -> list[PAPattern]:
    """偵測 Pin Bar。

    Bullish Pin Bar: 長下影線，實體在上方
      - 下影線 >= ratio * 全 K 棒範圍
      - 實體 <= (1 - ratio) * 全 K 棒範圍

    Bearish Pin Bar: 長上影線，實體在下方
      - 上影線 >= ratio * 全 K 棒範圍
      - 實體 <= (1 - ratio) * 全 K 棒範圍
    """
    patterns: list[PAPattern] = []
    n = len(opens)

    for i in range(n):
        bar_range = highs[i] - lows[i]
        if bar_range == 0:
            continue

        body = abs(closes[i] - opens[i])
        upper_wick = highs[i] - max(opens[i], closes[i])
        lower_wick = min(opens[i], closes[i]) - lows[i]

        body_ratio = body / bar_range

        # Bullish Pin Bar: 長下影線
        if lower_wick / bar_range >= ratio and body_ratio <= (1 - ratio):
            strength = lower_wick / bar_range
            patterns.append(PAPattern(
                index=i,
                signal=PASignal.BULLISH_PIN_BAR,
                strength=float(strength),
            ))

        # Bearish Pin Bar: 長上影線
        if upper_wick / bar_range >= ratio and body_ratio <= (1 - ratio):
            strength = upper_wick / bar_range
            patterns.append(PAPattern(
                index=i,
                signal=PASignal.BEARISH_PIN_BAR,
                strength=float(strength),
            ))

    return patterns


def detect_engulfing(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    engulf_ratio: float = 1.0,
) -> list[PAPattern]:
    """偵測吞噬 K 棒 (Engulfing)。

    Bullish Engulfing:
      - 前一根看跌（close < open）
      - 當前看漲（close > open）
      - 當前實體完全包覆前一根實體

    Bearish Engulfing:
      - 前一根看漲（close > open）
      - 當前看跌（close < open）
      - 當前實體完全包覆前一根實體
    """
    patterns: list[PAPattern] = []
    n = len(opens)

    for i in range(1, n):
        prev_body = abs(closes[i - 1] - opens[i - 1])
        curr_body = abs(closes[i] - opens[i])

        if prev_body == 0:
            continue

        # Bullish Engulfing
        if (closes[i - 1] < opens[i - 1]      # 前一根看跌
            and closes[i] > opens[i]            # 當前看漲
            and closes[i] >= opens[i - 1]       # 當前 close >= 前一根 open
            and opens[i] <= closes[i - 1]       # 當前 open <= 前一根 close
            and curr_body >= prev_body * engulf_ratio):
            strength = min(curr_body / prev_body, 2.0) / 2.0
            patterns.append(PAPattern(
                index=i,
                signal=PASignal.BULLISH_ENGULFING,
                strength=float(strength),
            ))

        # Bearish Engulfing
        if (closes[i - 1] > opens[i - 1]      # 前一根看漲
            and closes[i] < opens[i]            # 當前看跌
            and opens[i] >= closes[i - 1]       # 當前 open >= 前一根 close
            and closes[i] <= opens[i - 1]       # 當前 close <= 前一根 open
            and curr_body >= prev_body * engulf_ratio):
            strength = min(curr_body / prev_body, 2.0) / 2.0
            patterns.append(PAPattern(
                index=i,
                signal=PASignal.BEARISH_ENGULFING,
                strength=float(strength),
            ))

    return patterns


class PAAnalyzer:
    """Price Action 形態分析器。"""

    def __init__(self, pin_bar_ratio: float = 0.6, engulf_ratio: float = 1.0):
        self.pin_bar_ratio = pin_bar_ratio
        self.engulf_ratio = engulf_ratio

    def analyze(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
    ) -> list[PAPattern]:
        """偵測所有 PA 形態。"""
        pin_bars = detect_pin_bar(opens, highs, lows, closes, self.pin_bar_ratio)
        engulfings = detect_engulfing(opens, highs, lows, closes, self.engulf_ratio)
        all_patterns = pin_bars + engulfings
        all_patterns.sort(key=lambda p: p.index)
        return all_patterns

    def has_bullish_signal(self, patterns: list[PAPattern], index: int) -> bool:
        """指定 index 是否有看漲 PA 訊號。"""
        return any(
            p.index == index and p.signal in (
                PASignal.BULLISH_PIN_BAR, PASignal.BULLISH_ENGULFING
            )
            for p in patterns
        )

    def has_bearish_signal(self, patterns: list[PAPattern], index: int) -> bool:
        """指定 index 是否有看跌 PA 訊號。"""
        return any(
            p.index == index and p.signal in (
                PASignal.BEARISH_PIN_BAR, PASignal.BEARISH_ENGULFING
            )
            for p in patterns
        )
