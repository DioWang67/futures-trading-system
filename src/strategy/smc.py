"""
Smart Money Concepts (SMC) 結構分析模組

偵測：
- Swing High / Swing Low
- BOS (Break of Structure) — 趨勢延續
- CHoCH (Change of Character) — 趨勢反轉
- Order Block (OB) — BOS 前最後一根反向 K 棒
- FVG (Fair Value Gap) — 三根 K 棒間的缺口
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Trend(Enum):
    BULLISH = 1
    BEARISH = -1
    NEUTRAL = 0


class StructureType(Enum):
    BOS = "BOS"
    CHOCH = "CHoCH"


@dataclass
class SwingPoint:
    index: int
    price: float
    is_high: bool  # True = swing high, False = swing low


@dataclass
class StructureBreak:
    index: int          # 突破發生的 bar index
    break_type: StructureType
    direction: Trend    # 突破後的趨勢方向
    level: float        # 被突破的價位


@dataclass
class OrderBlock:
    start_index: int
    open_price: float
    high: float
    low: float
    close_price: float
    direction: Trend      # BULLISH OB = 看漲, BEARISH OB = 看跌
    is_valid: bool = True
    age: int = 0          # 已經過了幾根 K 棒

    @property
    def top(self) -> float:
        return self.high

    @property
    def bottom(self) -> float:
        return self.low


@dataclass
class FVG:
    index: int          # 中間那根 K 棒的 index
    top: float
    bottom: float
    direction: Trend    # BULLISH = 向上缺口, BEARISH = 向下缺口
    is_filled: bool = False


def detect_swing_points(
    highs: np.ndarray,
    lows: np.ndarray,
    lookback: int = 5,
) -> list[SwingPoint]:
    """偵測 swing high 和 swing low。

    Swing high: 該 bar 的 high 是前後 lookback 根中最高的
    Swing low:  該 bar 的 low 是前後 lookback 根中最低的
    """
    n = len(highs)
    points: list[SwingPoint] = []

    for i in range(lookback, n - lookback):
        # Swing High
        window_highs = highs[i - lookback: i + lookback + 1]
        if highs[i] == np.max(window_highs) and np.sum(window_highs == highs[i]) == 1:
            points.append(SwingPoint(index=i, price=float(highs[i]), is_high=True))

        # Swing Low
        window_lows = lows[i - lookback: i + lookback + 1]
        if lows[i] == np.min(window_lows) and np.sum(window_lows == lows[i]) == 1:
            points.append(SwingPoint(index=i, price=float(lows[i]), is_high=False))

    points.sort(key=lambda p: p.index)
    return points


def detect_structure_breaks(
    swing_points: list[SwingPoint],
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    min_move: float = 15.0,
) -> list[StructureBreak]:
    """偵測 BOS 和 CHoCH。

    BOS: 順勢突破前一個 swing high/low（趨勢延續）
    CHoCH: 逆勢突破前一個 swing high/low（趨勢反轉）
    """
    if len(swing_points) < 2:
        return []

    breaks: list[StructureBreak] = []
    current_trend = Trend.NEUTRAL

    # 追蹤最近的 swing high 和 swing low
    last_sh: Optional[SwingPoint] = None
    last_sl: Optional[SwingPoint] = None

    for sp in swing_points:
        if sp.is_high:
            # 檢查是否突破前一個 swing high
            if last_sh is not None and sp.price > last_sh.price + min_move:
                if current_trend == Trend.BULLISH or current_trend == Trend.NEUTRAL:
                    break_type = StructureType.BOS
                else:
                    break_type = StructureType.CHOCH
                breaks.append(StructureBreak(
                    index=sp.index,
                    break_type=break_type,
                    direction=Trend.BULLISH,
                    level=last_sh.price,
                ))
                current_trend = Trend.BULLISH
            last_sh = sp
        else:
            # 檢查是否跌破前一個 swing low
            if last_sl is not None and sp.price < last_sl.price - min_move:
                if current_trend == Trend.BEARISH or current_trend == Trend.NEUTRAL:
                    break_type = StructureType.BOS
                else:
                    break_type = StructureType.CHOCH
                breaks.append(StructureBreak(
                    index=sp.index,
                    break_type=break_type,
                    direction=Trend.BEARISH,
                    level=last_sl.price,
                ))
                current_trend = Trend.BEARISH
            last_sl = sp

    return breaks


def detect_order_blocks(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    structure_breaks: list[StructureBreak],
    body_ratio: float = 0.4,
    max_age: int = 20,
) -> list[OrderBlock]:
    """偵測 Order Block。

    Bullish OB: BOS 向上突破前，最後一根看跌 K 棒
    Bearish OB: BOS 向下突破前，最後一根看漲 K 棒
    """
    obs: list[OrderBlock] = []

    for brk in structure_breaks:
        if brk.break_type != StructureType.BOS:
            continue

        search_start = max(0, brk.index - max_age)

        if brk.direction == Trend.BULLISH:
            # 往回找最後一根看跌 K 棒（close < open）
            for i in range(brk.index - 1, search_start - 1, -1):
                bar_range = highs[i] - lows[i]
                if bar_range == 0:
                    continue
                body = abs(closes[i] - opens[i])
                if closes[i] < opens[i] and body / bar_range >= body_ratio:
                    obs.append(OrderBlock(
                        start_index=i,
                        open_price=float(opens[i]),
                        high=float(highs[i]),
                        low=float(lows[i]),
                        close_price=float(closes[i]),
                        direction=Trend.BULLISH,
                    ))
                    break

        elif brk.direction == Trend.BEARISH:
            # 往回找最後一根看漲 K 棒（close > open）
            for i in range(brk.index - 1, search_start - 1, -1):
                bar_range = highs[i] - lows[i]
                if bar_range == 0:
                    continue
                body = abs(closes[i] - opens[i])
                if closes[i] > opens[i] and body / bar_range >= body_ratio:
                    obs.append(OrderBlock(
                        start_index=i,
                        open_price=float(opens[i]),
                        high=float(highs[i]),
                        low=float(lows[i]),
                        close_price=float(closes[i]),
                        direction=Trend.BEARISH,
                    ))
                    break

    return obs


def detect_fvg(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    opens: np.ndarray,
    min_gap: float = 5.0,
) -> list[FVG]:
    """偵測 Fair Value Gap (FVG)。

    Bullish FVG: bar[i-1].high < bar[i+1].low （向上跳空）
    Bearish FVG: bar[i-1].low > bar[i+1].high （向下跳空）
    """
    n = len(highs)
    fvgs: list[FVG] = []

    for i in range(1, n - 1):
        # Bullish FVG
        gap_bottom = float(highs[i - 1])
        gap_top = float(lows[i + 1])
        if gap_top - gap_bottom >= min_gap:
            fvgs.append(FVG(
                index=i,
                top=gap_top,
                bottom=gap_bottom,
                direction=Trend.BULLISH,
            ))

        # Bearish FVG
        gap_top_bear = float(lows[i - 1])
        gap_bottom_bear = float(highs[i + 1])
        if gap_top_bear - gap_bottom_bear >= min_gap:
            fvgs.append(FVG(
                index=i,
                top=gap_top_bear,
                bottom=gap_bottom_bear,
                direction=Trend.BEARISH,
            ))

    return fvgs


def update_ob_validity(
    obs: list[OrderBlock],
    current_index: int,
    closes: np.ndarray,
    max_age: int = 20,
) -> list[OrderBlock]:
    """更新 Order Block 的有效性。

    OB 失效條件：
    1. 超過 max_age 根 K 棒
    2. 被 close 穿過（bullish OB 被跌破，bearish OB 被突破）
    """
    for ob in obs:
        if not ob.is_valid:
            continue
        ob.age = current_index - ob.start_index
        if ob.age > max_age:
            ob.is_valid = False
            continue
        if current_index < len(closes):
            if ob.direction == Trend.BULLISH and closes[current_index] < ob.low:
                ob.is_valid = False
            elif ob.direction == Trend.BEARISH and closes[current_index] > ob.high:
                ob.is_valid = False

    return obs


def check_fvg_filled(fvgs: list[FVG], current_index: int, highs: np.ndarray, lows: np.ndarray) -> list[FVG]:
    """檢查 FVG 是否已被回填。"""
    for fvg in fvgs:
        if fvg.is_filled:
            continue
        if current_index < len(highs):
            if fvg.direction == Trend.BULLISH and lows[current_index] <= fvg.bottom:
                fvg.is_filled = True
            elif fvg.direction == Trend.BEARISH and highs[current_index] >= fvg.top:
                fvg.is_filled = True
    return fvgs


class SMCAnalyzer:
    """SMC 分析器，整合所有 SMC 概念的偵測。"""

    def __init__(
        self,
        swing_lookback: int = 5,
        bos_min_move: float = 15.0,
        ob_max_age: int = 20,
        ob_body_ratio: float = 0.4,
        fvg_min_gap: float = 5.0,
    ):
        self.swing_lookback = swing_lookback
        self.bos_min_move = bos_min_move
        self.ob_max_age = ob_max_age
        self.ob_body_ratio = ob_body_ratio
        self.fvg_min_gap = fvg_min_gap

    def analyze(self, df: pd.DataFrame) -> dict:
        """分析整個 DataFrame，回傳所有 SMC 結構。

        df 需要有 columns: open, high, low, close
        """
        opens = df["open"].values.astype(float)
        highs = df["high"].values.astype(float)
        lows = df["low"].values.astype(float)
        closes = df["close"].values.astype(float)

        swing_points = detect_swing_points(highs, lows, self.swing_lookback)
        structure_breaks = detect_structure_breaks(
            swing_points, highs, lows, closes, self.bos_min_move
        )
        order_blocks = detect_order_blocks(
            opens, highs, lows, closes, structure_breaks,
            self.ob_body_ratio, self.ob_max_age
        )
        fvgs = detect_fvg(highs, lows, closes, opens, self.fvg_min_gap)

        return {
            "swing_points": swing_points,
            "structure_breaks": structure_breaks,
            "order_blocks": order_blocks,
            "fvgs": fvgs,
        }

    def get_htf_trend(self, structure_breaks: list[StructureBreak]) -> Trend:
        """從 HTF 結構突破判斷當前趨勢方向。"""
        if not structure_breaks:
            return Trend.NEUTRAL
        # 以最後一個 BOS/CHoCH 的方向為準
        return structure_breaks[-1].direction

    def get_valid_obs(
        self,
        order_blocks: list[OrderBlock],
        current_index: int,
        closes: np.ndarray,
    ) -> list[OrderBlock]:
        """取得當前仍有效的 Order Block。"""
        update_ob_validity(order_blocks, current_index, closes, self.ob_max_age)
        return [ob for ob in order_blocks if ob.is_valid]

    def get_unfilled_fvgs(
        self,
        fvgs: list[FVG],
        current_index: int,
        highs: np.ndarray,
        lows: np.ndarray,
    ) -> list[FVG]:
        """取得尚未被回填的 FVG。"""
        check_fvg_filled(fvgs, current_index, highs, lows)
        return [f for f in fvgs if not f.is_filled]
