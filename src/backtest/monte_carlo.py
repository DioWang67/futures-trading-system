"""
Monte Carlo 模擬模組

三種模擬方法：
1. Trade Shuffle — 打亂交易順序，模擬不同進場時序下的權益曲線
2. Bootstrap — 有放回隨機抽樣，模擬不同市場路徑
3. PnL Noise — 對每筆交易損益加入隨機擾動（±滑點/噪音）

輸出：
- N 條模擬權益曲線
- MDD 分布（中位數、95th percentile）
- 最終損益分布
- 破產機率（權益跌破門檻的比例）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloResult:
    """蒙地卡羅模擬結果。"""

    n_simulations: int = 0
    method: str = ""

    # 最終權益
    final_equity_mean: float = 0.0
    final_equity_median: float = 0.0
    final_equity_5th: float = 0.0       # 最差 5%
    final_equity_95th: float = 0.0      # 最好 5%
    final_equity_std: float = 0.0

    # 最大回撤
    max_dd_mean: float = 0.0
    max_dd_median: float = 0.0
    max_dd_95th: float = 0.0            # 95% 最差情況 MDD
    max_dd_worst: float = 0.0           # 最糟的一條

    # 破產機率
    ruin_probability: float = 0.0       # 跌破 ruin_threshold 的比例
    ruin_threshold: float = 0.0

    # 獲利機率
    profit_probability: float = 0.0     # 最終權益 > 初始資金的比例

    # 原始數據（供繪圖用）
    all_final_equities: list[float] = field(default_factory=list)
    all_max_drawdowns: list[float] = field(default_factory=list)
    equity_curves: list[list[float]] = field(default_factory=list)  # 前 100 條供繪圖

    def summary(self) -> str:
        """產生人類可讀的摘要。"""
        lines = [
            f"Monte Carlo Simulation ({self.method})",
            f"  Simulations:       {self.n_simulations:,}",
            f"  ---",
            f"  Final Equity:",
            f"    Mean:            ${self.final_equity_mean:>12,.2f}",
            f"    Median:          ${self.final_equity_median:>12,.2f}",
            f"    5th percentile:  ${self.final_equity_5th:>12,.2f}",
            f"    95th percentile: ${self.final_equity_95th:>12,.2f}",
            f"    Std Dev:         ${self.final_equity_std:>12,.2f}",
            f"  ---",
            f"  Max Drawdown:",
            f"    Mean:            {self.max_dd_mean:>11.2%}",
            f"    Median:          {self.max_dd_median:>11.2%}",
            f"    95th percentile: {self.max_dd_95th:>11.2%}",
            f"    Worst:           {self.max_dd_worst:>11.2%}",
            f"  ---",
            f"  Profit Probability:  {self.profit_probability:>7.1%}",
            f"  Ruin Probability:    {self.ruin_probability:>7.1%}  (threshold: ${self.ruin_threshold:,.0f})",
        ]
        return "\n".join(lines)


def _compute_max_drawdown_pct(equity_curve: np.ndarray) -> float:
    """計算權益曲線的最大回撤百分比。"""
    peak = np.maximum.accumulate(equity_curve)
    # 避免除以零
    mask = peak > 0
    dd = np.zeros_like(equity_curve)
    dd[mask] = (peak[mask] - equity_curve[mask]) / peak[mask]
    return float(np.max(dd)) if len(dd) > 0 else 0.0


def _build_equity_curve(pnl_sequence: np.ndarray, initial_capital: float) -> np.ndarray:
    """從損益序列建構權益曲線。"""
    return initial_capital + np.cumsum(pnl_sequence)


def monte_carlo_shuffle(
    trade_pnls: list[float],
    initial_capital: float = 100_000.0,
    n_simulations: int = 1000,
    ruin_threshold_pct: float = 0.5,
    seed: Optional[int] = None,
) -> MonteCarloResult:
    """Trade Shuffle 蒙地卡羅。

    保留原始交易損益不變，只打亂順序。
    測試策略是否對交易順序敏感。

    Parameters
    ----------
    trade_pnls : list[float]
        每筆交易的淨損益（已扣手續費）
    initial_capital : float
        初始資金
    n_simulations : int
        模擬次數
    ruin_threshold_pct : float
        破產門檻（佔初始資金的比例，例如 0.5 = 跌到剩 50%）
    """
    if not trade_pnls:
        return MonteCarloResult(method="shuffle")

    rng = np.random.default_rng(seed)
    pnls = np.array(trade_pnls)
    n_trades = len(pnls)
    ruin_threshold = initial_capital * ruin_threshold_pct

    all_final = np.zeros(n_simulations)
    all_mdd = np.zeros(n_simulations)
    ruin_count = 0
    curves = []

    for i in range(n_simulations):
        shuffled = rng.permutation(pnls)
        eq = _build_equity_curve(shuffled, initial_capital)
        all_final[i] = eq[-1]
        all_mdd[i] = _compute_max_drawdown_pct(eq)
        if np.min(eq) <= ruin_threshold:
            ruin_count += 1
        if i < 100:
            curves.append(eq.tolist())

    return _build_result(
        method="shuffle",
        n_simulations=n_simulations,
        initial_capital=initial_capital,
        ruin_threshold=ruin_threshold,
        all_final=all_final,
        all_mdd=all_mdd,
        ruin_count=ruin_count,
        curves=curves,
    )


def monte_carlo_bootstrap(
    trade_pnls: list[float],
    initial_capital: float = 100_000.0,
    n_simulations: int = 1000,
    n_trades_per_sim: Optional[int] = None,
    ruin_threshold_pct: float = 0.5,
    seed: Optional[int] = None,
) -> MonteCarloResult:
    """Bootstrap 蒙地卡羅。

    有放回隨機抽樣，每次模擬抽取 n_trades_per_sim 筆交易。
    模擬不同市場路徑下的績效分布。

    Parameters
    ----------
    n_trades_per_sim : int or None
        每次模擬抽取的交易數。None = 與原始交易數相同。
    """
    if not trade_pnls:
        return MonteCarloResult(method="bootstrap")

    rng = np.random.default_rng(seed)
    pnls = np.array(trade_pnls)
    n_trades = n_trades_per_sim or len(pnls)
    ruin_threshold = initial_capital * ruin_threshold_pct

    all_final = np.zeros(n_simulations)
    all_mdd = np.zeros(n_simulations)
    ruin_count = 0
    curves = []

    for i in range(n_simulations):
        sampled = rng.choice(pnls, size=n_trades, replace=True)
        eq = _build_equity_curve(sampled, initial_capital)
        all_final[i] = eq[-1]
        all_mdd[i] = _compute_max_drawdown_pct(eq)
        if np.min(eq) <= ruin_threshold:
            ruin_count += 1
        if i < 100:
            curves.append(eq.tolist())

    return _build_result(
        method="bootstrap",
        n_simulations=n_simulations,
        initial_capital=initial_capital,
        ruin_threshold=ruin_threshold,
        all_final=all_final,
        all_mdd=all_mdd,
        ruin_count=ruin_count,
        curves=curves,
    )


def monte_carlo_noise(
    trade_pnls: list[float],
    initial_capital: float = 100_000.0,
    n_simulations: int = 1000,
    noise_std_pct: float = 0.1,
    ruin_threshold_pct: float = 0.5,
    seed: Optional[int] = None,
) -> MonteCarloResult:
    """PnL Noise 蒙地卡羅。

    保留交易順序，但對每筆損益加入高斯噪音。
    模擬滑點/成交價變異對績效的影響。

    Parameters
    ----------
    noise_std_pct : float
        噪音標準差（佔每筆 |PnL| 的比例，例如 0.1 = ±10%）
    """
    if not trade_pnls:
        return MonteCarloResult(method="noise")

    rng = np.random.default_rng(seed)
    pnls = np.array(trade_pnls)
    n_trades = len(pnls)
    ruin_threshold = initial_capital * ruin_threshold_pct

    all_final = np.zeros(n_simulations)
    all_mdd = np.zeros(n_simulations)
    ruin_count = 0
    curves = []

    for i in range(n_simulations):
        noise = rng.normal(0, noise_std_pct, n_trades) * np.abs(pnls)
        noisy_pnls = pnls + noise
        eq = _build_equity_curve(noisy_pnls, initial_capital)
        all_final[i] = eq[-1]
        all_mdd[i] = _compute_max_drawdown_pct(eq)
        if np.min(eq) <= ruin_threshold:
            ruin_count += 1
        if i < 100:
            curves.append(eq.tolist())

    return _build_result(
        method="noise",
        n_simulations=n_simulations,
        initial_capital=initial_capital,
        ruin_threshold=ruin_threshold,
        all_final=all_final,
        all_mdd=all_mdd,
        ruin_count=ruin_count,
        curves=curves,
    )


def _build_result(
    method: str,
    n_simulations: int,
    initial_capital: float,
    ruin_threshold: float,
    all_final: np.ndarray,
    all_mdd: np.ndarray,
    ruin_count: int,
    curves: list,
) -> MonteCarloResult:
    return MonteCarloResult(
        n_simulations=n_simulations,
        method=method,
        final_equity_mean=float(np.mean(all_final)),
        final_equity_median=float(np.median(all_final)),
        final_equity_5th=float(np.percentile(all_final, 5)),
        final_equity_95th=float(np.percentile(all_final, 95)),
        final_equity_std=float(np.std(all_final)),
        max_dd_mean=float(np.mean(all_mdd)),
        max_dd_median=float(np.median(all_mdd)),
        max_dd_95th=float(np.percentile(all_mdd, 95)),
        max_dd_worst=float(np.max(all_mdd)),
        ruin_probability=ruin_count / n_simulations if n_simulations > 0 else 0.0,
        ruin_threshold=ruin_threshold,
        profit_probability=float(np.mean(all_final > initial_capital)),
        all_final_equities=all_final.tolist(),
        all_max_drawdowns=all_mdd.tolist(),
        equity_curves=curves,
    )
