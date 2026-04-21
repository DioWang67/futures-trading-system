"""
回測引擎模組

架構：
1. precompute_signals() 在 pandas 層完成所有 SMC/PA 分析（向量化，秒級完成）
2. Backtrader 只負責模擬下單、手續費、滑點和績效統計
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import backtrader as bt
import numpy as np
import pandas as pd

from src.strategy.bt_strategy import SMCPAStrategy, SMCPAData, precompute_signals

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """回測結果。"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0       # 百分比
    avg_rr: float = 0.0
    total_pnl: float = 0.0
    equity_curve: list[float] = field(default_factory=list)
    trade_log: list[dict] = field(default_factory=list)
    params: dict = field(default_factory=dict)

    @property
    def meets_threshold(self) -> bool:
        """是否達到所有績效門檻。"""
        return (
            self.win_rate > 0.55
            and self.profit_factor > 1.5
            and self.sharpe_ratio > 1.2
            and self.max_drawdown < 0.15
            and self.avg_rr > 1.5
        )

    def meets_custom_threshold(
        self,
        min_win_rate: float = 0.55,
        min_profit_factor: float = 1.5,
        min_sharpe: float = 1.2,
        max_mdd: float = 0.15,
        min_rr: float = 1.5,
    ) -> bool:
        return (
            self.win_rate >= min_win_rate
            and self.profit_factor >= min_profit_factor
            and self.sharpe_ratio >= min_sharpe
            and self.max_drawdown <= max_mdd
            and self.avg_rr >= min_rr
        )


class BacktestEngine:
    """Backtrader 回測引擎。"""

    def __init__(
        self,
        initial_cash: float = 1_000_000.0,
        commission: float = 22.0,
        slippage: float = 1.0,
        size: int = 1,
        point_value: float = 10.0,
    ):
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage = slippage
        self.size = size
        self.point_value = point_value

    def run(
        self,
        ltf_data: pd.DataFrame,
        htf_data: Optional[pd.DataFrame] = None,
        strategy_params: Optional[dict] = None,
    ) -> BacktestResult:
        """執行一次回測。

        1. 先用 precompute_signals 預算所有訊號
        2. 再丟進 Backtrader 模擬交易
        """
        params = strategy_params or {}

        # ---- Step 1: 預計算訊號 ----
        logger.info("預計算 SMC+PA 訊號...")
        signal_df = precompute_signals(
            ltf_df=ltf_data,
            htf_df=htf_data,
            swing_lookback=params.get("swing_lookback", 5),
            bos_min_move=params.get("bos_min_move", 15.0),
            ob_max_age=params.get("ob_max_age", 20),
            ob_body_ratio=params.get("ob_body_ratio", 0.4),
            fvg_min_gap=params.get("fvg_min_gap", 5.0),
            fvg_enabled=params.get("fvg_enabled", True),
            pin_bar_ratio=params.get("pin_bar_ratio", 0.6),
            engulf_ratio=params.get("engulf_ratio", 1.0),
            sl_buffer=params.get("sl_buffer", 2.0),
            rr_ratio=params.get("rr_ratio", 1.5),
            use_structure_tp=params.get("use_structure_tp", True),
            pa_confirm=params.get("pa_confirm", True),
            adx_period=params.get("adx_period", 14),
            adx_threshold=params.get("adx_threshold", 20.0),
            adx_filter_enabled=params.get("adx_filter_enabled", True),
        )

        signal_count = (signal_df["signal"] != 0).sum()
        logger.info(f"訊號預計算完成: {signal_count} 個進場訊號")

        # ---- Step 2: Backtrader 模擬 ----
        cerebro = bt.Cerebro()

        feed = self._create_feed(signal_df)
        cerebro.adddata(feed, name="ltf")

        cerebro.addstrategy(SMCPAStrategy)

        cerebro.broker.setcash(self.initial_cash)
        cerebro.broker.setcommission(
            commission=self.commission,
            margin=0,
            mult=self.point_value,
            commtype=bt.CommInfoBase.COMM_FIXED,
        )
        cerebro.broker.set_slippage_fixed(self.slippage)
        cerebro.addsizer(bt.sizers.FixedSize, stake=self.size)

        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

        logger.info("開始 Backtrader 模擬...")
        results = cerebro.run()
        strat = results[0]
        logger.info("Backtrader 模擬完成")

        return self._extract_result(strat, params)

    def _create_feed(self, df: pd.DataFrame) -> SMCPAData:
        """建立 Backtrader data feed。"""
        df = df.copy()
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime")
        # 確保訊號欄位存在
        for col in ("htf_trend", "signal", "sl", "tp"):
            if col not in df.columns:
                df[col] = 0.0 if col != "sl" and col != "tp" else np.nan
        return SMCPAData(dataname=df)

    def _extract_result(self, strat, params: dict) -> BacktestResult:
        """從策略結果中提取績效指標。"""
        result = BacktestResult(params=params)
        result.trade_log = strat.trade_log

        trade_analyzer = strat.analyzers.trades.get_analysis()
        total = trade_analyzer.get("total", {})
        result.total_trades = total.get("total", 0) or 0

        if result.total_trades == 0:
            return result

        won = trade_analyzer.get("won", {})
        lost = trade_analyzer.get("lost", {})
        result.winning_trades = won.get("total", 0) or 0
        result.losing_trades = lost.get("total", 0) or 0

        result.win_rate = result.winning_trades / result.total_trades if result.total_trades > 0 else 0

        gross_profit = won.get("pnl", {}).get("total", 0) or 0
        gross_loss = abs(lost.get("pnl", {}).get("total", 0) or 0)
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        sharpe_analysis = strat.analyzers.sharpe.get_analysis()
        result.sharpe_ratio = sharpe_analysis.get("sharperatio", 0) or 0

        dd_analysis = strat.analyzers.drawdown.get_analysis()
        result.max_drawdown = dd_analysis.get("max", {}).get("drawdown", 0) / 100.0

        if strat.trade_log:
            rrs = [t["rr"] for t in strat.trade_log if "rr" in t]
            result.avg_rr = float(np.mean(rrs)) if rrs else 0

            # Per-trade Sharpe（比 Backtrader 的每日 Sharpe 更適合低頻策略）
            trade_pnls = np.array([t["pnl"] for t in strat.trade_log])
            if len(trade_pnls) >= 2 and np.std(trade_pnls) > 0:
                trade_sharpe = np.mean(trade_pnls) / np.std(trade_pnls)
                # 取 per-trade Sharpe 和 daily Sharpe 中較高者
                result.sharpe_ratio = max(result.sharpe_ratio, trade_sharpe)
        else:
            avg_win = won.get("pnl", {}).get("average", 0) or 0
            avg_loss = abs(lost.get("pnl", {}).get("average", 0) or 0)
            result.avg_rr = avg_win / avg_loss if avg_loss > 0 else 0

        result.total_pnl = (gross_profit - gross_loss)

        return result


def run_backtest(
    ltf_data: pd.DataFrame,
    htf_data: Optional[pd.DataFrame] = None,
    config: Optional[dict] = None,
    strategy_params: Optional[dict] = None,
) -> BacktestResult:
    """便利函式：用 config 建立引擎並執行回測。"""
    if config is None:
        config = {}

    bt_cfg = config.get("backtest", {})
    engine = BacktestEngine(
        commission=bt_cfg.get("commission", 22.0),
        slippage=bt_cfg.get("slippage", 1.0),
        size=bt_cfg.get("size", 1),
        point_value=bt_cfg.get("point_value", 10.0),
    )

    return engine.run(ltf_data, htf_data, strategy_params)
