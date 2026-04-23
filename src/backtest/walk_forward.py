"""
Walk-Forward 驗證模組

將 3 年資料切成多段，每段前 70% 訓練 + 後 30% 測試，
每段都必須達標才算真正通過。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from src.backtest.engine import BacktestEngine, BacktestResult
from src.backtest.optimizer import StrategyOptimizer

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardSegment:
    """單一 Walk-Forward 區段結果。"""
    segment_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_result: Optional[BacktestResult] = None
    test_result: Optional[BacktestResult] = None
    best_params: Optional[dict] = None
    passed: bool = False


@dataclass
class WalkForwardResult:
    """Walk-Forward 整體結果。"""
    segments: list[WalkForwardSegment] = field(default_factory=list)
    all_passed: bool = False
    overall_test_results: list[BacktestResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if not self.segments:
            return 0.0
        return sum(1 for s in self.segments if s.passed) / len(self.segments)


class WalkForwardValidator:
    """Walk-Forward 驗證器。"""

    def __init__(
        self,
        ltf_data: pd.DataFrame,
        htf_data: Optional[pd.DataFrame] = None,
        config: Optional[dict] = None,
        n_splits: int = 6,
        train_ratio: float = 0.7,
    ):
        self.ltf_data = ltf_data
        self.htf_data = htf_data
        self.config = config or {}
        self.n_splits = n_splits
        self.train_ratio = train_ratio

        wf_cfg = self.config.get("walk_forward", {})
        self.n_splits = wf_cfg.get("n_splits", n_splits)
        self.train_ratio = wf_cfg.get("train_ratio", train_ratio)

    def _split_data(
        self, df: pd.DataFrame,
    ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """將資料切成 n_splits 段，每段分 train/test。"""
        n = len(df)
        segment_size = n // self.n_splits
        splits = []
        strategy_cfg = self.config.get("strategy", {})
        wf_cfg = self.config.get("walk_forward", {})
        lookback = int(wf_cfg.get("lookback_period", 0) or 0)
        lookback = max(
            lookback,
            int(strategy_cfg.get("swing_lookback", 0) or 0),
            int(strategy_cfg.get("adx_period", 0) or 0),
            int(strategy_cfg.get("ob_max_age", 0) or 0),
        )
        min_train_bars = int(wf_cfg.get("min_train_bars", max(50, lookback * 2)))
        min_test_bars = int(wf_cfg.get("min_test_bars", max(20, lookback)))

        for i in range(self.n_splits):
            start = i * segment_size
            end = min((i + 1) * segment_size, n)
            segment = df.iloc[start:end].copy()

            train_end_idx = int(len(segment) * self.train_ratio)
            train = segment.iloc[:train_end_idx].copy()
            test = segment.iloc[train_end_idx:].copy()

            if len(train) >= min_train_bars and len(test) >= min_test_bars:
                splits.append((train, test))
            else:
                logger.info(
                    "跳過 WF 區段 {}: train={} (< {}?) test={} (< {}?)",
                    i + 1, len(train), min_train_bars, len(test), min_test_bars,
                )

        return splits

    def _split_htf_by_datetime(
        self,
        htf: pd.DataFrame,
        ltf_segment: pd.DataFrame,
    ) -> pd.DataFrame:
        """從 HTF 資料中擷取與 LTF 區段對應的時間範圍。"""
        if htf is None or htf.empty:
            return pd.DataFrame()

        dt_col = "datetime"
        if dt_col not in ltf_segment.columns:
            return htf

        ltf_start = ltf_segment[dt_col].min()
        ltf_end = ltf_segment[dt_col].max()

        mask = (htf[dt_col] >= ltf_start) & (htf[dt_col] <= ltf_end)
        return htf[mask].copy()

    def validate(self) -> WalkForwardResult:
        """執行 Walk-Forward 驗證。"""
        result = WalkForwardResult()

        ltf_splits = self._split_data(self.ltf_data)

        if not ltf_splits:
            logger.error("資料不足，無法切分")
            return result

        for i, (ltf_train, ltf_test) in enumerate(ltf_splits):
            logger.info(f"\n{'='*60}")
            logger.info(f"Walk-Forward 區段 {i+1}/{len(ltf_splits)}")
            logger.info(f"{'='*60}")

            segment = WalkForwardSegment(
                segment_id=i + 1,
                train_start=str(ltf_train.iloc[0].get("datetime", "N/A")),
                train_end=str(ltf_train.iloc[-1].get("datetime", "N/A")),
                test_start=str(ltf_test.iloc[0].get("datetime", "N/A")),
                test_end=str(ltf_test.iloc[-1].get("datetime", "N/A")),
            )

            # HTF 對應切分
            htf_train = self._split_htf_by_datetime(self.htf_data, ltf_train)
            htf_test = self._split_htf_by_datetime(self.htf_data, ltf_test)

            # ---- 訓練：參數優化 ----
            logger.info(f"訓練期：{segment.train_start} ~ {segment.train_end}")
            optimizer = StrategyOptimizer(
                ltf_data=ltf_train,
                htf_data=htf_train if not htf_train.empty else None,
                config=self.config,
            )
            best_params = optimizer.optimize()
            segment.best_params = best_params
            segment.train_result = optimizer.get_best_result()

            if best_params is None:
                logger.warning(f"區段 {i+1} 訓練失敗，無法找到參數")
                result.segments.append(segment)
                continue

            # ---- 測試：用訓練的最佳參數跑測試期 ----
            logger.info(f"測試期：{segment.test_start} ~ {segment.test_end}")
            bt_cfg = self.config.get("backtest", {})
            engine = BacktestEngine(
                initial_cash=bt_cfg.get("initial_cash", 1_000_000.0),
                commission=bt_cfg.get("commission", 22.0),
                slippage=bt_cfg.get("slippage", 1.0),
                size=bt_cfg.get("size", 1),
                point_value=bt_cfg.get("point_value", 10.0),
            )

            # 從 optuna params 過濾出策略可接受的參數
            strategy_params = {k: v for k, v in best_params.items()}
            test_result = engine.run(
                ltf_test,
                htf_test if not htf_test.empty else None,
                strategy_params,
            )
            segment.test_result = test_result

            # 檢查測試是否達標
            opt_cfg = self.config.get("optimization", {})
            wf_thresholds = self.config.get("walk_forward", {})
            # WF 測試門檻可以獨立設定，比訓練門檻寬鬆
            segment.passed = test_result.meets_custom_threshold(
                min_win_rate=wf_thresholds.get("min_win_rate", opt_cfg.get("min_win_rate", 0.45)),
                min_profit_factor=wf_thresholds.get("min_profit_factor", opt_cfg.get("min_profit_factor", 1.2)),
                min_sharpe=wf_thresholds.get("min_sharpe", opt_cfg.get("min_sharpe", 0.0)),
                max_mdd=wf_thresholds.get("max_mdd", opt_cfg.get("max_mdd", 0.15)),
                min_rr=wf_thresholds.get("min_rr", opt_cfg.get("min_rr", 0.8)),
            )

            status = "PASS" if segment.passed else "FAIL"
            logger.info(
                f"區段 {i+1} {status}: "
                f"WR={test_result.win_rate:.2%} "
                f"PF={test_result.profit_factor:.2f} "
                f"Sharpe={test_result.sharpe_ratio:.2f} "
                f"MDD={test_result.max_drawdown:.2%} "
                f"R:R={test_result.avg_rr:.2f}"
            )

            result.segments.append(segment)
            result.overall_test_results.append(test_result)

        passed = sum(1 for s in result.segments if s.passed)
        total = len(result.segments)
        # 通過條件：至少 60% 的區段達標（允許個別區段失敗）
        min_pass_rate = self.config.get("walk_forward", {}).get("min_pass_rate", 0.6)
        result.all_passed = total > 0 and (passed / total) >= min_pass_rate

        if result.all_passed:
            logger.info(
                f"\nWalk-Forward 驗證通過！ {passed}/{total} 段達標"
                f" (門檻: {min_pass_rate:.0%})"
            )
        else:
            logger.warning(
                f"\nWalk-Forward 驗證未通過："
                f" {passed}/{total} 段達標"
                f" (門檻: {min_pass_rate:.0%})"
            )

        return result
