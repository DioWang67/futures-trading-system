"""
Optuna 參數優化模組

自動搜尋最佳策略參數，
同時達到勝率、獲利因子、夏普、MDD、R:R 門檻才算達標。
"""

from __future__ import annotations

import logging
from typing import Optional

import optuna
import pandas as pd

from src.backtest.engine import BacktestEngine, BacktestResult

logger = logging.getLogger(__name__)


class StrategyOptimizer:
    """使用 Optuna 優化 SMC+PA 策略參數。"""

    def __init__(
        self,
        ltf_data: pd.DataFrame,
        htf_data: Optional[pd.DataFrame] = None,
        config: Optional[dict] = None,
    ):
        self.ltf_data = ltf_data
        self.htf_data = htf_data
        self.config = config or {}

        bt_cfg = self.config.get("backtest", {})
        self.engine = BacktestEngine(
            commission=bt_cfg.get("commission", 22.0),
            slippage=bt_cfg.get("slippage", 1.0),
            size=bt_cfg.get("size", 1),
            point_value=bt_cfg.get("point_value", 10.0),
        )

        opt_cfg = self.config.get("optimization", {})
        self.min_win_rate = opt_cfg.get("min_win_rate", 0.55)
        self.min_profit_factor = opt_cfg.get("min_profit_factor", 1.5)
        self.min_sharpe = opt_cfg.get("min_sharpe", 1.2)
        self.max_mdd = opt_cfg.get("max_mdd", 0.15)
        self.min_rr = opt_cfg.get("min_rr", 1.5)
        self.n_trials = opt_cfg.get("n_trials", 500)
        self.timeout = opt_cfg.get("timeout", 7200)

        self.best_result: Optional[BacktestResult] = None
        self.best_params: Optional[dict] = None

    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna 目標函式。

        回傳 composite score，越高越好。
        如果達標，score > 0；不達標，score <= 0。
        """
        params = {
            "swing_lookback": trial.suggest_int("swing_lookback", 3, 10),
            "bos_min_move": trial.suggest_float("bos_min_move", 5.0, 50.0, step=5.0),
            "ob_max_age": trial.suggest_int("ob_max_age", 10, 40),
            "ob_body_ratio": trial.suggest_float("ob_body_ratio", 0.2, 0.7, step=0.05),
            "fvg_min_gap": trial.suggest_float("fvg_min_gap", 2.0, 15.0, step=1.0),
            "fvg_enabled": trial.suggest_categorical("fvg_enabled", [True, False]),
            "pin_bar_ratio": trial.suggest_float("pin_bar_ratio", 0.4, 0.8, step=0.05),
            "engulf_ratio": trial.suggest_float("engulf_ratio", 0.8, 1.5, step=0.1),
            "rr_ratio": trial.suggest_float("rr_ratio", 1.0, 3.0, step=0.25),
            "use_structure_tp": trial.suggest_categorical("use_structure_tp", [True, False]),
            "sl_buffer": trial.suggest_float("sl_buffer", 1.0, 5.0, step=0.5),
            "pa_confirm": trial.suggest_categorical("pa_confirm", [True, False]),
            # ADX trend filter
            "adx_period": trial.suggest_int("adx_period", 10, 28, step=2),
            "adx_threshold": trial.suggest_float("adx_threshold", 15.0, 35.0, step=2.5),
            "adx_filter_enabled": trial.suggest_categorical("adx_filter_enabled", [True, False]),
        }

        result = self.engine.run(self.ltf_data, self.htf_data, params)

        if result.total_trades < 5:
            return -100.0

        # Composite score: 各指標與門檻的差距加權總和
        score = 0.0
        score += (result.win_rate - self.min_win_rate) * 100
        score += (result.profit_factor - self.min_profit_factor) * 20
        score += (result.sharpe_ratio - self.min_sharpe) * 30
        score += (self.max_mdd - result.max_drawdown) * 200
        score += (result.avg_rr - self.min_rr) * 20
        # 交易數量獎勵：鼓勵更多交易以提高統計有效性
        score += min(result.total_trades, 100) * 0.5

        # 檢查是否達標
        if result.meets_custom_threshold(
            self.min_win_rate, self.min_profit_factor,
            self.min_sharpe, self.max_mdd, self.min_rr,
        ):
            logger.info(
                f"Trial {trial.number} 達標！"
                f" WR={result.win_rate:.2%} PF={result.profit_factor:.2f}"
                f" Sharpe={result.sharpe_ratio:.2f} MDD={result.max_drawdown:.2%}"
                f" R:R={result.avg_rr:.2f}"
            )
            # 保存最佳達標結果
            if self.best_result is None or score > self._best_score:
                self.best_result = result
                self.best_params = params
                self._best_score = score

        return score

    def optimize(self) -> Optional[dict]:
        """執行參數優化。

        Returns
        -------
        dict or None
            達標的最佳參數，如果沒達標回傳 None
        """
        self._best_score = float("-inf")

        study = optuna.create_study(
            direction="maximize",
            study_name="smc_pa_optimization",
        )

        # 設定日誌等級
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study.optimize(
            self._objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True,
        )

        if self.best_params is not None:
            logger.info(f"最佳參數: {self.best_params}")
            return self.best_params

        # 沒達標，回傳最佳嘗試的參數
        if study.best_trial is not None:
            logger.warning("未達標，回傳最佳嘗試參數")
            return study.best_trial.params

        return None

    def get_best_result(self) -> Optional[BacktestResult]:
        """取得最佳回測結果。"""
        return self.best_result
