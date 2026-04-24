"""
Optuna 參數優化模組

自動搜尋最佳策略參數，
同時達到勝率、獲利因子、夏普、MDD、R:R 門檻才算達標。
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import optuna
import pandas as pd

from src.backtest.engine import BacktestEngine, BacktestResult

logger = logging.getLogger(__name__)


PARAM_SPECS: dict[str, dict[str, Any]] = {
    "swing_lookback": {"kind": "int", "low": 3, "high": 10},
    "bos_min_move": {"kind": "float", "low": 5.0, "high": 50.0, "step": 5.0},
    "ob_max_age": {"kind": "int", "low": 10, "high": 40},
    "ob_body_ratio": {"kind": "float", "low": 0.2, "high": 0.7, "step": 0.05},
    "fvg_min_gap": {"kind": "float", "low": 2.0, "high": 15.0, "step": 1.0},
    "fvg_enabled": {"kind": "categorical", "choices": [True, False]},
    "pin_bar_ratio": {"kind": "float", "low": 0.4, "high": 0.8, "step": 0.05},
    "engulf_ratio": {"kind": "float", "low": 0.8, "high": 1.5, "step": 0.1},
    "rr_ratio": {"kind": "float", "low": 1.0, "high": 3.0, "step": 0.25},
    "use_structure_tp": {"kind": "categorical", "choices": [True, False]},
    "sl_buffer": {"kind": "float", "low": 1.0, "high": 5.0, "step": 0.5},
    "pa_confirm": {"kind": "categorical", "choices": [True, False]},
    "adx_period": {"kind": "int", "low": 10, "high": 28, "step": 2},
    "adx_threshold": {"kind": "float", "low": 15.0, "high": 35.0, "step": 2.5},
    "adx_filter_enabled": {"kind": "categorical", "choices": [True, False]},
    "atr_filter_enabled": {"kind": "categorical", "choices": [True, False]},
    "atr_period": {"kind": "int", "low": 10, "high": 28, "step": 2},
    "atr_min_points": {"kind": "float", "low": 5.0, "high": 60.0, "step": 5.0},
}


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
            initial_cash=bt_cfg.get("initial_cash", 1_000_000.0),
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
        self.base_strategy_params = dict(self.config.get("strategy", {}))
        self.fixed_params = dict(opt_cfg.get("fixed_params", {}))
        enabled_params = opt_cfg.get("enabled_params")
        self.enabled_params = set(enabled_params) if enabled_params else None
        overlap = set(self.fixed_params).intersection(self.enabled_params or set())
        if overlap:
            overlap_keys = ", ".join(sorted(overlap))
            raise ValueError(
                "fixed_params 與 enabled_params 不可重複鍵: "
                f"{overlap_keys}"
            )

        self.score_weights = {
            "win_rate": opt_cfg.get("score_weights", {}).get("win_rate", 100.0),
            "profit_factor": opt_cfg.get("score_weights", {}).get("profit_factor", 20.0),
            "sharpe": opt_cfg.get("score_weights", {}).get("sharpe", 30.0),
            "max_drawdown": opt_cfg.get("score_weights", {}).get("max_drawdown", 200.0),
            "avg_rr": opt_cfg.get("score_weights", {}).get("avg_rr", 20.0),
            "trades": opt_cfg.get("score_weights", {}).get("trades", 0.5),
        }
        self.profit_factor_cap = opt_cfg.get("profit_factor_cap", 5.0)

        self.best_result: Optional[BacktestResult] = None
        self.best_params: Optional[dict] = None

    def _suggest_param(self, trial: optuna.Trial, name: str):
        spec = PARAM_SPECS[name]
        if spec["kind"] == "int":
            return trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step", 1))
        if spec["kind"] == "float":
            return trial.suggest_float(name, spec["low"], spec["high"], step=spec.get("step"))
        if spec["kind"] == "categorical":
            return trial.suggest_categorical(name, spec["choices"])
        raise ValueError(f"Unsupported parameter spec for {name}: {spec}")

    def _build_trial_params(self, trial: optuna.Trial) -> dict:
        params = dict(self.base_strategy_params)
        params.update(self.fixed_params)

        for name in PARAM_SPECS:
            if name in self.fixed_params:
                continue
            if self.enabled_params is None or name in self.enabled_params:
                params[name] = self._suggest_param(trial, name)

        return params

    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna 目標函式。

        回傳 composite score，越高越好。
        如果達標，score > 0；不達標，score <= 0。
        """
        params = self._build_trial_params(trial)

        result = self.engine.run(self.ltf_data, self.htf_data, params)

        if result.total_trades < 5:
            return -100.0

        # Composite score: 各指標與門檻的差距加權總和
        pf_for_score = min(result.profit_factor, self.profit_factor_cap)
        score = 0.0
        score += (result.win_rate - self.min_win_rate) * self.score_weights["win_rate"]
        score += (pf_for_score - self.min_profit_factor) * self.score_weights["profit_factor"]
        score += (result.sharpe_ratio - self.min_sharpe) * self.score_weights["sharpe"]
        score += (self.max_mdd - result.max_drawdown) * self.score_weights["max_drawdown"]
        score += (result.avg_rr - self.min_rr) * self.score_weights["avg_rr"]
        # 交易數量獎勵：鼓勵更多交易以提高統計有效性
        score += min(result.total_trades, 100) * self.score_weights["trades"]

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
            merged_params = dict(self.base_strategy_params)
            merged_params.update(self.fixed_params)
            merged_params.update(study.best_trial.params)
            return merged_params

        return None

    def get_best_result(self) -> Optional[BacktestResult]:
        """取得最佳回測結果。"""
        return self.best_result
