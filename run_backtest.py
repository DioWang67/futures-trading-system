"""
主程式入口

用法：
    python run_backtest.py                 # 跑完整流程（資料 → 回測 → 優化 → WF 驗證 → 報告）
    python run_backtest.py --sample        # 用模擬資料跑
    python run_backtest.py --csv PATH      # 用本地 CSV 檔案（15min K 棒）
    python run_backtest.py --optimize-only # 只跑參數優化
    python run_backtest.py --wf-only       # 只跑 Walk-Forward 驗證

範例（使用 Back_Trader 現有資料）：
    python run_backtest.py --csv "E:/python_program/Back_Trader/data/TXF_15m_2020_2026_Merged.csv"
    python run_backtest.py --csv "E:/python_program/Back_Trader/data/MXF_15m_Extended.csv"
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from src.data.fetcher import load_config, fetch_and_cache, generate_sample_data, load_csv_data
from src.backtest.engine import run_backtest
from src.backtest.monte_carlo import monte_carlo_shuffle, monte_carlo_bootstrap, monte_carlo_noise
from src.backtest.optimizer import StrategyOptimizer
from src.backtest.walk_forward import WalkForwardValidator
from src.report.generator import (
    generate_backtest_report,
    generate_montecarlo_report,
    generate_walkforward_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="MX SMC+PA 回測系統")
    parser.add_argument("--sample", action="store_true", help="使用模擬資料")
    parser.add_argument("--csv", type=str, default=None,
                        help="15 分 K 棒 CSV 路徑（會自動重採樣產生 60 分 HTF）")
    parser.add_argument("--optimize-only", action="store_true", help="只跑參數優化")
    parser.add_argument("--wf-only", action="store_true", help="只跑 Walk-Forward 驗證")
    parser.add_argument("--mc-only", action="store_true", help="只跑 Monte Carlo 模擬")
    parser.add_argument("--no-mc", action="store_true", help="跳過 Monte Carlo 模擬")
    parser.add_argument("--mc-sims", type=int, default=1000, help="Monte Carlo 模擬次數 (預設 1000)")
    parser.add_argument("--config", default=None, help="設定檔路徑")
    args = parser.parse_args()

    # 讀取設定
    config = load_config(args.config)
    logger.info("設定載入完成")

    # ---- 取得資料 ----
    if args.csv:
        csv_path = args.csv
        logger.info(f"從 CSV 載入: {csv_path}")
        ltf_data = load_csv_data(csv_path)
        htf_data = load_csv_data(csv_path, resample_freq="60min")
        logger.info(f"LTF (15min): {len(ltf_data)} 筆")
        logger.info(f"HTF (60min): {len(htf_data)} 筆")
        logger.info(f"時間範圍: {ltf_data['datetime'].iloc[0]} ~ {ltf_data['datetime'].iloc[-1]}")

    elif args.sample:
        logger.info("使用模擬資料")
        ltf_data = generate_sample_data(n_bars=8000, freq="15T", seed=42)
        htf_data = generate_sample_data(n_bars=2000, freq="60T", seed=42)

    else:
        logger.info("從 Shioaji 拉取資料...")
        ltf_data = fetch_and_cache(config, freq="15T")
        htf_data = fetch_and_cache(config, freq="60T")

    if ltf_data.empty:
        logger.error("無法取得 LTF 資料，請檢查 API 設定或使用 --sample / --csv")
        sys.exit(1)

    logger.info(f"LTF 資料: {len(ltf_data)} 筆, HTF 資料: {len(htf_data)} 筆")

    # ---- 單次回測（使用預設參數） ----
    result = None
    if not args.optimize_only and not args.wf_only:
        logger.info("\n===== 使用預設參數回測 =====")
        strategy_cfg = config.get("strategy", {})
        result = run_backtest(ltf_data, htf_data, config, strategy_cfg)
        report_path = generate_backtest_report(result, "SMC+PA 預設參數回測")
        logger.info(f"回測報告: {report_path}")
        logger.info(
            f"結果: 交易數={result.total_trades} 勝率={result.win_rate:.2%} "
            f"PF={result.profit_factor:.2f} Sharpe={result.sharpe_ratio:.2f} "
            f"MDD={result.max_drawdown:.2%} R:R={result.avg_rr:.2f}"
        )

    # ---- Monte Carlo 模擬 ----
    if not args.no_mc and not args.optimize_only and not args.wf_only:
        if result and result.trade_log:
            trade_pnls = [t.get("pnl", 0.0) for t in result.trade_log]
            initial_capital = config.get("strategy", {}).get("initial_capital", 1_000_000)
            n_sims = args.mc_sims

            logger.info(f"\n===== Monte Carlo 模擬 ({n_sims:,} runs × 3 methods) =====")

            mc_shuffle = monte_carlo_shuffle(trade_pnls, initial_capital=initial_capital, n_simulations=n_sims, seed=42)
            mc_bootstrap = monte_carlo_bootstrap(trade_pnls, initial_capital=initial_capital, n_simulations=n_sims, seed=42)
            mc_noise = monte_carlo_noise(trade_pnls, initial_capital=initial_capital, n_simulations=n_sims, seed=42)

            for mc in [mc_shuffle, mc_bootstrap, mc_noise]:
                logger.info(f"\n{mc.summary()}")

            mc_report_path = generate_montecarlo_report(
                [mc_shuffle, mc_bootstrap, mc_noise],
                backtest_result=result,
                title="SMC+PA Monte Carlo 模擬報告",
            )
            logger.info(f"Monte Carlo 報告: {mc_report_path}")

            if args.mc_only:
                return
        else:
            logger.warning("無交易紀錄，跳過 Monte Carlo 模擬")
            if args.mc_only:
                return

    # ---- 參數優化 ----
    if not args.wf_only:
        logger.info("\n===== Optuna 參數優化 =====")
        optimizer = StrategyOptimizer(ltf_data, htf_data, config)
        best_params = optimizer.optimize()

        if best_params:
            logger.info(f"最佳參數: {best_params}")
            best_result = optimizer.get_best_result()
            if best_result:
                report_path = generate_backtest_report(best_result, "SMC+PA 優化後回測")
                logger.info(f"優化回測報告: {report_path}")

            if args.optimize_only:
                return

    # ---- Walk-Forward 驗證 ----
    logger.info("\n===== Walk-Forward 驗證 =====")
    validator = WalkForwardValidator(ltf_data, htf_data, config)
    wf_result = validator.validate()

    report_path = generate_walkforward_report(wf_result)
    logger.info(f"Walk-Forward 報告: {report_path}")

    if wf_result.all_passed:
        logger.info("Walk-Forward 驗證全部通過！策略可考慮實盤部署。")
    else:
        logger.warning(
            f"Walk-Forward 驗證未通過: {wf_result.pass_rate:.0%} 區段達標"
        )


if __name__ == "__main__":
    main()
