"""
Main backtest runner.

Usage:
    python -m backtest.run --data data/NQ_5min.csv
    python -m backtest.run --data data/NQ_5min.csv --exit-mode "固定止盈 (SL+TP)"
    python -m backtest.run --data data/NQ_5min.csv --export results.csv

CSV format expected:
    datetime, open, high, low, close, volume
    2024-01-02 09:30:00, 16800.25, 16810.50, 16795.00, 16805.75, 1234

datetime column should be in America/New_York timezone (or UTC — will be converted).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.config import StrategyConfig
from backtest.engine import BacktestEngine
from backtest.indicators import compute_all
from backtest.signals import compute_scores


# ============================================================================
# Data loading
# ============================================================================

def load_csv(path: str, tz: str = "America/New_York") -> pd.DataFrame:
    """Load OHLCV CSV into a timezone-aware DatetimeIndex DataFrame."""
    df = pd.read_csv(path, parse_dates=[0], index_col=0)

    # Normalize column names
    col_map = {}
    for col in df.columns:
        cl = col.strip().lower()
        if cl in ("open", "high", "low", "close", "volume"):
            col_map[col] = cl
    df = df.rename(columns=col_map)

    required = ["open", "high", "low", "close", "volume"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    df.index.name = "datetime"
    df = df.sort_index()

    # Ensure timezone
    if df.index.tz is None:
        df.index = df.index.tz_localize(tz)
    else:
        df.index = df.index.tz_convert(tz)

    # Drop duplicates and NaN rows
    df = df[~df.index.duplicated(keep="first")]
    df = df.dropna(subset=required)

    return df


# ============================================================================
# Reporting
# ============================================================================

def print_report(trades: list, equity: list[float], cfg: StrategyConfig):
    """Print a summary report comparable to TradingView Strategy Tester."""
    if not trades:
        print("\n  No trades executed.\n")
        return

    df_trades = pd.DataFrame([
        {
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "direction": t.direction,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "qty": t.quantity,
            "pnl": t.pnl,
            "commission": t.commission,
            "net_pnl": t.net_pnl,
            "exit_reason": t.exit_reason,
        }
        for t in trades
    ])

    total_trades = len(df_trades)
    winners = df_trades[df_trades["net_pnl"] > 0]
    losers = df_trades[df_trades["net_pnl"] < 0]
    win_count = len(winners)
    loss_count = len(losers)
    win_rate = win_count / total_trades * 100 if total_trades else 0

    gross_profit = winners["net_pnl"].sum() if len(winners) else 0
    gross_loss = losers["net_pnl"].sum() if len(losers) else 0
    net_profit = df_trades["net_pnl"].sum()
    total_commission = df_trades["commission"].sum()

    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float("inf")

    avg_win = winners["net_pnl"].mean() if len(winners) else 0
    avg_loss = losers["net_pnl"].mean() if len(losers) else 0
    largest_win = winners["net_pnl"].max() if len(winners) else 0
    largest_loss = losers["net_pnl"].min() if len(losers) else 0

    # Max drawdown from equity curve
    equity_arr = np.array(equity)
    peak = np.maximum.accumulate(equity_arr)
    drawdown = peak - equity_arr
    max_dd = drawdown.max()
    max_dd_pct = (max_dd / peak[np.argmax(drawdown)]) * 100 if peak[np.argmax(drawdown)] > 0 else 0

    # Long / Short breakdown
    longs = df_trades[df_trades["direction"] == "long"]
    shorts = df_trades[df_trades["direction"] == "short"]

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  SMC x GainzAlgo v4.9 — Python Backtest Report")
    print(f"  Exit Mode: {cfg.exit_mode}")
    print(f"{sep}")
    print(f"  Initial Capital:     ${cfg.initial_capital:,.2f}")
    print(f"  Final Equity:        ${equity[-1]:,.2f}")
    print(f"  Net Profit:          ${net_profit:,.2f}")
    print(f"  Total Commission:    ${total_commission:,.2f}")
    print(f"  Profit Factor:       {profit_factor:.2f}")
    print(f"  Max Drawdown:        ${max_dd:,.2f} ({max_dd_pct:.1f}%)")
    print(f"{'-' * 60}")
    print(f"  Total Trades:        {total_trades}")
    print(f"  Winners:             {win_count} ({win_rate:.1f}%)")
    print(f"  Losers:              {loss_count} ({100 - win_rate:.1f}%)")
    print(f"  Avg Win:             ${avg_win:,.2f}")
    print(f"  Avg Loss:            ${avg_loss:,.2f}")
    print(f"  Largest Win:         ${largest_win:,.2f}")
    print(f"  Largest Loss:        ${largest_loss:,.2f}")
    print(f"{'-' * 60}")
    print(f"  Long Trades:         {len(longs)}  (PnL: ${longs['net_pnl'].sum():,.2f})")
    print(f"  Short Trades:        {len(shorts)}  (PnL: ${shorts['net_pnl'].sum():,.2f})")
    print(f"{sep}")

    # Exit reason breakdown
    print(f"\n  Exit Reasons:")
    for reason, group in df_trades.groupby("exit_reason"):
        print(f"    {reason:20s}  count={len(group):3d}  PnL=${group['net_pnl'].sum():>10,.2f}")

    # Last 10 trades
    print(f"\n  Last 10 Trades:")
    print(f"  {'Entry Time':20s} {'Dir':5s} {'Entry':>10s} {'Exit':>10s} {'Reason':15s} {'Net PnL':>10s}")
    print(f"  {'-'*75}")
    for _, r in df_trades.tail(10).iterrows():
        et = r["entry_time"].strftime("%Y-%m-%d %H:%M") if r["entry_time"] else ""
        print(f"  {et:20s} {r['direction']:5s} {r['entry_price']:10.2f} {r['exit_price']:10.2f} "
              f"{r['exit_reason']:15s} {r['net_pnl']:10.2f}")
    print()

    return df_trades


def plot_equity(equity: list[float], df_index: pd.DatetimeIndex):
    """Plot equity curve (optional — requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [matplotlib not installed — skipping equity plot]")
        return

    # Align equity to the bars that were actually simulated
    # equity starts from struct_lookback+1
    idx = df_index[-len(equity):]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(idx, equity, linewidth=1.0, color="#00BFFF")
    ax.set_title("Equity Curve — SMC v4.9 Python Backtest")
    ax.set_ylabel("Equity ($)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig("equity_curve.png", dpi=150)
    print("  Equity curve saved to equity_curve.png")
    plt.show()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="SMC v4.9 Python Backtest")
    parser.add_argument("--data", required=True, help="Path to OHLCV CSV file")
    parser.add_argument("--tz", default="America/New_York", help="Timezone of CSV data")
    parser.add_argument("--exit-mode", default=None,
                        help="Exit mode override: '固定止盈 (SL+TP)' | '移動止盈 (SL+Trailing)' | '分批出場 (1:1 + 2.5R)'")
    parser.add_argument("--trade-mode", default=None,
                        help="Trade mode: '多空皆可' | '只做多' | '只做空'")
    parser.add_argument("--qty", type=int, default=None, help="Override default quantity")
    parser.add_argument("--point-value", type=float, default=None,
                        help="Point value (NQ=20, MNQ=2, MES=5, MX=10)")
    parser.add_argument("--max-sl-points", type=float, default=None,
                        help="Max SL distance in points (0=disable filter, default=100)")
    parser.add_argument("--tick-size", type=float, default=None,
                        help="Tick size (NQ/MNQ=0.25, MX=1)")
    parser.add_argument("--capital", type=float, default=None,
                        help="Initial capital")
    parser.add_argument("--export", default=None, help="Export trades to CSV")
    parser.add_argument("--no-plot", action="store_true", help="Skip equity plot")
    args = parser.parse_args()

    # ── Config ──
    cfg = StrategyConfig()
    if args.exit_mode:
        cfg.exit_mode = args.exit_mode
    if args.trade_mode:
        cfg.trade_mode = args.trade_mode
    if args.qty:
        cfg.default_qty = args.qty
    if args.point_value:
        cfg.point_value = args.point_value
    if args.max_sl_points is not None:
        cfg.max_sl_points = args.max_sl_points
    if args.tick_size is not None:
        cfg.tick_size = args.tick_size
    if args.capital is not None:
        cfg.initial_capital = args.capital

    # ── Load data ──
    print(f"Loading data from {args.data} ...")
    df = load_csv(args.data, tz=args.tz)
    print(f"  Loaded {len(df)} bars  [{df.index[0]} → {df.index[-1]}]")

    # ── Compute indicators ──
    print("Computing indicators ...")
    df = compute_all(df, cfg)

    # ── Compute scores ──
    print("Computing signals & scores ...")
    df = compute_scores(df, cfg)

    # Signal summary
    buy_signals = df["smart_buy"].sum()
    sell_signals = df["smart_sell"].sum()
    print(f"  Raw smart_buy signals:  {buy_signals}")
    print(f"  Raw smart_sell signals: {sell_signals}")

    # ── Run engine ──
    print("Running backtest engine ...")
    engine = BacktestEngine(df, cfg)
    trades = engine.run()
    print(f"  Completed: {len(trades)} trades")

    # ── Report ──
    df_trades = print_report(trades, engine.equity_curve, cfg)

    # ── Export ──
    if args.export and df_trades is not None:
        df_trades.to_csv(args.export, index=False)
        print(f"  Trades exported to {args.export}")

    # ── Plot ──
    if not args.no_plot and engine.equity_curve:
        plot_equity(engine.equity_curve, df.index)


if __name__ == "__main__":
    main()
