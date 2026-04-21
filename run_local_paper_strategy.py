from __future__ import annotations

import argparse
import asyncio
import sys

from loguru import logger

from brokers import ShioajiBroker
from config import settings
from notifier import Notifier
from risk_manager import RiskConfig, RiskManager
from src.data.fetcher import load_config
from src.live.paper_strategy import LocalPaperStrategyRunner
from trade_store import TradeStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the local SMC+PA strategy against Shioaji paper trading"
    )
    parser.add_argument("--config", default=None, help="Path to config/settings.yaml")
    parser.add_argument("--symbol", default=None, help="Ticker to trade, default from config")
    parser.add_argument("--quantity", type=int, default=None, help="Contracts per entry")
    parser.add_argument("--ltf", default="15min", help="Low timeframe resample frequency")
    parser.add_argument("--htf", default="60min", help="High timeframe resample frequency")
    parser.add_argument("--lookback-days", type=int, default=10, help="Startup history window in days for the initial 1-minute fetch")
    parser.add_argument("--poll-seconds", type=int, default=30, help="Fixed polling interval when --fixed-poll is enabled")
    parser.add_argument("--close-grace-seconds", type=int, default=3, help="Seconds to wait after each LTF bar close before checking")
    parser.add_argument("--fixed-poll", action="store_true", help="Use a fixed poll interval instead of aligning checks to bar closes")
    parser.add_argument("--state-path", default=".tmp/local_paper_state.json", help="Runner state file")
    parser.add_argument("--once", action="store_true", help="Run one strategy cycle and exit")
    parser.add_argument("--allow-live", action="store_true", help="Bypass the safety check that requires simulation mode")
    return parser


async def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    symbol = str(
        args.symbol
        or config.get("instrument", {}).get("symbol", settings.shioaji.futures_symbol)
    ).upper()
    quantity = int(args.quantity or config.get("backtest", {}).get("size", 1) or 1)

    if not settings.shioaji.simulation and not args.allow_live:
        logger.error(
            "Refusing to start local strategy against a live broker. "
            "Set SHIOAJI_SIMULATION=true or pass --allow-live explicitly."
        )
        return 2

    settings.require_shioaji()

    risk_cfg = RiskConfig(
        max_daily_loss=settings.risk.max_daily_loss,
        max_daily_loss_pct=settings.risk.max_daily_loss_pct,
        max_position_size=settings.risk.max_position_size,
        max_order_qty=settings.risk.max_order_qty,
        max_drawdown_pct=settings.risk.max_drawdown_pct,
        cooldown_after_consecutive_losses=settings.risk.cooldown_after_consecutive_losses,
        cooldown_seconds=settings.risk.cooldown_seconds,
        initial_capital=settings.risk.initial_capital,
    )
    risk_manager = RiskManager(risk_cfg)
    trade_store = TradeStore()
    notifier = Notifier(
        telegram_token=settings.notification.telegram_bot_token.get_secret_value(),
        telegram_chat_id=settings.notification.telegram_chat_id,
        line_notify_token=settings.notification.line_notify_token.get_secret_value(),
    )

    broker = ShioajiBroker(
        risk_manager=risk_manager,
        trade_store=trade_store,
        notifier=notifier,
    )
    broker.attach_event_loop(asyncio.get_running_loop())
    await asyncio.to_thread(broker.login)

    runner = LocalPaperStrategyRunner(
        broker=broker,
        symbol=symbol,
        quantity=quantity,
        strategy_params=config.get("strategy", {}),
        ltf_freq=args.ltf,
        htf_freq=args.htf,
        lookback_days=args.lookback_days,
        poll_seconds=args.poll_seconds,
        close_grace_seconds=args.close_grace_seconds,
        align_to_bar_close=not args.fixed_poll,
        state_path=args.state_path,
    )

    logger.info(
        "Local paper strategy ready: symbol={} quantity={} mode={}",
        symbol,
        quantity,
        "simulation" if settings.shioaji.simulation else "live",
    )
    await notifier.send_system(
        "Local paper strategy started: "
        f"{symbol} x{quantity} "
        f"({'simulation' if settings.shioaji.simulation else 'live'})"
    )
    watchdog_task = asyncio.create_task(
        broker.run_watchdog(),
        name="shioaji-protective-watchdog",
    )

    try:
        if args.once:
            result = await runner.cycle()
            logger.info("Single cycle result: {}", result)
            return 0
        await runner.run_forever()
        return 0
    finally:
        watchdog_task.cancel()
        try:
            await watchdog_task
        except asyncio.CancelledError:
            pass
        await asyncio.to_thread(broker.logout)
        await notifier.send_system(f"Local paper strategy stopped: {symbol}")


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
