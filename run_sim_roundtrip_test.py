from __future__ import annotations

import argparse
import asyncio
from typing import Any

from loguru import logger

from brokers import ShioajiBroker
from config import settings
from notifier import Notifier
from risk_manager import RiskConfig, RiskManager
from trade_store import TradeStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fast end-to-end simulation roundtrip for TMF without waiting for strategy signals"
    )
    parser.add_argument(
        "--side",
        choices=("long", "short"),
        default="long",
        help="Direction for the temporary test entry",
    )
    parser.add_argument("--quantity", type=int, default=1, help="Contracts to open")
    parser.add_argument(
        "--stop-loss-points",
        type=float,
        default=20.0,
        help="Distance from entry price to the protective stop",
    )
    parser.add_argument(
        "--take-profit-points",
        type=float,
        default=40.0,
        help="Distance from entry price to the protective take profit",
    )
    parser.add_argument(
        "--trigger-after",
        type=float,
        default=5.0,
        help="Seconds to wait before manually triggering the protective exit",
    )
    parser.add_argument(
        "--trigger-reason",
        choices=("stop_loss", "take_profit", "manual_test"),
        default="stop_loss",
        help="Reason label to use for the manual protective trigger",
    )
    parser.add_argument(
        "--quote-timeout",
        type=float,
        default=15.0,
        help="Seconds to wait for a fresh quote before aborting",
    )
    parser.add_argument(
        "--fill-timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for the open/close fills to arrive",
    )
    parser.add_argument(
        "--leave-open",
        action="store_true",
        help="Only open and arm protection; do not auto-trigger the protective exit",
    )
    parser.add_argument(
        "--allow-live",
        action="store_true",
        help="Bypass the safety check that requires simulation mode",
    )
    return parser


def _build_risk_manager() -> RiskManager:
    return RiskManager(
        RiskConfig(
            max_daily_loss=settings.risk.max_daily_loss,
            max_daily_loss_pct=settings.risk.max_daily_loss_pct,
            max_position_size=settings.risk.max_position_size,
            max_order_qty=settings.risk.max_order_qty,
            max_drawdown_pct=settings.risk.max_drawdown_pct,
            cooldown_after_consecutive_losses=settings.risk.cooldown_after_consecutive_losses,
            cooldown_seconds=settings.risk.cooldown_seconds,
            initial_capital=settings.risk.initial_capital,
        )
    )


async def _wait_for_position(
    broker: ShioajiBroker,
    expected_side: str,
    expected_qty: int,
    timeout_seconds: float,
) -> bool:
    deadline = asyncio.get_running_loop().time() + max(0.1, timeout_seconds)
    while asyncio.get_running_loop().time() < deadline:
        pos = broker.position.get_position()
        if pos.side == expected_side and pos.quantity >= expected_qty:
            return True
        await asyncio.sleep(0.2)
    return False


async def _wait_for_flat(
    broker: ShioajiBroker,
    timeout_seconds: float,
) -> bool:
    deadline = asyncio.get_running_loop().time() + max(0.1, timeout_seconds)
    while asyncio.get_running_loop().time() < deadline:
        pos = broker.position.get_position()
        if pos.side == "flat" or pos.quantity == 0:
            return True
        await asyncio.sleep(0.2)
    return False


async def main() -> int:
    args = build_parser().parse_args()

    if not settings.shioaji.simulation and not args.allow_live:
        logger.error(
            "Refusing to run the roundtrip test against a live broker. "
            "Set SHIOAJI_SIMULATION=true or pass --allow-live explicitly."
        )
        return 2

    settings.require_shioaji()
    trade_store = TradeStore()
    notifier = Notifier(
        telegram_token=settings.notification.telegram_bot_token.get_secret_value(),
        telegram_chat_id=settings.notification.telegram_chat_id,
    )
    broker = ShioajiBroker(
        risk_manager=_build_risk_manager(),
        trade_store=trade_store,
        notifier=notifier,
    )
    broker.attach_event_loop(asyncio.get_running_loop())

    await asyncio.to_thread(broker.login)
    try:
        quote = await broker.wait_for_fresh_quote(timeout_seconds=args.quote_timeout)
        last_price = float(quote.get("last_tick_price") or 0.0)
        if quote.get("is_stale") or last_price <= 0:
            logger.error("No fresh TMF quote received in time: {}", quote)
            return 3

        action = "buy" if args.side == "long" else "sell"
        result = await broker.place_order(
            action,
            int(args.quantity),
            ticker=broker.futures_symbol,
            sentiment=args.side,
            idempotency_key=f"manual-roundtrip-{args.side}-{int(asyncio.get_running_loop().time())}",
        )
        logger.info("Open order result: {}", result)
        filled = await _wait_for_position(
            broker,
            args.side,
            int(args.quantity),
            timeout_seconds=args.fill_timeout,
        )
        if not filled:
            logger.error(
                "Test entry did not fill within {}s; current position={}",
                args.fill_timeout,
                broker.position.get_position(),
            )
            return 4

        position = broker.position.get_position()
        entry_price = float(position.entry_price or last_price)
        if args.side == "long":
            stop_loss = entry_price - float(args.stop_loss_points)
            take_profit = entry_price + float(args.take_profit_points)
        else:
            stop_loss = entry_price + float(args.stop_loss_points)
            take_profit = entry_price - float(args.take_profit_points)

        broker.arm_protective_exit(
            broker.futures_symbol,
            args.side,
            position.quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        logger.info(
            "Protective exit armed for {} x{} @ {:.1f} SL {:.1f} TP {:.1f}",
            args.side,
            position.quantity,
            entry_price,
            stop_loss,
            take_profit,
        )

        if args.leave_open:
            logger.warning(
                "Leaving the simulation position open by request. "
                "Protective exit remains application-side and requires this process to stay alive."
            )
            while True:
                await asyncio.sleep(3600)

        await asyncio.sleep(max(0.0, float(args.trigger_after)))
        quote = await broker.wait_for_fresh_quote(timeout_seconds=3.0)
        trigger_price = float(quote.get("last_tick_price") or entry_price)
        trigger = await broker.trigger_protective_exit(
            broker.futures_symbol,
            reason=args.trigger_reason,
            trigger_price=trigger_price,
        )
        logger.info("Manual protective trigger result: {}", trigger)

        closed = await _wait_for_flat(broker, timeout_seconds=args.fill_timeout)
        if not closed:
            logger.error(
                "Protective exit did not flatten the position within {}s; current position={}",
                args.fill_timeout,
                broker.position.get_position(),
            )
            return 5

        recent_events = trade_store.get_recent_protective_events(limit=3)
        recent_fills = trade_store.get_recent_fills(limit=4)
        logger.info("Recent protective events: {}", recent_events)
        logger.info("Recent fills: {}", recent_fills)
        return 0
    finally:
        await asyncio.to_thread(broker.logout)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
