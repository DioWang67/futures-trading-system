"""
Notification module — Telegram + LINE Notify.

Sends alerts for:
  - Trade executions (fills)
  - Risk events (daily loss limit, drawdown halt, cooldown)
  - System events (broker disconnect, reconnect, startup/shutdown)
  - Error alerts

Usage:
    notifier = Notifier()
    await notifier.send_fill("shioaji", "Buy", 1, 20500.0, pnl=1200.0)
    await notifier.send_risk_alert("Daily loss limit reached: -$50,000")
"""

from __future__ import annotations

import asyncio
from typing import Optional

import httpx
from loguru import logger

# Telegram Bot API
TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"

# LINE Notify API
LINE_NOTIFY_API = "https://notify-api.line.me/api/notify"


class Notifier:
    """Async notification sender for Telegram and LINE."""

    def __init__(
        self,
        telegram_token: str = "",
        telegram_chat_id: str = "",
        line_notify_token: str = "",
    ) -> None:
        self._telegram_token = telegram_token
        self._telegram_chat_id = telegram_chat_id
        self._line_notify_token = line_notify_token
        self._enabled = bool(telegram_token or line_notify_token)

        if not self._enabled:
            logger.warning(
                "[Notifier] No notification channels configured "
                "(set TELEGRAM_BOT_TOKEN or LINE_NOTIFY_TOKEN)"
            )

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    async def _send_telegram(self, text: str) -> bool:
        if not self._telegram_token or not self._telegram_chat_id:
            return False
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    TELEGRAM_API.format(token=self._telegram_token),
                    json={
                        "chat_id": self._telegram_chat_id,
                        "text": text,
                        "parse_mode": "HTML",
                    },
                )
                if resp.status_code != 200:
                    logger.error(
                        "[Notifier] Telegram send failed: {} {}",
                        resp.status_code, resp.text,
                    )
                    return False
                return True
        except Exception as e:
            logger.error("[Notifier] Telegram error: {}", e)
            return False

    async def _send_line(self, text: str) -> bool:
        if not self._line_notify_token:
            return False
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    LINE_NOTIFY_API,
                    headers={
                        "Authorization": f"Bearer {self._line_notify_token}",
                    },
                    data={"message": text},
                )
                if resp.status_code != 200:
                    logger.error(
                        "[Notifier] LINE send failed: {} {}",
                        resp.status_code, resp.text,
                    )
                    return False
                return True
        except Exception as e:
            logger.error("[Notifier] LINE error: {}", e)
            return False

    async def _send(self, text: str) -> None:
        """Send to all configured channels."""
        if not self._enabled:
            return
        tasks = []
        if self._telegram_token:
            tasks.append(self._send_telegram(text))
        if self._line_notify_token:
            tasks.append(self._send_line(text))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # ------------------------------------------------------------------
    # High-level notification methods
    # ------------------------------------------------------------------

    async def send_fill(
        self,
        broker: str,
        action: str,
        quantity: int,
        price: float,
        pnl: Optional[float] = None,
    ) -> None:
        emoji = "\U0001f7e2" if action == "Buy" else "\U0001f534"
        pnl_str = ""
        if pnl is not None:
            pnl_emoji = "\u2705" if pnl >= 0 else "\u274c"
            pnl_str = f"\nPnL: ${pnl:,.0f} {pnl_emoji}"
        text = (
            f"{emoji} <b>Fill</b> [{broker}]\n"
            f"{action} x{quantity} @ {price:,.1f}"
            f"{pnl_str}"
        )
        await self._send(text)

    async def send_protective_event(
        self,
        broker: str,
        ticker: str,
        side: str,
        quantity: int,
        trigger_reason: str,
        status: str,
        trigger_price: float = 0.0,
        submit_price: float = 0.0,
        fill_price: float = 0.0,
        slippage_points: float | None = None,
        execution_price_type: str = "",
    ) -> None:
        icon_map = {
            "triggered": "\u26a0\ufe0f",
            "submitted": "\U0001f6e1\ufe0f",
            "filled": "\u2705",
            "failed": "\u274c",
        }
        icon = icon_map.get(status, "\u2139\ufe0f")
        lines = [
            f"{icon} <b>Protective Exit</b> [{broker}]",
            f"{ticker} {side} x{quantity}",
            f"Reason: {trigger_reason}",
            f"Status: {status}",
        ]
        if trigger_price > 0:
            lines.append(f"Trigger: {trigger_price:,.1f}")
        if submit_price > 0:
            suffix = f" ({execution_price_type})" if execution_price_type else ""
            lines.append(f"Submit: {submit_price:,.1f}{suffix}")
        if fill_price > 0:
            lines.append(f"Fill: {fill_price:,.1f}")
        if slippage_points is not None:
            lines.append(f"Slippage: {slippage_points:+.1f} pts")
        await self._send("\n".join(lines))

    async def send_risk_alert(self, message: str) -> None:
        text = f"\u26a0\ufe0f <b>RISK ALERT</b>\n{message}"
        await self._send(text)

    async def send_halt(self, reason: str) -> None:
        text = f"\U0001f6d1 <b>TRADING HALTED</b>\n{reason}"
        await self._send(text)

    async def send_system(self, message: str) -> None:
        text = f"\u2699\ufe0f <b>System</b>\n{message}"
        await self._send(text)

    async def send_error(self, message: str) -> None:
        text = f"\u274c <b>ERROR</b>\n{message}"
        await self._send(text)

    async def send_daily_summary(self, summary: dict) -> None:
        total_pnl = summary.get("total_pnl", 0)
        total_trades = summary.get("total_trades", 0)
        pnl_emoji = "\U0001f7e2" if total_pnl >= 0 else "\U0001f534"

        lines = [
            f"\U0001f4ca <b>Daily Summary</b> ({summary.get('date', 'today')})",
            f"Total PnL: ${total_pnl:,.0f} {pnl_emoji}",
            f"Total Trades: {total_trades}",
        ]
        for b in summary.get("brokers", []):
            lines.append(
                f"  {b['broker']}: {b['trade_count']} trades, "
                f"PnL=${b.get('total_pnl', 0):,.0f}"
            )
        await self._send("\n".join(lines))
