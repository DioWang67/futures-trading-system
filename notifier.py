"""
Notification module — Telegram.

Sends alerts for:
  - Trade executions (fills)
  - Risk events (daily loss limit, drawdown halt, cooldown)
  - System events (broker disconnect, reconnect, startup/shutdown)
  - Error alerts

LINE Notify was retired by LINE Corporation on 2025-03-31 so it is no longer
a supported channel. Teams that still want a LINE path should migrate to the
official Messaging API; that's out of scope for this module.

Usage:
    notifier = Notifier(telegram_token="...", telegram_chat_id="...")
    await notifier.send_fill("shioaji", "Buy", 1, 20500.0, pnl=1200.0)
    await notifier.send_risk_alert("Daily loss limit reached: -$50,000")
"""

from __future__ import annotations

from typing import Optional

import httpx
from loguru import logger

# Telegram Bot API
TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


class Notifier:
    """Async notification sender (Telegram)."""

    def __init__(
        self,
        telegram_token: str = "",
        telegram_chat_id: str = "",
    ) -> None:
        self._telegram_token = telegram_token
        self._telegram_chat_id = telegram_chat_id
        self._enabled = bool(telegram_token and telegram_chat_id)

        if not self._enabled:
            logger.warning(
                "[Notifier] No notification channel configured "
                "(set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID); "
                "risk/halt alerts will only appear in local logs"
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

    async def _send(self, text: str) -> None:
        """Send to Telegram if configured."""
        if not self._enabled:
            return
        await self._send_telegram(text)

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
            pnl_emoji = "✅" if pnl >= 0 else "❌"
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
            "triggered": "⚠️",
            "submitted": "\U0001f6e1️",
            "filled": "✅",
            "failed": "❌",
        }
        icon = icon_map.get(status, "ℹ️")
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
        text = f"⚠️ <b>RISK ALERT</b>\n{message}"
        await self._send(text)

    async def send_halt(self, reason: str) -> None:
        text = f"\U0001f6d1 <b>TRADING HALTED</b>\n{reason}"
        await self._send(text)

    async def send_system(self, message: str) -> None:
        text = f"⚙️ <b>System</b>\n{message}"
        await self._send(text)

    async def send_error(self, message: str) -> None:
        text = f"❌ <b>ERROR</b>\n{message}"
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
