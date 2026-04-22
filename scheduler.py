"""
Background schedulers:
  - Shioaji token refresh with exponential backoff
  - Broker connection health monitor
  - Daily risk reset
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from loguru import logger

REFRESH_INTERVAL = 23 * 60 * 60  # 23 hours
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 60  # 1 minute

HEALTH_CHECK_INTERVAL = 30  # seconds

# Taiwan market opens at 08:45 TAIFEX time; reset counters 15 minutes before
# so the new-day windows are ready when orders can start flowing. We use an
# explicit tzinfo because Docker / systemd containers typically run in UTC,
# and datetime.now() there would reset 8 hours off.
RISK_RESET_TZ = ZoneInfo("Asia/Taipei")
RISK_RESET_HOUR = 8
RISK_RESET_MINUTE = 30


async def shioaji_token_refresh_loop(shioaji_broker: object) -> None:
    """Re-login to Shioaji every 23 hours to prevent 24hr token expiry."""
    while True:
        await asyncio.sleep(REFRESH_INTERVAL)
        logger.info("[Scheduler] Starting Shioaji token refresh...")

        delay = INITIAL_RETRY_DELAY
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                await asyncio.to_thread(shioaji_broker.reconnect)
                logger.info("[Scheduler] Shioaji token refresh completed")
                break
            except Exception as e:
                logger.error(
                    "[Scheduler] Shioaji token refresh failed (attempt {}/{}): {}",
                    attempt, MAX_RETRIES, e,
                )
                if attempt == MAX_RETRIES:
                    logger.critical(
                        "[Scheduler] Shioaji token refresh exhausted all {} retries",
                        MAX_RETRIES,
                    )
                    # Notify
                    notifier = getattr(shioaji_broker, "_notifier", None)
                    if notifier:
                        await notifier.send_error(
                            "Shioaji token refresh failed after all retries!"
                        )
                    break
                logger.info("[Scheduler] Retrying in {}s...", delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 900)  # cap at 15 minutes


async def broker_health_monitor(
    shioaji_broker: object,
    rithmic_broker: object,
    notifier: object = None,
) -> None:
    """Monitor broker connections and attempt reconnection on failure."""
    # Seed from the actual live state so disabled / stub brokers don't get
    # reported as "disconnected" 30 seconds after startup.
    was_connected = {
        "shioaji": bool(getattr(shioaji_broker, "is_connected", False)),
        "rithmic": bool(getattr(rithmic_broker, "is_connected", False)),
    }

    while True:
        await asyncio.sleep(HEALTH_CHECK_INTERVAL)

        # Check Shioaji
        sj_connected = getattr(shioaji_broker, "is_connected", False)
        if was_connected["shioaji"] and not sj_connected:
            logger.error("[HealthMonitor] Shioaji disconnected!")
            if notifier:
                await notifier.send_error("Shioaji broker disconnected!")
            # Attempt reconnect
            try:
                await asyncio.to_thread(shioaji_broker.reconnect)
                logger.info("[HealthMonitor] Shioaji reconnected")
                if notifier:
                    await notifier.send_system("Shioaji broker reconnected")
            except Exception as e:
                logger.error("[HealthMonitor] Shioaji reconnect failed: {}", e)
        elif not was_connected["shioaji"] and sj_connected:
            logger.info("[HealthMonitor] Shioaji connection restored")
        was_connected["shioaji"] = sj_connected

        # Check Rithmic
        rt_connected = getattr(rithmic_broker, "is_connected", False)
        if was_connected["rithmic"] and not rt_connected:
            logger.error("[HealthMonitor] Rithmic disconnected!")
            if notifier:
                await notifier.send_error("Rithmic broker disconnected!")
            # Attempt reconnect
            reconnect = getattr(rithmic_broker, "reconnect", None)
            if reconnect:
                try:
                    await reconnect()
                    logger.info("[HealthMonitor] Rithmic reconnected")
                    if notifier:
                        await notifier.send_system("Rithmic broker reconnected")
                except Exception as e:
                    logger.error("[HealthMonitor] Rithmic reconnect failed: {}", e)
        elif not was_connected["rithmic"] and rt_connected:
            logger.info("[HealthMonitor] Rithmic connection restored")
        was_connected["rithmic"] = rt_connected


async def daily_risk_reset_loop(risk_manager: object) -> None:
    """Reset daily risk counters at 08:30 Asia/Taipei."""
    while True:
        now = datetime.now(tz=RISK_RESET_TZ)
        reset_time = now.replace(
            hour=RISK_RESET_HOUR,
            minute=RISK_RESET_MINUTE,
            second=0,
            microsecond=0,
        )
        if now >= reset_time:
            reset_time = reset_time + timedelta(days=1)

        wait_seconds = (reset_time - now).total_seconds()
        logger.info(
            "[Scheduler] Daily risk reset scheduled in {:.0f}s (at {} {})",
            wait_seconds,
            reset_time.strftime("%Y-%m-%d %H:%M"),
            RISK_RESET_TZ.key,
        )
        await asyncio.sleep(wait_seconds)

        try:
            risk_manager.reset_daily()
            logger.info("[Scheduler] Daily risk counters reset")
        except Exception as e:
            logger.error("[Scheduler] Daily reset failed: {}", e)
