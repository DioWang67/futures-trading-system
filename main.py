"""
Trading Bot — FastAPI application entry point.

Industrial-grade features:
  - Pydantic-validated config with fail-fast on missing secrets
  - Secret masking in logs
  - Health / readiness / risk-status endpoints
  - Risk management with daily loss limits & drawdown circuit breaker
  - Trade persistence (SQLite) with crash recovery
  - Telegram notifications
  - Graceful shutdown with timeout
"""

from __future__ import annotations

import asyncio
import hmac
import re
import sys
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from loguru import logger

from config import settings
from notifier import Notifier
from risk_manager import RiskConfig, RiskManager
from scheduler import (
    shioaji_token_refresh_loop,
    broker_health_monitor,
    daily_risk_reset_loop,
)
from trade_store import TradeStore
from webhook.router import router

# ---------------------------------------------------------------------------
# Secret masking filter
# ---------------------------------------------------------------------------
_SECRET_PATTERNS: list[re.Pattern] = []


def _build_secret_patterns() -> None:
    """Collect non-empty secret values and compile regex patterns."""
    secrets = [
        settings.shioaji.api_key.get_secret_value(),
        settings.shioaji.secret_key.get_secret_value(),
        settings.shioaji.ca_password.get_secret_value(),
        settings.rithmic.password.get_secret_value(),
        settings.webhook.secret.get_secret_value(),
        settings.admin.secret.get_secret_value(),
        settings.notification.telegram_bot_token.get_secret_value(),
    ]
    # Require a meaningful length before we start wholesale-replacing
    # substrings in logs; a 4-char secret like "test" would otherwise
    # mask every occurrence of "test" in file paths, tracebacks, etc.
    for s in secrets:
        if s and len(s) >= 16:
            _SECRET_PATTERNS.append(re.compile(re.escape(s)))


def _mask_secrets(message: str) -> str:
    for pat in _SECRET_PATTERNS:
        message = pat.sub("***REDACTED***", message)
    return message


def _secret_filter(record):
    record["message"] = _mask_secrets(record["message"])
    return True


def _require_admin_secret(x_admin_secret: str = Header(default="")) -> None:
    """Protect control-plane endpoints with a shared admin secret."""
    expected = settings.admin.secret.get_secret_value()
    if not expected:
        raise HTTPException(
            status_code=500,
            detail="Admin secret not configured on server",
        )
    if not x_admin_secret or not hmac.compare_digest(x_admin_secret, expected):
        raise HTTPException(status_code=403, detail="Invalid admin secret")


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
def _setup_logging() -> None:
    logger.remove()
    logger.add(
        sys.stdout,
        level=settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        filter=_secret_filter,
    )
    logger.add(
        "logs/trading.log",
        level=settings.log_level,
        rotation="10 MB",
        retention="30 days",
        compression="gz",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} - {message}",
        filter=_secret_filter,
    )


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------
SHUTDOWN_TIMEOUT = 10  # seconds


@asynccontextmanager
async def lifespan(app: FastAPI):
    _build_secret_patterns()
    _setup_logging()

    logger.info("=== Trading Bot starting ===")

    # Validate required secrets at startup (fail fast)
    try:
        settings.require_webhook()
        settings.require_admin()
    except ValueError as e:
        logger.critical("Config error: {}", e)
        raise SystemExit(1) from e

    # --- Shared services ---
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
    )

    app.state.risk_manager = risk_manager
    app.state.trade_store = trade_store
    app.state.notifier = notifier

    # --- Shioaji ---
    from brokers import ShioajiBroker
    shioaji_broker = ShioajiBroker(
        risk_manager=risk_manager,
        trade_store=trade_store,
        notifier=notifier,
    )
    shioaji_broker.attach_event_loop(asyncio.get_running_loop())
    try:
        settings.require_shioaji()
        await asyncio.to_thread(shioaji_broker.login)
    except ValueError as e:
        logger.warning("Shioaji not configured, skipping: {}", e)
    except Exception as e:
        logger.error("Shioaji startup failed (will continue without it): {}", e)

    # --- Rithmic ---
    rithmic_broker = None
    try:
        from brokers import RithmicBroker
        rithmic_broker = RithmicBroker(
            risk_manager=risk_manager,
            trade_store=trade_store,
            notifier=notifier,
        )
        settings.require_rithmic()
        await rithmic_broker.connect()
    except ImportError:
        logger.warning("async_rithmic not installed, Rithmic broker disabled")
    except ValueError as e:
        logger.warning("Rithmic not configured, skipping: {}", e)
    except Exception as e:
        logger.error("Rithmic startup failed (will continue without it): {}", e)

    if rithmic_broker is None:
        # Stub broker that always returns "not connected". Leave
        # supported_tickers empty so the webhook router rejects MES
        # traffic outright rather than advertising a path that always
        # errors.
        from position_state import PositionState

        class _StubBroker:
            position = PositionState("rithmic")
            broker_name = "rithmic"
            is_connected = False
            supported_tickers: set[str] = set()
            async def place_order(self, *a, **kw):
                return {"status": "error", "broker": "rithmic", "reason": "not installed"}
            async def disconnect(self):
                pass
        rithmic_broker = _StubBroker()

    app.state.shioaji_broker = shioaji_broker
    app.state.rithmic_broker = rithmic_broker

    # Background tasks
    refresh_task = asyncio.create_task(
        shioaji_token_refresh_loop(shioaji_broker, notifier),
        name="shioaji-token-refresh",
    )
    protective_watchdog_task = asyncio.create_task(
        shioaji_broker.run_watchdog(),
        name="shioaji-protective-watchdog",
    )
    health_task = asyncio.create_task(
        broker_health_monitor(shioaji_broker, rithmic_broker, notifier),
        name="broker-health-monitor",
    )
    risk_reset_task = asyncio.create_task(
        daily_risk_reset_loop(risk_manager, trade_store),
        name="daily-risk-reset",
    )

    logger.info("=== Trading Bot ready ===")
    await notifier.send_system("Trading Bot started")
    yield

    # --- Shutdown ---
    logger.info("=== Trading Bot shutting down ===")
    refresh_task.cancel()
    protective_watchdog_task.cancel()
    health_task.cancel()
    risk_reset_task.cancel()

    # Save final position snapshots
    for broker in [shioaji_broker, rithmic_broker]:
        pos = broker.position.get_position()
        trade_store.save_position_snapshot(
            broker.broker_name, pos.side, pos.quantity, pos.entry_price
        )

    async def _shutdown():
        for task in [refresh_task, protective_watchdog_task, health_task, risk_reset_task]:
            try:
                await task
            except asyncio.CancelledError:
                pass
        try:
            await asyncio.to_thread(shioaji_broker.logout)
        except Exception as e:
            logger.error("Shioaji logout error: {}", e)
        try:
            await rithmic_broker.disconnect()
        except Exception as e:
            logger.error("Rithmic disconnect error: {}", e)

    try:
        await asyncio.wait_for(_shutdown(), timeout=SHUTDOWN_TIMEOUT)
    except asyncio.TimeoutError:
        logger.warning("Shutdown timed out after {}s", SHUTDOWN_TIMEOUT)

    await notifier.send_system("Trading Bot stopped")
    logger.info("=== Trading Bot stopped ===")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Trading Bot",
    version="3.0.0",
    lifespan=lifespan,
    docs_url=None,      # disable Swagger in production
    redoc_url=None,
)
app.include_router(router)


@app.get("/health")
async def health():
    """Liveness probe — always returns 200 if the process is running."""
    return {"status": "ok"}


@app.get("/ready")
async def readiness():
    """Readiness probe — reports broker connection status."""
    shioaji_ok = getattr(app.state, "shioaji_broker", None)
    rithmic_ok = getattr(app.state, "rithmic_broker", None)

    shioaji_connected = shioaji_ok.is_connected if shioaji_ok else False
    rithmic_connected = rithmic_ok.is_connected if rithmic_ok else False

    return {
        "ready": shioaji_connected or rithmic_connected,
        "brokers": {
            "shioaji": shioaji_connected,
            "rithmic": rithmic_connected,
        },
    }


@app.get("/risk")
async def risk_status(_: None = Depends(_require_admin_secret)):
    """Risk manager status — current drawdown, daily PnL, halt state."""
    risk_manager = getattr(app.state, "risk_manager", None)
    if not risk_manager:
        return {"error": "risk manager not initialized"}
    return risk_manager.get_status()


@app.post("/risk/resume")
async def risk_resume(_: None = Depends(_require_admin_secret)):
    """Manually resume trading after a halt."""
    risk_manager = getattr(app.state, "risk_manager", None)
    if not risk_manager:
        return {"error": "risk manager not initialized"}
    risk_manager.resume_trading()
    notifier = getattr(app.state, "notifier", None)
    if notifier:
        await notifier.send_system("Trading resumed by operator")
    return {"status": "resumed"}


@app.get("/trades/today")
async def today_trades(_: None = Depends(_require_admin_secret)):
    """Get today's trade fills."""
    trade_store = getattr(app.state, "trade_store", None)
    if not trade_store:
        return {"error": "trade store not initialized"}
    return trade_store.get_daily_summary()


@app.get("/trades/recent")
async def recent_trades(_: None = Depends(_require_admin_secret)):
    """Get recent trade fills."""
    trade_store = getattr(app.state, "trade_store", None)
    if not trade_store:
        return {"error": "trade store not initialized"}
    return {"fills": trade_store.get_recent_fills(50)}


@app.get("/positions")
async def positions(_: None = Depends(_require_admin_secret)):
    """Get current positions across all brokers."""
    shioaji = getattr(app.state, "shioaji_broker", None)
    rithmic = getattr(app.state, "rithmic_broker", None)

    result = {}
    if shioaji:
        pos = shioaji.position.get_position()
        result["shioaji"] = {
            "side": pos.side, "quantity": pos.quantity, "entry_price": pos.entry_price
        }
    if rithmic:
        pos = rithmic.position.get_position()
        result["rithmic"] = {
            "side": pos.side, "quantity": pos.quantity, "entry_price": pos.entry_price
        }
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level="warning",  # let loguru handle app logging
    )
