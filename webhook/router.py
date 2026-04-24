"""
Webhook router — receives TradingView signals and dispatches to brokers.

Security:
  - HMAC-SHA256 signature verification (X-Signature header)
  - Timestamp replay protection (X-Timestamp header)
  - Falls back to shared-secret check if HMAC headers are absent
  - Idempotency via X-Idempotency-Key header (deduplicates retries)
"""

from __future__ import annotations

import hmac
import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request
from loguru import logger
from pydantic import ValidationError

from config import settings
from webhook.validator import (
    WebhookPayload,
    verify_hmac_signature,
    verify_timestamp,
)

router = APIRouter()


def _generate_idempotency_key(body: bytes, timestamp: str) -> str:
    """Generate a deterministic key from request body + signed timestamp."""
    return hashlib.sha256(body + timestamp.encode()).hexdigest()[:32]


def _get_supported_tickers(broker: Any) -> set[str]:
    tickers = getattr(broker, "supported_tickers", set())
    return {str(t).upper() for t in tickers}


def _resolve_broker_for_ticker(
    ticker: str,
    shioaji_broker: Any,
    rithmic_broker: Any,
) -> tuple[Any | None, dict, dict]:
    """Resolve ``ticker`` to the one broker that truly supports it."""
    normalized = ticker.upper()
    shioaji_tickers = _get_supported_tickers(shioaji_broker)
    rithmic_tickers = _get_supported_tickers(rithmic_broker)

    shioaji_match = normalized in shioaji_tickers
    rithmic_match = normalized in rithmic_tickers

    if shioaji_match and rithmic_match:
        reason = f"ticker {normalized} is ambiguously configured on multiple brokers"
        return (
            None,
            {"status": "error", "reason": reason},
            {"status": "error", "reason": reason},
        )
    if shioaji_match:
        return (
            shioaji_broker,
            {},
            {"status": "skipped", "reason": f"{normalized} routed to shioaji only"},
        )
    if rithmic_match:
        return (
            rithmic_broker,
            {"status": "skipped", "reason": f"{normalized} routed to rithmic only"},
            {},
        )

    reason = f"unsupported ticker: {normalized}"
    return (
        None,
        {"status": "rejected", "reason": reason},
        {"status": "rejected", "reason": reason},
    )


async def _verify_request(
    body: bytes,
    x_signature: str,
    x_timestamp: str,
    x_webhook_secret: str,
) -> None:
    """Verify webhook authenticity. Raises HTTPException on failure."""
    secret = settings.webhook.secret.get_secret_value()
    if not secret:
        # Secret absent is an operator error, not a client error — signal
        # unavailability so load balancers back off instead of thinking
        # the service is malfunctioning.
        raise HTTPException(
            status_code=503,
            detail="Webhook secret not configured on server",
        )

    if not x_timestamp:
        logger.warning("Webhook rejected: missing timestamp header")
        raise HTTPException(status_code=401, detail="Missing timestamp")

    max_drift = settings.webhook.max_timestamp_drift_seconds
    if not verify_timestamp(x_timestamp, max_drift):
        logger.warning(
            "Webhook rejected: timestamp drift too large (ts={})", x_timestamp
        )
        raise HTTPException(status_code=403, detail="Timestamp expired")

    # Prefer strict HMAC signature verification.
    if x_signature:
        if not verify_hmac_signature(body, secret, x_signature, x_timestamp):
            logger.warning("Webhook rejected: invalid HMAC signature")
            raise HTTPException(status_code=403, detail="Invalid signature")
        return

    # Fallback is disabled by default because it cannot bind timestamp+body.
    if settings.webhook.allow_legacy_secret_header and x_webhook_secret:
        if not hmac.compare_digest(x_webhook_secret, secret):
            logger.warning("Webhook rejected: invalid secret")
            raise HTTPException(status_code=403, detail="Invalid webhook secret")
        return

    logger.warning("Webhook rejected: missing signature header")
    raise HTTPException(status_code=401, detail="Missing signature")


@router.post("/webhook")
async def handle_webhook(
    request: Request,
    x_signature: str = Header(default=""),
    x_timestamp: str = Header(default=""),
    x_webhook_secret: str = Header(default=""),
    x_idempotency_key: str = Header(default=""),
) -> dict:
    """Receive TradingView webhook and route to the matching broker.

    HMAC is verified on the raw bytes **before** we let Pydantic parse
    them, so an unauthenticated caller can't probe the schema by
    watching the gap between 422 (validation) and 401/403 (auth) errors.
    """

    # Read raw body for HMAC verification
    body = await request.body()
    await _verify_request(body, x_signature, x_timestamp, x_webhook_secret)

    # Only parse the body after auth passes.
    try:
        payload = WebhookPayload.model_validate_json(body)
    except ValidationError as exc:
        logger.warning("Webhook rejected: invalid payload schema")
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    received_at = datetime.now(timezone.utc).isoformat()

    # Generate a stable idempotency key if not provided. Because timestamp is
    # mandatory and signed, the same signed request maps to the same key.
    idempotency_key = x_idempotency_key or _generate_idempotency_key(
        body, x_timestamp
    )

    logger.info(
        "Webhook received: action={}, sentiment={}, qty={}, ticker={}, idem_key={}",
        payload.action,
        payload.sentiment,
        payload.quantity,
        payload.ticker,
        idempotency_key[:8],
    )

    # Get broker instances from app state
    shioaji_broker = request.app.state.shioaji_broker
    rithmic_broker = request.app.state.rithmic_broker
    trade_store = getattr(request.app.state, "trade_store", None)
    source_ip = ""
    if request.client and request.client.host:
        source_ip = request.client.host
    if trade_store:
        try:
            trade_store.record_signal_event(
                idempotency_key=idempotency_key,
                source_path=str(request.url.path),
                source_ip=source_ip,
                action=payload.action,
                sentiment=payload.sentiment,
                quantity=payload.quantity,
                ticker=payload.ticker,
                payload_json=body.decode("utf-8", errors="replace"),
            )
        except Exception as exc:
            logger.warning("Signal event recording failed: {}", exc)

    # Check risk manager halt — but let exit signals through so operators
    # can still flatten positions while the system is halted.
    risk_manager = getattr(request.app.state, "risk_manager", None)
    if risk_manager and risk_manager.is_halted and not payload.is_exit:
        logger.warning(
            "Webhook rejected: trading halted — {}",
            risk_manager.halt_reason,
        )
        return {
            "received_at": received_at,
            "action": payload.action,
            "status": "rejected",
            "reason": f"trading halted: {risk_manager.halt_reason}",
        }

    # Route to broker based on ticker
    # TXF/MXF → Shioaji (台灣期交所), NQ/MNQ/ES/MES → Rithmic (CME)
    selected_broker, shioaji_result, rithmic_result = _resolve_broker_for_ticker(
        payload.ticker, shioaji_broker, rithmic_broker
    )
    if selected_broker is shioaji_broker:
        shioaji_result = await _safe_execute(shioaji_broker, payload, idempotency_key)
        if trade_store:
            trade_store.update_signal_event_result(
                idempotency_key=idempotency_key,
                broker="shioaji",
                status=str(shioaji_result.get("status") or ""),
                reason=str(shioaji_result.get("reason") or ""),
                result_json=json.dumps(shioaji_result, ensure_ascii=False),
            )
    elif selected_broker is rithmic_broker:
        rithmic_result = await _safe_execute(rithmic_broker, payload, idempotency_key)
        if trade_store:
            trade_store.update_signal_event_result(
                idempotency_key=idempotency_key,
                broker="rithmic",
                status=str(rithmic_result.get("status") or ""),
                reason=str(rithmic_result.get("reason") or ""),
                result_json=json.dumps(rithmic_result, ensure_ascii=False),
            )

    logger.info(
        "Webhook processed: shioaji={}, rithmic={}",
        shioaji_result.get("status"),
        rithmic_result.get("status"),
    )

    # Notify on risk rejection
    notifier = getattr(request.app.state, "notifier", None)
    for result, name in [(shioaji_result, "shioaji"), (rithmic_result, "rithmic")]:
        if result.get("status") == "risk_rejected" and notifier:
            sent = await notifier.send_risk_alert(
                f"[{name}] Order rejected: {result.get('reason', 'unknown')}"
            )
            if not sent:
                logger.error(
                    "Risk rejection alert delivery failed for {}: {}",
                    name,
                    result.get("reason", "unknown"),
                )

    return {
        "received_at": received_at,
        "action": payload.action,
        "idempotency_key": idempotency_key,
        "shioaji": shioaji_result,
        "rithmic": rithmic_result,
    }


async def _safe_execute(
    broker: object,
    payload: WebhookPayload,
    idempotency_key: str = "",
) -> dict:
    """Execute broker order with full error isolation."""
    broker_name = str(getattr(broker, "broker_name", "") or "unknown")
    try:
        position = getattr(broker, "position", None)
        if broker_name == "unknown" and position is not None:
            broker_name = str(getattr(position, "broker_name", "") or "unknown")

        # Pass full payload to broker
        place_order = getattr(broker, "place_order", None)
        if place_order is None:
            return {"status": "error", "broker": broker_name, "reason": "no place_order method"}

        # Determine effective action: if type=exit/eod_close, override action to "exit"
        effective_action = payload.action
        if payload.is_exit:
            effective_action = "exit"

        return await place_order(
            effective_action,
            payload.quantity,
            ticker=payload.ticker,
            sentiment=payload.sentiment,
            idempotency_key=idempotency_key,
        )
    except Exception as e:
        # Deliberately no retry here: re-submitting place_order without the
        # idempotency key can duplicate a live order. Surface the error.
        logger.error("[{}] Unhandled error in place_order: {}", broker_name, e)
        return {"status": "error", "broker": broker_name, "reason": "broker_error"}
