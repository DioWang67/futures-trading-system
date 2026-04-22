"""
Webhook payload validation and HMAC-SHA256 signature verification.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from typing import Literal, Optional

from pydantic import BaseModel, field_validator


class WebhookPayload(BaseModel):
    action: Literal["buy", "sell", "exit"]
    sentiment: Literal["long", "short", "flat", "bullish", "bearish"]
    quantity: int = 1
    ticker: str
    price: Optional[str] = None
    time: Optional[str] = None
    type: Optional[Literal["entry", "exit", "eod_close"]] = None

    @field_validator("quantity")
    @classmethod
    def quantity_must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("quantity must be positive")
        return v

    @field_validator("action", mode="before")
    @classmethod
    def normalize_action(cls, v: str) -> str:
        return v.strip().lower()

    @field_validator("sentiment", mode="before")
    @classmethod
    def normalize_sentiment(cls, v: str) -> str:
        v = v.strip().lower()
        # Map Pine Script sentiment to internal format
        mapping = {"bullish": "long", "bearish": "short"}
        return mapping.get(v, v)

    @property
    def is_entry(self) -> bool:
        """Check if this is an entry signal (vs exit/close)."""
        if self.type:
            return self.type == "entry"
        return self.action in ("buy", "sell")

    @property
    def is_exit(self) -> bool:
        """Check if this is an exit signal."""
        if self.type:
            return self.type in ("exit", "eod_close")
        return self.action == "exit"


def build_signature_payload(body: bytes, timestamp: str) -> bytes:
    """Canonical bytes covered by the webhook HMAC."""
    if not timestamp:
        return b""
    return timestamp.encode("utf-8") + b"." + body


def verify_hmac_signature(
    body: bytes,
    secret: str,
    signature: str,
    timestamp: str,
) -> bool:
    """Verify HMAC-SHA256 signature.

    The expected signature format is: sha256=<hex_digest>
    over the canonical payload ``<timestamp>.<raw_body>``.
    """
    signed_payload = build_signature_payload(body, timestamp)
    if not secret or not signature or not signed_payload:
        return False

    expected = hmac.new(
        secret.encode("utf-8"),
        signed_payload,
        hashlib.sha256,
    ).hexdigest()

    # Support both "sha256=xxx" and bare "xxx" formats
    if signature.startswith("sha256="):
        signature = signature[7:]

    return hmac.compare_digest(expected, signature)


def verify_timestamp(
    timestamp: str,
    max_drift_seconds: int = 120,
) -> bool:
    """Reject webhooks older than max_drift_seconds (replay protection).

    Accepts either seconds or milliseconds since epoch — TradingView's
    ``{{timenow}}`` format isn't contractually stable between chart
    versions, and some senders use millisecond precision.
    """
    try:
        ts = int(timestamp)
    except (ValueError, TypeError):
        try:
            ts = int(float(timestamp))
        except (ValueError, TypeError):
            return False

    # Heuristic: anything > ~year 2286 in seconds is almost certainly ms.
    if ts > 10_000_000_000:
        ts = ts // 1000

    now = int(time.time())
    return abs(now - ts) <= max_drift_seconds
