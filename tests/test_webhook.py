"""Webhook validator & security tests."""

import hashlib
import hmac
import time

import pytest

from webhook.validator import (
    WebhookPayload,
    build_signature_payload,
    verify_hmac_signature,
    verify_timestamp,
)


# ============================================================
# HMAC Signature
# ============================================================

class TestHMACSignature:
    def test_valid_signature(self):
        secret = "test-secret-key"
        timestamp = "1712345678"
        body = b'{"action":"buy","sentiment":"long","quantity":1,"ticker":"MXF"}'
        signed = build_signature_payload(body, timestamp)
        sig = hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()
        assert verify_hmac_signature(body, secret, f"sha256={sig}", timestamp)

    def test_valid_signature_bare(self):
        secret = "test-secret-key"
        timestamp = "1712345678"
        body = b'{"action":"buy"}'
        signed = build_signature_payload(body, timestamp)
        sig = hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()
        assert verify_hmac_signature(body, secret, sig, timestamp)

    def test_invalid_signature(self):
        secret = "test-secret-key"
        body = b'{"action":"buy"}'
        assert not verify_hmac_signature(body, secret, "sha256=badhex", "1712345678")

    def test_empty_secret_rejected(self):
        assert not verify_hmac_signature(b"body", "", "sha256=abc", "1712345678")

    def test_empty_signature_rejected(self):
        assert not verify_hmac_signature(b"body", "secret", "", "1712345678")

    def test_tampered_body(self):
        secret = "key"
        timestamp = "1712345678"
        original = b'{"quantity":1}'
        signed = build_signature_payload(original, timestamp)
        sig = hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()
        tampered = b'{"quantity":999}'
        assert not verify_hmac_signature(tampered, secret, f"sha256={sig}", timestamp)

    def test_tampered_timestamp(self):
        secret = "key"
        body = b"data"
        signed = build_signature_payload(body, "1712345678")
        sig = hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()
        assert not verify_hmac_signature(body, secret, f"sha256={sig}", "1712345679")

    def test_timing_safe_comparison(self):
        """verify_hmac_signature should use constant-time comparison."""
        secret = "key"
        timestamp = "1712345678"
        body = b"data"
        signed = build_signature_payload(body, timestamp)
        sig = hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()
        almost = sig[:-1] + ("0" if sig[-1] != "0" else "1")
        assert not verify_hmac_signature(body, secret, f"sha256={almost}", timestamp)


# ============================================================
# Timestamp replay protection
# ============================================================

class TestTimestampVerification:
    def test_current_timestamp_valid(self):
        ts = str(int(time.time()))
        assert verify_timestamp(ts, max_drift_seconds=30)

    def test_old_timestamp_rejected(self):
        ts = str(int(time.time()) - 120)
        assert not verify_timestamp(ts, max_drift_seconds=30)

    def test_future_timestamp_rejected(self):
        ts = str(int(time.time()) + 120)
        assert not verify_timestamp(ts, max_drift_seconds=30)

    def test_invalid_timestamp_rejected(self):
        assert not verify_timestamp("not-a-number")
        assert not verify_timestamp("")
        assert not verify_timestamp(None)

    def test_float_timestamp(self):
        ts = str(time.time())  # float string
        assert verify_timestamp(ts, max_drift_seconds=30)


# ============================================================
# Payload validation
# ============================================================

class TestWebhookPayload:
    def test_valid_payload(self):
        p = WebhookPayload(
            action="buy", sentiment="long", quantity=1, ticker="MXF"
        )
        assert p.action == "buy"
        assert p.quantity == 1

    def test_action_normalized(self):
        p = WebhookPayload(
            action="  BUY  ", sentiment="long", quantity=1, ticker="MXF"
        )
        assert p.action == "buy"

    def test_sentiment_normalized(self):
        p = WebhookPayload(
            action="sell", sentiment="  SHORT ", quantity=1, ticker="MXF"
        )
        assert p.sentiment == "short"

    def test_zero_quantity_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            WebhookPayload(
                action="buy", sentiment="long", quantity=0, ticker="MXF"
            )

    def test_negative_quantity_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            WebhookPayload(
                action="buy", sentiment="long", quantity=-1, ticker="MXF"
            )

    def test_invalid_action_rejected(self):
        with pytest.raises(ValueError):
            WebhookPayload(
                action="invalid", sentiment="long", quantity=1, ticker="MXF"
            )

    def test_optional_fields(self):
        p = WebhookPayload(
            action="exit", sentiment="flat", quantity=1, ticker="MXF",
            price="16500.0", time="2024-01-01 09:30:00",
        )
        assert p.price == "16500.0"
        assert p.time is not None

    def test_pine_script_format(self):
        """Pine Script sends bullish/bearish sentiment and no quantity."""
        p = WebhookPayload(
            action="buy", sentiment="bullish", ticker="MNQ"
        )
        assert p.sentiment == "long"  # mapped from bullish
        assert p.quantity == 1  # default

    def test_pine_script_bearish(self):
        p = WebhookPayload(
            action="sell", sentiment="bearish", ticker="MNQ"
        )
        assert p.sentiment == "short"

    def test_type_field_entry(self):
        p = WebhookPayload(
            action="buy", sentiment="bullish", ticker="MNQ", type="entry"
        )
        assert p.is_entry is True
        assert p.is_exit is False

    def test_type_field_exit(self):
        p = WebhookPayload(
            action="sell", sentiment="flat", ticker="MNQ", type="exit"
        )
        assert p.is_entry is False
        assert p.is_exit is True

    def test_type_field_eod_close(self):
        p = WebhookPayload(
            action="sell", sentiment="flat", ticker="MNQ", type="eod_close"
        )
        assert p.is_exit is True

    def test_default_quantity(self):
        """Quantity defaults to 1 when not provided."""
        p = WebhookPayload(
            action="buy", sentiment="long", ticker="MXF"
        )
        assert p.quantity == 1
