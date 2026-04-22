"""Security-focused regression tests for webhook/auth/control-plane fixes."""

from __future__ import annotations

import hashlib
import hmac
import time
from types import SimpleNamespace
import importlib
from pathlib import Path

import pytest
from fastapi import HTTPException
from pydantic import SecretStr

import main
webhook_router = importlib.import_module("webhook.router")
from webhook.validator import build_signature_payload


class TestWebhookAuthHardening:
    @pytest.mark.asyncio
    async def test_verify_request_requires_timestamp(self, monkeypatch):
        monkeypatch.setattr(
            webhook_router.settings.webhook,
            "secret",
            SecretStr("webhook-secret"),
        )

        body = b'{"action":"buy","ticker":"MXF"}'
        signed = build_signature_payload(body, "1712345678")
        sig = hmac.new(b"webhook-secret", signed, hashlib.sha256).hexdigest()

        with pytest.raises(HTTPException, match="Missing timestamp"):
            await webhook_router._verify_request(body, sig, "", "")

    @pytest.mark.asyncio
    async def test_verify_request_rejects_missing_signature_when_legacy_disabled(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            webhook_router.settings.webhook,
            "secret",
            SecretStr("webhook-secret"),
        )
        monkeypatch.setattr(
            webhook_router.settings.webhook,
            "allow_legacy_secret_header",
            False,
        )

        with pytest.raises(HTTPException, match="Missing signature"):
            await webhook_router._verify_request(
                b'{"action":"buy"}',
                "",
                str(int(time.time())),
                "webhook-secret",
            )

    @pytest.mark.asyncio
    async def test_verify_request_accepts_legacy_header_only_when_explicitly_enabled(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            webhook_router.settings.webhook,
            "secret",
            SecretStr("webhook-secret"),
        )
        monkeypatch.setattr(
            webhook_router.settings.webhook,
            "allow_legacy_secret_header",
            True,
        )

        await webhook_router._verify_request(
            b'{"action":"buy"}',
            "",
            str(int(time.time())),
            "webhook-secret",
        )

    def test_generated_idempotency_key_is_stable_for_same_signed_request(self):
        body = b'{"action":"buy","ticker":"MXF"}'
        timestamp = "1712345678"
        first = webhook_router._generate_idempotency_key(body, timestamp)
        second = webhook_router._generate_idempotency_key(body, timestamp)
        assert first == second


class TestBrokerTickerResolution:
    def test_ticker_routes_only_to_true_supported_broker(self):
        shioaji = SimpleNamespace(supported_tickers={"MXF"})
        rithmic = SimpleNamespace(supported_tickers={"MES"})

        broker, shioaji_result, rithmic_result = webhook_router._resolve_broker_for_ticker(
            "MES", shioaji, rithmic
        )
        assert broker is rithmic
        assert shioaji_result["status"] == "skipped"
        assert rithmic_result == {}

    def test_unsupported_ticker_is_rejected_instead_of_silently_remapped(self):
        shioaji = SimpleNamespace(supported_tickers={"MXF"})
        rithmic = SimpleNamespace(supported_tickers={"MES"})

        broker, shioaji_result, rithmic_result = webhook_router._resolve_broker_for_ticker(
            "NQ", shioaji, rithmic
        )
        assert broker is None
        assert shioaji_result["status"] == "rejected"
        assert rithmic_result["status"] == "rejected"


class TestAdminControlAuth:
    def test_admin_secret_accepts_valid_header(self, monkeypatch):
        monkeypatch.setattr(main.settings.admin, "secret", SecretStr("admin-secret"))
        assert main._require_admin_secret("admin-secret") is None

    def test_admin_secret_rejects_invalid_header(self, monkeypatch):
        monkeypatch.setattr(main.settings.admin, "secret", SecretStr("admin-secret"))
        with pytest.raises(HTTPException, match="Invalid admin secret"):
            main._require_admin_secret("wrong-secret")


class TestSharedSecretValidation:
    def test_require_webhook_rejects_short_secret(self, monkeypatch):
        monkeypatch.setattr(main.settings.webhook, "secret", SecretStr("short-secret"))
        with pytest.raises(ValueError, match="at least 16 characters"):
            main.settings.require_webhook()

    def test_require_admin_rejects_short_secret(self, monkeypatch):
        monkeypatch.setattr(main.settings.admin, "secret", SecretStr("short-admin"))
        with pytest.raises(ValueError, match="at least 16 characters"):
            main.settings.require_admin()


class TestDeployFiles:
    def test_dockerfile_installs_tzdata_for_zoneinfo(self):
        dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
        assert "tzdata" in dockerfile
