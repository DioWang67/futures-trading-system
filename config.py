"""
Unified configuration — single source of truth.

Uses pydantic-settings to:
  1. Load .env automatically
  2. Validate all values at startup (fail fast)
  3. Mask secrets in repr / logs
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

MIN_SHARED_SECRET_LENGTH = 16


class ShioajiSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SHIOAJI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: SecretStr = Field(default=SecretStr(""), description="Shioaji API key")
    secret_key: SecretStr = Field(default=SecretStr(""), description="Shioaji secret")
    simulation: bool = Field(
        default=False,
        description="Use Sinopac Shioaji simulation / paper-trading mode",
    )
    runtime_dir: str = Field(
        default=".tmp/shioaji_runtime",
        description="Runtime directory for Shioaji cache/log files in simulation mode",
    )
    futures_symbol: str = Field(
        default="TMF",
        description="Default TAIFEX futures symbol to trade on Shioaji",
    )
    point_value: float = Field(
        default=10.0,
        description="TWD value per index point for the configured futures symbol",
    )
    enable_quote_monitor: bool = Field(
        default=True,
        description="Subscribe tick quotes so protective exits can trigger immediately",
    )
    quote_stale_seconds: int = Field(
        default=15,
        description="Mark quote feed stale when no tick arrives within this many seconds",
    )
    protective_watchdog_interval: int = Field(
        default=5,
        description="Seconds between protective-exit watchdog checks",
    )
    protective_exit_price_type: str = Field(
        default="MKT",
        description="Execution mode for protective exits: MKT, MKP, or LMT",
    )
    protective_limit_offset_points: float = Field(
        default=2.0,
        description="For LMT protective exits, how many points through the trigger to price the order",
    )
    ca_path: str = Field(default="", description="Path to CA certificate (.pfx)")
    ca_password: SecretStr = Field(default=SecretStr(""), description="CA password")

    @field_validator("ca_path")
    @classmethod
    def validate_ca_path(cls, v: str) -> str:
        if v and not Path(v).exists():
            raise ValueError(f"CA certificate not found: {v}")
        return v

    @field_validator("futures_symbol")
    @classmethod
    def validate_futures_symbol(cls, v: str) -> str:
        return str(v or "").strip().upper()

    @field_validator("point_value")
    @classmethod
    def validate_point_value(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("SHIOAJI_POINT_VALUE must be > 0")
        return float(v)

    @field_validator("quote_stale_seconds", "protective_watchdog_interval")
    @classmethod
    def validate_positive_ints(cls, v: int) -> int:
        if int(v) <= 0:
            raise ValueError("watchdog settings must be > 0")
        return int(v)

    @field_validator("protective_exit_price_type")
    @classmethod
    def validate_protective_exit_price_type(cls, v: str) -> str:
        value = str(v or "").strip().upper()
        allowed = {"MKT", "MKP", "LMT"}
        if value not in allowed:
            raise ValueError(
                f"SHIOAJI_PROTECTIVE_EXIT_PRICE_TYPE must be one of {allowed}"
            )
        return value

    @field_validator("protective_limit_offset_points")
    @classmethod
    def validate_limit_offset(cls, v: float) -> float:
        if float(v) < 0:
            raise ValueError("SHIOAJI_PROTECTIVE_LIMIT_OFFSET_POINTS must be >= 0")
        return float(v)


class RithmicSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RITHMIC_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    user: str = Field(default="", description="Rithmic username")
    password: SecretStr = Field(default=SecretStr(""), description="Rithmic password")
    system_name: str = Field(default="Rithmic Test")
    app_name: str = Field(default="my_trading_bot")
    url: str = Field(default="rituz00100.rithmic.com:443")


class WebhookSettings(BaseSettings):
    """Webhook security settings."""

    model_config = SettingsConfigDict(
        env_prefix="WEBHOOK_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    secret: SecretStr = Field(
        default=SecretStr(""),
        description="Shared secret for HMAC-SHA256 verification",
        alias="WEBHOOK_SECRET",
    )
    max_timestamp_drift_seconds: int = Field(
        default=120,
        description=(
            "Max allowed age of webhook timestamp (replay protection). "
            "TradingView webhooks commonly land 10-30 seconds after the "
            "bar closes and server clocks drift a few seconds between "
            "pods; 30s was too tight in practice."
        ),
    )
    allow_legacy_secret_header: bool = Field(
        default=False,
        description="Allow insecure X-Webhook-Secret fallback for legacy senders",
        alias="WEBHOOK_ALLOW_LEGACY_SECRET_HEADER",
    )


class AdminSettings(BaseSettings):
    """Control-plane authentication settings."""

    model_config = SettingsConfigDict(
        env_prefix="ADMIN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    secret: SecretStr = Field(
        default=SecretStr(""),
        description="Shared secret required for admin/control endpoints",
        alias="ADMIN_SECRET",
    )


class RiskSettings(BaseSettings):
    """Risk management settings."""

    model_config = SettingsConfigDict(
        env_prefix="RISK_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    max_daily_loss: float = Field(default=50_000.0, description="Max daily loss ($)")
    max_daily_loss_pct: float = Field(default=0.05, description="Max daily loss (% of capital)")
    max_position_size: int = Field(default=10, description="Max contracts per broker")
    max_order_qty: int = Field(default=5, description="Max contracts per order")
    max_drawdown_pct: float = Field(default=0.15, description="Drawdown circuit breaker (%)")
    cooldown_after_consecutive_losses: int = Field(default=3, description="Losses before cooldown")
    cooldown_seconds: int = Field(default=300, description="Cooldown duration (seconds)")
    initial_capital: float = Field(default=1_000_000.0, description="Starting capital")


class NotificationSettings(BaseSettings):
    """Notification channel settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    telegram_bot_token: SecretStr = Field(
        default=SecretStr(""),
        alias="TELEGRAM_BOT_TOKEN",
    )
    telegram_chat_id: str = Field(default="", alias="TELEGRAM_CHAT_ID")


class Settings(BaseSettings):
    """Top-level settings — aggregates all sub-configs."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Sub-configs
    shioaji: ShioajiSettings = Field(default_factory=ShioajiSettings)
    rithmic: RithmicSettings = Field(default_factory=RithmicSettings)
    webhook: WebhookSettings = Field(default_factory=WebhookSettings)
    admin: AdminSettings = Field(default_factory=AdminSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    notification: NotificationSettings = Field(default_factory=NotificationSettings)

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # App
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v = v.upper()
        if v not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {allowed}, got '{v}'")
        return v

    def require_shioaji(self) -> None:
        """Validate that Shioaji credentials are present. Call at startup."""
        if not self.shioaji.api_key.get_secret_value():
            raise ValueError("SHIOAJI_API_KEY is required but not set")
        if not self.shioaji.secret_key.get_secret_value():
            raise ValueError("SHIOAJI_SECRET_KEY is required but not set")

    def require_rithmic(self) -> None:
        """Validate that Rithmic credentials are present. Call at startup."""
        if not self.rithmic.user:
            raise ValueError("RITHMIC_USER is required but not set")
        if not self.rithmic.password.get_secret_value():
            raise ValueError("RITHMIC_PASSWORD is required but not set")

    def require_webhook(self) -> None:
        """Validate that webhook secret is set. Call at startup."""
        secret = self.webhook.secret.get_secret_value()
        if not secret:
            raise ValueError(
                "WEBHOOK_SECRET is required but not set. "
                "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
            )
        if len(secret) < MIN_SHARED_SECRET_LENGTH:
            raise ValueError(
                "WEBHOOK_SECRET must be at least "
                f"{MIN_SHARED_SECRET_LENGTH} characters long. "
                "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
            )

    def require_admin(self) -> None:
        """Validate that admin endpoint authentication is configured."""
        secret = self.admin.secret.get_secret_value()
        if not secret:
            raise ValueError(
                "ADMIN_SECRET is required but not set. "
                "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
            )
        if len(secret) < MIN_SHARED_SECRET_LENGTH:
            raise ValueError(
                "ADMIN_SECRET must be at least "
                f"{MIN_SHARED_SECRET_LENGTH} characters long. "
                "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()


# Backwards-compatible alias
settings = get_settings()
