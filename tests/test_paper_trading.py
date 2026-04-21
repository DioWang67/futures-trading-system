from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import SecretStr

from config import Settings
import brokers.shioaji_broker as shioaji_broker_module
from risk_manager import RiskConfig, RiskManager
from trade_store import TradeStore


class TestSettingsEnvLoading:
    def test_nested_shioaji_env_vars_load_into_settings(self, monkeypatch):
        monkeypatch.setenv("SHIOAJI_API_KEY", "paper-api-key")
        monkeypatch.setenv("SHIOAJI_SECRET_KEY", "paper-secret-key")
        monkeypatch.setenv("SHIOAJI_SIMULATION", "true")
        settings = Settings(_env_file=None)

        assert settings.shioaji.api_key.get_secret_value() == "paper-api-key"
        assert settings.shioaji.secret_key.get_secret_value() == "paper-secret-key"
        assert settings.shioaji.simulation is True


class _FakeShioajiApi:
    def __init__(self, simulation: bool = False, place_order_exc: Exception | None = None):
        self.simulation = simulation
        self.place_order_exc = place_order_exc
        self.activate_ca_calls = []
        self.login_calls = []
        self.order_requests = []
        self.callback = None
        self.session_down_callback = None
        self.quote_subscriptions = []
        self.tick_callback = None
        self.quote = SimpleNamespace(
            set_on_tick_fop_v1_callback=self._set_tick_callback,
            subscribe=self._subscribe,
        )
        self.Contracts = SimpleNamespace(
            Futures={"TMF": [SimpleNamespace(code="TMFE6", delivery_date="2026/05/20")]}
        )
        self.futopt_account = SimpleNamespace(account_type="F")

    def _set_tick_callback(self, cb):
        self.tick_callback = cb

    def _subscribe(self, contract, **kwargs):
        self.quote_subscriptions.append((contract, kwargs))

    def login(self, **kwargs):
        self.login_calls.append(kwargs)
        return ["ok"]

    def activate_ca(self, **kwargs):
        self.activate_ca_calls.append(kwargs)

    def set_order_callback(self, cb):
        self.callback = cb

    def set_session_down_callback(self, cb):
        self.session_down_callback = cb

    def list_positions(self, account):
        return []

    def logout(self):
        return True

    def Order(self, **kwargs):
        return SimpleNamespace(**kwargs)

    def place_order(self, contract, order):
        if self.place_order_exc:
            raise self.place_order_exc
        self.order_requests.append((contract, order))
        return {
            "contract": contract.code,
            "action": order.action,
            "quantity": order.quantity,
            "order": {"id": f"trade-id-{len(self.order_requests)}"},
        }


def _db_path(name: str) -> Path:
    path = Path(".tmp") / "test_paper_trading" / f"{name}-{uuid4().hex}.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


class TestShioajiBrokerPaperMode:
    def test_login_uses_simulation_and_skips_ca(self, monkeypatch):
        fake_api = _FakeShioajiApi(simulation=True)
        ctor_args = []

        monkeypatch.setattr(shioaji_broker_module.settings.shioaji, "simulation", True)
        monkeypatch.setattr(
            shioaji_broker_module.settings.shioaji, "runtime_dir", ".tmp/test-shioaji-runtime"
        )
        monkeypatch.setattr(
            shioaji_broker_module.settings.shioaji, "api_key", SecretStr("api-key")
        )
        monkeypatch.setattr(
            shioaji_broker_module.settings.shioaji, "secret_key", SecretStr("secret-key")
        )
        monkeypatch.setattr(
            shioaji_broker_module.settings.shioaji, "ca_path", "C:/fake/live-only.pfx"
        )
        monkeypatch.setattr(
            shioaji_broker_module.settings.shioaji, "ca_password", SecretStr("ca-pass")
        )
        monkeypatch.setattr(
            shioaji_broker_module.settings.shioaji, "futures_symbol", "TMF"
        )
        def fake_ctor(simulation=False):
            ctor_args.append(simulation)
            return fake_api

        monkeypatch.setattr(shioaji_broker_module.sj, "Shioaji", fake_ctor)
        monkeypatch.setattr(
            shioaji_broker_module.ShioajiBroker, "_sync_position", lambda self: None
        )

        broker = shioaji_broker_module.ShioajiBroker()
        broker.login()

        assert broker.is_simulation is True
        assert ctor_args == [True]
        assert len(fake_api.login_calls) == 1
        assert fake_api.login_calls[0]["contracts_timeout"] == 10_000
        assert fake_api.activate_ca_calls == []
        assert broker.is_connected is True
        assert broker._contracts["TMF"].code == "TMFE6"
        assert fake_api.tick_callback is not None
        assert fake_api.quote_subscriptions[0][0].code == "TMFE6"

    def test_live_login_activates_ca(self, monkeypatch):
        fake_api = _FakeShioajiApi(simulation=False)
        ctor_args = []

        monkeypatch.setattr(shioaji_broker_module.settings.shioaji, "simulation", False)
        monkeypatch.setattr(
            shioaji_broker_module.settings.shioaji, "api_key", SecretStr("api-key")
        )
        monkeypatch.setattr(
            shioaji_broker_module.settings.shioaji, "secret_key", SecretStr("secret-key")
        )
        monkeypatch.setattr(
            shioaji_broker_module.settings.shioaji, "ca_path", "C:/live/live-cert.pfx"
        )
        monkeypatch.setattr(
            shioaji_broker_module.settings.shioaji, "ca_password", SecretStr("ca-pass")
        )
        monkeypatch.setattr(
            shioaji_broker_module.settings.shioaji, "futures_symbol", "TMF"
        )
        def fake_ctor(simulation=False):
            ctor_args.append(simulation)
            return fake_api

        monkeypatch.setattr(shioaji_broker_module.sj, "Shioaji", fake_ctor)
        monkeypatch.setattr(
            shioaji_broker_module.ShioajiBroker, "_sync_position", lambda self: None
        )

        broker = shioaji_broker_module.ShioajiBroker()
        broker.login()

        assert broker.is_simulation is False
        assert ctor_args == [False]
        assert fake_api.activate_ca_calls[0]["ca_path"] == "C:/live/live-cert.pfx"

    def test_simulation_place_order_exception_is_treated_as_submitted(self, monkeypatch):
        fake_api = _FakeShioajiApi(
            simulation=True,
            place_order_exc=Exception("Topic: api/v1/paper/place_order, payload: {...}"),
        )

        monkeypatch.setattr(shioaji_broker_module.settings.shioaji, "simulation", True)
        monkeypatch.setattr(
            shioaji_broker_module.settings.shioaji, "api_key", SecretStr("api-key")
        )
        monkeypatch.setattr(
            shioaji_broker_module.settings.shioaji, "secret_key", SecretStr("secret-key")
        )
        monkeypatch.setattr(
            shioaji_broker_module.settings.shioaji, "ca_path", ""
        )
        monkeypatch.setattr(
            shioaji_broker_module.settings.shioaji, "futures_symbol", "TMF"
        )
        monkeypatch.setattr(
            shioaji_broker_module.sj, "Shioaji", lambda simulation=False: fake_api
        )
        monkeypatch.setattr(
            shioaji_broker_module.ShioajiBroker, "_sync_position", lambda self: None
        )

        broker = shioaji_broker_module.ShioajiBroker()
        broker.login()
        result = broker._submit_order_sync("Buy", 1, "TMF")

        assert result["status"] == "submitted"
        assert "simulation submit accepted" in result["reason"]

    def test_watchdog_enters_close_only_without_protective_exit(self, monkeypatch):
        fake_api = _FakeShioajiApi(simulation=True)

        monkeypatch.setattr(shioaji_broker_module.settings.shioaji, "simulation", True)
        monkeypatch.setattr(
            shioaji_broker_module.settings.shioaji, "api_key", SecretStr("api-key")
        )
        monkeypatch.setattr(
            shioaji_broker_module.settings.shioaji, "secret_key", SecretStr("secret-key")
        )
        monkeypatch.setattr(
            shioaji_broker_module.settings.shioaji, "ca_path", ""
        )
        monkeypatch.setattr(
            shioaji_broker_module.settings.shioaji, "futures_symbol", "TMF"
        )
        monkeypatch.setattr(
            shioaji_broker_module.settings.shioaji, "quote_stale_seconds", 999
        )
        monkeypatch.setattr(
            shioaji_broker_module.sj, "Shioaji", lambda simulation=False: fake_api
        )
        monkeypatch.setattr(
            shioaji_broker_module.ShioajiBroker, "_sync_position", lambda self: None
        )

        risk = RiskManager(RiskConfig())
        broker = shioaji_broker_module.ShioajiBroker(risk_manager=risk)
        broker.login()
        broker.position.update_position("long", 1, entry_price=20000.0)

        status = broker.evaluate_protective_watchdog()

        assert status["close_only"] is True
        assert "no protective exit" in status["watchdog_reason"]

    def test_watchdog_accepts_fresh_quote_with_armed_protection(self, monkeypatch):
        fake_api = _FakeShioajiApi(simulation=True)

        monkeypatch.setattr(shioaji_broker_module.settings.shioaji, "simulation", True)
        monkeypatch.setattr(
            shioaji_broker_module.settings.shioaji, "api_key", SecretStr("api-key")
        )
        monkeypatch.setattr(
            shioaji_broker_module.settings.shioaji, "secret_key", SecretStr("secret-key")
        )
        monkeypatch.setattr(
            shioaji_broker_module.settings.shioaji, "ca_path", ""
        )
        monkeypatch.setattr(
            shioaji_broker_module.settings.shioaji, "futures_symbol", "TMF"
        )
        monkeypatch.setattr(
            shioaji_broker_module.settings.shioaji, "quote_stale_seconds", 15
        )
        monkeypatch.setattr(
            shioaji_broker_module.sj, "Shioaji", lambda simulation=False: fake_api
        )
        monkeypatch.setattr(
            shioaji_broker_module.ShioajiBroker, "_sync_position", lambda self: None
        )

        risk = RiskManager(RiskConfig())
        broker = shioaji_broker_module.ShioajiBroker(risk_manager=risk)
        broker.login()
        broker.position.update_position("long", 1, entry_price=20000.0)
        broker.arm_protective_exit("TMF", "long", 1, stop_loss=19950.0, take_profit=20100.0)
        tick = SimpleNamespace(code="TMFE6", close=20010.0)
        broker._tick_callback(None, tick)

        status = broker.evaluate_protective_watchdog()

        assert status["close_only"] is False
        assert status["watchdog_reason"] == ""
        assert status["quote"]["is_stale"] is False

    @pytest.mark.asyncio
    async def test_wait_for_fresh_quote_returns_latest_tick(self, monkeypatch):
        fake_api = _FakeShioajiApi(simulation=True)

        monkeypatch.setattr(shioaji_broker_module.settings.shioaji, "simulation", True)
        monkeypatch.setattr(
            shioaji_broker_module.settings.shioaji, "api_key", SecretStr("api-key")
        )
        monkeypatch.setattr(
            shioaji_broker_module.settings.shioaji, "secret_key", SecretStr("secret-key")
        )
        monkeypatch.setattr(shioaji_broker_module.settings.shioaji, "ca_path", "")
        monkeypatch.setattr(
            shioaji_broker_module.settings.shioaji, "futures_symbol", "TMF"
        )
        monkeypatch.setattr(
            shioaji_broker_module.settings.shioaji, "quote_stale_seconds", 15
        )
        monkeypatch.setattr(
            shioaji_broker_module.sj, "Shioaji", lambda simulation=False: fake_api
        )
        monkeypatch.setattr(
            shioaji_broker_module.ShioajiBroker, "_sync_position", lambda self: None
        )

        broker = shioaji_broker_module.ShioajiBroker()
        broker.login()
        broker._tick_callback(None, SimpleNamespace(code="TMFE6", close=20012.0))

        quote = await broker.wait_for_fresh_quote(timeout_seconds=1.0)

        assert quote["is_stale"] is False
        assert quote["last_tick_price"] == 20012.0

    @pytest.mark.asyncio
    async def test_manual_trigger_protective_exit_uses_public_hook(self, monkeypatch):
        fake_api = _FakeShioajiApi(simulation=True)

        monkeypatch.setattr(shioaji_broker_module.settings.shioaji, "simulation", True)
        monkeypatch.setattr(
            shioaji_broker_module.settings.shioaji, "api_key", SecretStr("api-key")
        )
        monkeypatch.setattr(
            shioaji_broker_module.settings.shioaji, "secret_key", SecretStr("secret-key")
        )
        monkeypatch.setattr(shioaji_broker_module.settings.shioaji, "ca_path", "")
        monkeypatch.setattr(
            shioaji_broker_module.settings.shioaji, "futures_symbol", "TMF"
        )
        monkeypatch.setattr(
            shioaji_broker_module.sj, "Shioaji", lambda simulation=False: fake_api
        )
        monkeypatch.setattr(
            shioaji_broker_module.ShioajiBroker, "_sync_position", lambda self: None
        )

        broker = shioaji_broker_module.ShioajiBroker()
        broker.login()
        broker.position.update_position("long", 1, entry_price=20000.0)
        broker.arm_protective_exit("TMF", "long", 1, stop_loss=19950.0, take_profit=20100.0)

        captured: dict[str, object] = {}

        async def fake_submit(ticker: str, reason: str, trigger_price: float) -> None:
            captured["ticker"] = ticker
            captured["reason"] = reason
            captured["trigger_price"] = trigger_price

        monkeypatch.setattr(broker, "_submit_protective_exit", fake_submit)
        result = await broker.trigger_protective_exit("TMF", reason="manual_test", trigger_price=19980.0)
        protective = broker.get_protective_exit("TMF")

        assert result["status"] == "ok"
        assert captured == {
            "ticker": "TMF",
            "reason": "manual_test",
            "trigger_price": 19980.0,
        }
        assert protective is not None
        assert protective["status"] == "triggered"
        assert protective["trigger_source"] == "manual_test"
        assert protective["trigger_price"] == 19980.0

    def test_submit_order_sync_uses_limit_price_for_protective_lmt(self, monkeypatch):
        fake_api = _FakeShioajiApi(simulation=True)

        monkeypatch.setattr(shioaji_broker_module.settings.shioaji, "simulation", True)
        monkeypatch.setattr(
            shioaji_broker_module.settings.shioaji, "api_key", SecretStr("api-key")
        )
        monkeypatch.setattr(
            shioaji_broker_module.settings.shioaji, "secret_key", SecretStr("secret-key")
        )
        monkeypatch.setattr(
            shioaji_broker_module.settings.shioaji, "ca_path", ""
        )
        monkeypatch.setattr(
            shioaji_broker_module.settings.shioaji, "futures_symbol", "TMF"
        )
        monkeypatch.setattr(
            shioaji_broker_module.sj, "Shioaji", lambda simulation=False: fake_api
        )
        monkeypatch.setattr(
            shioaji_broker_module.ShioajiBroker, "_sync_position", lambda self: None
        )

        broker = shioaji_broker_module.ShioajiBroker()
        broker.login()
        result = broker._submit_order_sync("Sell", 1, "TMF", price_type="LMT", limit_price=19945.0)

        assert result["status"] == "submitted"
        assert result["price_type"] == "LMT"
        assert result["limit_price"] == 19945.0
        _, order = fake_api.order_requests[-1]
        assert order.price == 19945.0

    @pytest.mark.asyncio
    async def test_protective_event_records_fill_slippage(self, monkeypatch):
        fake_api = _FakeShioajiApi(simulation=True)

        monkeypatch.setattr(shioaji_broker_module.settings.shioaji, "simulation", True)
        monkeypatch.setattr(
            shioaji_broker_module.settings.shioaji, "api_key", SecretStr("api-key")
        )
        monkeypatch.setattr(
            shioaji_broker_module.settings.shioaji, "secret_key", SecretStr("secret-key")
        )
        monkeypatch.setattr(
            shioaji_broker_module.settings.shioaji, "ca_path", ""
        )
        monkeypatch.setattr(
            shioaji_broker_module.settings.shioaji, "futures_symbol", "TMF"
        )
        monkeypatch.setattr(
            shioaji_broker_module.settings.shioaji, "protective_exit_price_type", "LMT"
        )
        monkeypatch.setattr(
            shioaji_broker_module.settings.shioaji, "protective_limit_offset_points", 2.0
        )
        monkeypatch.setattr(
            shioaji_broker_module.sj, "Shioaji", lambda simulation=False: fake_api
        )
        monkeypatch.setattr(
            shioaji_broker_module.ShioajiBroker, "_sync_position", lambda self: None
        )

        trade_store = TradeStore(":memory:")
        broker = shioaji_broker_module.ShioajiBroker(trade_store=trade_store)
        broker.login()
        broker.position.update_position("long", 1, entry_price=20000.0)
        broker.arm_protective_exit("TMF", "long", 1, stop_loss=19950.0, take_profit=20100.0)

        await broker._submit_protective_exit("TMF", "stop_loss", 19948.0)
        protective = broker.get_protective_exit("TMF")

        assert protective is not None
        assert protective["status"] == "submitted"
        assert protective["execution_price_type"] == "LMT"
        assert protective["submit_price"] == 19946.0

        broker._order_callback(
            None,
            {
                "order": {"action": "Sell", "id": protective["broker_order_id"]},
                "status": {
                    "deal_quantity": 1,
                    "deal_price": 19944.0,
                    "status": "Filled",
                },
                "operation": {"op_code": "00", "op_type": "New"},
            },
        )

        events = trade_store.get_recent_protective_events(limit=1)

        assert len(events) == 1
        assert events[0]["status"] == "filled"
        assert events[0]["execution_price_type"] == "LMT"
        assert events[0]["submit_price"] == 19946.0
        assert events[0]["fill_price"] == 19944.0
        assert events[0]["slippage_points"] == pytest.approx(-4.0)

    def test_futures_deal_callback_updates_position_from_top_level_fields(self, monkeypatch):
        fake_api = _FakeShioajiApi(simulation=True)

        monkeypatch.setattr(shioaji_broker_module.settings.shioaji, "simulation", True)
        monkeypatch.setattr(
            shioaji_broker_module.settings.shioaji, "api_key", SecretStr("api-key")
        )
        monkeypatch.setattr(
            shioaji_broker_module.settings.shioaji, "secret_key", SecretStr("secret-key")
        )
        monkeypatch.setattr(shioaji_broker_module.settings.shioaji, "ca_path", "")
        monkeypatch.setattr(
            shioaji_broker_module.settings.shioaji, "futures_symbol", "TMF"
        )
        monkeypatch.setattr(
            shioaji_broker_module.sj, "Shioaji", lambda simulation=False: fake_api
        )
        monkeypatch.setattr(
            shioaji_broker_module.ShioajiBroker, "_sync_position", lambda self: None
        )

        trade_store = TradeStore(":memory:")
        broker = shioaji_broker_module.ShioajiBroker(trade_store=trade_store)
        broker.login()
        broker._submit_order_sync("Buy", 1, "TMF")

        broker._order_callback(
            "OrderState.FuturesDeal",
            {
                "trade_id": "trade-id-1",
                "seqno": "trade-id-1",
                "ordno": "000C08",
                "action": "Buy",
                "code": "TMFE6",
                "price": 37796.0,
                "quantity": 1,
            },
        )

        position = broker.position.get_position()
        fills = trade_store.get_recent_fills(limit=1)

        assert position.side == "long"
        assert position.quantity == 1
        assert position.entry_price == 37796.0
        assert fills[0]["fill_price"] == 37796.0
