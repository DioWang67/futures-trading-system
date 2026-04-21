from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import paper_dashboard


class _FakeTradeStore:
    def __init__(self, db_path=None):
        self.db_path = db_path

    def get_latest_position(self, broker: str):
        return {
            "side": "long",
            "quantity": 1,
            "entry_price": 20000.0,
            "snapshot_at": "2026-04-21T09:15:05+00:00",
        }

    def get_daily_summary(self):
        return {
            "date": "2026-04-21",
            "brokers": [
                {
                    "broker": "shioaji",
                    "trade_count": 1,
                    "total_pnl": 500.0,
                    "total_commission": 0.0,
                    "wins": 1,
                    "losses": 0,
                }
            ],
            "total_pnl": 500.0,
            "total_trades": 1,
        }

    def get_recent_fills(self, limit: int):
        return [
            {
                "broker": "shioaji",
                "action": "Sell",
                "filled_qty": 1,
                "fill_price": 20100.0,
                "pnl": 500.0,
                "filled_at": "2026-04-21T09:30:00+00:00",
            }
        ]

    def get_recent_protective_events(self, limit: int):
        return [
            {
                "ticker": "TMF",
                "side": "long",
                "quantity": 1,
                "trigger_reason": "stop_loss",
                "trigger_price": 19948.0,
                "submit_price": 19946.0,
                "fill_price": 19944.0,
                "slippage_points": -4.0,
                "execution_price_type": "LMT",
                "status": "filled",
                "triggered_at": "2026-04-21T09:20:00+00:00",
            }
        ]

    def get_recent_risk_events(self, limit: int):
        return [
            {
                "event_type": "order_rejected",
                "broker": "shioaji",
                "details": "close-only mode active",
                "created_at": "2026-04-21T09:25:00+00:00",
            }
        ]


def test_dashboard_payload_reads_runner_state_and_trade_store(monkeypatch):
    root = Path(".tmp") / "test_paper_dashboard" / uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "runner_state.json"

    state_path.write_text(
        json.dumps(
            {
                "last_processed_bar_time": "2026-04-21T09:15:00",
                "managed_trade": {
                    "status": "active",
                    "side": "long",
                    "quantity": 1,
                    "sl": 19900.0,
                    "tp": 20200.0,
                    "entry_bar_time": "2026-04-21T09:15:00",
                    "submitted_at": "2026-04-21T09:15:02+00:00",
                    "exit_reason": "",
                },
                "metadata": {
                    "symbol": "MXF",
                    "ltf_freq": "15min",
                    "htf_freq": "60min",
                    "quantity": 1,
                },
                "runtime": {
                    "status": "sleeping",
                    "last_cycle_at": paper_dashboard._utc_now().isoformat(),
                    "next_cycle_eta": paper_dashboard._utc_now().isoformat(),
                    "last_cycle_result": {"status": "skipped", "reason": "no action"},
                    "protective_exit": {
                        "status": "armed",
                        "side": "long",
                        "quantity": 1,
                        "stop_loss": 19900.0,
                        "take_profit": 20200.0,
                    },
                    "watchdog": {
                        "close_only": False,
                        "close_only_reason": "",
                        "watchdog_reason": "",
                    },
                    "quote": {
                        "last_tick_price": 20010.0,
                        "age_seconds": 1.2,
                        "is_stale": False,
                        "stale_after_seconds": 15,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(paper_dashboard, "DEFAULT_STATE_PATH", state_path)
    monkeypatch.setattr(paper_dashboard, "TradeStore", _FakeTradeStore)

    payload = paper_dashboard._build_dashboard_payload()

    assert payload["runner"]["status"] == "sleeping"
    assert payload["runner"]["metadata"]["symbol"] == "MXF"
    assert payload["runner"]["health"] == "ok"
    assert payload["runner"]["protective_exit"]["status"] == "armed"
    assert payload["runner"]["quote"]["last_tick_price"] == 20010.0
    assert payload["position"]["side"] == "long"
    assert payload["summary"]["total_trades"] == 1
    assert payload["recent_fills"][0]["broker"] == "shioaji"
    assert payload["recent_protective_events"][0]["status"] == "filled"
    assert payload["recent_risk_events"][0]["event_type"] == "order_rejected"
