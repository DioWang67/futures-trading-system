from __future__ import annotations

import pytest

from notifier import Notifier


class _CaptureNotifier(Notifier):
    def __init__(self) -> None:
        super().__init__(telegram_token="token", telegram_chat_id="chat")
        self.messages: list[str] = []

    async def _send(self, text: str) -> None:
        self.messages.append(text)


@pytest.mark.asyncio
async def test_send_protective_event_formats_slippage():
    notifier = _CaptureNotifier()

    await notifier.send_protective_event(
        broker="shioaji",
        ticker="TMF",
        side="long",
        quantity=1,
        trigger_reason="stop_loss",
        status="filled",
        trigger_price=19948.0,
        submit_price=19946.0,
        fill_price=19944.0,
        slippage_points=-4.0,
        execution_price_type="LMT",
    )

    assert len(notifier.messages) == 1
    message = notifier.messages[0]
    assert "Protective Exit" in message
    assert "TMF long x1" in message
    assert "Status: filled" in message
    assert "Slippage: -4.0 pts" in message
