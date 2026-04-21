from __future__ import annotations

import asyncio
from typing import Any, Optional

from async_rithmic import RithmicClient, ExchangeOrderNotificationType
from loguru import logger

from config import settings
from position_state import PositionState
from risk_manager import RiskManager
from trade_store import TradeStore
from brokers.base import compute_realized_pnl, route_order


# MES contract multiplier: 5 USD per index point
RITHMIC_POINT_VALUE = 5.0


class RithmicBroker:
    """Rithmic broker for MES."""

    SUPPORTED_TICKERS = {"MES"}

    def __init__(
        self,
        risk_manager: Optional[RiskManager] = None,
        trade_store: Optional[TradeStore] = None,
        notifier: Optional[Any] = None,
    ) -> None:
        self.client: RithmicClient | None = None
        self.account_id: str = ""
        self._contracts: dict[str, Any] = {}
        self.position = PositionState("rithmic")
        self._connected = False
        self._risk_manager = risk_manager
        self._trade_store = trade_store
        self._notifier = notifier

    @property
    def broker_name(self) -> str:
        return "rithmic"

    @property
    def supported_tickers(self) -> set[str]:
        return set(self.SUPPORTED_TICKERS)

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
    async def connect(self) -> None:
        cfg = settings.rithmic
        try:
            self.client = RithmicClient(
                user=cfg.user,
                password=cfg.password.get_secret_value(),
                system_name=cfg.system_name,
                app_name=cfg.app_name,
                url=cfg.url,
            )

            await self.client.connect()

            accounts = await self.client.get_accounts()
            if not accounts:
                logger.error("[Rithmic] No accounts found")
                return
            self.account_id = accounts[0]
            logger.info("[Rithmic] Using account: {}", self.account_id)

            self._contracts["MES"] = await self.client.get_front_month_contract(
                "MES", "CME"
            )
            logger.info("[Rithmic] MES contract: {}", self._contracts["MES"])

            self.client.on_exchange_order_notification += self._on_fill

            # Restore position from trade store on startup
            self._restore_position()

            self._connected = True
            logger.info("[Rithmic] Connected successfully")
        except Exception as e:
            logger.error("[Rithmic] Connection failed: {}", e)
            self._connected = False
            raise

    async def disconnect(self) -> None:
        if self.client:
            try:
                # Save position before disconnect
                if self._trade_store:
                    pos = self.position.get_position()
                    self._trade_store.save_position_snapshot(
                        "rithmic", pos.side, pos.quantity, pos.entry_price
                    )
                await self.client.disconnect()
                logger.info("[Rithmic] Disconnected")
            except Exception as e:
                logger.error("[Rithmic] Disconnect error: {}", e)
            finally:
                self._connected = False

    async def reconnect(self) -> None:
        """Reconnect after connection drop."""
        logger.info("[Rithmic] Reconnecting...")
        try:
            await self.disconnect()
        except Exception:
            pass
        await self.connect()

    def _restore_position(self) -> None:
        """Restore position from trade store on startup."""
        if not self._trade_store:
            return
        snapshot = self._trade_store.get_latest_position("rithmic")
        if snapshot and snapshot["side"] != "flat":
            self.position.update_position(
                snapshot["side"], snapshot["quantity"], snapshot["entry_price"]
            )
            logger.info(
                "[Rithmic] Restored position from DB: {} x{}",
                snapshot["side"], snapshot["quantity"],
            )

    # ------------------------------------------------------------------
    # Fill callback
    # ------------------------------------------------------------------
    async def _on_fill(self, notification: Any) -> None:
        try:
            if notification.type != ExchangeOrderNotificationType.FILL:
                return

            logger.info(
                "[Rithmic] Fill notification: side={}, qty={}, price={}",
                notification.side,
                notification.filled_quantity,
                notification.fill_price,
            )

            side = str(notification.side)
            # Normalize Rithmic side (may arrive as "B"/"S" or "BUY"/"SELL")
            if side.upper().startswith("B"):
                action = "Buy"
            elif side.upper().startswith("S"):
                action = "Sell"
            else:
                logger.error("[Rithmic] Unknown fill side: {}", side)
                return

            fill_info = self.position.apply_fill(
                action,
                notification.filled_quantity,
                notification.fill_price,
            )
            realized_pnl = compute_realized_pnl(
                action=action,
                closed_qty=fill_info["closed_qty"],
                fill_price=notification.fill_price,
                prev_side=fill_info["prev_side"],
                prev_entry=fill_info["prev_entry"],
                point_value=RITHMIC_POINT_VALUE,
            )
            is_closing = fill_info["closed_qty"] > 0

            # Persist fill (linked to the oldest still-open order for this side)
            if self._trade_store:
                order_id = self._trade_store.claim_pending_order(
                    "rithmic", action, fill_qty=notification.filled_quantity,
                )
                self._trade_store.record_fill(
                    "rithmic",
                    action,
                    notification.filled_quantity,
                    notification.fill_price,
                    pnl=realized_pnl,
                    order_id=order_id,
                )
                pos = self.position.get_position()
                self._trade_store.save_position_snapshot(
                    "rithmic", pos.side, pos.quantity, pos.entry_price
                )

            # Feed realized PnL back into risk manager.
            if self._risk_manager and is_closing:
                self._risk_manager.record_fill(
                    "rithmic", realized_pnl, is_closing_trade=True,
                )

            # Notify
            if self._notifier:
                await self._notifier.send_fill(
                    "rithmic",
                    action,
                    notification.filled_quantity,
                    notification.fill_price,
                )

        except Exception as e:
            logger.error("[Rithmic] Fill callback error: {}", e)

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------
    async def _submit_order(self, action: str, quantity: int, ticker: str) -> dict:
        contract = self._contracts.get(ticker)
        if contract is None:
            return {"status": "error", "reason": f"unsupported ticker: {ticker}"}

        order = await self.client.submit_order(
            contract=contract,
            account_id=self.account_id,
            side=action,
            quantity=quantity,
            order_type="Market",
        )
        logger.info(
            "[Rithmic] Order submitted: {} {} x{} -> {}",
            ticker, action, quantity, order,
        )
        return {
            "status": "submitted",
            "action": action,
            "quantity": quantity,
            "ticker": ticker,
            "order": str(order),
        }

    async def place_order(
        self, action: str, quantity: int, idempotency_key: str = "", **kwargs
    ) -> dict:
        ticker = str(kwargs.get("ticker") or "").upper()
        if not ticker:
            ticker = next(iter(self.SUPPORTED_TICKERS))
        if ticker not in self.SUPPORTED_TICKERS:
            return {
                "status": "error",
                "broker": self.broker_name,
                "reason": f"unsupported ticker: {ticker}",
            }

        return await route_order(
            broker_name=self.broker_name,
            position=self.position,
            action=action,
            quantity=quantity,
            submit_fn=lambda order_action, order_qty: self._submit_order(
                order_action, order_qty, ticker
            ),
            is_connected=self._connected and self.client is not None,
            risk_manager=self._risk_manager,
            trade_store=self._trade_store,
            idempotency_key=idempotency_key,
        )
