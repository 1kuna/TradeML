"""Alpaca brokerage client for paper/live trading."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional

import pandas as pd
from loguru import logger

try:  # Optional dependency
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest
except Exception:  # pragma: no cover - alpaca dependency may be absent
    TradingClient = None  # type: ignore
    MarketOrderRequest = None  # type: ignore
    OrderSide = None  # type: ignore
    TimeInForce = None  # type: ignore


@dataclass
class AlpacaCredentials:
    api_key: str
    api_secret: str
    paper: bool = True


class AlpacaBrokerClient:
    """Thin wrapper around alpaca-py trading client with safe fallbacks."""

    def __init__(self, creds: AlpacaCredentials):
        self.creds = creds
        if TradingClient is None:
            logger.warning("alpaca-py not installed; broker client will operate in dry-run mode")
            self._client = None
        else:
            self._client = TradingClient(creds.api_key, creds.api_secret, paper=creds.paper)
            logger.info("Alpaca trading client initialized (%s)", "paper" if creds.paper else "live")

    def submit_orders(
        self,
        asof: datetime,
        target_weights: pd.DataFrame,
        policy_cfg: Optional[Dict] = None,
    ) -> List[Dict]:
        """Translate target weights into market orders and submit to Alpaca."""
        policy_cfg = policy_cfg or {}
        notional = float(policy_cfg.get("notional", 1_000_000.0))
        price_map = policy_cfg.get("price_map") or {}
        tif = policy_cfg.get("time_in_force", "day").upper()
        allow_fractional = bool(policy_cfg.get("fractional", True))

        if target_weights.empty:
            logger.info("No positions to trade on %s", asof.date())
            return []

        orders: List[Dict] = []
        for _, row in target_weights.iterrows():
            symbol = row["symbol"]
            weight = float(row.get("target_w", 0.0))
            if abs(weight) < 1e-6:
                continue
            price = price_map.get(symbol)
            if price is None:
                logger.warning("Missing price for %s; skipping order", symbol)
                continue
            qty = weight * notional / price
            if not allow_fractional:
                qty = int(round(qty))
                if qty == 0:
                    continue
            order_payload = {
                "symbol": symbol,
                "qty": float(qty),
                "side": "buy" if weight > 0 else "sell",
                "notional": abs(weight) * notional,
                "time_in_force": tif,
            }
            orders.append(order_payload)

            if self._client and MarketOrderRequest and OrderSide and TimeInForce:
                request = MarketOrderRequest(
                    symbol=symbol,
                    qty=abs(float(qty)),
                    side=OrderSide.BUY if weight > 0 else OrderSide.SELL,
                    time_in_force=getattr(TimeInForce, tif, TimeInForce.DAY),
                    extended_hours=bool(policy_cfg.get("extended_hours", False)),
                )
                try:
                    resp = self._client.submit_order(request)
                    order_payload["alpaca_order_id"] = getattr(resp, "id", None)
                except Exception as exc:  # pragma: no cover - network side-effect
                    logger.exception(f"Alpaca order submission failed for {symbol}: {exc}")
            else:
                logger.debug("Dry run order: %s", order_payload)

        return orders

    def cancel_open_orders(self, symbols: Optional[Iterable[str]] = None) -> int:
        if not self._client:
            logger.warning("Dry run cancel; Alpaca client unavailable")
            return 0
        try:
            open_orders = self._client.get_orders()
        except Exception as exc:  # pragma: no cover
            logger.exception(f"Failed to list Alpaca orders: {exc}")
            return 0
        cancelled = 0
        for order in open_orders:
            if symbols and getattr(order, "symbol", None) not in symbols:
                continue
            try:
                self._client.cancel_order(order.id)
                cancelled += 1
            except Exception as exc:  # pragma: no cover
                logger.warning(f"Failed to cancel order {order.id}: {exc}")
        logger.info("Cancelled %s Alpaca orders", cancelled)
        return cancelled
