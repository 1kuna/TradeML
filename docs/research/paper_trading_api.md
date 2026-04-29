# Paper Trading API

TradeML uses Alpaca as the default paper-money API target.

Official Alpaca docs checked on 2026-04-29:

- Paper trading is free and available to all Alpaca users: <https://docs.alpaca.markets/docs/paper-trading>
- Alpaca Trading API paper mode uses the paper endpoint: `https://paper-api.alpaca.markets`: <https://docs.alpaca.markets/docs/paper-trading>
- Orders are submitted to the Trading API `/v2/orders` endpoint, and Alpaca SDK examples use `paper=True`: <https://docs.alpaca.markets/docs/working-with-orders>

Repo policy:

- Research artifacts may write paper/shadow signals, target weights, paper order deltas, and mature paper PnL.
- Alpaca paper order submission is disabled by default via `paper_policy.broker.submit_orders_enabled: false`.
- The paper client refuses non-paper Alpaca base URLs.
- No live Alpaca trading endpoint is configured or used by the research autopilot.

