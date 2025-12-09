# Testing TradeML

## Suites
- Unit: logic-level coverage (`pytest tests/unit -q`)
- Integration: synthetic data + worker/fetchers/training (`pytest tests/integration -q`)
- Live endpoints: real vendor calls (`pytest tests/integration/test_live_endpoints.py -m liveapi -q`)

## Env
- For local/synthetic runs, no API keys required.
- Live endpoints: set `ALPACA_API_KEY/SECRET`, `FINNHUB_API_KEY`, `FRED_API_KEY`, `ALPHA_VANTAGE_API_KEY`, `MASSIVE_API_KEY` (optional).

## Notes
- Tests isolate state under `DATA_ROOT` temp dirs; no cleanup needed.
- Live tests skip automatically if credentials are missing.
