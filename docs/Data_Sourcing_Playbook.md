# Data Sourcing Playbook

TradeML Phase 1 uses free or low-cost vendors with explicit point-in-time and entitlement constraints.

## Primary roles

- `Alpaca`: primary daily equities bars. Requires both API key and secret for live market-data requests.
- `Massive (Polygon.io)`: cross-check bars plus splits/dividends/ticker reference.
- `Finnhub`: backup reference and earnings calendar. Candle access may depend on entitlement.
- `Alpha Vantage`: listing status and corporate-action reference.
- `FRED / ALFRED`: macro series plus vintage dates for PIT-aware research.
- `FMP`: delistings and earnings via current stable endpoints.
- `SEC EDGAR`: authoritative filing timeline and company submissions.

## PIT rules

- Raw vendor files remain immutable.
- Corporate actions are applied only in curated outputs.
- Earnings timing is risk-control-only unless sourced from PIT-safe filings.
- Training must gate on GREEN coverage in the actual training window.

## Current live-smoke notes

- Massive, Alpha Vantage, FRED, FMP stable endpoints, Finnhub profile/earnings, and SEC EDGAR passed live smoke with the current `.env`.
- Alpaca live bars are blocked unless `ALPACA_API_SECRET` is present.
- Finnhub candle access may skip cleanly when the account is not entitled.

## Endpoint summary

- Alpaca: `GET /v2/stocks/bars`
- Massive: `GET /v2/aggs/ticker/{symbol}/range/1/day/{from}/{to}`
- Finnhub: `GET /api/v1/stock/candle`, `GET /api/v1/calendar/earnings`, `GET /api/v1/stock/profile2`
- Alpha Vantage: `GET /query?function=LISTING_STATUS`
- FRED: `GET /fred/series/observations`, `GET /fred/series/vintagedates`
- FMP stable: `GET /stable/delisted-companies`, `GET /stable/earnings-calendar`
- SEC EDGAR: `GET /submissions/CIK{cik}.json`
