# Data Sourcing Playbook

TradeML Phase 1 uses free or low-cost vendors with explicit point-in-time and entitlement constraints.

See also: `docs/Provider_Role_Matrix.md` for the docs-backed role, saturation, and QC policy that the runtime now follows.

## Primary roles

- `Alpaca`: primary forward daily equities bars, recent-history backfill, assets, and corporate actions. Best fit for multi-symbol bar pulls and same-day collection.
- `Tiingo`: deep-history daily equities bars. Best fit for long-tail historical backfill, not broad reference fanout.
- `Massive (Polygon.io)`: independent bar QC plus splits/dividends/ticker reference.
- `Twelve Data`: supplemental bar QC plus dividends, splits, earnings, stocks, and financial statements.
  - Batch `/time_series` requests where possible; spend credits on bars first, then corp-actions/statements.
- `Finnhub`: reference/events only with the current entitlement. Keep for company profile and earnings calendar; do not use for canonical bars unless a live audit re-verifies candle access.
- `Alpha Vantage`: listing status and corporate-action reference.
- `FRED / ALFRED`: macro series plus vintage dates for PIT-aware research.
- `FMP`: delistings and earnings via current stable endpoints; historical-price and symbol-change lanes stay audit-gated.
- `SEC EDGAR`: authoritative filing timeline, company ticker map, submissions, and companyfacts.

## PIT rules

- Raw vendor files remain immutable.
- Corporate actions are applied only in curated outputs.
- Earnings timing is risk-control-only unless sourced from PIT-safe filings.
- Training must gate on GREEN coverage in the actual training window.

## Current live-smoke notes

- Alpaca bars/assets/corporate-actions, Twelve Data core market/reference endpoints, FRED, FMP stable endpoints, Finnhub profile/earnings, and SEC EDGAR passed live smoke with the current `.env`.
- Finnhub candle access is not reliable enough for canonical bars on the current entitlement and should be treated as audit-gated.
- Tiingo is valuable for deep-history bars, but its real usable limits appear tighter than the modeled global budget, so it should stay focused on history rather than generic QC/reference work.

## Canonical bar and QC policy

- Canonical forward bars:
  - `Alpaca` primary
  - `Tiingo`, `Twelve Data`, `Massive` supplemental where planner-eligible
- Canonical deep-history backfill:
  - `Tiingo` primary
  - `Alpaca` recent-history supplement
  - `Twelve Data` then `Massive` as overflow lanes
- Price QC:
  - compare primary output against `Massive`, `Twelve Data`, and optionally `Tiingo`
  - do not use `Finnhub` for canonical bar QC unless a fresh audit explicitly restores candle entitlement

## Endpoint summary

- Alpaca: `GET /v2/stocks/bars`
- Massive: `GET /v2/aggs/ticker/{symbol}/range/1/day/{from}/{to}`
- Finnhub: `GET /api/v1/stock/candle`, `GET /api/v1/calendar/earnings`, `GET /api/v1/stock/profile2`
- Alpha Vantage: `GET /query?function=LISTING_STATUS`
- FRED: `GET /fred/series/observations`, `GET /fred/series/vintagedates`
- FMP stable: `GET /stable/delisted-companies`, `GET /stable/earnings-calendar`
- SEC EDGAR: `GET /submissions/CIK{cik}.json`
