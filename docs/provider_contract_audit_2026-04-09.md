# Provider Contract Audit 2026-04-09

This ledger is the docs-backed source note for the provider contract table in
`src/trademl/data_node/provider_contracts.py`.

## Canonical Vendors

### Alpaca
- Docs:
  - `https://docs.alpaca.markets/reference/stockbars`
  - `https://docs.alpaca.markets/reference/getassets-1`
- Runtime decisions:
  - canonical critical path enabled
  - multi-symbol bars with `page_token` pagination
  - follow documented `X-RateLimit-*` reset headers on throttles
  - safe batch size held at `100`
  - treat entitlement-plan markers as permanent

### Tiingo
- Docs:
  - `https://www.tiingo.com/documentation/end-of-day`
  - `https://www.tiingo.com/documentation/corporate-actions/dividends`
  - `https://www.tiingo.com/documentation/corporate-actions/splits`
- Runtime decisions:
  - canonical critical path enabled
  - daily bars remain ticker-scoped
  - adjusted fields preserved as supplemental metadata

### Massive / Polygon
- Docs:
  - `https://polygon.io/docs/rest/stocks/aggregates/custom-bars`
  - `https://polygon.io/docs/rest/stocks/tickers/all-tickers`
- Runtime decisions:
  - keep off frozen-window critical path
  - aggregate bars remain ticker-scoped
  - both aggregate bars and reference tickers follow documented `next_url` pagination
  - bars use documented `limit <= 50000`; reference tickers use `limit <= 1000`
  - classify plan/authorization failures as permanent

### Twelve Data
- Docs:
  - `https://support.twelvedata.com/en/articles/5203360-batch-api-requests`
  - `https://support.twelvedata.com/en/articles/5609168-introduction-to-twelve-data`
  - `https://support.twelvedata.com/en/articles/9935903-us-equities-market-data`
- Runtime decisions:
  - keep off frozen-window critical path
  - comma-separated symbol requests remain enabled on the time-series lane
  - batch cost is accounted per requested symbol / endpoint credit
  - treat documented partial batch responses under quota exhaustion as valid telemetry, not transport failure
  - batch size held at `8`

## Auxiliary / Reference Vendors

### FRED
- Docs:
  - `https://fred.stlouisfed.org/docs/api/fred/series_observations.html`
  - `https://fred.stlouisfed.org/docs/api/fred/series_vintagedates.html`
- Runtime decisions:
  - treat observations and vintages as separate lanes
  - never let empty observations suppress vintage writes

### SEC EDGAR
- Docs:
  - `https://www.sec.gov/search-filings/edgar-application-programming-interfaces`
- Runtime decisions:
  - preserve user-agent requirement
  - treat as reference-only paced lane

### FMP
- Docs:
  - `https://site.financialmodelingprep.com/developer/docs`
- Runtime decisions:
  - low-throughput reference-only

### Finnhub
- Docs:
  - `https://finnhub.io/docs/api`
- Runtime decisions:
  - research/reference supplemental lane
  - not a canonical closer

### Alpha Vantage
- Docs:
  - `https://www.alphavantage.co/documentation/`
- Runtime decisions:
  - low-throughput reference-only

## Runtime Rewrite Notes
- The budget manager now tracks:
  - outbound requests
  - logical units served
  - request cost units
  - local budget blocks
  - remote rate limits
  - permanent failures
  - valid empty successes
- Canonical planner/runtime now consume the provider contract table for:
  - batch size
  - critical-path eligibility
  - history-depth gating
  - empty-result semantics
