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
  - supported tickers metadata is collected as reference state
  - documented `startDate` / `endDate` bounds can veto impossible canonical requests before they hit the connector
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
  - observations use explicit deterministic `limit=100000` and ascending sort
  - never let empty observations suppress vintage writes

### SEC EDGAR
- Docs:
  - `https://www.sec.gov/search-filings/edgar-application-programming-interfaces`
- Runtime decisions:
  - preserve user-agent requirement
  - fetch archived submissions segments when the primary submissions payload points to additional history files
  - treat as reference-only paced lane

### FMP
- Docs:
  - `https://site.financialmodelingprep.com/developer/docs`
- Runtime decisions:
  - low-throughput reference-only
  - delistings follow documented `page` / `limit` pagination with limit `100`

### Finnhub
- Docs:
  - `https://finnhub.io/docs/api`
- Runtime decisions:
  - research/reference supplemental lane
  - treat `s=no_data` on candles as a valid empty result
  - not a canonical closer

### Alpha Vantage
- Docs:
  - `https://www.alphavantage.co/documentation/`
- Runtime decisions:
  - low-throughput reference-only
  - `LISTING_STATUS` stays CSV with explicit `date` / `state` filters
  - corp-actions normalize documented named arrays (`dividends`, `splits`) in addition to prior `data` payloads

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
