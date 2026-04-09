# Provider Doc Audit

Docs-backed audit of what each active vendor can provide beyond the currently wired minimum. This is the "go deeper" backlog for continuing saturation/QC improvements.

## Core takeaway

- `Alpaca`: strongest same-day and recent-history bar source; assets and corp-actions belong here.
- `Tiingo`: use as the deep-history specialist, not as a general-purpose auxiliary burner.
- `Twelve Data`: biggest immediate upside from additional docs-driven exploitation because it supports batch requests and has broad event/fundamental coverage.
- `Massive`: best independent validation lane; keep it focused on QC and targeted reference corroboration.
- `Finnhub`: useful event/reference surface, but still not trustworthy enough for canonical bars on current entitlement.
- `SEC EDGAR`: biggest non-bar upside from docs is bulk ingestion, not more per-symbol JSON fanout.

## By vendor

### Alpaca

Currently used:
- daily stock bars
- assets
- corporate actions

Docs-confirmed additional value:
- same family of market-data endpoints can stay focused on recent-history fill and same-day validation

Priority:
- keep as primary forward and recent-history canonical bar source
- keep corp-actions in the merged reference set

### Tiingo

Currently used:
- daily prices

Docs-confirmed additional value:
- fundamentals
- supported tickers metadata

Priority:
- keep focused on deep-history bars first
- only spend on non-bar Tiingo lanes once canonical history is no longer the bottleneck

### Twelve Data

Currently used:
- daily bars
- dividends
- splits
- earnings
- stocks
- financial statements

Docs-confirmed additional value:
- batch API requests for bars
- price targets
- insider transactions

Priority:
- maximize batch bar pulls first
- then spend leftover credits on corp-actions, earnings, and statements
- keep research-only lanes gated behind Phase 1 data readiness

### Massive / Polygon

Currently used:
- daily bars
- reference tickers
- dividends
- splits

Docs-confirmed additional value:
- deeper ticker/reference detail and independent reference corroboration

Priority:
- keep as the independent QC/check vendor
- use for reference corroboration and targeted overflow only

### Finnhub

Currently used:
- earnings calendar
- company profile

Docs-confirmed additional value:
- peers
- recommendation trends
- company news
- financials-reported

Priority:
- stay reference/events only
- if we add more Finnhub lanes, add them to supplemental/research buckets, not canonical bars

### SEC EDGAR

Currently used:
- company tickers
- filing index
- companyfacts

Docs-confirmed additional value:
- bulk issuer/submission/companyfacts archives

Priority:
- next scale-up should be bulk SEC ingestion rather than more tiny per-CIK loops

### FRED / ALFRED

Currently used:
- observations
- vintagedates

Docs-confirmed additional value:
- broader PIT-safe macro pack

Priority:
- keep macro saturated independently because it does not compete with bar budgets

### FMP

Currently used:
- delistings
- earnings calendar

Docs-confirmed but still audit-gated:
- symbol changes
- broader historical price endpoints

Priority:
- leave audit-gated until live entitlement behavior is stable

### Alpha Vantage

Currently used:
- listing status
- corp actions

Priority:
- keep as low-rate corroborating reference only

## Immediate next upgrades

1. Add weighted budget accounting for credit-based vendors, especially `twelve_data`.
2. Add SEC bulk ingestion path for higher-throughput filings/companyfacts collection.
3. Expand Finnhub supplemental lanes only after the training-critical corpus is healthy.
4. Re-check Tiingo non-bar lanes with a docs-first + live-entitlement pass before enabling any of them.

## Official docs used

- [Alpaca Stock Bars](https://docs.alpaca.markets/reference/stockbars)
- [Alpaca Assets](https://docs.alpaca.markets/reference/get-v2-assets)
- [Alpaca Corporate Actions](https://docs.alpaca.markets/reference/corporateactions-1)
- [Tiingo EOD Docs / KB](https://www.tiingo.com/kb/article/the-fastest-method-to-ingest-tiingo-end-of-day-stock-api-data/)
- [Twelve Data Request Format](https://support.twelvedata.com/en/articles/5620512-how-to-create-a-request)
- [Twelve Data Batch Requests](https://support.twelvedata.com/en/articles/5203360-batch-api-requests)
- [Twelve Data Stocks](https://support.twelvedata.com/en/articles/5620513-how-to-find-all-available-symbols-at-twelve-data)
- [Finnhub API](https://finnhub.io/docs/api)
- [SEC EDGAR APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)
- [FRED Observations](https://fred.stlouisfed.org/docs/api/fred/series_observations.html)
- [FRED Vintage Dates](https://fred.stlouisfed.org/docs/api/fred/series_vintagedates.html)
