# Future Data Archive Vendor Map

## Purpose

Bank rolling-window and supplemental data now without contaminating the current Phase 1 daily training lane.

Rules for this archive lane:

- collect now
- persist raw with vendor and timestamp lineage
- keep minute/news/event archives out of current Phase 1 features until PIT-safe curation exists

## Recommended vendor split

### Minute bars

- Primary: `alpaca`
  - Docs-backed strengths:
    - multi-symbol stock bars
    - `1Min` timeframe
    - `page_token` pagination
    - `limit` up to `10000`
  - Best use:
    - rolling archive for liquid equities minute bars
    - lowest-friction first archive lane
- Secondary: `twelve_data`
  - Docs-backed strengths:
    - `time_series` supports `1min`
    - `next_api_query` pagination
  - Best use:
    - overflow or targeted supplemental archive after Alpaca lane is proven
- Future/optional: `tiingo` IEX intraday
  - Best use:
    - targeted validation or alternate sample source

### Ticker-linked news

- Primary: `tiingo`
  - Docs-backed strengths:
    - `/tiingo/news`
    - ticker-tagged article feed
  - Best use:
    - main rolling company-news archive
- Supplemental: `finnhub`
  - Docs-backed strengths:
    - `/company-news`
    - explicit `symbol`, `from`, `to`
  - Best use:
    - per-ticker backfill and corroborating event/news coverage
- Optional: `alpaca` news
  - Docs-backed strengths:
    - `/v1beta1/news`
    - symbol-filterable news
  - Best use:
    - later third source if we want broader redundancy

### Filing and event timing

- Primary: `sec_edgar`
  - Best use:
    - filing index
    - companyfacts
    - authoritative PIT filing timeline
- Supplemental: `finnhub`, `fmp`, `twelve_data`
  - Best use:
    - earnings calendars and event corroboration

### Macro and regime

- Primary: `fred`
  - Best use:
    - observation history
    - ALFRED vintages
    - regime descriptors

### Fundamentals and corporate history

- Primary mix:
  - `sec_edgar` for companyfacts
  - `twelve_data` for financial statements
  - existing corp-actions mix across `alpaca`, `alpha_vantage`, `massive`, `twelve_data`

## What to collect now

### Safe to archive immediately

- `equities_minute`
  - primary vendor: `alpaca`
  - output: `data/raw/equities_minute/date=YYYY-MM-DD/data.parquet`
- `ticker_news`
  - primary vendor: `tiingo`
  - supplemental vendor: `finnhub`
  - output: `data/raw/ticker_news/date=YYYY-MM-DD/data.parquet`
- continue existing:
  - macro vintages
  - SEC filing timelines
  - companyfacts
  - earnings calendars
  - corp actions

### Collect later after dedicated curation design

- intraday feature-ready curated bars
- sentiment-scored news features
- event surprise features
- minute-level microstructure features

## Promotion guidance

Do not wire these archive lanes into the current Phase 1 model yet.

Promotion order:

1. archive raw data
2. build PIT-safe curated tables
3. validate timestamp/availability rules
4. expose as new `feature_family` or `data_profile`
5. let the autonomous research loop compare them against the daily baseline

## Current implementation status

- active archive lanes added:
  - `alpaca.equities_minute.research`
  - `tiingo.news.research`
  - `finnhub.company_news.research`
- planner windows intentionally short:
  - minute bars: last 5 days
  - news: last 7 days
- these lanes are `research_only`, not critical-path Phase 1 lanes
