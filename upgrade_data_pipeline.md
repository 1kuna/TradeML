# TradeML Data Pipeline — Multi‑Provider Expansion & Scheduler Integration (Implementation Outline v1.0)

> Goal: plug in **all free‑tier data providers** we have keys for (Alpaca, Polygon, Finnhub, Alpha Vantage, FMP, FRED), route intelligently to **maximize coverage & accuracy** under rate limits, and standardize **forward ingest + audit/backfill** so research consumes **stable curated tables**. This is a handoff for the dev team.

---

## 0) Executive summary
- Keep the storage/QC/SSOT you already built. Add **provider coverage** not speed.
- Implement a **capability‑aware router** + **per‑vendor executors** (already scaffolded) so each source works on what it’s uniquely good at, while high‑limit sources carry the shared loads.
- Introduce a single **`configs/endpoints.yml`** (capabilities + budgets + uniqueness flags) and wire producers/backfill to it.
- Extend connectors minimally (Finnhub daily candles, Alpha Vantage historical options, FMP fundamentals bulk). Add producers and backfill targets for new tables.

---

## 1) Where we are (current state)
- **Connectors present** (extend, don’t replace):
  - `data_layer/connectors/alpaca_connector.py`: equities day/minute, options (bars/trades/chain), corporate actions.
  - `data_layer/connectors/polygon_connector.py`: equities aggregates (day/minute), reference splits/dividends, tickers.
  - `data_layer/connectors/finnhub_connector.py`: options chain; (add candles + fundamentals).
  - `data_layer/connectors/alpha_vantage_connector.py`: corporate actions + listing status; (add historical options + overview flow if needed).
  - `data_layer/connectors/fmp_connector.py`: delistings, historical price, symbols, statements.
  - `data_layer/connectors/fred_connector.py`: series, treasuries, vintages.
- **Scheduler**
  - `scripts/scheduler/per_vendor.py` + `scripts/scheduler/producers.py`: per‑vendor runners, tokenized units, EOD gating hooks; env‑driven caps.
  - `configs/edge.yml`: task list & per‑vendor daily budgets.
  - Global pacing (`utils/pacing.py`) and token‑bucket budget (`ops/ssot/budget.py`).
- **SSOT (audit/backfill/curate)**
  - `ops/ssot/audit.py` computes completeness ledger → `data_layer/qc/partition_status.parquet`.
  - `ops/ssot/backfill.py` queue + safety‑net window sweeps.
  - `configs/backfill.yml` budgets, priorities, targets, QC thresholds.
  - `scripts/curator.py` & `configs/curator.yml` incremental curation.
- **State & resume**: S3 leases, bookmarks, bad‑symbols cache; `.env.template` already exposes knobs.

**Implication:** We don’t need a new framework; we add **capabilities**, **missing producers**, and a small **router** on top of what exists.

---

## 2) What to add (gaps → concrete deliverables)

### 2.1 Capabilities registry (new)
Create `configs/endpoints.yml` to declare, per provider, what datasets they cover, limits, and whether the dataset is **unique** (no other source covers it as well on free tier):

```yaml
# configs/endpoints.yml
providers:
  alpaca:
    rpm: 200
    datasets:
      equities_eod:
        endpoint: alpaca.bars
        unique: false
        weight: 3          # prefer when high‑limit or cheap
      equities_minute:
        endpoint: alpaca.minute
        unique: true       # only practical free‑tier minute at scale
      options_bars:
        endpoint: alpaca.options.bars
        unique: partial    # indicative unless OPRA; mark quality
      corp_actions:
        endpoint: alpaca.corporate_actions
        unique: false
  finnhub:
    rpm: 60
    datasets:
      options_chain:
        endpoint: finnhub.option_chain
        unique: true       # chains breadth on free tier
      equities_eod:
        endpoint: finnhub.candle.daily
        unique: false
  polygon:
    rpm: 5
    datasets:
      equities_eod:
        endpoint: polygon.aggs.day
        unique: false
      options_contracts:
        endpoint: polygon.options.contracts
        unique: true
  alpha_vantage:
    rpm: 0.4    # 25/day
    datasets:
      corp_actions:
        endpoint: av.corp_actions
        unique: true
      delistings:
        endpoint: av.listing_status
        unique: true
      options_chain_hist:
        endpoint: av.historical_options
        unique: true
  fmp:
    rpm: 0.17   # 250/day
    datasets:
      symbols_bulk:
        endpoint: fmp.stock_list
        unique: true
      statements:
        endpoint: fmp.statements
        unique: true
  fred:
    rpm: 120
    datasets:
      treasuries:
        endpoint: fred.treasury_curve
        unique: true
      series:
        endpoint: fred.series
        unique: true
```

### 2.2 Connector deltas
Small, focused additions; retain current schemas and `_add_metadata()` usage.

- **Finnhub**: add `fetch_candle_daily(symbol, start_date, end_date)` returning `EQUITY_BARS_SCHEMA`.
- **Alpha Vantage**: add `fetch_historical_options(symbol, expiry=None)` returning greeks/IV columns mapped to `OPTIONS_IV/OPTIONS_TRADES` (pick one schema and document fields). Keep a per‑day cap from `backfill.yml`.
- **FMP**: add bulk endpoints wrappers for statements (`income`, `balance`, `cashflow`), plus `fetch_available_symbols(exchange="NYSE,NASDAQ")` (already present; reuse) and a thin `fetch_company_profile` if missing.
- **Polygon**: keep aggregates as supplemental; ensure contracts snapshot + small aggregates sampler path exists (already present under ops helpers).

### 2.3 Producers (forward ingest) — extend tasks
Add producers to `scripts/scheduler/producers.py` and wire them in `per_vendor.py`:

- `finnhub_daily_units`: walk universe by day windows; tokens ≈ `ceil(n_syms / 100)`.
- `av_corp_actions_units` (low tokens, date‑sliced).
- `av_options_hist_units` (seed chains/greeks sparsely; honor 25/day budget).
- `fmp_fundamentals_units` (statements per symbol; small daily budget; rotate symbols).
- Keep existing: `alpaca_bars`, `alpaca_minute`, `alpaca_options_chain/bars`, `polygon_bars`, `finnhub_options`, `fred_treasury`.

### 2.4 Backfill targets
Extend `configs/backfill.yml`:
- Add targets: `options_chain_hist` (AV), `fundamentals` (FMP), `equities_eod_finnhub` (optional), `options_contracts_polygon`.
- For each, set `earliest`, `chunk_days`, and **daily budgets** (align with free limits).

### 2.5 Router (lightweight)
Add `ops/ssot/router.py` (if not already) with a pure function:
- Inputs: dataset (`equities_eod`, `equities_minute`, `options_chain`, …), provider availability, `endpoints.yml`, **partition_status** (missing vs present), and **budgets**.
- Output: **ordered provider plan** for that dataset today (unique‑first → highest‑limit → fallbacks).
- Producers call `route(dataset)` to choose which provider’s unit to emit first; if budget/token depleted, they yield fallback units.

### 2.6 QC/curation
- Ensure curated tables for new inputs (e.g., `curated/equities_ohlcv_adj` already present; add curated for fundamentals if needed, or keep in `reference/`).
- Expand `data_layer/qc/data_quality.py` checks to include row bounds for new tables.

---

## 3) Implementation steps (ordered for PRs)

### PR‑1 — Capabilities & config
1) Add `configs/endpoints.yml` (above).  
2) Add loader `utils/capabilities.py` exposing `cap_for(dataset)` and `providers_for(dataset)`.
3) Expand `.env.template` (already has most knobs) with any new flags you add.

### PR‑2 — Connector extensions
1) **Finnhub**: implement `fetch_candle_daily`; unit test with fake payload.  
2) **Alpha Vantage**: implement `fetch_historical_options` + CSV/JSON parsing; mark per‑day guard.  
3) **FMP**: implement `fetch_statements(symbol, kind, period)` and ensure pagination; add `fetch_company_profile` if missing.  

### PR‑3 — Producers + per‑vendor hooks
1) Add `finnhub_daily_units`, `av_corp_actions_units`, `av_options_hist_units`, `fmp_fundamentals_units`.  
2) Wire in `scripts/scheduler/per_vendor.py::_producers_for_vendor()` with new task names.  
3) Extend `configs/edge.yml -> tasks:` to include the new tasks; keep budgets conservative.

### PR‑4 — Router integration (minimal)
1) Add `ops/ssot/router.py` function `route(dataset)` using `endpoints.yml` + **unique‑first scoring**:  
   `score = 3*is_unique + 2*coverage_gap + 1*budget_headroom + 1*recency_need − 2*duplication_risk`  
2) In each producer, before yielding a unit, call `route()` to pick provider ordering for that dataset; if provider A is frozen/out of budget, yield B’s unit instead. (Small shim; no redesign.)

### PR‑5 — Backfill
1) Extend `configs/backfill.yml` with new targets/budgets; map to connector calls (windowed).  
2) Add queue handlers for new tables in `ops/ssot/backfill.py` (copy the equities pattern).  
3) Update acceptance tests: synthetic gaps for each new table flip RED→GREEN.

### PR‑6 — Curator/QC & smoke
1) Add/adjust curated writers if needed (fundamentals may stay in `reference/` only).  
2) Extend `scripts/smoke.py` to touch one unit from each new provider/task and print counts.  
3) Update coverage heatmap to include new tables.

---

## 4) Config diffs (copy‑paste)

### 4.1 `configs/edge.yml`
```yaml
tasks:
  - alpaca_bars
  - alpaca_minute
  - alpaca_options_chain
  - alpaca_options_bars
  - polygon_bars
  - finnhub_options
  - finnhub_daily         # NEW
  - fred_treasury
  - av_corp_actions       # NEW (low volume)
  - av_options_hist       # NEW (very low volume)
  - fmp_fundamentals      # NEW (rotate symbols)

policy:
  daily_api_budget:
    alpaca:  250000
    polygon: 7000
    finnhub: 86400
    fred:    10000
    av:      25
    fmp:     250
```

### 4.2 `configs/backfill.yml` (snippets)
```yaml
policy:
  daily_api_budget:
    av: 25
    fmp: 250
    polygon: 7000

priorities:
  - { table: corp_actions,       weight: 100 }
  - { table: delistings,         weight: 100 }
  - { table: equities_eod,       weight: 90 }
  - { table: equities_minute,    weight: 80 }
  - { table: options_chains,     weight: 70 }
  - { table: fundamentals,       weight: 60 }   # NEW

targets:
  fundamentals:      { earliest: '2000-01-01', chunk_days: 1 }
  options_chain_hist:{ earliest: '2015-01-01', chunk_days: 1 }
  equities_eod_fn:   { earliest: '2000-01-01', chunk_days: 30 }  # Finnhub alt
```

---

## 5) Data & lineage rules (no surprises downstream)
- Keep **raw/** append‑only; one day per partition; include `source_name`, `source_uri`, `ingested_at` on every row.
- Options datasets: distinguish **indicative** vs **OPRA** with a `quote_quality` column and `indicative` boolean (already followed by Alpaca options snapshot path).
- Corporate actions/delistings must be **backfilled to full history first** before training; equities curation applies adj factors deterministically.

---

## 6) Acceptance criteria (ops + research)
- Coverage heatmap shows **≥98% GREEN** last 2y for `equities_eod` across the universe; `equities_minute` GREEN for top‑N; options chains present for target underliers at least 1 snapshot/day.
- Router honors uniqueness/budgets: when AV budget=0, chains still get captured via Finnhub/Alpaca paths; when Polygon is slow/frozen, other vendors keep advancing.
- Smoke test completes end‑to‑end on local (or Pi) with at least one partition per new table.

---

## 7) Rollout plan
1) Land capabilities + producers behind feature flags; run on dev with `EDGE_SCHEDULER_MODE=per_vendor` (already defaulted).  
2) Enable nightly node loop to include new tasks; watch logs for freezes/429s; tune caps.  
3) Once coverage stabilizes, flip backfill targets on for history.  
4) Gate training on GREEN thresholds for any new model consuming the new data.

---

## 8) File‑by‑file TODO (short list)
- **Connectors**: add Finnhub candles, AV historical options, FMP statements/profile helpers.
- **Producers**: add `finnhub_daily_units`, `av_corp_actions_units`, `av_options_hist_units`, `fmp_fundamentals_units`.
- **Scheduler**: wire new tasks in `per_vendor.py`; keep caps in env.
- **Configs**: add `configs/endpoints.yml`; extend `edge.yml`/`backfill.yml` as above.
- **Backfill**: add handlers for `fundamentals`, `options_chain_hist`.
- **Smoke/Tests**: extend unit tests for new connector methods; update `scripts/smoke.py`.

---

## 9) Notes on “accuracy over speed”
- **Unique‑first**: prioritize datasets only one provider offers on free tier (AV delistings/CA, Polygon contracts, Finnhub chains, FRED vintages).
- **Verification**: for shared datasets (EOD), assign a **canonical primary** per dataset (e.g., Finnhub→primary for daily) and keep alternates only for audit; do not average vendors.
- **Budgets**: treat AV/FMP as scarce; rotate symbols/day and backfill slowly in background.

---

## 10) Optional: small code stubs (signatures)
- `FinnhubConnector.fetch_candle_daily(symbol: str, start: date, end: date) -> pd.DataFrame`
- `AlphaVantageConnector.fetch_historical_options(symbol: str, expiry: Optional[str]) -> pd.DataFrame`
- `FMPConnector.fetch_statements(symbol: str, kind: Literal['income','balance','cashflow'], period: Literal['annual','quarter']) -> pd.DataFrame`
- `router.route(dataset: str, want_date: date, universe: list[str]) -> list[str]  # provider order`

---

**Done.** Hand this file to the team; I can also generate the initial `endpoints.yml` and PR skeletons if you want next.

