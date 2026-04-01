# TradeML — Development Guide (for Coding Agents)

**Read this first.** This document tells you what to build, in what order, and how to know when each phase is done. It is the instruction set for autonomous implementation.

**Reference documents:**
- `SSOT_V3.md` — the canonical spec. Every design decision, schema, and contract lives there. When in doubt, the SSOT wins.
- `Data_Sourcing_Playbook.md` — vendor details and PIT practices
- `Data_Sources_Detail.pdf` — exact API endpoint paths and parameters

**Conventions:**
- **Phase**: a group of work that produces a testable, working milestone.
- **Gate**: a concrete checklist of pass/fail criteria. All items must pass before moving to the next phase.
- **Your call**: marks where multiple valid implementations exist. Pick one, document why in a code comment.
- **SSOT §X.Y**: references a specific section in SSOT_V3.md.

**General rules:**
- Test-first where practical. Write the test, then make it pass.
- Every module uses dependency injection (constructor args, not global imports).
- Every external dependency (API, filesystem, database) gets a mockable wrapper.
- Commit after each passing gate. Tag the commit with the phase name.
- If harder than expected, ship a simplified version that passes the gate. Mark limitations with TODO.
- If genuinely blocked, stop and explain what's needed.
- A working 80% solution that passes tests beats a perfect one that doesn't exist.

---

## Repo Structure (create this first)

```
trademl/
├── SSOT_V3.md                          # Canonical spec (copy in)
├── CLAUDE.md                           # Agent instructions (copy in)
├── README.md                           # Brief project overview
├── pyproject.toml                      # Dependencies, project config
├── .env.template                       # Environment variable template
│
├── src/
│   ├── trademl/
│   │   ├── __init__.py
│   │   │
│   │   ├── connectors/                 # Vendor API clients
│   │   │   ├── __init__.py
│   │   │   ├── base.py                 # BaseConnector protocol + retry logic
│   │   │   ├── alpaca.py
│   │   │   ├── massive.py              # Polygon.io
│   │   │   ├── finnhub.py
│   │   │   ├── alpha_vantage.py
│   │   │   ├── fred.py
│   │   │   ├── fmp.py
│   │   │   └── sec_edgar.py
│   │   │
│   │   ├── data_node/                  # Pi data collection service
│   │   │   ├── __init__.py
│   │   │   ├── service.py              # Main loop (collect → audit → backfill → curate → sync)
│   │   │   ├── db.py                   # SQLite helpers (queue, partition_status)
│   │   │   ├── auditor.py              # Gap detection via calendar comparison
│   │   │   ├── curator.py              # Raw → curated pipeline
│   │   │   ├── budgets.py              # Per-vendor rate limiting + daily caps
│   │   │   └── __main__.py             # Entry point: python -m trademl.data_node
│   │   │
│   │   ├── calendars/                  # Exchange calendar generation
│   │   │   ├── __init__.py
│   │   │   └── exchange.py             # Wrapper around exchange_calendars lib
│   │   │
│   │   ├── features/                   # Feature computation
│   │   │   ├── __init__.py
│   │   │   ├── equities.py             # Cross-sectional feature builder
│   │   │   └── preprocessing.py        # Rank normalization, missing data
│   │   │
│   │   ├── labels/                     # Label construction
│   │   │   ├── __init__.py
│   │   │   └── returns.py              # Universe-relative forward returns
│   │   │
│   │   ├── models/                     # Model training
│   │   │   ├── __init__.py
│   │   │   ├── ridge.py                # Ridge baseline
│   │   │   └── lgbm.py                 # LightGBM challenger
│   │   │
│   │   ├── validation/                 # Walk-forward, CPCV, PBO, DSR
│   │   │   ├── __init__.py
│   │   │   ├── walk_forward.py         # Primary: expanding walk-forward
│   │   │   ├── cpcv.py                 # Secondary: combinatorially purged CV
│   │   │   ├── pbo.py                  # Probability of backtest overfitting
│   │   │   ├── dsr.py                  # Deflated Sharpe ratio
│   │   │   └── diagnostics.py          # IC by year/sector, placebo, cost stress
│   │   │
│   │   ├── portfolio/                  # Portfolio construction
│   │   │   ├── __init__.py
│   │   │   └── build.py                # Equal-weight top quintile, weekly rebalance
│   │   │
│   │   ├── backtest/                   # Backtesting engine
│   │   │   ├── __init__.py
│   │   │   └── engine.py               # Deterministic event-driven backtester
│   │   │
│   │   ├── costs/                      # Cost modeling
│   │   │   ├── __init__.py
│   │   │   └── models.py               # Spread + impact
│   │   │
│   │   └── reports/                    # Output generation
│   │       ├── __init__.py
│   │       └── emitter.py              # JSON + markdown signal reports
│   │
│   └── scripts/
│       ├── pi_data_node_wizard.py      # Pi first-time setup
│       ├── train.py                    # Training entry point
│       └── backtest.py                 # Backtest runner
│
├── tests/
│   ├── unit/
│   │   ├── test_calendars.py
│   │   ├── test_features.py
│   │   ├── test_labels.py
│   │   ├── test_preprocessing.py
│   │   ├── test_portfolio.py
│   │   ├── test_costs.py
│   │   ├── test_walk_forward.py
│   │   ├── test_cpcv.py
│   │   ├── test_curator.py
│   │   ├── test_db.py
│   │   └── test_budgets.py
│   ├── integration/
│   │   ├── test_pipeline_e2e.py        # Synthetic data → features → labels → train → validate
│   │   ├── test_backtest_e2e.py        # Full backtest on synthetic data
│   │   └── test_live_endpoints.py      # Real API calls (skips if no creds)
│   └── fixtures/
│       ├── sample_bars.parquet         # Small synthetic dataset
│       ├── sample_corp_actions.parquet
│       └── sample_calendar.parquet
│
├── configs/
│   ├── equities_xs.yml                 # Training config (SSOT §10.1)
│   └── node.yml                        # Pi node config (SSOT §10.2)
│
└── docs/
    ├── SSOT_V3.md
    ├── Data_Sourcing_Playbook.md
    ├── Data_Sources_Detail.pdf
    └── archive/                        # All old docs go here
```

---

## Phase 0 — Scaffolding

**Goal:** Empty repo with structure, dependencies, CI skeleton, and one passing test.

### Steps

1. **Create repo structure** — every directory and `__init__.py` from the tree above.

2. **`pyproject.toml`** — dependencies:
   - Core: `pandas`, `pyarrow`, `numpy`, `scipy`, `scikit-learn`, `lightgbm`
   - Data: `requests`, `exchange_calendars`
   - Testing: `pytest`, `pytest-cov`
   - Optional: `optuna` (HPO), `rich` (Pi dashboard)

3. **`.env.template`** — copy from SSOT §9.2.

4. **Seed test** — `tests/unit/test_scaffold.py`:
   ```python
   def test_import():
       import trademl
       assert trademl is not None
   ```

5. **Copy SSOT_V3.md** into `docs/`.

### Gate 0
- [ ] `pip install -e .` succeeds
- [ ] `pytest tests/unit/test_scaffold.py -v` passes
- [ ] All directories from the tree exist
- [ ] `.env.template` exists with all variables from SSOT §9.2

---

## Phase 1 — Calendars & Core Utilities

**Goal:** Exchange calendar generation works. SQLite helpers work. Budget manager works. These are dependencies for everything else.

### Steps (parallel OK — these are independent)

1. **Exchange calendars** — `src/trademl/calendars/exchange.py`
   - Wrap `exchange_calendars` library
   - `get_trading_days(exchange, start, end) -> list[date]`
   - `is_trading_day(exchange, date) -> bool`
   - `is_early_close(exchange, date) -> bool`
   - Generate and save parquet to `reference/calendars/XNYS.parquet` etc.
   - See SSOT §1.8

2. **SQLite helpers** — `src/trademl/data_node/db.py`
   - Create `backfill_queue` and `partition_status` tables (schemas from SSOT §1.4)
   - `enqueue_task(dataset, symbol, start_date, end_date, kind, priority)`
   - `lease_next_task() -> Task | None` (FIFO within priority, respects `next_not_before`)
   - `mark_task_done(task_id)`
   - `mark_task_failed(task_id, error, backoff_minutes)`
   - `update_partition_status(source, dataset, date, status, row_count, ...)`
   - All operations transactional

3. **Budget manager** — `src/trademl/data_node/budgets.py`
   - Load per-vendor RPM and daily caps from config
   - `can_spend(vendor) -> bool`
   - `record_spend(vendor)`
   - `reset_daily()` (called at midnight)
   - FORWARD tasks get a reserved 10% of daily budget (SSOT §1.4)

### Gate 1
- [ ] `pytest tests/unit/test_calendars.py -v` — all pass
  - Known holidays correct (July 4, Thanksgiving, Christmas)
  - Early closes correct (day after Thanksgiving)
  - No trading days on weekends
  - DST transitions handled
- [ ] `pytest tests/unit/test_db.py -v` — all pass
  - Enqueue + lease + done cycle works
  - Failed tasks respect backoff
  - Duplicate tasks rejected by unique constraint
  - Priority ordering: 0 before 1 before 5
- [ ] `pytest tests/unit/test_budgets.py -v` — all pass
  - RPM limiting works
  - Daily cap enforcement works
  - FORWARD tasks can spend when others can't (10% reserve)

---

## Phase 2 — Connectors

**Goal:** Every Phase 1 vendor connector can fetch data and return it as a DataFrame. Mocked for tests, real for smoke tests.

### Steps

1. **Base connector** — `src/trademl/connectors/base.py`
   - Protocol: `fetch(dataset, symbols, start_date, end_date) -> pd.DataFrame`
   - Retry logic: exponential backoff with jitter on 429/5xx
   - Budget integration: check `budgets.can_spend()` before each request
   - Logging: vendor, endpoint, symbols, rows returned, elapsed time
   - NOT_ENTITLED / NOT_SUPPORTED → raise permanent failure (no retry)

2. **Alpaca connector** — `src/trademl/connectors/alpaca.py` (SSOT Appendix A)
   - `/v2/stocks/bars` — multi-symbol batch (chunks of 100)
   - Returns DataFrame with SSOT §1.1 columns
   - Handle pagination (next_page_token)

3. **Massive (Polygon) connector** — `src/trademl/connectors/massive.py`
   - `/v2/aggs/ticker/{symbol}/range/1/day/{from}/{to}` — one symbol at a time
   - Reference: `get_tickers`, `get_stock_dividends`, `get_stock_splits`
   - Returns same column schema as Alpaca for price bars

4. **Finnhub connector** — `src/trademl/connectors/finnhub.py`
   - `/stock/candle` — daily candles (backup)
   - `/stock/earnings-calendar` — earnings dates
   - `/stock/profile2` — company info

5. **Alpha Vantage connector** — `src/trademl/connectors/alpha_vantage.py`
   - `LISTING_STATUS` — active/delisted symbols
   - Splits/dividends functions

6. **FRED connector** — `src/trademl/connectors/fred.py`
   - `/fred/series/observations` — time series
   - `/fred/series/vintagedates` — ALFRED vintage dates

7. **FMP connector** — `src/trademl/connectors/fmp.py`
   - `/api/v3/delisted-company` — delistings
   - `/api/v3/earnings-calendar` — earnings (backup)

8. **SEC EDGAR connector** — `src/trademl/connectors/sec_edgar.py`
   - Filing index for 8-K (earnings announcements)
   - Respect rate limit (10 req/sec) and User-Agent policy

### Gate 2
- [ ] Each connector has a mock that satisfies the BaseConnector protocol
- [ ] `pytest tests/unit/` — all connector unit tests pass with mocks
- [ ] `pytest tests/integration/test_live_endpoints.py -m liveapi` — passes for each vendor with valid API keys (skips cleanly without keys)
- [ ] Each connector returns a DataFrame matching the expected schema from SSOT §1.1

---

## Phase 3 — Data Node Service

**Goal:** Pi data node collects daily bars, writes parquet to a path (local or NAS), audits completeness, fills gaps, and tracks state.

### Steps

1. **Auditor** — `src/trademl/data_node/auditor.py`
   - Compare exchange calendar vs `partition_status`
   - Flag missing dates as RED
   - Generate GAP tasks for missing dates
   - Mark weekends/holidays as GREEN with `NO_SESSION`

2. **Curator** — `src/trademl/data_node/curator.py`
   - Read raw bars + corp actions reference
   - Apply split adjustments (price × ratio, volume ÷ ratio)
   - Apply dividend adjustments: ratio = `(close_before_ex - dividend) / close_before_ex` — see SSOT §1.9
   - Write curated parquet
   - Log every adjustment applied

3. **Service loop** — `src/trademl/data_node/service.py`
   - Main loop per SSOT §6.1:
     - After close: fetch today's bars (FORWARD tasks)
     - Audit: check completeness
     - Backfill: process GAP/BOOTSTRAP tasks from queue within budget
     - Curate: rebuild any raw dates that changed
     - Sync: copy partition_status.parquet to data path
   - Graceful shutdown on SIGINT/SIGTERM
   - All state in local SQLite

4. **`__main__.py`** — `python -m trademl.data_node` entry point

### Gate 3
- [ ] Synthetic test: create fake raw data, run curator, verify adjusted prices are correct around a known split date
- [ ] Synthetic test: remove a date from raw, run auditor, verify it creates a GAP task
- [ ] Integration test: full cycle (collect with mock connector → audit → curate) produces expected parquet output
- [ ] Service starts and stops cleanly
- [ ] `partition_status.parquet` is written with correct GREEN/AMBER/RED values

---

## Phase 4 — Features & Labels

**Goal:** Given curated OHLCV, produce rank-normalized features and universe-relative labels.

### Steps

1. **Feature builder** — `src/trademl/features/equities.py`
   - `build_features(panel: pd.DataFrame, config: dict) -> pd.DataFrame`
   - Implements all Phase 1 features from SSOT §2.1:
     - Momentum: 5, 20, 60, 126-day log returns
     - Reversal: 1-day, 5-day log returns
     - Drawdown: 20, 60-day max loss from peak
     - Gap: overnight return
     - Realized vol: 20, 60-day
     - Idiosyncratic vol: 60-day residual after market beta
     - Dollar volume: 20-day (⚠ IEX-only, rough proxy)
     - Amihud: 20-day
     - Log price

2. **Rank normalization** — `src/trademl/features/preprocessing.py`
   - `rank_normalize(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame`
   - Per date: rank each feature, map to [-1, +1]
   - Skip date-level features (not cross-sectional)
   - Handle missing data: fill with 0.5 rank (cross-sectional median)
   - Drop features with >30% missing on a given date

3. **Label builder** — `src/trademl/labels/returns.py`
   - `build_labels(panel: pd.DataFrame, horizon: int = 5) -> pd.DataFrame`
   - Universe-relative: `label = log_return(t, t+horizon) - mean(log_returns across universe)`
   - Also compute raw forward returns for comparison
   - Count trading days using calendar (not calendar days)

### Gate 4
- [ ] Features: all SSOT §2.1 features compute without error on sample data
- [ ] Features: output shape is (n_dates × n_symbols, n_features)
- [ ] Rank normalization: output values in [-1, 1], no NaN for features with sufficient coverage
- [ ] Labels: universe-relative returns sum to approximately 0 on each date
- [ ] Labels: horizon counts trading days, not calendar days
- [ ] PIT test: shifting input data doesn't change features/labels outside expected window
- [ ] `pytest tests/unit/test_features.py tests/unit/test_labels.py tests/unit/test_preprocessing.py -v` — all pass

---

## Phase 5 — Validation Framework

**Goal:** Walk-forward validation, CPCV, and diagnostic tests all work on synthetic data.

### Steps

1. **Walk-forward** — `src/trademl/validation/walk_forward.py`
   - `expanding_walk_forward(X, y, model_fn, config) -> list[FoldResult]`
   - Expanding window: initial 2 years, step 3 months (configurable)
   - **Purge last 5 trading days from each training fold** (SSOT §4.1 — label overlap)
   - Each fold returns: rank IC, decile spreads, hit rate
   - Support any model that implements `fit(X, y)` and `predict(X)`

2. **CPCV** — `src/trademl/validation/cpcv.py`
   - 8-fold combinatorially purged CV with 10-day embargo
   - Symbol-aware purging
   - Returns per-fold metrics + OOF predictions

3. **PBO** — `src/trademl/validation/pbo.py`
   - Compute probability of backtest overfitting from CPCV results

4. **DSR** — `src/trademl/validation/dsr.py`
   - Deflated Sharpe ratio adjusting for skew/kurtosis/multiple testing

5. **Diagnostics** — `src/trademl/validation/diagnostics.py`
   - `ic_by_year(predictions, actuals, dates) -> dict`
   - `ic_by_sector(predictions, actuals, sectors) -> dict` (Phase 2; stub for now)
   - `placebo_test(X, y, model_fn, n_shuffles=5) -> list[float]` — train on shuffled labels, return ICs
   - `cost_stress_test(results, multiplier=2.0) -> dict` — recompute metrics with doubled costs

### Gate 5
- [ ] Walk-forward: produces correct number of folds for a 5-year dataset with 3-month steps
- [ ] Walk-forward: training window does NOT include last 5 days (purge check)
- [ ] Walk-forward: each fold produces rank IC, decile spreads
- [ ] CPCV: multi-symbol purging retains ~70%+ of training data (not 0%)
- [ ] Placebo test: shuffled labels produce IC near 0 (±0.01)
- [ ] `pytest tests/unit/test_walk_forward.py tests/unit/test_cpcv.py -v` — all pass

---

## Phase 6 — Models & Portfolio

**Goal:** Ridge baseline and LightGBM challenger can be trained via walk-forward. Portfolio construction produces weekly trades.

### Steps

1. **Ridge baseline** — `src/trademl/models/ridge.py`
   - Wraps sklearn Ridge with `fit(X, y)` / `predict(X)` interface
   - Alpha tuned via walk-forward (not separately cross-validated — walk-forward IS the validation)

2. **LightGBM challenger** — `src/trademl/models/lgbm.py`
   - Wraps LightGBM with same interface
   - HPO via Optuna: small grid from SSOT §3.2, ≤20 trials
   - Time-decay sample weighting (18-month half-life)

3. **Portfolio builder** — `src/trademl/portfolio/build.py`
   - `build_portfolio(scores: pd.Series, config: dict) -> pd.DataFrame`
   - Rank scores, take top quintile, equal-weight
   - Weekly rebalance (configurable day)
   - Return target_weights DataFrame

4. **Cost model** — `src/trademl/costs/models.py`
   - `apply_costs(trades: pd.DataFrame, config: dict) -> pd.DataFrame`
   - Spread: 5 bps default (configurable)
   - Impact: √-law (available but likely negligible at retail size)
   - Stress multiplier: 2× for robustness checks

5. **Backtest engine** — `src/trademl/backtest/engine.py`
   - `run_backtest(prices, target_weights, cost_model, config) -> BacktestResult`
   - Deterministic: same inputs = same outputs
   - Weekly rebalance, position tracking
   - Outputs: equity curve, trade log, cost attribution

### Gate 6
- [ ] Ridge: trains on sample data, produces predictions
- [ ] LightGBM: trains on sample data, produces predictions, HPO runs ≤20 trials
- [ ] Portfolio: top quintile correct size (20% of universe), weights sum to 1
- [ ] Cost model: spread applied correctly, stress multiplier works
- [ ] Backtest: deterministic (run twice → identical output)
- [ ] Integration test: synthetic data → features → labels → Ridge → walk-forward → portfolio → backtest → produces metrics
- [ ] `pytest tests/ -v` — all pass

---

## Phase 7 — End-to-End Pipeline & Reports

**Goal:** The full pipeline runs on real data. Produces a report with IC, decile spreads, and diagnostic tests.

### Steps

1. **Training script** — `src/scripts/train.py`
   - Load curated data from NAS path
   - Check GREEN coverage
   - Build features + labels
   - Run Ridge via walk-forward
   - Run LightGBM via walk-forward
   - Compare on OOS metrics
   - Run diagnostics (IC by year, placebo, cost stress)
   - Save results to NAS

2. **Report emitter** — `src/trademl/reports/emitter.py`
   - Produce JSON + markdown summary
   - Include: fold-by-fold IC, average IC, decile spreads chart data, placebo results, cost stress results
   - Phase 1 go/no-go assessment per SSOT §4.4

3. **Pi wizard** — `src/scripts/pi_data_node_wizard.py`
   - Interactive setup per SSOT §9.1
   - NAS mount, API keys, SQLite init, Stage 0 seeding, schedule config

### Gate 7 (Phase 1 complete)
- [ ] Pi data node runs for 24+ hours without crashing, collecting real data to NAS
- [ ] ≥98% GREEN coverage on 100 symbols × available history
- [ ] Training script runs end-to-end on real data and produces a report
- [ ] Ridge walk-forward IC is computed for all folds
- [ ] LightGBM walk-forward IC is computed and compared to Ridge
- [ ] Placebo test passes (shuffled IC ≈ 0)
- [ ] Cost stress test (2× costs) results included
- [ ] Report is saved to NAS as JSON + markdown
- [ ] All tests pass: `pytest tests/ -v`

**After Gate 7:** Read the report. Does Phase 1 pass the go/no-go criteria in SSOT §4.4? If yes → start Phase 2 (buy data, expand universe). If no → investigate why (data quality? feature set? the thesis itself?).

---

## When You're Stuck

- If a vendor API doesn't work as documented, implement what works and document the difference.
- If a phase is taking too long, ship the simplest version that passes the gate. TODO the rest.
- If a test is flaky, make it deterministic or skip it with a clear reason.
- If you need to deviate from the SSOT, do it, comment why, and flag it for human review.
- Prefer working code over perfect code. Ship the 80% solution.
