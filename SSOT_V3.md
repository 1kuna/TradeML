# TradeML — Single Source of Truth (SSOT) v3

**Purpose.** This is the canonical specification for the TradeML system. All code, configs, and design decisions must conform to this document. It supersedes all previous specs, blueprints, and architecture docs.

**Philosophy.** Prove the core thesis first: can a cross-sectional daily model find meaningful signal in US equities using free-tier data? Start with the simplest credible baseline, validate honestly, and add complexity only when prior results justify it. Every layer of abstraction must earn its place.

**Key constraint.** We start with free-tier data. This has real limitations (single-venue, no permanent IDs, limited corporate-action history). Phase 1 accepts these limitations to prove the pipeline works. If signal exists, we upgrade data quality as the first investment — not hardware, not model complexity.

---

## 0. System Overview

### 0.1 What We're Building

An autonomous research and trading system that:
1. Collects US equities market data from free/low-cost APIs
2. Builds cross-sectional features and trains models to predict short-horizon relative stock returns
3. Constructs portfolios with realistic cost modeling and risk controls
4. Validates with strict anti-overfitting governance
5. Generates ranked trade signals with confidence levels

### 0.2 Hardware & Network

| Machine | Role | Phase 1 Use |
|---------|------|-------------|
| **Raspberry Pi** | Data collection node (runs 24/7) | Collects daily bars, writes to NAS |
| **NAS (Synology/QNAP/TrueNAS)** | Central data storage (SMB share) | Stores all parquet data files |
| **Windows Desktop (RTX 4090)** | Primary training/research host | Runs training scripts — GPU not needed for Phase 1 |
| **Linux Workstation (4070 Ti)** | Secondary training host | Backup / parallel experiments |
| **DGX Spark** | Heavy experimentation | **Not used until Phase 3+** |

All machines on the same LAN. NAS is the single shared filesystem for data. Pi keeps its own local state database.

### 0.3 Storage Architecture

**Simple: parquet files on a NAS via SMB. State database local to each machine.**

```
//nas/trademl/                           # SMB share root
  data/
    raw/                                 # Immutable vendor parquet files
      equities_bars/                     # date=YYYY-MM-DD/data.parquet (multi-symbol per file)
      macros_fred/                       # series=XXX/data.parquet
      sec_filings/                       # date=YYYY-MM-DD/data.parquet (filing index)
    curated/                             # Adjusted, PIT-safe derived tables
      equities_ohlcv_adj/               # date=YYYY-MM-DD/data.parquet
    reference/                           # Corp actions, delistings, calendars, universe, events
      corp_actions.parquet
      delistings.parquet
      calendars/                         # Per-exchange session files
        XNYS.parquet                     # NYSE sessions, holidays, early closes
        XNAS.parquet                     # NASDAQ
      universe.parquet
      earnings_calendar.parquet          # Earnings dates by symbol
      splits_dividends.parquet           # From Polygon/AV, cross-referenced
    qc/                                  # Quality control ledger
      partition_status.parquet
  models/                                # Trained model artifacts
    equities_xs/
  reports/                               # Signal outputs, backtests
    daily/
  logs/

Pi local (NOT on NAS):
  ~/trademl/
    control/
      node.sqlite                        # backfill_queue + partition_status mirror
      stage.yml
      bookmarks.json
```

**Mount points:**
- Pi: `sudo mount -t cifs //nas/trademl /mnt/trademl -o credentials=/etc/nas-creds,uid=pi,gid=pi`
- Windows: Map `\\nas\trademl` as a network drive
- Linux: `/etc/fstab` entry with CIFS mount

**Why not S3/MinIO?** We don't need an object-store API. SMB on a LAN is simpler, faster for our data volumes, and every machine can access it natively. If we later need cloud sync, we add it on top.

**Why SQLite stays local on the Pi?** SQLite does not support reliable concurrent access over network filesystems. The Pi owns its queue and state database locally. It publishes immutable parquet files to the NAS. Training hosts read parquet from the NAS — they never touch the Pi's SQLite.

**Partitioning rules:**
- Equities daily bars: `date=YYYY-MM-DD/data.parquet` — one file per date, all symbols in the file. This avoids small-file overhead on NAS/SMB that one-file-per-symbol-per-day would create.
- Macro/series data: `series=XXX/data.parquet`
- Reference tables: single parquet files (corp_actions.parquet, etc.)
- All raw files carry metadata columns: `ingested_at`, `source_name`, `source_uri`, `vendor_ts`

### 0.4 Phasing

| Phase | Scope | Gate to Next Phase |
|-------|-------|--------------------|
| **Phase 1: Prove the Pipeline** | Pi collects daily EOD bars (free Alpaca) → NAS. Build features. Train Ridge baseline + LightGBM challenger. Walk-forward validation. Long-only, weekly rebalance, equal-weight buckets. | Stable positive rank IC across multiple years. Monotone decile spreads. Signal survives doubled costs. |
| **Phase 2: Prove the Signal** | Buy clean data (Norgate, Databento, or Sharadar). Proper security master with permanent IDs. Expand to 500+ symbols with 10+ years. Add sector/fundamental features. Paper trade (long-only or ETF-hedged). | Go/no-go bars met on clean data with realistic costs. Paper trading operational for 4+ weeks. |
| **Phase 3: Add Complexity** | Options sleeve (with paid OPRA data). Meta-stacker. Champion/challenger automation. Long/short if borrow permits. | Options delta-hedged PnL passes go/no-go. Stacker outperforms single sleeves. |
| **Phase 4: Scale** | Intraday models (only if justified). Offline RL for sizing (only if justified). DGX for heavy experiments. Live trading. | Prior phases fully stable. |

**Rule: don't start Phase N+1 until Phase N passes its gate.**

### 0.5 Non-Negotiable Invariants

1. **Point-in-time (PIT) discipline**: no future information in features, labels, or universe construction. Every dataset timestamped to public availability.
2. **Raw immutability**: vendor data is never modified after write. All adjustments happen in curated tables with full lineage.
3. **Honest validation**: walk-forward as primary method. CPCV/PBO/DSR as secondary diagnostics. Every discretionary choice logged and counted.
4. **Cost realism**: all backtests include spread, market impact, and fees. No "zero-cost" results.
5. **Simple first**: no technique is adopted until the simpler alternative has been tried and failed.

---

## 1. Data Layer

### 1.1 Phase 1 Datasets & Schemas

**Raw — Equities Daily Bars**

Source: Alpaca `/v2/stocks/bars` (free plan, IEX feed only)

| Column | Type | Notes |
|--------|------|-------|
| `date` | DATE | Trading day (exchange local) |
| `symbol` | TEXT | Ticker (⚠ not a permanent ID — see §1.4) |
| `open` | FLOAT | |
| `high` | FLOAT | |
| `low` | FLOAT | |
| `close` | FLOAT | |
| `vwap` | FLOAT | ⚠ IEX-venue only, not consolidated |
| `volume` | INT | ⚠ IEX-venue only |
| `trade_count` | INT | ⚠ IEX-venue only |
| `ingested_at` | TIMESTAMP | UTC |
| `source_name` | TEXT | "alpaca" |
| `source_uri` | TEXT | API endpoint |

**⚠ Free-tier data limitations (accepted for Phase 1, fixed in Phase 2):**
- Volume, VWAP, trade_count, and any derived liquidity features are from the IEX exchange only (~3% of US equity volume). These are biased and not comparable to consolidated SIP data.
- For Phase 1, price-based features (momentum, returns, volatility from close prices) are the most reliable. Liquidity features should be treated as rough proxies only.
- Proper consolidated data requires a paid vendor (Phase 2 upgrade).

**Raw — Macro/Treasuries**

Source: FRED API (free, generous limits)

| Column | Type |
|--------|------|
| `series_id` | TEXT |
| `observation_date` | DATE |
| `value` | FLOAT |
| `vintage_date` | DATE (for ALFRED revisions) |
| `ingested_at` | TIMESTAMP |

**Reference — Corporate Actions** (Phase 1: basic; Phase 2: proper security master)

Phase 1 (from Alpha Vantage / FMP):

| Column | Type |
|--------|------|
| `symbol` | TEXT |
| `event_type` | TEXT (split, dividend) |
| `ex_date` | DATE |
| `ratio` | FLOAT |
| `source` | TEXT |

**⚠ Phase 1 security-master limitations (accepted, fixed in Phase 2):**
- No permanent identifiers — symbols can be reused after delistings
- No ticker/name change history
- No merger/spinoff handling
- No terminal delisting return data
- No historical index constituent membership

These limitations mean Phase 1 backtests have residual survivorship bias and imprecise corporate-action adjustments. This is acceptable for pipeline validation but NOT for concluding whether alpha exists. Phase 2 addresses this by purchasing a proper reference dataset (Norgate, Databento, or Sharadar/Nasdaq Data Link) that provides PIT IDs, delisted coverage, and corporate-action history.

**Reference — Delistings** (Phase 1: basic)

| Column | Type |
|--------|------|
| `symbol` | TEXT |
| `delist_date` | DATE |
| `reason` | TEXT |
| `source` | TEXT |

**Curated — Equities OHLCV Adjusted**

Same column schema as raw bars, with our own split/dividend adjustments applied. Rebuilt deterministically from raw + reference. In Phase 1, adjustments are best-effort given the reference data limitations.

### 1.2 Vendors & Rate Limits (Phase 1)

| Vendor | Datasets | Free Rate Limit | Daily Cap |
|--------|----------|-----------------|-----------|
| **Alpaca** | Equities EOD bars (primary) | ~200 req/min | Unpublished (generous) |
| **Massive (Polygon.io)** | Equities EOD bars (secondary/cross-check), reference data (tickers, splits, dividends, financials) | 5 req/min | Unpublished (limited) |
| **Finnhub** | Equities daily candles (backup), earnings calendar, analyst estimates, company profiles, sector data, market news | 60 req/min overall | Unlimited |
| **Alpha Vantage** | Corp actions, listing status, splits/dividends, historical options (limited) | 5 req/min | 500/day |
| **FRED** | Macro, treasuries, economic data, ALFRED vintages | 120 req/min | Unlimited |
| **FMP** | Delistings, fundamentals, historical prices (backup), earnings calendar, sector performance | ~3 req/min practical | 250/day |
| **SEC EDGAR** | Filing index (8-K, 10-K, 10-Q), authoritative event dates, company facts | 10 req/sec | Unlimited (be polite) |

**Conservative Pi budgets** (~75% of published limits):

| Vendor | Pi RPM Cap | Pi Daily Cap |
|--------|-----------|-------------|
| Alpaca | 150 | 10,000 |
| Massive | 4 | 300 |
| Finnhub | 50 | 10,000 |
| Alpha Vantage | 4 | 400 |
| FRED | 80 | 5,000 |
| FMP | 3 | 200 |
| SEC EDGAR | 8 req/sec | 5,000 |

**Vendor roles:**
- **Price bars**: Alpaca (primary), Massive (cross-validation), Finnhub (backup)
- **Corporate actions / reference**: Alpha Vantage + Massive (splits, dividends, listings), FMP (delistings), SEC EDGAR (authoritative filing dates)
- **Events / calendar**: Finnhub (earnings calendar, analyst estimates), FMP (earnings calendar, economic calendar)
- **Macro**: FRED / ALFRED (treasuries, macro time series, vintage dates for PIT)

Having multiple sources for the same data enables cross-vendor validation — if Alpaca and Massive disagree on a close price, we flag it for review.

**Phase 2 data upgrade candidates** (in order of value-for-money):
- **Norgate Data**: survivorship-bias-free US stocks, delisted names, historical constituents, corporate actions. Gold standard for backtesting.
- **Databento**: PIT security master, corporate actions, official closing prices. API-native. New but comprehensive.
- **Sharadar / Nasdaq Data Link**: EOD prices + fundamentals + corporate actions with delisted coverage.
- **Tiingo**: EOD data with splits/dividends, decent API. Cheaper option.

### 1.3 Completeness Model (GREEN / AMBER / RED)

Every partition `(source, dataset, date)` is graded:
- **GREEN**: exists, QC passes (row count in expected range, calendar-aligned)
- **AMBER**: exists but marginal (short session, fewer symbols than expected)
- **RED**: missing

Tracked in Pi-local SQLite (`partition_status` table) and mirrored to `//nas/trademl/data/qc/partition_status.parquet` during daily sync.

Non-trading days: marked GREEN with `expected_rows=0`, `qc_code='NO_SESSION'`. No gap tasks for these.

### 1.4 Data Collection (Pi)

The Pi runs a single Python service that:

1. **Collects today's EOD bars** after market close (Alpaca primary)
2. **Collects reference data** on a weekly schedule (corp actions from AV, delistings from FMP, earnings from Finnhub/FMP, splits/dividends from Massive/AV)
3. **Collects macro data** daily (FRED treasuries and key series)
4. **Audits** what's been collected vs what should exist (using exchange calendar)
5. **Fills gaps** by re-requesting missing dates, prioritized by recency
6. **Runs cross-vendor price checks** weekly (compare Alpaca vs Massive/Finnhub for a sample)
7. **Curates** raw data into adjusted tables (daily, after collection)
8. **Writes parquet** to the NAS
9. **Tracks state** in local SQLite
10. **Syncs partition_status.parquet** to NAS for training host visibility

**Task queue** (local SQLite on Pi):

```sql
CREATE TABLE backfill_queue (
  id               INTEGER PRIMARY KEY AUTOINCREMENT,
  dataset          TEXT NOT NULL,
  symbol           TEXT,
  start_date       DATE NOT NULL,
  end_date         DATE NOT NULL,
  kind             TEXT NOT NULL,       -- BOOTSTRAP | GAP | FORWARD
  priority         INTEGER NOT NULL,
  status           TEXT NOT NULL,       -- PENDING | LEASED | DONE | FAILED
  attempts         INTEGER DEFAULT 0,
  next_not_before  TIMESTAMP,
  last_error       TEXT,
  created_at       TIMESTAMP NOT NULL,
  updated_at       TIMESTAMP NOT NULL,
  UNIQUE(dataset, symbol, start_date, end_date, kind)
);
```

**Vendor error handling:** NOT_ENTITLED or NOT_SUPPORTED responses mark the task as permanently failed for that vendor — no infinite retry. Budget exhaustion defers tasks with backoff, doesn't block the process.

### 1.5 Staged Universe

| Stage | Universe | EOD History | Promotion Trigger |
|-------|----------|-------------|-------------------|
| **Stage 0** (Phase 1) | 100 liquid US large/mid-caps | 5 years | ≥98% GREEN on EOD bars |
| **Stage 1** (Phase 2) | 500+ US equities (incl. delisted, with paid data) | 10–15 years | Manual, after data vendor purchased |

Universe selection (Phase 1): US common stocks above a liquidity floor, based on Alpaca's available symbols. Accept that this is survivorship-biased — the universe is names that are still listed today. Phase 2 fixes this with a proper security master.

### 1.6 QC Checks

**Per-date structural checks:**
- Expected columns present and correctly typed
- Row count within expected range (number of symbols × 1 row each)
- Date matches exchange calendar (not a weekend/holiday)
- No duplicate symbol entries per date

**Cross-date continuity:**
- No multi-day gaps without explanation (holiday/weekend)
- Price continuity checks around known corporate action dates

### 1.7 PIT Discipline

- Features and labels use only data available at prediction time
- Macro features lagged to public release timestamp
- Corporate actions applied downstream, never modifying raw data
- FRED series: store vintage dates where available (ALFRED) for revision-aware research
- Earnings dates: use the *announcement date* (not the report date) for distance-to-earnings features and for avoiding rebalancing into known events
- **Phase 1 honest limitation**: universe is survivorship-biased (current tickers only). Acknowledged and accepted. Fixed in Phase 2.

### 1.8 Exchange Calendars

Trading calendars drive the audit system (which dates should have data?), feature engineering (how many trading days in a window?), and label construction (5 trading days forward, not 5 calendar days).

**Source**: Use `exchange_calendars` (Python library, maintained, covers NYSE/NASDAQ and global exchanges) or `pandas_market_calendars` as a fallback.

**What's generated:**
- Per exchange: full list of trading sessions, holidays, early-close dates, and DST transitions
- Stored as parquet at `reference/calendars/XNYS.parquet`, `reference/calendars/XNAS.parquet`
- Columns: `date`, `market_open` (UTC), `market_close` (UTC), `is_early_close`, `is_holiday`

**Rules:**
- All feature/label window calculations count trading days, not calendar days (e.g., "5-day return" = 5 trading sessions)
- Calendar generation must be unit-tested against known dates (specific holidays, early closes like day after Thanksgiving, DST transitions)
- No ad-hoc daylight saving logic in feature/label code — only the calendar is authoritative
- Calendar is regenerated at the start of each year and whenever exchange schedule changes are announced

### 1.9 Curation Pipeline (Raw → Curated)

Curated tables are deterministic rebuilds from raw data plus reference data. They exist so that research code never touches raw vendor payloads directly.

**`curated/equities_ohlcv_adj/`** — the primary research dataset:

**Build process:**
1. Read raw daily bars for the date range being rebuilt
2. Load `reference/corp_actions.parquet` (splits and dividends)
3. Apply proportional adjustments backward from most recent date:
   - Splits: adjust prices by split ratio, adjust volume inversely
   - Cash dividends: compute adjustment ratio as `(close_before_ex - dividend) / close_before_ex` where `close_before_ex` is the closing price on the trading day immediately before the ex-date. All prices on dates before the ex-date are multiplied by this ratio. **Do not use the ex-date close** — the ex-date price already reflects the dividend drop, so using it would let normal ex-date price movement leak into the adjustment factor.
4. Log every adjustment applied: `{symbol, date, event_type, ratio, source}`
5. Write curated parquet to NAS

**Rebuild triggers:**
- New raw data arrives (daily, after collection)
- Reference data updates (corp actions added or corrected)
- Manual request (re-curation command)

**Rebuild scope:** Only rebuild dates affected by the change, not the entire history. A new split affects all dates before the ex-date for that symbol; a new daily bar only affects that date.

**Lineage:** Each curated file records `curated_at` timestamp and the set of raw + reference files that contributed to it.

**Phase 1 limitations:** Adjustments are best-effort. Without a proper security master, we may miss some corporate actions, handle ticker changes incorrectly, or fail to account for mergers/spinoffs. Accepted for Phase 1.

### 1.10 Events & Earnings Data

Earnings dates and other corporate events matter for two reasons:
1. **Features**: distance-to-earnings is a known cross-sectional predictor
2. **Risk management**: rebalancing into a name right before earnings is a known source of uncompensated vol

**Phase 1 sources:**
- **Finnhub** `/stock/earnings-calendar` and `/calendar/earnings`: upcoming and historical earnings dates
- **FMP** `/api/v3/earnings-calendar`: backup earnings calendar
- **SEC EDGAR** 8-K filing dates: authoritative announcement timing (more work to parse, but PIT-accurate)

**Storage:** `reference/earnings_calendar.parquet`

| Column | Type | Notes |
|--------|------|-------|
| `symbol` | TEXT | |
| `earnings_date` | DATE | Announcement date |
| `fiscal_quarter` | TEXT | e.g., "Q3 2025" |
| `time` | TEXT | "BMO" (before open), "AMC" (after close), "unknown" |
| `source` | TEXT | |
| `ingested_at` | TIMESTAMP | |

**Usage in features:** Compute `days_to_next_earnings` as a feature (§2.1). Consider excluding names within N days of earnings from rebalancing if vol impact is significant.

### 1.11 Training Host Data Discovery

The training host reads from the NAS. It needs to know what data is available and whether it's complete enough to train on.

**Mechanism:** The Pi syncs `qc/partition_status.parquet` to the NAS after each audit cycle. The training workflow reads this file to check GREEN coverage before proceeding.

**Workflow:**
1. Training script reads `//nas/trademl/data/qc/partition_status.parquet`
2. Computes GREEN fraction over the desired training window
3. If below threshold (e.g., < 0.98), abort with a message showing which dates/symbols are missing
4. If sufficient, proceed to load curated data from `//nas/trademl/data/curated/`

No live database connection needed. The parquet file is the contract between the Pi (data producer) and the workstation (data consumer).

---

## 2. Features & Labels

### 2.1 Cross-Sectional Features (Phase 1)

Per `(date, symbol)`, computed from curated OHLCV:

**Price-based (most reliable with free-tier data):**
- Multi-horizon momentum: 5, 20, 60, 126-day log returns
- Short-term reversal: 1-day and 5-day log returns (separate from longer momentum)
- Recent drawdown: max loss from peak over trailing 20 and 60 days
- Gap statistics: overnight return (close-to-open)

**Volatility:**
- 20-day and 60-day realized volatility (close-to-close)
- Idiosyncratic volatility: residual vol after removing market beta (trailing 60-day OLS vs market index)

**Liquidity (⚠ IEX-venue only, treat as rough proxies):**
- 20-day average daily dollar volume
- Amihud illiquidity ratio (use with caution — single-venue volume)

**Size/Price (important controls):**
- Log price level

**⚠ Dropped from Phase 1 — market cap:** Computing market cap requires historical shares outstanding as a dated PIT series. If shares come from a latest company profile (e.g., Finnhub `/stock/profile2`), that is straight lookahead bias — you'd be using today's share count for past dates. Market cap features are deferred to Phase 2 when a proper security master with historical shares is available.

**Event proximity (risk control only — not a predictive feature in Phase 1):**
- Binary flag: earnings within next 5 trading days → **used to exclude names from rebalancing**, not as a model input
- ⚠ `days_to_next_earnings` is only PIT-safe if the historical earnings calendar preserves what was known as of each date. Finnhub and FMP calendars may backfill corrected dates, which leaks future info. Only SEC EDGAR 8-K filing dates are clearly defensible for historical PIT timing. In Phase 1, use the earnings flag **only as a risk/exclusion filter** (don't trade into imminent earnings), not as a predictive feature. Promote to a feature in Phase 2 if/when you have a verified PIT earnings calendar.

**Phase 2 additions (with paid data):**
- Industry-relative features (momentum, vol, returns relative to sector average)
- Residual momentum (return after stripping market + sector + size factors)
- Fundamental features if data permits: valuation (E/P, B/P), profitability (ROE, gross margins), investment (asset growth), accruals, issuance

### 2.2 Feature Preprocessing

**Rank normalization (not z-scores).**

Cross-sectional z-scores are fragile: sensitive to outliers, and collapse date-level features (like calendar dummies) to zero. Following the empirical ML literature, we use cross-sectional rank transforms:

1. Each date, rank all symbols by each feature
2. Map ranks to [-1, +1] (or [0, 1]) via percentile
3. This is robust to outliers, preserves monotone relationships, and handles missing data gracefully

**Date-level features** (if any, e.g., macro regime indicators): kept as-is, NOT rank-normalized within date. Used as interaction terms or model conditioning variables.

**Missing data:** Features with >30% missing on a given date are dropped for that date. Individual missing values filled with cross-sectional median (i.e., rank 0.5).

### 2.3 Labels

**Primary target: universe-relative forward returns (cross-sectional demeaning).**

For each symbol on date t:
```
label(t) = log_return(t, t+5) - mean(log_return(t, t+5) across all symbols in universe)
```

This is equal-weighted cross-sectional demeaning — we subtract the mean 5-day forward return of our universe on each date. This is NOT the same as subtracting SPY or a market-cap-weighted index; it is a universe-relative measure that removes the common daily return component.

**Why universe-relative?** Raw returns are dominated by market beta. Cross-sectional demeaning improves signal-to-noise for a relative-value portfolio. We use our own universe mean rather than an external benchmark because (a) our Phase 1 universe is only 100 names and may not track broad indices well, and (b) this is consistent with equal-weight top-quintile portfolio construction.

**Secondary targets (for comparison):**
- Raw 5-day forward returns (as a check — useful for seeing how much of the signal is beta vs stock-specific)
- 20-day forward returns (for comparison with 5-day horizon)

**Triple-barrier labels** (Phase 2): Profit-take/stop-loss labels scaled by volatility. Deferred because they add complexity and the continuous regression target is cleaner for initial research.

### 2.4 Feature Engineering Rules

- All price-derived features lagged at least one bar
- Feature list persisted with every trained model
- New features require demonstrating incremental value via walk-forward before inclusion
- Every discretionary choice (feature set, target, window, etc.) is logged and counted toward the multiple-testing budget

---

## 3. Model Architecture

### 3.1 Phase 1 Baseline: Ridge Regression

**Start here. If this doesn't work, fancier models probably won't save you.**

A cross-sectional Ridge regression each week:
- Input: rank-normalized features for all symbols on rebalance date
- Target: universe-relative 5-day forward return
- Regularization: Ridge penalty tuned via walk-forward

This is roughly equivalent to a Fama-MacBeth regression approach and serves as the "can a linear model find signal" test. If rank IC is consistently near zero, the feature set needs work before adding nonlinear models.

### 3.2 Phase 1 Challenger: LightGBM

Only after the Ridge baseline establishes whether there's signal at all:

- Same inputs and target as Ridge
- Small HPO grid via Optuna (≤20 trials, logged):
  - `num_leaves`: {31, 63}
  - `learning_rate`: {0.01, 0.05}
  - `min_data_in_leaf`: {50, 200}
  - `colsample_bytree`: {0.6, 0.9}
  - `subsample`: {0.6, 0.9}
- Time-decay sample weighting: 18-month half-life (recent data weighted more)
- Must demonstrably beat Ridge on out-of-sample IC to justify its use

### 3.3 Phase 2: CatBoost + Expanded Features

- CatBoost as second nonlinear challenger
- Expanded feature set including sector-relative, fundamental (with paid data)
- Regime masks: exclude pre-2012 data from training, use for stress testing only
- Wider universe (500+ symbols, 10+ years)

### 3.4 Phase 3: Options Vol Sleeve

**Only with paid OPRA-grade data (Databento or equivalent).**

- Compute IV from NBBO mids; fit SVI/SSVI surfaces
- Features: ATM IV level, skew, curvature, term-structure slope, IV rank
- Model: GBDT on surface features
- Target: delta-hedged PnL
- Separate validation (same walk-forward framework)

### 3.5 Phase 3: Meta-Stacker

When multiple sleeves exist, blend OOF predictions via stacking:
- Input: out-of-fold predictions from each sleeve, aligned by `(date, symbol)`
- Model: Ridge or small GBDT
- Must respect temporal fold boundaries (no leakage)

### 3.6 Phase 4: Optional (only if justified)

**Intraday sleeve**: Only if minute-bar data demonstrates a clear sequential edge that daily features can't capture. Requires proper consolidated minute data.

**Offline RL for sizing**: Only if a stable alpha signal exists and sizing optimization demonstrably improves net performance beyond simple heuristics.

**These are explicitly not planned until Phase 1 and 2 are complete and successful.**

---

## 4. Validation & Anti-Overfitting

### 4.1 Primary: Expanding Walk-Forward

The main validation method. Mimics how the model would actually be deployed.

**Configuration:**
- Initial training window: at least 2 years
- Expanding window: each refit adds 1 quarter (or 1 month) of new data
- Test on the next period (1 quarter or 1 month)
- **Purge the last 5 trading days of the training window** before fitting. Because labels use 5-day forward returns, the last 5 days of training data have labels that overlap with the test window. Dropping them prevents leakage. This is not optional.

**Metrics computed each fold:**
- Rank IC (Spearman correlation between predicted and realized rankings)
- Decile/quintile spread returns (long top bucket minus short bottom bucket)
- Hit rate (% of top-bucket stocks that outperform)

**What we look for:**
- Stable positive rank IC across folds (not just average — look at distribution)
- Monotone relationship: higher predicted score → higher realized return across buckets
- Signal survives across different years, sectors, and liquidity buckets
- Signal survives when costs are doubled as a stress test

### 4.2 Secondary: CPCV / PBO / DSR

These are useful diagnostics, not the primary validation gate.

**CPCV**: 8-fold combinatorially purged CV with 10-day embargo. Used to check for overfitting across multiple path combinations.

**PBO**: Probability of backtest overfitting. A high PBO is a red flag, but a low PBO on a short dataset is not strong evidence of a real signal (pseudo-precision on non-stationary data).

**DSR**: Deflated Sharpe Ratio adjusting for skew/kurtosis/multiple testing. Useful as a sanity check.

**Important caveat (from adversarial review):** CPCV/PBO/DSR only account for the trials you explicitly count. The real multiple-testing problem is every discretionary choice: universe, target, features, cost assumptions, rebalance frequency, etc. Recent research finds that design-choice variation exceeds standard statistical error by ~59%. We mitigate this by:
- Logging every design choice as part of the experiment record
- Running placebo tests (scrambled labels) to check if the validation framework itself is leaky
- Testing robustness across years, sectors, size buckets, and cost assumptions

### 4.3 Additional Validation Tests

- **IC by year**: Is the signal stable over time or concentrated in one period?
- **IC by sector/size bucket**: Does the signal work broadly or only in one corner?
- **Placebo test**: Train on shuffled labels. IC should be ~0. If not, something is leaking.
- **Cost stress test**: Double all cost assumptions. Does the strategy survive?
- **Decay test**: How quickly does the signal decay after the rebalance date? (IC at t+1, t+2, ..., t+10)

### 4.4 Phase 1 Go/No-Go Criteria

Phase 1 is about proving the pipeline works and there's signal worth investigating. The bars are diagnostic, not definitive:

| Metric | What We're Looking For | Notes |
|--------|----------------------|-------|
| Mean rank IC | Consistently positive (e.g., > 0.02) | Across walk-forward folds |
| IC stability | Positive in most years, not just one period | Check IC by year |
| Decile monotonicity | Higher predicted → higher realized | At least approximate monotonicity |
| Survives doubled costs | Strategy still marginally profitable | Cost stress test |
| Placebo passes | IC ≈ 0 on shuffled labels | Validation isn't leaky |

**If Phase 1 passes these criteria**, it justifies investing in better data (Phase 2) and more rigorous testing.

**If Phase 1 fails**, the problem is almost certainly data quality or feature set, not model complexity. The response should be to upgrade data, not to add more layers.

### 4.5 Phase 2 Go/No-Go Bars (with clean data)

| Metric | Target | Notes |
|--------|--------|-------|
| Sharpe Ratio (net) | ≥ 0.7 (stretch: ≥ 1.0) | After all costs |
| Max Drawdown | ≤ 25% | Depends on leverage |
| Mean Rank IC | > 0.03 | Stable across time |
| PBO | ≤ 10% | Treat as diagnostic, not gospel |
| Survives cost doubling | Still profitable | Stress test |
| Sector/size robustness | Signal in most buckets | Not a single-corner artifact |

### 4.6 Dark Holdout

Reserve the most recent 6 months as an untouched holdout. Only used once for final sign-off before paper trading. Don't peek.

On a 5-year dataset this is expensive (10% of data). Accept the cost — it's the only clean independent test.

---

## 5. Portfolio Construction & Execution

### 5.1 Phase 1: Simple Long-Only

**Equal-weight top bucket.**

1. Model scores all symbols each rebalance date
2. Rank scores from best to worst
3. Go long the top quintile (20% of universe) with equal weights
4. Rebalance weekly (every Friday close, or staggered across the week)
5. Benchmark: equal-weight universe return

**Why this simple?** The adversarial review correctly identified that z-score → weights → vol target → Kelly is too much estimation error stacked on estimation error when the signal itself is unproven. Equal-weight top bucket is the cleanest test of "does the ranking work."

### 5.2 Phase 2: Long + ETF Hedge

When signal is established:
1. Long top quintile (equal-weight)
2. Short a broad market ETF (SPY or similar) to hedge market beta
3. Size the hedge to target approximate market neutrality
4. Rebalance weekly

**Why not long/short individual names?** Alpaca's free plan only allows shorting easy-to-borrow names. The interesting shorts (small, illiquid, hard-to-borrow) are exactly the ones you can't short. ETF hedging captures most of the market-neutral benefit without borrow constraints.

### 5.3 Cost Model

Every backtest must account for:

| Cost Component | Phase 1 Default | Notes |
|----------------|-----------------|-------|
| Commission | $0 | US equities via most brokers |
| Effective spread | 5 bps (one-way) | Conservative for large/mid-cap |
| Market impact | Negligible at retail size | √-law model available but likely irrelevant at <$1M |
| Stress test | Double all costs (10 bps spread) | Must survive this |

**Phase 2 additions:** Borrow cost for ETF hedge (typically very low for SPY). Real slippage calibration requires live fills — paper trading is for operational testing only (see §5.5).

**Rebalance frequency consideration:** Weekly rebalancing with a 5-day target means you're rebalancing at roughly the label horizon. This is intentional — it matches the prediction window to the holding period. If turnover is too high, stagger rebalances (e.g., rebalance 1/5 of positions each day).

### 5.4 Backtest Engine

- Deterministic: same config + data = identical results
- Position-aware, tracks holdings over time
- Corporate actions (splits/dividends) applied on ex-dates
- Weekly rebalance at configurable day/time
- Outputs: equity curve, trade log, cost attribution, IC time series, decile returns

### 5.5 Paper Trading (Phase 2)

Broker: Alpaca (paper trading account, free)
- Execute weekly rebalance orders
- Compare simulated vs paper fills (operational smoke test only — not for calibrating slippage models)
- Run for ≥ 4 weeks before considering live

**Caveat (from Alpaca docs):** Paper trading results may differ significantly from live market conditions. Paper fills are for operational testing, not slippage calibration.

---

## 6. Operational Flow

### 6.1 Pi Data Node (runs continuously)

```
Pi (runs 24/7):
  1. After market close: fetch today's EOD bars from Alpaca
  2. Write parquet to //nas/trademl/data/raw/equities_bars/date=YYYY-MM-DD/data.parquet
  3. Audit: compare collected dates vs exchange calendar
  4. Fill gaps: re-request any missing dates
  5. Update local SQLite (partition_status, backfill_queue)
  6. Sync partition_status.parquet to NAS for training host visibility
  7. Sleep → repeat
```

**Schedule:**
- EOD collection: once daily, ~30 min after market close (allow for data settlement)
- Gap audit: daily after collection
- Reference data refresh (corp actions, delistings): weekly

### 6.2 Training Workflow (workstation, manual in Phase 1)

```
1. Read data from //nas/trademl/data/curated/
2. Check GREEN coverage (≥98% of expected dates in training window)
3. Build rank-normalized features
4. Compute universe-relative forward returns (labels)
5. Run expanding walk-forward with Ridge baseline
6. Evaluate IC stability, decile spreads, placebo test, cost stress
7. If promising: run LightGBM challenger with same walk-forward
8. Compare Ridge vs LightGBM on OOS metrics
9. Save model + results to //nas/trademl/models/
```

Phase 1: run manually from a script or notebook.
Phase 2: schedule nightly or weekly.

### 6.3 Monitoring (Phase 2+)

- Coverage heatmap: date × dataset GREEN/AMBER/RED
- Feature drift: change in rank distributions over time
- Model staleness: IC trend since last retrain
- Tripwires: halt signals if data quality degrades

---

## 7. Testing Strategy

### 7.1 Test Suites

| Suite | Scope | How to Run |
|-------|-------|------------|
| **Unit** | Feature builders, label construction, calendar logic, portfolio math, rank normalization, cost models | `pytest tests/unit -q` |
| **Integration** | End-to-end pipeline on synthetic data: ingest → curate → features → labels → train → validate → backtest | `pytest tests/integration -q` |
| **Live endpoints** | Real vendor API calls (Alpaca, Finnhub, FRED, etc.) — skips automatically if credentials missing | `pytest tests/integration/test_live_endpoints.py -m liveapi -q` |
| **Validation** | CPCV purging/embargo correctness, walk-forward fold logic, placebo test framework | `pytest tests/validation -q` |

### 7.2 Required Tests per Component

**Data layer:**
- Calendar: known holidays, early closes, DST transitions produce correct sessions
- Connectors: pagination, rate-limit handling, normalization, UTC conversion (per vendor)
- Curation: split/dividend adjustments produce expected price continuity around known ex-dates
- Round-trip: re-ingesting a historical day produces identical parquet (determinism)

**Features & labels:**
- PIT safety: shifting input data forward/backward in time doesn't change labels/features outside the expected window
- Rank normalization: output is in [-1, 1], no NaN for features with sufficient coverage, date-level features are NOT rank-normalized
- Market-relative labels: sum to approximately zero cross-sectionally on each date (sanity check)

**Validation:**
- CPCV: multi-symbol purging + embargo produces expected retention rates
- Walk-forward: expanding window grows correctly, no overlap between train and test
- Placebo: shuffled labels → IC ≈ 0 (validates that the framework isn't leaky)

**Portfolio & backtest:**
- Equal-weight top quintile: correct number of positions, correct weights
- Weekly rebalance: fires on expected dates
- Cost model: spread + impact applied correctly
- Determinism: same config + data → identical equity curve

### 7.3 CI Expectations

Any change to SSOT-governed code (features, labels, validation, portfolio, backtest, curation) should run unit + integration tests. Breaking a test requires either fixing the code or explicitly updating the SSOT with rationale.

---

## 8. Logging & Observability

### 8.1 Pi Data Node Logging

- All log output to `~/trademl/logs/node.log` (rotated daily, kept 30 days)
- Structured log fields: `timestamp` (UTC), `level`, `vendor`, `dataset`, `action`, `rows`, `elapsed_ms`, `error`
- Key events logged:
  - Collection start/finish per vendor per date (with row counts)
  - Gap audit results (dates flagged RED/AMBER)
  - Backfill task start/complete/fail (with error messages)
  - Budget status (spent vs cap per vendor, daily)
  - Curation rebuild triggers and completion
  - Partition status sync to NAS

### 8.2 Training Workflow Logging

- Each training run logs to `//nas/trademl/logs/training/run_YYYYMMDD_HHMMSS.log`
- Captures: GREEN coverage check result, feature build summary, model hyperparameters, walk-forward fold metrics (IC, decile spreads), final summary, artifact paths
- Model artifacts saved alongside the log so each run is fully reproducible

### 8.3 Alerts / Failure Handling

Phase 1 is manual-check. Log files are the primary diagnostic tool.

Phase 2+: consider lightweight alerting (email, Slack webhook, or Pushover notification) for:
- Pi collection failure (no data for today by cutoff time)
- GREEN coverage dropping below threshold
- Walk-forward IC turning negative for N consecutive folds

---

## 9. Pi Setup & Environment

### 9.1 Pi Data Node Setup

A wizard script (`scripts/pi_data_node_wizard.py`) handles first-time setup:

1. **Detect environment**: ARM vs x86, available memory, external storage
2. **Configure NAS mount**: Prompt for NAS path, test write access, add to `/etc/fstab` for persistence
3. **Collect API keys**: Alpaca, Finnhub, Alpha Vantage, FRED, FMP, Massive — write to `.env`
4. **Initialize local state**: Create `~/trademl/control/`, initialize `node.sqlite` with empty tables
5. **Seed Stage 0**: Build initial 100-symbol universe, enqueue BOOTSTRAP tasks for 5y EOD
6. **Configure schedule**: Set collection time (default: 16:30 ET), maintenance time (default: 02:00 local)
7. **Start node**: Optionally launch `python -m data_node` at the end

**Resume:** The wizard reads existing state and continues from where it left off. Safe to re-run.

### 9.2 Environment Variables (`.env`)

```bash
# Role
TRADEML_ENV=local                    # local | dev | prod

# NAS mount (Pi writes here)
NAS_MOUNT=/mnt/trademl

# Pi local state (NOT on NAS)
LOCAL_STATE=~/trademl/control

# API keys
ALPACA_API_KEY=...
ALPACA_BASE_URL=https://paper-api.alpaca.markets/v2
FINNHUB_API_KEY=...
ALPHA_VANTAGE_API_KEY=...
FRED_API_KEY=...
FMP_API_KEY=...
MASSIVE_API_KEY=...

# Node identity
EDGE_NODE_ID=rpi-01

# Collection schedule
COLLECTION_TIME_ET=16:30             # 30 min after close
MAINTENANCE_HOUR_LOCAL=2             # 02:00 local time

# Budget overrides (optional — defaults from config)
# NODE_DAILY_CAP_ALPACA=10000
# NODE_RPM_ALPACA=150
```

### 9.3 Workstation Setup

Simpler than the Pi — no collection, just read from NAS:

1. Mount NAS share (Windows: map network drive; Linux: CIFS mount)
2. Clone repo, create venv, install dependencies
3. Copy `.env` with NAS path (no API keys needed for training-only use)
4. Run training scripts pointing at NAS data path

---

## 10. Configuration

### 10.1 Training Config (equities_xs.yml)

```yaml
model: ridge                    # Phase 1 baseline
challenger: lightgbm            # only after baseline evaluated

target: universe_relative_5d      # universe-relative 5-day forward return

data:
  dependencies: [equities_ohlcv_adj]
  green_threshold: 0.98
  window_years: 5               # Stage 0

features:
  price:
    momentum: [5, 20, 60, 126]
    reversal: [1, 5]
    drawdown: [20, 60]
    gap: [overnight]
  volatility:
    realized: [20, 60]
    idiosyncratic: [60]         # residual after market beta removal
  liquidity:                     # ⚠ IEX-only, rough proxies
    adv_dollar: [20]
    amihud: [20]
  controls:
    log_price: true
    log_market_cap: true         # if available

preprocessing:
  method: rank_normalize         # cross-sectional rank → [-1, 1]
  missing_threshold: 0.30        # drop feature if >30% missing on a date
  missing_fill: cross_sectional_median

validation:
  primary: expanding_walk_forward
  initial_train_years: 2
  step: 3_months
  secondary: [cpcv, pbo, dsr]
  cpcv_folds: 8
  embargo_days: 10
  additional: [ic_by_year, ic_by_sector, placebo, cost_stress]

portfolio:
  method: equal_weight_top_quintile
  rebalance: weekly
  hedge: none                    # Phase 1: long-only
  cost_spread_bps: 5
  cost_stress_multiplier: 2.0
```

### 10.2 Pi Node Config

```yaml
node:
  nas_mount: /mnt/trademl
  local_state: ~/trademl/control/
  collection_time_et: "16:30"    # 30 min after close (ET)
  maintenance_hour_local: 2      # 02:00 local time
  
stage:
  current: 0
  stage_0:
    symbols: 100
    eod_years: 5
    green_promote_threshold: 0.98

vendors:
  alpaca:
    rpm: 150
    daily_cap: 10000
    datasets: [equities_eod]
    role: primary_bars
  massive:
    rpm: 4
    daily_cap: 300
    datasets: [equities_eod, reference_tickers, reference_splits, reference_dividends]
    role: cross_validation_bars, reference
  finnhub:
    rpm: 50
    daily_cap: 10000
    datasets: [equities_eod, earnings_calendar, company_profile]
    role: backup_bars, events
  alpha_vantage:
    rpm: 4
    daily_cap: 400
    datasets: [corp_actions, listings]
    role: reference
  fred:
    rpm: 80
    daily_cap: 5000
    datasets: [macros_treasury]
    role: macro
  fmp:
    rpm: 3
    daily_cap: 200
    datasets: [delistings, earnings_calendar, fundamentals]
    role: reference, events
  sec_edgar:
    rpm: 8                       # requests per second, not per minute
    daily_cap: 5000
    datasets: [filing_index]
    role: authoritative_events
```

---

## 11. Current State (honest assessment, March 2026)

### What Exists

- **Data connectors**: Alpaca, Massive, Finnhub, FRED, Alpha Vantage, FMP — basic versions exist with known bugs
- **Data node**: Partially built; forward ingest works, backfill has critical bugs (see below)
- **Feature store**: Basic 8-feature set (needs rework per §2)
- **Labeling**: Horizon returns (needs switch to universe-relative per §2.3)
- **Models**: Ridge/Logistic baselines only (LightGBM never implemented)
- **Validation**: CPCV with symbol-aware purging fix, PBO, DSR (needs walk-forward as primary)
- **Portfolio**: Basic construction (needs simplification to equal-weight buckets)
- **Backtest**: Event-driven engine, tested on small universe
- **Execution**: Square-root impact model, Almgren-Chriss simulator
- **Options**: IV calculator, SVI fitting (defer to Phase 3)
- **Monitoring**: PSI/KL drift detection, tripwire system

### Known Bugs (from code review)

1. **CRITICAL**: Budget limiter double-counts spend and can block indefinitely
2. **CRITICAL**: Raw storage partitions are date-only; files can overwrite across symbols
3. **HIGH**: Worker shutdown can hang
4. **HIGH**: NOT_ENTITLED errors cause infinite retry
5. **HIGH**: Maintenance exports can OOM on large histories
6. **MEDIUM**: QC sampling weights inverted

### What Needs to Be Built/Fixed for Phase 1

1. Fix critical data node bugs (#1, #2, #3, #4 above)
2. Switch to NAS storage (parquet on SMB, SQLite local to Pi)
3. Set up NAS mount on Pi + workstation; verify read/write from both
4. Generate and store exchange calendars (XNYS, XNAS) using `exchange_calendars`
5. Build curation pipeline (raw → adjusted OHLCV with split/dividend handling)
6. Wire all Phase 1 vendors: Alpaca (bars), Massive (cross-check + reference), Finnhub (backup bars + earnings), AV (corp actions), FRED (macro), FMP (delistings), SEC EDGAR (filing dates)
7. Build earnings calendar collection and storage
8. Rework features: rank normalization, fix seasonality bug, add reversal + idiosyncratic vol + distance-to-earnings
9. Switch labels to universe-relative forward returns
10. Implement expanding walk-forward as primary validation
11. Implement Ridge baseline
12. Implement LightGBM challenger
13. Simplify portfolio to equal-weight top quintile, weekly rebalance
14. Seed 100-symbol universe and collect 5 years of EOD data
15. Implement cross-vendor price validation (weekly Alpaca vs Massive/Finnhub spot-check)
16. Build partition_status sync from Pi to NAS
17. Write unit + integration tests per §7

---

## 12. Open Questions & Known Risks

### Acknowledged Risks

1. **Free-tier data quality**: IEX-only feed has venue bias. Price-based features are likely okay; liquidity features are compromised. We accept this for Phase 1 and upgrade in Phase 2.

2. **Survivorship bias**: Phase 1 universe is current tickers only. Backtests are biased. We know this and don't make go/no-go decisions based on Phase 1 Sharpe alone — we look at IC stability and robustness instead.

3. **Small sample**: 5 years × 100 names with overlapping 5-day labels gives limited effective sample size. Phase 1 results should be interpreted cautiously. The real test is Phase 2 with 10+ years and 500+ names.

4. **Alpha may not exist**: Cross-sectional equity alpha from daily free-tier data may simply not be there at a meaningful level, especially after costs. The honest answer is "we don't know yet." The purpose of Phase 1 is to find out cheaply.

### Questions for Future Investigation

- What is the realistic alpha potential for a retail system with this kind of data and setup?
- Is weekly rebalancing optimal, or would biweekly or monthly be better given costs?
- At what point (AUM, capacity) does market impact become the binding constraint?
- Does the signal improve meaningfully with consolidated SIP data vs IEX-only?
- Are there orthogonal data sources (alternative data, NLP, etc.) that would add value at low cost?

---

## 13. Acceptance Criteria

### Phase 1 Complete When:

- [ ] Pi collecting 100-symbol daily EOD bars to NAS reliably (≥98% GREEN)
- [ ] Corporate actions applied (best-effort with free data)
- [ ] Rank-normalized features computed correctly
- [ ] Market-relative labels computed correctly
- [ ] Ridge baseline trained via expanding walk-forward
- [ ] Walk-forward IC is consistently positive across most folds
- [ ] Placebo test passes (shuffled labels → IC ≈ 0)
- [ ] Decile returns approximately monotone
- [ ] LightGBM challenger compared against Ridge
- [ ] Results survive doubled cost assumptions
- [ ] Results are reproducible (same data + config → same output)

### Phase 2 Complete When:

- [ ] Paid data vendor integrated (security master, delisted names, PIT corp actions)
- [ ] Universe expanded to 500+ with 10+ years of history
- [ ] Features expanded (sector-relative, fundamental where available)
- [ ] Net Sharpe ≥ 0.7 on walk-forward with realistic costs
- [ ] Signal robust across years, sectors, size buckets
- [ ] Paper trading operational for 4+ weeks

### Phase 3 Complete When:

- [ ] Options data (OPRA-grade) integrated
- [ ] Delta-hedged vol sleeve passes go/no-go bars
- [ ] Meta-stacker outperforms single sleeves
- [ ] Champion/challenger automation working

---

## Appendix A: Connector Specifications

All connectors must implement:
- Exponential backoff with jitter on 429/5xx errors
- Respect per-vendor budget from config (RPM + daily cap)
- Raw payload → parquet with metadata columns (`ingested_at`, `source_name`, `source_uri`, `vendor_ts`)
- Write to NAS via mount path (not S3 API)
- Log every request: vendor, endpoint, symbols requested, rows returned, elapsed time

**Alpaca** — Primary equities bars
- Base URL: `https://paper-api.alpaca.markets/v2` (paper/market data endpoint)
- Auth: API key only via `APCA-API-KEY-ID` header (no secret key needed for market data)
- Endpoint: `/v2/stocks/bars` (multi-symbol batch, chunks of ≤100 symbols)
- Data: Daily OHLCV bars. Free plan = IEX exchange feed only.
- Also available: `/v2/stocks/{symbol}/corporate_actions/announcements` for splits/dividends
- Pagination: next-page token based

**Massive (Polygon.io)** — Cross-validation bars + reference data
- Bars: `/v2/aggs/ticker/{symbol}/range/1/day/{from}/{to}` (one symbol at a time on free tier)
- Reference: `get_tickers` (active/delisted ticker metadata), `get_stock_dividends`, `get_stock_splits`, `get_stock_financials_vX`
- Free tier: 5 req/min, EOD data only (delayed). Useful for cross-checking Alpaca and for reference data that Alpaca doesn't provide.
- Note: Free tier is slow. Use primarily for reference data and weekly cross-vendor price validation, not bulk daily collection.

**Finnhub** — Backup bars + events/earnings
- Bars: `/stock/candle` (one symbol at a time, daily candles)
- Events: `/stock/earnings-calendar`, `/calendar/earnings` (upcoming + historical earnings dates)
- Reference: `/stock/profile2` (company info, sector, industry), `/stock/peers`, `/stock/recommendation` (analyst estimates)
- Fundamentals: `/stock/financials-reported`, `/stock/metrics` (basic financial ratios)
- Market: `/stock/market-status`, `/stock/market-holiday`
- Rate limit: 60 req/min overall; 900/min for market data; 300/min for fundamentals

**Alpha Vantage** — Corporate actions + listings
- Listings: `LISTING_STATUS` (active and delisted symbols with IPO/delist dates)
- Corp actions: splits and dividend functions
- Options: `HISTORICAL_OPTIONS` (limited, free tier — useful for exploration only)
- Note: Extremely rate-limited (5/min, 500/day). Use for reference data updates, not bulk collection.

**FRED / ALFRED** — Macro + treasuries
- Series: `/fred/series/observations` for time series data (treasuries, CPI, unemployment, etc.)
- Vintages: `/fred/series/vintagedates` for revision-aware PIT data via ALFRED
- Categories: `/fred/category/series` for discovering related series
- All endpoints free. 120 req/min.

**FMP** — Delistings + backup fundamentals
- Delistings: `/api/v3/delisted-company` (list of delisted companies)
- Prices: `/api/v3/historical-price-full/{symbol}` (backup daily prices with splits/dividends)
- Earnings: `/api/v3/earnings-calendar` (upcoming earnings)
- Fundamentals: `/api/v3/income-statement/{symbol}`, `/api/v3/balance-sheet-statement/{symbol}`
- Economics: `/api/v3/treasury-rate`, `/api/v3/economic-calendar`
- Note: 250 req/day on free tier. Prioritize delistings and earnings calendar.

**SEC EDGAR** — Authoritative filing dates
- Filing index: `/cgi-bin/browse-edgar` or bulk archives for 8-K, 10-K, 10-Q filings
- Company facts: `/api/xbrl/companyfacts/` for structured financial data
- Rate limit: 10 req/sec (be polite — use `User-Agent` header with contact email per SEC policy)
- Use for: authoritative earnings announcement dates (8-K filings), PIT event timing
- Note: Parsing SEC data is more work than vendor APIs but is the gold standard for event timing.

See **Data_Sourcing_Playbook.md** for full vendor details and **Data_Sources_Detail.pdf** for exact endpoint paths and parameters.

---

## Appendix B: Glossary

| Term | Definition |
|------|-----------|
| **Rank IC** | Spearman correlation between predicted rankings and realized return rankings |
| **CPCV** | Combinatorially Purged Cross-Validation — addresses temporal dependence in financial CV |
| **PBO** | Probability of Backtest Overfitting — fraction of CV paths where in-sample best is below median OOS |
| **DSR** | Deflated Sharpe Ratio — adjusts Sharpe for non-normality and multiple testing |
| **Walk-forward** | Expanding-window train/test where the model is periodically retrained on all available data up to each test point |
| **PIT** | Point-in-time — ensures no future information is used |
| **GREEN/AMBER/RED** | Data quality grades: complete / marginal / missing |
| **SIP** | Securities Information Processor — consolidated US equity data feed from all exchanges |
| **IEX** | Investors Exchange — a single venue; Alpaca free plan data comes from this exchange only |
| **SVI/SSVI** | Stochastic Volatility Inspired — parametric model for fitting implied volatility surfaces |
| **ADV** | Average Daily Volume — typically 20-day trailing average |
| **OPRA** | Options Price Reporting Authority — the consolidated US options data feed |
| **Fama-MacBeth** | Cross-sectional regression approach: regress returns on characteristics each period, average the coefficients |
| **NAS** | Network Attached Storage — shared filesystem accessible from all machines on the LAN |

---

## Appendix C: Phase 2 Data Vendor Comparison

When Phase 1 results justify upgrading data quality, evaluate these vendors:

| Vendor | PIT IDs | Delisted | Corp Actions | Constituents | EOD Prices | Fundamentals | Approx. Cost |
|--------|---------|----------|-------------|--------------|------------|-------------|-------------|
| **Norgate Data** | ✅ | ✅ | ✅ | ✅ (S&P, Russell) | ✅ | ❌ | ~$50–100/mo |
| **Databento** | ✅ | ✅ | ✅ | Partial | ✅ | ❌ | Pay-per-use |
| **Sharadar (Nasdaq Data Link)** | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ~$30–50/mo |
| **Tiingo** | Partial | Partial | ✅ | ❌ | ✅ | ❌ | Free–$30/mo |

Also consider for Phase 2+:
- **IBKR Stock Loan Dashboard**: indicative borrow fees and availability (free for IBKR clients). Useful for short-leg realism if/when we move beyond long-only + ETF hedge.

---

*This document is the single source of truth for the TradeML project. All previous specs, blueprints, and architecture docs are archived under `docs/archive/`. Changes to this SSOT require a version bump and documented rationale.*
