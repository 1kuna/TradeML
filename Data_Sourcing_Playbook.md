# Ultimate U.S. Equities & Options Data Sourcing Playbook (Free → Prosumer, PIT‑Safe)

**Purpose.** This playbook is the canonical, hand‑off guide for sourcing, validating, and storing all market‑adjacent data required by the Autonomous Trading Agent blueprint. It prioritizes **point‑in‑time (PIT) accuracy first**, then **free/low‑cost availability**, then **generous rate limits/coverage**, and finally **upgrade paths** to prosumer/enterprise vendors. Nothing here assumes paid feeds to begin with; every category includes a viable free path and a realistic upgrade.

> **Prime directives**
> 1) Never leak the future: every dataset must be **PIT‑true** and timestamped to **public availability** (not just file write time).  
> 2) Raw first: keep **raw** vendor payloads immutable; derive adjusted views downstream with full lineage.  
> 3) Universes must include **delisted** names and **corporate actions** applied by *our* logic (not vendor black boxes).  
> 4) Calendars/halfdays/DST handled with real exchange calendars.  
> 5) Every file has checksums + `ingested_at`, `source_uri`, `transform_id`.

---

## 0) Coverage Map (what we need to power stocks **and** options)

**Equities**  
- TAQ (trades/quotes), minute/hourly bars, EOD OHLCV  
- Corporate actions (splits, cash dividends, special distributions)  
- Delistings & symbol changes  
- Index membership history (e.g., S&P 500 adds/deletes)  
- Costs/constraints: commissions, tick size regime, spreads, **borrow fees/locates**  
- Risk‑free curve & macro time series; FX reference rates

**Options (OPRA universe)**  
- NBBO quotes/trades, chain snapshots  
- Implied volatility & Greeks (we can compute from NBBO mids)  
- IV surface parameters (SVI/SSVI) + no‑arbitrage flags  
- Options OI/volume (if available), expiries, strikes, contract specs

**Reference**  
- Exchange calendars, halts, circuit breakers  
- Corporate event dates (earnings, guidance, dividends)  
- Vendor rate‑limit and outage metadata (for resiliency tests)

---

## 1) Storage & Schema Contracts

**Layout (Parquet + S3/MinIO):**
```
/data_layer
  raw/
    equities_ticks/    date=YYYY-MM-DD/symbol=XYZ/part-*.parquet
    equities_bars/     date=YYYY-MM-DD/symbol=XYZ/part-*.parquet
    options_nbbo/      date=YYYY-MM-DD/underlier=XYZ/part-*.parquet
    filings_sec/       date=YYYY-MM-DD/cik=.../part-*.jsonl
    macros_fred/       series=.../date=YYYY-MM-DD/part-*.parquet
  reference/
    corp_actions/      event_type in {split, div}
    delistings/
    calendars/
    index_membership/
    tick_size_regime/
  curated/
    equities_ohlcv_adj/
    options_iv/
    options_svi_surface/
```

**Equities (ticks)**: `ts_ns`, `symbol`, `price`, `size`, `side`, `venue`, `seq`

**Equities (bars)**: `date`, `symbol`, `open`, `high`, `low`, `close`, `vwap`, `volume`, `nbbo_spread`, `trades`, `session_id`

**Corporate actions**: `symbol`, `event_type`, `ex_date`, `record_date`, `pay_date`, `ratio`

**Delistings**: `symbol`, `delist_date`, `reason`

**Options NBBO**: `ts_ns`, `underlier`, `expiry`, `strike`, `cp_flag`, `bid`, `ask`, `bid_size`, `ask_size`, `nbbo_mid`, `exchs_bitmap`

**Options IV & surface**: `date`, `underlier`, `expiry`, `k_logmoneyness`, `iv`, `total_var`, `svi_a`, `svi_b`, `svi_rho`, `svi_m`, `svi_sigma`, `no_arb_flags`, `fit_rmse`

All raw files carry metadata: `ingested_at`, `source_uri`, `vendor_ts`, `api_version`.

---

## 2) Providers by Category (Free → Upgrade)

### 2.1 Equities – TAQ / Minute / EOD
**Free (single venue, T+1):**
- **~~IEX Exchange — HIST/DEEP/TOPS~~**: ❌ **SHUT DOWN** as of late 2024. No longer available.

**Free (minute/EOD dev):**
- **Alpaca** (via `/v2` endpoint): free minute bars and live stream with generous quotas; fine for development and backtests at minute/EOD horizons.
- **`yfinance`**: exploration only (adjusted series are not PIT‑safe). Never for final research.

**Prosumer/Enterprise upgrades:**
- **Databento**: historical SIP TAQ and **OPRA** options NBBO as flat binary/Parquet; pay‑as‑you‑go; excellent for research reproducibility.
- **Polygon**: consolidated equities minute/EOD, options chains, some tick coverage; friendly APIs.
- **Tiingo / Intrinio / Quandl (Nasdaq Data Link)**: alternative EOD and fundamentals.

**Notes:** always record exchange timestamps; keep splits/divs separate; never rely on vendor pre‑adjusted OHLCV without logging the algorithm.

### 2.2 Corporate Actions & Delistings
**Free:**
- **Alpha Vantage — `LISTING_STATUS`** for active/delisted symbols; split/dividend endpoints for proportional adjustments.
- **Financial Modeling Prep (FMP) — Delisted Companies** (cross‑check against AV and SEC).
- **SEC EDGAR**: authoritative filings; use for event dates and symbol changes.

**Upgrade:** paid corporate action histories from primary vendors (e.g., CRSP/Refinitiv) when budget allows and precision is paramount.

### 2.3 Costs/Constraints & Shorting Realism
**Free:**
- Broker public **commission schedules** (U.S. equities often $0 commission; options per‑contract fees vary). Persist the schedule as a versioned YAML.
- **Tick size regime** (post‑2024 sub‑penny/half‑penny quoting for certain names). Maintain a date×symbol mapping to drive spread/impact models.
- **Interactive Brokers (IBKR) Stock Loan & Borrow Dashboard**: indicative borrow fees/availability; use for HTB realism and sanity checks.

**Upgrade:** desk‑quality securities lending datasets if short alpha is core.

### 2.4 Risk‑Free, Macro, FX
**Free:**
- **FRED API** for Treasuries & macro; prefer **ALFRED vintages** for revision‑aware backtests.
- **U.S. Treasury Fiscal Data** for official rates.
- **ECB reference FX** (daily EUR‑centric rates) or your equities vendor’s FX for consistency.

### 2.5 Index Membership History
**Free (manual/lightweight):**
- Maintain a canonical table from public S&P press releases and reputable mirrors for S&P 500 adds/deletes.
- For Russell, use FTSE Russell’s public notices and rebalance calendars.

**Upgrade:** licensed index history from S&P/FTSE when budget allows.

### 2.6 Options – Chains, NBBO, Greeks
**Free (exploratory):**
- **Finnhub** options chains, IV, and selected Greeks (rate‑limited; suitable for v0 experiments).

**Prosumer/Enterprise:**
- **Databento — OPRA** consolidated NBBO trades/quotes (research‑grade).  
- **Polygon Options** NBBO snapshots, OI/volume; good API ergonomics.  
- **Cboe DataShop** official EOD files and Greeks snapshots for benchmarking.

**Policy:** compute **IV/Greeks yourselves** from NBBO mids; fit daily **SVI/SSVI** surfaces with no‑arbitrage constraints and persist QC flags.

### 2.7 Earnings/Corporate Events
**Free:**
- Company press releases, EDGAR 8‑K, and exchange notices (halt, ex‑div calendars).

**Upgrade:** structured earnings calendars/Surprise datasets (e.g., Nasdaq, Refinitiv) when you need systematic coverage.

---

## 3) PIT Alignment & Leakage Controls

- **Release timestamps:** every macro/fundamental/news feature must be lagged to its *public release* time; do not assume midnight availability. Persist both `event_time` and `ingested_at`.
- **Restatements:** for macro series (GDP/CPI/etc.), store both **current** and **vintaged** values; train on vintages available at prediction time.
- **Index reconstitutions:** rebuild constituent lists by date; never use today’s index members for past backtests.
- **Corporate actions:** apply proportional adjustments downstream with explicit logs; verify prices around ex‑dates.
- **DST & half‑days:** generate sessions from exchange calendars; unit‑test random dates.

---

## 4) QC & Sanity Checks (automated)

1. **Schema & monotonicity**: timestamps non‑decreasing, sizes positive, venues in known set.  
2. **Outlier prints**: median filters/quantile clamps; flag >5σ moves; review especially near corporate actions.  
3. **Coverage diffs**: compare free minute bars vs a second source on a sample week; alert if >X% diff.  
4. **PIT audits**: periodic back‑resolves of historical payloads to ensure no silent vendor retro‑edits creep into raw.  
5. **Calendar conformance**: bars align to session schedule; half‑days correct.  
6. **Borrow realism**: if no borrow info, haircut short alpha or cap gross short exposure.  
7. **Tick size regime**: confirm symbol is in correct regime by date; feed this into spread/impact modeling.

---

## 5) Connector Specifications (implement exactly)

**Common client behavior**  
- Exponential backoff with jitter; 429/5xx retries; circuit‑breaker on persistent failures.  
- Pagination, resume tokens, and partial‑day replays.  
- Raw payload write → checksum → manifest entry; only then derive curated tables.  
- Clock skew guard: log vendor `server_time` when available.

**~~IEX (HIST/DEEP/TOPS)~~** ❌ **REMOVED** - Service shut down late 2024

**Alpaca (minute/EOD via `/v2` API)**
- Paths: `/v2/stocks/bars` for historical, `/v2/stocks/quotes/latest` for real-time.
- Output: OHLCV with exchange timestamp; ensure symbol normalization (corporate actions applied downstream).

**Alpha Vantage (corp actions, listings)**  
- Paths: `LISTING_STATUS`, splits/dividends.  
- Output: active/delisted flags; event tables.

**SEC EDGAR**  
- Paths: company facts/filings index; press releases via 8‑K.  
- Output: filing dates/times, links; use for authoritative event timing.

**FRED/ALFRED (macro)**  
- Paths: series by ID; ALFRED for vintages.  
- Output: time series with `observation_date`, `value`, `vintage_date` (for ALFRED).

**Finnhub (options)**  
- Paths: chain by underlier/date; IV/Greeks; subject to rate limits.  
- Output: per‑contract fields; store raw and compute IV ourselves for consistency.

**Databento/Polygon/Cboe (upgrade)**  
- Ensure ability to drop‑in replace the free sources with identical curated schemas; keep vendor‑specific quirks confined to `connectors/`.

---

## 6) Universe Construction (bias‑proof)

- Start with **all U.S. common stocks** above a liquidity floor (e.g., median 60‑day ADV threshold), **including delisted**.  
- Exclude ETFs, preferreds, rights, units unless explicitly needed.  
- Maintain per‑date **eligibility flags** (PIT): listing exchange, shortable, borrow availability tier.

---

## 7) Options IV Surface (our policy)

- Compute **Black‑Scholes IV** from NBBO mids (guard against crossed/locked markets with sensible filters).  
- Fit daily **SVI/SSVI** per expiry; enforce no‑arbitrage (butterfly/vertical/calendar).  
- Persist: fit parameters, QC metrics, arbitrage flags; discard surfaces failing QC.

---

## 8) Cost & Impact Inputs (data you must persist)

- Commission schedule (by venue/instrument) as versioned YAML.  
- Effective spread estimates by symbol×date×time‑of‑day.  
- Borrow fee snapshots & availability tiers.  
- Impact coefficients for square‑root law (estimated from our own fills once live; seed with literature defaults).  
- Options contract multipliers, exercise style (A/E), and assignment calendars.

---

## 9) Migration Triggers (when to upgrade data)

- **Microstructure research** needs consolidated TAQ → move to Databento SIP.  
- **Options trading beyond toy scale** → move to OPRA NBBO (Databento or Cboe DataShop).  
- **Frequent corporate‑action edge cases** → license higher‑fidelity corp‑action history.  
- **Turnover/capacity scaling** → require better impact calibration → ingest broker slippage logs and richer depth data.

---

## 10) Acceptance Tests (per source)

- **Round‑trip determinism:** re‑ingesting a historical day yields identical Parquet hashes.  
- **Session alignment:** bars begin/end exactly at calendar boundaries; half‑days handled.  
- **Event correctness:** split/div days reproduce expected proportional adjustments in curated OHLCV.  
- **Delisting realism:** backtests include names that later delisted; no survivorship bias.  
- **IV sanity:** IV is monotone with price/strike; surfaces pass no‑arb checks within thresholds.  
- **Rate‑limit robustness:** simulated 429 storms do not corrupt raw stores; jobs resume cleanly.

---

## 11) Quick Start (today)

1) Stand up MinIO + Postgres with docker‑compose; create `/raw`, `/curated`, `/reference` buckets.
2) Implement connectors for Alpaca (v2), Alpha Vantage, FMP, FRED, Finnhub with backoff & checksums.
3) Ingest last 2 years of minute/EOD from Alpaca.
4) Build corporate‑action + delisting reference tables; apply to curated OHLCV.
5) Pull a month of options chains for 50 large caps from Finnhub; compute IV from mids; test SVI fit & QC.
6) Run QC suite; fix any schema/lifecycle issues before model work starts.

---

### Final notes
- Keep the **free→prosumer** substitution boundary at the curated layer so upgrading a vendor does **not** ripple through research code.  
- Always log vendor outages and response latencies; these power resilience tests and can explain weird backtest/staging gaps later.  
- Treat this playbook as **source‑of‑truth** for connectors and PIT practices; changes here require a pull request and sign‑off from Data Eng + Quant.

