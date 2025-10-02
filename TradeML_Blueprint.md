# Autonomous Equities + Options Trading Agent — Full Blueprint (Free‑Data First)

**Mission:** Build an autonomous research & trading system that (a) learns cross‑sectional stock edges and options volatility/term‑structure edges from **free‑tier data first**, (b) simulates and then trades with realistic costs/impact, (c) self‑evaluates and self‑improves within strict anti‑overfitting governance, and (d) produces **daily ranked stock picks and options strategies** with explicit risk/capacity.

**Non‑goals (v1):** broker integration, taxable lots accounting, portfolio margin optimization, and HFT/latency arbitrage.

**Go/No‑Go Bars (all OOS, net of costs/impact):**
- **Equities sleeve:** Sharpe ≥ 1.0; Max Drawdown ≤ 20%; Deflated Sharpe Ratio (DSR) > 0; Probability of Backtest Overfitting (PBO) ≤ 5%.
- **Options sleeve:** Same bars on **delta‑hedged PnL** (vol sleeve) and on **total PnL** (spread sleeve); **net Vega exposure bounded** by policy.

---

## 1) System Architecture (single workstation → scalable)

```
/repo
  /infra
    docker-compose.yml           # Postgres, MinIO (S3‑compatible), message queue
    terraform/                   # optional cloud IaC (S3/EC2/Batch/Athena)
  /data_layer
    connectors/                  # IEX, Alpaca, FRED, SEC, AV, Finnhub...
    raw/                         # immutable Parquet, partitioned by date/symbol/source
    curated/                     # cleaned, corporate‑action adjusted, no look‑ahead
    reference/                   # calendars, tick sizes, delistings, rate curves
  /feature_store
    equities/                    # x‑section & time‑series features
    options/                     # NBBO → IV → SVI/SSVI params + QC flags
  /labeling
    horizon/                     # k‑step returns
    triple_barrier/              # TP/SL/T labels (configurable)
  /validation
    cpcv/                        # purged, embargoed CV + PBO, DSR
  /models
    equities_xs/                 # ridge/logit, LightGBM baselines → challengers
    equities_ts/                 # optional TCN/LSTM/Transformer if justified
    options_vol/                 # IV‑surface dynamics models
  /portfolio
    sizing/                      # vol targeting, fractional‑Kelly
    constraints/                 # turnover, sector, gross/net caps
  /execution
    cost_models/                 # fees, spread, square‑root impact, borrow fees
    simulators/                  # Almgren–Chriss scheduler; Avellaneda–Stoikov MM
  /backtest
    engine/                      # event‑driven, position‑aware, deterministic
  /ops
    monitoring/                  # drift, slippage vs sim, factor exposures, tripwires
    reports/                     # daily/weekly scorecards
```

**Storage:** Parquet (columnar, immutable) + MinIO/S3; Postgres for metadata/run logs.  
**Orchestration:** Prefect/Airflow.  
**Versioning:** git + DVC for data artifacts; MLflow for model registry + lineage.  
**Calendars:** `exchange_calendars` or `pandas_market_calendars` for sessions (incl. early closes/DST).  
**Security:** read‑only keys for data APIs; secrets via `.env` / Vault; checksum every flat file.

---

## 2) Data Sources & Contracts (free‑first)

> Use the dedicated **Data Sourcing Playbook** as the canonical map. Implement connectors with retry/backoff, paging, and integrity checks.

### 2.1 Equities (free tier now, upgrade later)
- **Prices/volume:** IEX HIST/DEEP (T+1 ticks, single venue) and Alpaca (minute/EOD). `yfinance` only for exploration (not point‑in‑time safe).
- **Corporate actions & delistings:** Alpha Vantage `LISTING_STATUS` + splits/divs; Financial Modeling Prep Delisted Companies; SEC/EDGAR filings for authoritative dates.
- **Risk‑free & macro:** FRED API + U.S. Treasury Fiscal Data; ECB FX ref rates (optional).
- **Tick size regime:** include post‑2024 half‑penny quoting for eligible names.

### 2.2 Options (free‑first; accurate path later)
- **Free/exploratory:** Finnhub options chains/IV/greeks (rate‑limited). Option Strategist weekly IV/HV snapshots for reality checks.
- **Upgrade path:** Polygon Options or Databento OPRA (consolidated NBBO trades/quotes). Cboe DataShop EOD/Greeks snapshots for benchmarks.

### 2.3 Point‑in‑Time Discipline
- Store **raw OHLCV** and **event feeds** separately; apply proportional adjustments yourselves; every artifact carries `ingested_at` and `source_uri`.
- Lag fundamentals/news features to **arrival timestamps**; never use adjusted close without documented method.

---

## 3) Data Schemas (Parquet; partitioned by `date`/`symbol`)

### 3.1 Equities — ticks and bars
**Ticks**: `ts_ns`, `symbol`, `price`, `size`, `side`, `venue`, `seq`  
**Bars**: `date`, `symbol`, `open`, `high`, `low`, `close`, `vwap`, `volume`, `nbbo_spread`, `trades`, `session_id`

### 3.2 Corporate actions & delistings
- `symbol`, `event_type` (split/div), `ex_date`, `record_date`, `pay_date`, `ratio`  
- `delist_date`, `reason`

### 3.3 Options NBBO (from OPRA or vendor equivalent)
- `ts_ns`, `underlier`, `expiry`, `strike`, `cp_flag`, `bid`, `ask`, `bid_size`, `ask_size`, `nbbo_mid`, `exchs_bitmap`

### 3.4 Implied Volatility & Surface (SVI/SSVI)
- By `date` & `underlier`: per‑expiry slice params `{a,b,rho,m,sigma}`; derived fields: `iv`, `total_var`, `k_logmoneyness`, `no_arb_flags`, QC metrics (fit RMSE, butterfly/vertical/Calendar arbitrage checks).

---

## 4) Labeling (execution‑aligned targets)

### 4.1 Equities (daily)
- **Regression target:** excess return \( r_{t→t+5} \), \( r_{t→t+20} \).
- **Classification target:** **triple‑barrier** labels with profit‑take/stop scaled by rolling volatility and max‑holding horizon (e.g., 10 sessions).

### 4.2 Options (two sleeves)
- **Volatility sleeve (delta‑hedged):** label future IV change \(Δσ\) or **variance risk premium** (VRP = IV² – RV²) over fixed horizons; evaluate on **delta‑hedged PnL** using fitted SVI surfaces.
- **Directional/spread sleeve:** combine underlier move labels + surface shape features; map to **verticals/diagonals/calendars** with max expected edge and bounded Greeks.

**General rules:** All labels are built on **PIT‑safe** data; horizons/thresholds are YAML‑configurable; ensure labels do not overlap training folds (see CPCV below).

---

## 5) Features (robust first, fancy later)

### 5.1 Equities (cross‑section + time‑series)
- **Baselines (must‑beat & include as features):** multi‑scale momentum (TS/XS) and value proxies.
- Volatility metrics (realized, downside, quarticity), liquidity (turnover, Amihud), dispersion, rolling skew/kurtosis, seasonality/time‑of‑day, earnings/event flags, simple cross‑asset spillovers.
- **If intraday:** microprice, order‑book imbalance, signed‑trade imbalance, short‑horizon RV from IEX DEEP (single venue) to prototype microstructure features.

### 5.2 Options
- IV **level/slope/curvature** from SVI; **term structure** slopes; **skew** (e.g., 25Δ RR); **IV rank/percentile**; **distance‑to‑earnings**; OI/volume changes; VRP.

**Feature engineering hygiene:** Standardize by rolling stats; winsorize/clamp outliers; lag **all** price‑derived features at least one bar; ensure no future leakage via overlapping labels.

---

## 6) Models (theory vs practice)

### 6.1 Cross‑sectional (many assets per time)
- **Why here:** higher effective breadth; interactions matter.
- **Start with:** regularized linear/logistic and **LightGBM/XGBoost** (monotonicity constraints where logical). These typically beat linear factors while staying data‑efficient.
- **Only after beating baselines net:** try shallow NNs/autoencoders for interaction capture; document gains.

### 6.2 Time‑series (single asset)
- Low SNR. Begin with regularized linear/logit + GBDT using strong features.  
- Sequence models (**TCN/LSTM/Transformer**) only if a sequential edge is demonstrated; use short contexts, dropout, weight decay; measure stability.

### 6.3 Options (IV surface dynamics)
- Tabular features → **GBDT** for IV level/slope/skew forecasts.  
- Optional shallow nets on SVI parameters if clear nonlinearity emerges.  
- Trading maps forecasts to delta‑hedged structures or spreads with explicit Greek caps.

---

## 7) Validation & Anti‑Overfitting (non‑negotiable)

- **Combinatorially Purged Cross‑Validation (CPCV)** with **purging** and **embargo** to block leakage from overlapping labels.  
- Report **PBO** (Probability of Backtest Overfitting) and **DSR** (Deflated Sharpe Ratio) for the final selection.  
- **Walk‑forward** (rolling) as a realism check after CPCV.  
- Maintain a **dark hold‑out period** untouched until final sign‑off.  
- **Multiplicity accounting:** every model/feature/hyper‑param attempt is logged; PBO/DSR incorporate the experiment count.

**Default CPCV (daily):** 8 folds, 10‑day embargo, OOS windows ≥ 6 months each (slide by 1 month).

---

## 8) Portfolio Construction & Execution (where edges live or die)

### 8.1 Position sizing
- Convert scores to expected returns / z‑scores; scale by inverse forecast uncertainty.  
- **Volatility targeting** at sleeve and book levels to stabilize Sharpe.  
- **Fractional Kelly (10–25%)** on conservative edge estimates; hard caps for turnover, sector, and single‑name limits; **borrow/HTB** constraints on shorts.

### 8.2 Costs & Impact
- Fees + **effective spread** + slippage modeled per symbol/maturity.
- **Impact:** concave **square‑root law** for meta‑orders; cross‑check with **Almgren–Chriss** schedules; **Obizhaeva–Wang** for LOB resilience dynamics (optional).  
- **Options:** wider effective spreads; apply impact scaled by option ADV; model assignment risk for American options around ex‑div.

### 8.3 Execution simulators
- **EOD scheduler:** Almgren–Chriss with temporary/permanent impact; participation caps (e.g., ≤ 10% ADV).  
- **Intraday queue sim (optional):** LOB‑aware partial fills and delay.  
- **Market‑making experiments:** Avellaneda–Stoikov (inventory control + arrival rates) on microstructure datasets.

---

## 9) Backtest Engine (deterministic, position‑aware)

**Requirements:**
- Event‑driven with **latency**, **partial fills**, **fees/borrow**, **corporate actions**, **short‑sale constraints**, and **order types** (MKT/LMT).  
- **Options PnL** with daily delta‑re‑hedging; Greeks attribution (Δ/Γ/Θ/ν).  
- **Position accounting:** FIFO; proportional adjustments on splits/divs; dividends credited on ex‑date if short/long as appropriate.  
- Deterministic: run ID = (data snapshot hash + config YAML).

**Acceptance tests:**
- Reproduce AC optimal schedule cost frontier within tolerance.  
- Reproduce expected inventory behavior under Avellaneda–Stoikov in symmetric arrival scenarios.

---

## 10) Trading Sleeves (build both; deploy one at a time)

### 10.1 Equities: Daily Cross‑Sectional Stock Selection
- **Objective:** rank top/bottom deciles by expected 5–20 day **excess return**; long/short with vol targeting and turnover caps.  
- **Baselines:** cross‑sectional momentum & value (must beat net).  
- **Models:** Ridge/Logit → LightGBM (monotone constraints where logical).  
- **Features:** multi‑scale momentum, vol, liquidity, seasonality, earnings flags, cross‑asset.  
- **Validation:** CPCV (8 folds, 10‑day embargo), PBO, DSR.  
- **Costs:** fee + half‑spread + square‑root impact (participation ≤ 10% ADV).

**Daily JSON output**
```json
{
  "asof": "YYYY-MM-DD",
  "universe": ["AAPL", "MSFT", "..."],
  "positions": [
    {"symbol":"AAPL","target_w":0.015,"horizon_days":10,
     "expected_alpha_bps":35,"conf":0.62,
     "tp_bps":120,"sl_bps":-60}
  ],
  "risk": {"target_vol_ann":0.12, "gross_cap":1.5, "sector_caps": {"TECH":0.25}}
}
```

### 10.2 Options: Volatility & Spread Strategies
- **Vol track (delta‑hedged):** forecast IV surface shifts; trade delta‑hedged straddles/strangles; cap net Vega/Gamma; re‑hedge daily or when |Δ| > threshold.  
- **Directional/spreads track:** map underlier signal + surface shape to verticals/diagonals/calendars maximizing expected edge under cost/impact.

**Daily JSON output**
```json
{
  "asof": "YYYY-MM-DD",
  "strategies": [
    {"underlier":"AAPL","type":"delta_hedged_straddle",
     "expiry":"YYYY-MM-DD","qty":10,"target_vega":5000,
     "expected_pnl_bps":40,"conf":0.58,
     "hedge_rules":{"rebalance":"daily","delta_threshold":0.2}}
  ],
  "risk": {"net_vega":0, "vega_cap": 0.2, "gamma_cap": 0.1}
}
```

---

## 11) Continuous Improvement (autonomous, but governed)

**Champion–Challenger:** nightly job generates next‑day signals from the **champion**. Challengers (new features/models) train under CPCV + PBO/DSR and **shadow‑trade** for ≥ N weeks. Promotion only if shadow **beats champion net** and meets DSR/PBO bars.

**Drift/Regime Monitoring:**
- **Data drift:** PSI/KL on key features;  
- **Forecast calibration:** Brier/log‑loss;  
- **Performance stability:** rolling IR by month; time‑under‑water;  
- **Risk:** factor betas; net Greeks;  
- **Execution:** realized vs modeled impact; slippage distribution;  
- **Change‑point** tests trigger re‑estimation/down‑weighting.

**Tripwires:** soft de‑risk bands on drawdown; hard kill if DD > cap or slippage > model by X% for Y days.

**Hyper‑param governance:** Bayesian optimization with a fixed experiment budget per quarter; selection by **net** OOS + DSR; auto‑archive failed lines.

---

## 12) Live Runbook (hands‑off, safe‑by‑default)

1) **07:00** Build sessions from exchange calendars; verify data freshness.  
2) **07:05** Construct labels/features with correct lags; re‑fit intraday models if any.  
3) **07:15** Score universe; build portfolios & option strategies; enforce risk caps and costs.  
4) **07:20** Emit **Trade Blotter** (stocks & options) with rationales/expected edge; human‑readable markdown + JSON.  
5) **RTH** Execute via simulator (paper/live) using AC schedule or quoting engine; log fills and slippage.  
6) **After close** Compute realized PnL; update drift/health dashboards; log to MLflow; evaluate challengers; schedule promotions if eligible.

---

## 13) Metrics & Reporting (every run)

- **Performance:** net Sharpe/Sortino, Max DD, time‑under‑water, turnover, capacity, calibration plots.  
- **Stat‑safety:** DSR (with skew/kurt/sample length), PBO (with split diagram).  
- **Execution:** fees, spread cost, estimated vs realized **impact** (square‑root residuals).  
- **Risk:** sector/country exposures; **options net Greeks**; factor betas.  
- **Ops:** data freshness, outages, failed checks.

---

## 14) Engineer Checklists & Acceptance Criteria

### A) Data & Reference
- [ ] Implement IEX/Alpaca/FRED/SEC/AlphaVantage connectors with retry/backoff; write PIT metadata.  
  **Done when:** replaying a day reconstructs identical files (hash‑match).
- [ ] Corporate actions pipeline: proportional adjustments + logged transforms; delisting applied to historical universes.  
  **Done when:** backtests include delisted names and pass split/div days.

### B) Calendars & Sessions
- [ ] Session generator with early closes/DST; unit tests for random dates.  
  **Done when:** any date maps to correct open/close & half‑day rules.

### C) Feature Store
- [ ] Equities library (momentum/vol/liquidity/seasonality).  
  **Done when:** features are deterministic under re‑ingestion and lag‑safe.  
- [ ] Options: BS IV from NBBO mids; fit **SVI** slices with no‑arb checks; QC metrics persisted.  
  **Done when:** arbitrage flags ≤ threshold; fit residuals within tolerance.

### D) Labeling
- [ ] Triple‑barrier + horizon returns; thresholds in YAML; alignment tests.  
  **Done when:** CPCV leakage tests pass on synthetic overlapping windows.

### E) Validation Suite
- [ ] **CPCV** with purge & embargo; **PBO** exporter; **DSR** calculator.  
  **Done when:** unit tests replicate textbook examples; fold diagrams logged.

### F) Models
- [ ] Baselines: ridge/logit and LightGBM for equities; LightGBM for options.  
  **Done when:** baselines beat naive momentum/value **net** or are rejected with reasons; feature ablation logged.

### G) Portfolio & Risk
- [ ] Vol targeting with rolling vol; **fractional Kelly** sizing; turnover & exposure caps; borrow fees/HTB rules.  
  **Done when:** synthetic tests show target vol achieved and Kelly fraction respected under shock scenarios.

### H) Execution Simulator
- [ ] Almgren–Chriss scheduler + square‑root impact; queue‑aware fills; Avellaneda–Stoikov MM (for experiments).  
  **Done when:** reproduces known cost frontiers & inventory dynamics.

### I) Backtester
- [ ] Deterministic engine; corporate actions; borrow fees; options Greeks PnL; delta‑hedging logic.  
  **Done when:** identical configs → identical reports; edge‑case corporate‑action days pass.

### J) Ops & Governance
- [ ] MLflow registry; champion‑challenger; shadow trading; drift dashboards; promotion rules codified.  
  **Done when:** nightly cycle runs end‑to‑end without human input and blocks unsafe promotions.

---

## 15) Minimal APIs (implement exactly)

### `features.compute_equity_features(date, universe) -> pd.DataFrame`
Inputs: PIT OHLCV + events.  
Output columns: `symbol`, `asof`, `feature_*` (standardized, lag‑safe).

### `labels.triple_barrier(date, universe, tp_sigma, sl_sigma, max_h) -> pd.DataFrame`
Output columns: `symbol`, `entry_date`, `exit_date`, `label`, `outcome`, `meta`.

### `validation.run_cpcv(X, y, groups, embargo_days) -> dict`
Returns: per‑fold metrics, trained models, **PBO**, **DSR**, fold diagrams.

### `portfolio.build(scores_df, risk_cfg) -> dict`
Returns: `target_weights`, achieved target vol, Kelly fraction used, constraints applied.

### `execution.simulate(orders, cost_model, market_data) -> (fills, slippage_report)`
Supports: AC schedule, queue sim (optional), square‑root impact, partial fills.

### `options.fit_svi_surface(date, underlier) -> dict`
Returns: slice params `{a,b,rho,m,sigma}` per expiry, no‑arb flags, QC metrics.

### `reporting.emit_daily(asof, positions, strategies, metrics) -> md, json`
Human‑readable blotter + machine‑readable JSON in `/reports/`.

---

## 16) Phase Plan (enforceable timeline)

**Phase 1 (Weeks 1–3): Data & Skeleton**  
• Connectors; calendars; corp actions; delistings; PIT storage; unit tests.  
• Minimal backtester (daily), fee+spread costs; basic report.

**Phase 2 (Weeks 3–6): Baselines & CV**  
• Build momentum/value baselines; implement CPCV + PBO + DSR; run equity sleeve; set pass bars.

**Phase 3 (Weeks 6–9): Options Foundations**  
• Chains → NBBO mids → IV → SVI fitting; vol features; first delta‑hedged straddle strategy with AC execution.

**Phase 4 (Weeks 9–12): Portfolio, Execution & Ops**  
• Vol targeting + fractional Kelly; AC simulator; drift dashboards; champion‑challenger & shadow trading.

**Phase 5 (Week 12+): Scale & Refine**  
• Upgrade data (Polygon/Databento) as needed; test shallow sequence models **only if** GBDT beats baselines net.

---

## 17) Guardrails (from day one)

- No model promotion without **CPCV + PBO + DSR** and a multi‑week **shadow** that matches live‑like slippage.  
- Data lineage on every row: `ingested_at`, `source_uri`, `transform_id`.  
- Capacity realism: square‑root impact estimates & participation caps are **hard limits**.  
- Universe realism: delisted names present; splits/dividends applied by our documented logic.  
- Explicit shorting realism: borrow fees/HTB indicators; haircut short alpha if borrow unknown.

---

## 18) What You Build First (tonight)

1) **Universe:** top‑1000 US equities by 60‑day ADV computed **including** delisted names.  
2) **Labels:** 5/20‑day horizon returns + triple‑barrier (TP/SL scaled by rolling vol).  
3) **Features:** momentum/value/vol/liquidity/seasonality.  
4) **Models:** Ridge/Logit vs LightGBM (monotone constraints when logical).  
5) **Validation:** CPCV (8 folds, 10‑day embargo) + PBO & DSR.  
6) **Portfolio:** vol targeting + 0.2× Kelly; fee+spread+sqrt‑impact costs; turnover cap.  
7) **Reports:** ranked list + blotter and rationale per trade.

---

## 19) Daily Deliverables to Stakeholder

1) **Top 20 long/short equities** with expected alpha, confidence, TP/SL, capital use, factor exposures, and attribution.  
2) **Top 10 options strategies** with structure, expected edge, net Greeks, worst‑case slippage, and hedge rules.  
3) **One‑page scorecard**: net PnL, Sharpe, DD, turnover, capacity, realized vs simulated impact, DSR/PSR trend.

---

## 20) Team Roles & Ownership

- **Data Engineering:** connectors, PIT storage, corp action/delisting handling, calendars, QC.  
- **Quant Research:** features, labeling, CPCV/PBO/DSR, ablations, model selection.  
- **Execution/Sim:** cost models, AC/queue sims, backtester integration, slippage monitoring.  
- **Risk & Portfolio:** sizing/constraints, vol targeting, Kelly policy, tripwires.  
- **MLOps/Platform:** MLflow registry, champion–challenger, shadow trading, dashboards, CI/CD.  
- **QA:** determinism, reproducibility, acceptance tests, chaos days (action/rate curve outages, corp actions).  
- **Owner (PM):** scope governance; promotion rules; sign‑off at each phase gate.

---

## 21) Testing Strategy (must pass before promotion)

- **Unit tests:** calendars, corp action adjustments, feature lagging, label generation, CPCV splits, DSR/PBO math.  
- **Property tests:** determinism of pipelines; invariants (no negative volumes, etc.).  
- **Backtest parity:** known toy strategies replicate textbook results (AC frontier; MM inventory).  
- **Resilience tests:** data‑gap handling; late arrivals; rate‑limit throttling; API failover.  
- **Security tests:** secrets not logged; PII none; S3 bucket policies; least‑privilege API keys.

---

## 22) Scalability Notes (when compute limits stop mattering)

- **Column pruning & partitioning** keep memory in check; use **DuckDB/Polars** for fast local analytics.  
- **Ray/Dask** for embarrassingly parallel CPCV and feature calc.  
- **Athena/Parquet** on S3 for serverless scale; **AWS Batch** for heavy re‑fits.  
- **Vector DB optional** for alt‑data embeddings (news), gated by leakage controls.

---

### Final Remark
Follow the blueprint **exactly**: treat CPCV/PBO/DSR, delistings, and impact as *hard* requirements. Complexity is a **budgeted** resource—only escalate model complexity after beating robust baselines **net** of all costs. This is how we get a durable, self‑improving agent instead of a backtest demo.

