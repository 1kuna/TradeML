# TradeML — Autonomous Equities + Options Trading Agent

![Status](https://img.shields.io/badge/status-Phase%201%20Complete%20(~75%25)-green)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**Mission:** Build an autonomous research & trading system that learns cross-sectional stock edges and options volatility/term-structure edges from free-tier data, simulates and trades with realistic costs, and self-improves within strict anti-overfitting governance.

**Non-goals (v1):** broker integration, taxable lots accounting, portfolio margin optimization, HFT/latency arbitrage.

## 📌 Current Status

- Phase 1 complete (~75%). See `PROGRESS.md` for the full completion report, validation results (CPCV/PBO/DSR), and next steps for Phase 2.


## 🎯 Go/No-Go Criteria (all OOS, net of costs/impact)

- **Equities sleeve:** Sharpe ≥ 1.0; Max Drawdown ≤ 20%; Deflated Sharpe Ratio (DSR) > 0; Probability of Backtest Overfitting (PBO) ≤ 5%
- **Options sleeve:** Same bars on delta-hedged PnL (vol sleeve) and total PnL (spread sleeve); net Vega exposure bounded by policy

---

## 🏗️ Architecture

```
/repo
  /infra              # Docker-compose, Terraform (PostgreSQL, MinIO, Redis, MLflow)
  /data_layer         # Raw/curated/reference data (Parquet, partitioned by date/symbol)
  /feature_store      # Equities & options features (PIT-safe, lag-enforced)
  /labeling           # Horizon returns, triple-barrier labels
  /validation         # CPCV, PBO, DSR (anti-overfitting governance)
  /models             # Baselines (Ridge/LightGBM) → challengers
  /portfolio          # Vol targeting, fractional Kelly, constraints
  /execution          # Cost models, Almgren-Chriss, Avellaneda-Stoikov
  /backtest           # Event-driven, position-aware, deterministic engine
  /ops                # Monitoring, drift, tripwires, champion-challenger
```

---

## 🚀 Quick Start

### 1. Prerequisites

- **Python 3.11+**
- **Docker & Docker Compose**
- **Git & DVC** (optional, for data versioning)

### 2. Clone & Setup

```bash
git clone <your-repo-url> TradeML
cd TradeML

# Copy environment template and fill in API keys
cp .env.template .env
# Edit .env with your API keys (see API Keys section below)

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Start Infrastructure

```bash
cd infra
docker-compose up -d

# Verify services are running
docker-compose ps

# Services will be available at:
# - PostgreSQL: localhost:5432
# - MinIO Console: http://localhost:9001 (minioadmin/minioadmin)
# - MLflow UI: http://localhost:5000
# - Redis: localhost:6379
```

### 4. API Keys Required (Free Tier)

Edit `.env` and add your API keys:

| Service | URL | Free Tier | Purpose |
|---------|-----|-----------|---------|
| **IEX Cloud** | https://iexcloud.io/ | 50k messages/month | Tick data (single venue) |
| **Alpaca Markets** | https://alpaca.markets/ | Unlimited paper trading | Minute/EOD bars, live stream |
| **Alpha Vantage** | https://www.alphavantage.co/ | 25 requests/day | Corporate actions, delistings |
| **FRED** | https://fred.stlouisfed.org/docs/api/ | Unlimited (rate-limited) | Risk-free rates, macro data |
| **Finnhub** | https://finnhub.io/ | 60 calls/min | Options chains, IV, Greeks |
| **FMP (optional)** | https://financialmodelingprep.com/ | 250 requests/day | Alternative delisting data |

### 5. Initialize Database

Database schema is automatically initialized via `infra/init-db/01-init-schema.sql` when PostgreSQL container starts.

To manually run migrations:
```bash
docker exec -i trademl_postgres psql -U trademl -d trademl < infra/init-db/01-init-schema.sql
```

---

## 📊 Data Pipeline

### Storage Layout

All data stored in MinIO (S3-compatible) as Parquet, partitioned by `date/symbol`:

```
/raw              # Immutable vendor payloads with checksums
  /equities_bars
  /equities_ticks
  /options_nbbo
  /macros_fred
/curated          # Cleaned, corporate-action adjusted, PIT-safe
  /equities_ohlcv_adj
  /options_iv
/reference        # Calendars, delistings, corporate actions, tick sizes
```

### Running Data Ingestion

```python
# Example: Ingest Alpaca EOD bars for last 2 years
python -m data_layer.connectors.alpaca_connector \
  --start-date 2022-01-01 \
  --end-date 2024-01-01 \
  --symbols SPY,AAPL,MSFT \
  --timeframe 1Day
```

---

## 🧪 Phase 1: Data & Skeleton (Weeks 1-3)

**Status:** 🟡 In Progress

- [x] Infrastructure (Docker-compose, PostgreSQL, MinIO, MLflow)
- [x] Directory structure
- [ ] Exchange calendars with DST/half-day handling
- [ ] Data connectors (IEX, Alpaca, Alpha Vantage, FRED, Finnhub)
- [ ] Corporate actions pipeline (PIT-safe adjustments)
- [ ] Delistings database (survivorship bias elimination)
- [ ] Minimal backtester (daily, fee+spread costs)

---

## 📈 Phase 2: Baselines & Validation (Weeks 3-6)

**Status:** ⚪ Pending

- [ ] Equity features (momentum, vol, liquidity, seasonality)
- [ ] Labeling (5/20-day horizon returns, triple-barrier)
- [ ] **CPCV with purging & embargo** (8 folds, 10-day embargo)
- [ ] **PBO & DSR calculators** (anti-overfitting governance)
- [ ] Baseline models: Ridge/Logistic vs LightGBM
- [ ] Portfolio construction (vol targeting, fractional Kelly)
- [ ] Execution simulator (Almgren-Chriss, square-root impact)

**Target:** Beat naive momentum/value baselines **net of all costs** or reject with documented reasons.

---

## 🎲 Phase 3: Options Foundations (Weeks 6-9)

**Status:** ⚪ Pending

- [ ] Options NBBO → IV computation (Black-Scholes)
- [ ] SVI/SSVI surface fitting with no-arbitrage checks
- [ ] Options features (IV level/slope/skew, term structure, VRP)
- [ ] Options labeling (vol sleeve: ΔIV, VRP; directional sleeve: underlier + surface)
- [ ] Options models (LightGBM for IV forecasts)
- [ ] Delta-hedged PnL framework with daily re-hedging

---

## ⚙️ Phase 4: Operations & Governance (Weeks 9-12)

**Status:** ⚪ Pending

- [ ] MLflow model registry
- [ ] Champion-challenger framework
- [ ] Shadow trading (≥4 weeks before promotion)
- [ ] Drift monitoring (PSI/KL on features, forecast calibration)
- [ ] Tripwires (DD thresholds, slippage vs model)
- [ ] Daily production runbook automation
- [ ] Reporting APIs (trade blotter, scorecard)

---

## 📐 Non-Negotiable Guardrails

1. **No model promotion without CPCV + PBO + DSR** and multi-week shadow trading beating champion net
2. **Data lineage on every row:** `ingested_at`, `source_uri`, `transform_id`
3. **Capacity realism:** square-root impact estimates, participation caps (≤10% ADV)
4. **Universe realism:** delisted names included; splits/dividends applied by our logic
5. **Shorting realism:** borrow fees/HTB indicators; haircut short alpha if unknown

---

## 🔬 Testing Strategy

```bash
# Run unit tests
pytest tests/unit -v

# Run integration tests (requires Docker services)
pytest tests/integration -v

# Run acceptance tests (backtest parity checks)
pytest tests/acceptance -v

# Code quality
black .
ruff check .
mypy data_layer/ models/ validation/
```

---

## 📚 Documentation

- **[TradeML_Blueprint.md](TradeML_Blueprint.md):** Full system blueprint with acceptance criteria
- **[Data_Sourcing_Playbook.md](Data_Sourcing_Playbook.md):** Canonical guide for data sources, PIT discipline, QC checks
- **API Contracts:** See section 15 of blueprint for minimal API specs
- **Runbook:** `ops/runbook.md` (coming soon)

---

## 📦 Daily Deliverables (Production)

### Equities Sleeve
```json
{
  "asof": "YYYY-MM-DD",
  "positions": [
    {
      "symbol": "AAPL",
      "target_w": 0.015,
      "horizon_days": 10,
      "expected_alpha_bps": 35,
      "conf": 0.62,
      "tp_bps": 120,
      "sl_bps": -60
    }
  ],
  "risk": {
    "target_vol_ann": 0.12,
    "gross_cap": 1.5,
    "sector_caps": {"TECH": 0.25}
  }
}
```

### Options Sleeve
```json
{
  "asof": "YYYY-MM-DD",
  "strategies": [
    {
      "underlier": "AAPL",
      "type": "delta_hedged_straddle",
      "expiry": "YYYY-MM-DD",
      "qty": 10,
      "target_vega": 5000,
      "expected_pnl_bps": 40,
      "conf": 0.58,
      "hedge_rules": {
        "rebalance": "daily",
        "delta_threshold": 0.2
      }
    }
  ],
  "risk": {
    "net_vega": 0,
    "vega_cap": 0.2,
    "gamma_cap": 0.1
  }
}
```

---

## 🤝 Contributing

This is a private research project. If you're collaborating:

1. **Branching:** `feature/short-description` or `fix/issue-name`
2. **Commits:** Conventional commits (`feat:`, `fix:`, `docs:`, etc.)
3. **Testing:** All new code must include unit tests; CPCV for models
4. **Code review:** Required for champion promotion, data connector changes, validation logic

---

## 📜 License

MIT License (modify as needed)

---

## ⚠️ Disclaimer

**This software is for educational and research purposes only.** Trading securities and derivatives involves substantial risk of loss. Past performance (simulated or real) does not guarantee future results. No representation is being made that any account will or is likely to achieve profits or losses similar to those shown. You should not use this system for live trading without thorough testing, regulatory compliance, and professional advice.

---

## 🛠️ Troubleshooting

### Docker services won't start
```bash
# Check logs
docker-compose logs postgres
docker-compose logs minio

# Restart services
docker-compose down
docker-compose up -d
```

### API rate limits exceeded
- Upgrade to paid tiers (see Data_Sourcing_Playbook.md upgrade path)
- Implement caching and request batching
- Use longer data refresh intervals

### Database connection errors
```bash
# Verify PostgreSQL is ready
docker exec trademl_postgres pg_isready -U trademl

# Reset database (WARNING: destroys all data)
docker-compose down -v
docker-compose up -d
```

---

## 📞 Support & Contact

For issues, feature requests, or questions:
- Open a GitHub issue
- Email: your-email@example.com
- Documentation: See `docs/` folder (coming soon)

---

**Built with discipline. Governed by statistics. Validated out-of-sample.**
