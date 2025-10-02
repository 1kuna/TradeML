# TradeML Quick Reference Card

## 🚀 Quick Start

```bash
# 1. Start Docker
cd infra && docker-compose up -d

# 2. Activate Python
cd .. && ./venv/Scripts/activate

# 3. Test connection
python -m data_layer.connectors.alpaca_connector --help
```

---

## 📊 Data Connectors

### Alpaca (Equities Bars) ✅ TESTED
```bash
python -m data_layer.connectors.alpaca_connector \
  --symbols AAPL MSFT GOOGL \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --timeframe 1Day \
  --output data_layer/raw/equities_bars
```

### FRED (Treasury Rates) ✅ TESTED
```bash
# Single series
python -m data_layer.connectors.fred_connector \
  --series DGS10 \
  --start-date 2024-01-01 \
  --end-date 2024-12-31

# Full Treasury curve
python -m data_layer.connectors.fred_connector \
  --treasury \
  --start-date 2024-01-01 \
  --end-date 2024-12-31
```

### Alpha Vantage (Corporate Actions)
```bash
# All corporate actions for a symbol
python -m data_layer.connectors.alpha_vantage_connector \
  --symbol AAPL \
  --action all

# Just splits
python -m data_layer.connectors.alpha_vantage_connector \
  --symbol AAPL \
  --action splits

# Delisted symbols
python -m data_layer.connectors.alpha_vantage_connector \
  --action delisted
```

### FMP (Delistings & Prices)
```bash
# Fetch all delisted companies
python -m data_layer.connectors.fmp_connector \
  --action delisted

# Fetch available symbols
python -m data_layer.connectors.fmp_connector \
  --action symbols

# Historical prices
python -m data_layer.connectors.fmp_connector \
  --action price \
  --symbol AAPL \
  --start-date 2024-01-01 \
  --end-date 2024-12-31
```

### Finnhub (Options)
```bash
# Options chain for symbol
python -m data_layer.connectors.finnhub_connector \
  --symbol AAPL

# Specific expiry
python -m data_layer.connectors.finnhub_connector \
  --symbol AAPL \
  --expiry 2024-12-20
```

---

## 🐳 Docker Commands

```bash
# Start all services
cd infra && docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f [service_name]

# Stop services
docker-compose down

# Reset database (⚠️ destroys data)
docker-compose down -v
docker-compose up -d
```

**Services:**
- PostgreSQL: `localhost:5432` (trademl/trademl_dev_pass)
- MinIO: `http://localhost:9001` (minioadmin/minioadmin)
- MLflow: `http://localhost:5000`
- Redis: `localhost:6379`

---

## 📁 Data Paths

```
data_layer/
├── raw/                    # Immutable vendor data
│   ├── equities_bars/      # OHLCV from Alpaca
│   ├── equities_ticks/     # Tick data (future)
│   └── options_nbbo/       # Options from Finnhub
├── curated/                # Processed, adjusted data
│   ├── equities_ohlcv_adj/ # Corporate action adjusted
│   └── options_iv/         # Computed IV and Greeks
└── reference/              # Calendars, delistings, etc.
    ├── corp_actions/
    ├── delistings/
    ├── calendars/
    └── macro/              # FRED data
```

---

## 🔍 Quick Data Checks

```python
# Read Parquet
import pandas as pd
df = pd.read_parquet("data_layer/raw/equities_bars/")
print(df.head())

# Check metadata
print(df[['ingested_at', 'source_name', 'source_uri']].head())

# Summary stats
print(df.describe())
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Symbols: {df['symbol'].unique()}")
```

---

## 🧪 Testing Calendar

```python
from data_layer.reference.calendars import get_calendar
from datetime import date

cal = get_calendar("XNYS")

# Is today a trading day?
print(cal.is_trading_day(date.today()))

# Next 5 trading days
next_day = date.today()
for i in range(5):
    next_day = cal.next_trading_day(next_day)
    print(next_day)

# Early close?
print(cal.is_early_close(date(2024, 11, 29)))  # Thanksgiving Friday
```

---

## 📊 Database Queries

```bash
# Connect to PostgreSQL
docker exec -it trademl_postgres psql -U trademl -d trademl

# View tables
\dt

# Sample queries
SELECT * FROM data_ingestion_log ORDER BY ingestion_start_ts DESC LIMIT 10;
SELECT * FROM model_runs WHERE status = 'completed';
SELECT * FROM corporate_actions WHERE symbol = 'AAPL';
```

---

## 🔑 API Key Check

```bash
# View configured keys (without showing values)
cat .env | grep -E "API_KEY|SECRET" | sed 's/=.*/=***/'

# Test if key is loaded
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('Alpaca:', 'OK' if os.getenv('ALPACA_API_KEY') else 'MISSING')"
```

---

## Validation Suite

```bash
# Run anti-overfitting validation suite
python validation/test_anti_overfitting_suite.py

# Or run individual components
python validation/cpcv/cpcv.py
python validation/pbo/pbo.py
python validation/dsr/dsr.py
```

---

## 📈 Next Phase Checklist

**Phase 1 Remaining:**
- [ ] Test Alpha Vantage connector
- [ ] Test FMP connector
- [ ] Test Finnhub connector
- [ ] Build corporate actions pipeline
- [ ] Create delistings database
- [ ] Implement universe constructor
- [ ] Build QC suite
- [ ] Implement minimal backtester

**Phase 2 Ready:**
- [ ] Equity features
- [ ] Labeling module
- [ ] CPCV validation ⚠️ CRITICAL
- [ ] PBO calculator ⚠️ CRITICAL
- [ ] DSR calculator ⚠️ CRITICAL
- [ ] Baseline models
- [ ] Portfolio construction
- [ ] Execution simulation

---

## 🛠️ Common Issues

**Import errors:**
```bash
# Reinstall packages
./venv/Scripts/pip install -r requirements.txt
```

**Docker not starting:**
```bash
docker-compose down
docker system prune -a  # Clean up
docker-compose up -d
```

**Data not loading:**
```bash
# Check .env file
cat .env | head -50

# Reload environment
from dotenv import load_dotenv
load_dotenv()
```

---

## 📚 Key Files

- `README.md` - Project overview
- `PROGRESS.md` - Phase 1 completion report
- `QUICK_REFERENCE.md` - Command cheat sheet
- `Data_Sourcing_Playbook.md` - Data sourcing guide
- `TradeML_Blueprint.md` - Full specification

---

## 🎯 Remember

**Non-negotiables:**
1. CPCV + PBO + DSR on every model
2. Dark holdout untouched until final
3. All costs/impact modeled
4. Delisting survivorship eliminated
5. PIT discipline everywhere

**Go/No-Go:**
- Sharpe ≥ 1.0 (OOS, net)
- Max DD ≤ 20%
- DSR > 0
- PBO ≤ 5%
