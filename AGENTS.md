# CLAUDE.md — Agent Instructions

## Project

TradeML: autonomous equities research system. Pi collects daily bars → NAS → workstation trains cross-sectional models.

## Canonical Spec

`docs/SSOT_V3.md` is the single source of truth. Every design decision lives there. When in doubt, the SSOT wins over comments, old code, or your own intuition.

## Build Order

Follow `DEV_GUIDE.md` phase by phase. Don't skip ahead. Each phase has a gate — all items must pass before moving on.

## Key Rules

1. **Test first.** Write the test, then make it pass.
2. **Dependency injection.** Every external dep (API, filesystem, database) gets a mockable wrapper passed via constructor. No module-level imports of concrete implementations.
3. **No future leakage.** This is a financial system. Features and labels must never use data from after the prediction date. When in doubt, lag more.
4. **Determinism.** Same inputs → same outputs. Always. No random seeds without explicit setting. No floating-point-order-dependent operations without sorting first.
5. **Rank normalization, not z-scores.** Cross-sectional features are rank-normalized to [-1, 1]. Never z-score cross-sectionally (it kills date-level features and is sensitive to outliers).
6. **Universe-relative labels.** Target = stock return minus equal-weighted universe mean return. Not raw returns. Not SPY-relative.
7. **Walk-forward purge.** The last 5 trading days of every training window must be dropped before fitting. Labels on those days overlap the test window.
8. **Costs always on.** Every backtest includes 5 bps spread. No "zero cost" results ever.
9. **SQLite is local to the Pi.** Never put SQLite on a network filesystem. Parquet files go to the NAS. SQLite stays on the machine that owns it.
10. **Log everything.** Every API call, every training run, every design choice. This is a research system — reproducibility is non-negotiable.

## Style

- Python 3.11+
- Type hints on all public functions
- Docstrings on all public functions (one-liner minimum)
- `black` formatting, `ruff` linting
- Pandas for data manipulation, PyArrow for parquet I/O
- pytest for all testing

## File Conventions

- Raw data: `data/raw/{dataset}/date=YYYY-MM-DD/data.parquet` (one file per date, multi-symbol)
- Curated data: `data/curated/{dataset}/date=YYYY-MM-DD/data.parquet`
- Reference data: `data/reference/{name}.parquet` (single files)
- Models: `models/{model_name}/run_{timestamp}/` (model.pkl + feature_list.json + config.json + metrics.json)
- Reports: `reports/daily/{date}.json` + `{date}.md`

## What NOT to Do

- Don't add S3/MinIO/object-store code. Storage is NAS via filesystem paths.
- Don't add Docker orchestration. The Pi runs a Python service directly.
- Don't add Prefect/Airflow. Scripts are the orchestration layer.
- Don't add options/SVI/intraday models. Those are Phase 3+.
- Don't add Kelly sizing, vol targeting, or z-score weighting. Phase 1 is equal-weight top quintile.
- Don't use market cap as a feature. We don't have historical shares outstanding.
- Don't use days-to-earnings as a predictive feature. PIT safety is unproven. Use only as a risk exclusion filter.
- Don't build champion/challenger automation. Phase 1 is manual comparison.
