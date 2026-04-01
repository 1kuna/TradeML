# TradeML

TradeML is a spec-driven equities research system for collecting daily market data, curating a point-in-time-safe research dataset, training cross-sectional models, and emitting reproducible validation reports.

## Layout

- `src/trademl/`: core package
- `src/scripts/`: user-facing scripts
- `tests/`: unit and integration coverage
- `configs/`: training and node configuration
- `docs/`: canonical spec and supporting documents

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/unit/test_scaffold.py -v
```

## Status

This repository follows `DEV_GUIDE.md` phase-by-phase. The canonical design contract lives in `docs/SSOT_V3.md`.
