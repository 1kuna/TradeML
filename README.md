# TradeML

TradeML is a spec-driven equities research system for collecting daily market data, curating a point-in-time-safe research dataset, training cross-sectional models, and emitting reproducible validation reports.

The completed `DEV_GUIDE.md` implementation lives under `src/trademl/` and `src/scripts/`. Older research modules under `data_layer/`, `feature_store/`, `validation/`, `models/`, and `ops/` still exist on `main`; they remain available during the transition, but the canonical design contract for the delivered build is `docs/SSOT_V3.md`.

## Layout

- `src/trademl/`: core package
- `src/scripts/`: user-facing scripts
- `data_layer/`, `feature_store/`, `validation/`, `models/`, `ops/`: legacy modules retained on `main`
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

This repository now contains the merged phase-by-phase implementation from `DEV_GUIDE.md`. The canonical design contract lives in `docs/SSOT_V3.md`.
