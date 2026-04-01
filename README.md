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

## Dashboard

Install the operator dashboard with one script:

```bash
./install_dashboard.sh
```

That installs the package in a local virtualenv, drops a `trademl` wrapper into `~/.local/bin`, and lets you launch the node UI from anywhere:

```bash
trademl dashboard
```

The dashboard provides:

- Node start / stop / restart controls
- Live queue and partition-status metrics
- NAS-backed fleet view for workers, shard leases, and recent events
- NAS reachability and mount write checks
- Editable NAS share / mount and schedule settings
- Log tailing and restart-safe progress inspection from SQLite + parquet state
- Cluster join / rebuild / leave controls and systemd install integration
- Local worker lifecycle controls for update / reset / uninstall

The intended operator flow is:

- Use `trademl dashboard` to configure the worker, join the cluster, monitor progress, and run the node day to day.
- Use the CLI only for install/update/reset/uninstall and launching the dashboard.

Minimal lifecycle commands:

```bash
trademl dashboard
trademl node update
trademl node reset --passphrase 'your-passphrase'
trademl node uninstall
trademl node install-service
```

The dashboard now exposes the rest of the worker lifecycle:

- Start / stop / restart
- Join cluster
- Rebuild local state from NAS
- Leave cluster
- Edit NAS and schedule settings
- Inspect fleet leases, workers, and recent events
- Rotate cluster passphrase and update encrypted shared secrets

For Raspberry Pi / Linux workers, you can also install the systemd unit:

```bash
./install_systemd.sh
```

## Status

This repository follows `DEV_GUIDE.md` phase-by-phase. The canonical design contract lives in `SSOT.md`.
