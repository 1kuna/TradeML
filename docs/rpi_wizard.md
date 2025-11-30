# Raspberry Pi Data Collection Wizard

Single-command, interactive setup for edge ingestion that matches SSOT v2. Run from repo root:

```bash
python rpi_wizard.py              # guided setup
python rpi_wizard.py --resume     # reuse saved state if present
python rpi_wizard.py --dry-run    # show actions without executing
```

What it does
- Picks a storage path (external SSD recommended), symlinks data/log paths to it, and writes Parquet + zstd for maximum compression.
- Creates/activates a Python venv, installs ingest dependencies, and patches `.env` (edge role, local storage, node id, pacing and budgets).
- Runs node self-checks, launches the node loop, and writes logs from process start.
- Persists wizard state under `logs/rpi_wizard_state.json` and `<data_root>/trademl_state/` so restarts months later resume from the same storage/bookmarks.

Outputs & logs
- Wizard log: `logs/rpi_wizard_<timestamp>.log`
- Node log: `<data_root>/logs/node.log` (tail with `tail -f <data_root>/logs/node.log`)

Behavior notes
- Storage backend is local-first on the SSD; remote replication is a future addition.
- Compression codec is controlled by `PARQUET_COMPRESSION` (default `zstd`, falls back automatically if unavailable).
- Failures that are not known transient issues (e.g., vendor rate limits) are loud and blocking; check the logs above for remediation hints.
