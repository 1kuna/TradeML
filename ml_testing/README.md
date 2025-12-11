# ML Testing (Manual-Only)

Tiny intraday fixture + manual harness to sanity-check the real pipeline locally (never in CI). Outputs stay under `ml_testing/`.

## 1) Build the fixture subset (from archive.zip)
Source: `archive.zip` in repo root (`1_min_SPY_2008-2021.csv`, times look like 07:30–13:59 MT).
```bash
python ml_testing/build_intraday_fixture.py \
  --zip-path archive.zip \
  --start-date 2021-05-03 \
  --end-date 2021-05-06
```
Outputs:
- `ml_testing/fixtures/curated/equities_minute/date=YYYY-MM-DD/data.parquet`
- `ml_testing/fixtures/curated/equities_ohlcv_adj/SPY.parquet`

Flags: `--source-tz` (default `America/Denver`); `--limit-rows` for debugging only.

## 2) Run the manual intraday harness
Uses real `ops.pipelines.intraday_xs.run_intraday` with short CPCV/epochs.
```bash
python ml_testing/run_intraday_smoke.py \
  --start-date 2021-05-03 \
  --end-date 2021-05-05 \
  --minute-dir ml_testing/fixtures/curated/equities_minute \
  --daily-dir ml_testing/fixtures/curated/equities_ohlcv_adj
```
Notes:
- Torch + GPU if available; else sklearn fallback.
- Redirect outputs via `--artifact-dir` / `--report-dir`.
- Keep `min_train_samples` small for the tiny fixture; raise for larger local runs.

## 3) Scope
- Not wired into pytest/CI.
- Meant for quick local smoke of the intraday stack, not full training.
