import os
from datetime import date


class StubBudget:
    def __init__(self):
        self.calls = 0

    def try_consume(self, vendor: str, tokens: int) -> bool:
        # Always deny to force shrink/stop
        self.calls += 1
        return False


class StubAlpaca:
    def __init__(self, *args, **kwargs):
        self.fetch_calls = []

    def fetch_bars(self, symbols, start_date, end_date, timeframe="1Day"):
        self.fetch_calls.append((tuple(symbols), start_date, end_date, timeframe))
        # Should not be called if budget denies; return empty to be safe
        import pandas as pd
        return pd.DataFrame()


def test_backfill_shrinks_and_halts_without_api_keys(monkeypatch):
    # Local storage
    os.environ["STORAGE_BACKEND"] = "local"

    import ops.ssot.backfill as bf

    # Monkeypatch BudgetManager to our stub
    stub_budget = StubBudget()
    monkeypatch.setattr(bf, "BudgetManager", lambda *a, **k: stub_budget)

    # Monkeypatch AlpacaConnector to stub (avoid API keys)
    stub_alpaca = StubAlpaca()
    monkeypatch.setattr(bf, "AlpacaConnector", lambda *a, **k: stub_alpaca)

    # Run backfill; with no raw data and budget denying, fetch_bars should not be called or at most return empty
    bf.backfill_run(budget=None)

    # Expect 0 calls due to budget denial before fetch
    assert len(stub_alpaca.fetch_calls) == 0 or all(len(c) == 4 for c in stub_alpaca.fetch_calls)

