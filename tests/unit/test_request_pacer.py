import time

import pytest

from utils.pacing import RequestPacer


def test_request_pacer_waits_between_calls(monkeypatch):
    monkeypatch.setenv("REQUEST_PACING_ENABLED", "true")
    monkeypatch.setenv("REQUEST_PACING_JITTER_MS", "0,0")
    RequestPacer._instance = None  # reset singleton

    pacer = RequestPacer.instance()
    pacer.acquire("alpaca", rps=2)  # first call should not sleep

    start = time.time()
    pacer.acquire("alpaca", rps=2)  # should pace ~0.5s later
    elapsed = time.time() - start
    RequestPacer._instance = None

    assert elapsed >= 0.45  # respects pacing interval
    assert elapsed < 2.0    # but does not hang


def test_request_pacer_can_be_disabled(monkeypatch):
    monkeypatch.setenv("REQUEST_PACING_ENABLED", "false")
    RequestPacer._instance = None

    pacer = RequestPacer.instance()
    start = time.time()
    for _ in range(3):
        pacer.acquire("alpaca", rps=0.5)
    elapsed = time.time() - start
    RequestPacer._instance = None

    assert elapsed < 0.05  # no pacing when disabled
