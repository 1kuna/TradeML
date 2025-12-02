from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass
from itertools import cycle
from typing import Dict, Iterator, List, Optional
import json
from pathlib import Path

from loguru import logger

from utils.concurrency import max_inflight_for


def _vendor_lease_name(vendor: str) -> str:
    mapping = {
        "alpaca": "edge-alpaca-collector",
        "polygon": "edge-polygon-collector",
        "finnhub": "edge-finnhub-collector",
        "fred": "edge-fred-collector",
        "av": "edge-av-collector",
        "fmp": "edge-fmp-collector",
    }
    return mapping.get(vendor, f"edge-{vendor}-collector")


def _vendor_freeze_seconds(vendor: str) -> int:
    env_key = f"NODE_VENDOR_FREEZE_SECONDS_{vendor.upper()}"
    try:
        return max(1, int(os.getenv(env_key, "60")))
    except Exception:
        return 60


def _vendor_cap(vendor: str) -> int:
    defaults = {"alpaca": 2, "polygon": 1, "finnhub": 2, "fred": 2, "av": 1, "fmp": 1}
    return max(1, max_inflight_for(vendor, default=defaults.get(vendor, 1)))


@dataclass
class RunnerStats:
    submitted: int = 0
    completed_ok: int = 0
    errors: int = 0
    ratelimited: int = 0


class VendorRunner:
    """Runs all tasks for a single vendor with its own executor and lease."""

    def __init__(self, edge, vendor: str, tasks: List[str], budget_mgr=None) -> None:
        self.edge = edge
        self.vendor = vendor
        self.tasks = list(tasks)
        self.budget = budget_mgr

        self._shutdown = False
        self._executor: Optional[ThreadPoolExecutor] = None
        self._lease_name = _vendor_lease_name(vendor)
        self._renew_thread: Optional[threading.Thread] = None
        self._runner_thread: Optional[threading.Thread] = None

        self._freeze_until: float = 0.0
        self._stats = RunnerStats()

    def start(self):
        if self.edge.lease_mgr:
            if not self.edge._acquire_lease(self._lease_name):
                logger.warning(f"{self.vendor}: lease held elsewhere; skipping")
                return
            renew_every = int(self.edge.config.get("locks", {}).get("renew_seconds", 45))
            self._renew_thread = threading.Thread(
                target=self.edge._renew_lease_loop, args=(self._lease_name, renew_every), daemon=True
            )
            self._renew_thread.start()

        self._runner_thread = threading.Thread(target=self._run_loop, name=f"runner-{self.vendor}", daemon=True)
        self._runner_thread.start()

    def stop(self, timeout: float = 5.0):
        self._shutdown = True
        try:
            if self._executor:
                self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        try:
            if self._runner_thread:
                self._runner_thread.join(timeout=timeout)
        except Exception:
            pass
        if self.edge.lease_mgr:
            try:
                self.edge.lease_mgr.release(self._lease_name)
            except Exception:
                pass
            if self._renew_thread:
                try:
                    self._renew_thread.join(timeout=1)
                except Exception:
                    pass

    def _release_lease_now(self):
        if self.edge.lease_mgr:
            try:
                self.edge.lease_mgr.release(self._lease_name)
            except Exception:
                pass
        if self._renew_thread:
            try:
                self._renew_thread.join(timeout=1)
            except Exception:
                pass
            self._renew_thread = None

    def _producers_for_vendor(self) -> List[Iterator[dict]]:
        from .producers import (
            alpaca_bars_units,
            alpaca_minute_units,
            alpaca_options_bars_units,
            alpaca_options_chain_units,
            alpaca_corporate_actions_units,
            polygon_bars_units,
            finnhub_options_units,
            fred_treasury_units,
            finnhub_daily_units,
            av_corp_actions_units,
            av_options_hist_units,
            fmp_fundamentals_units,
        )

        prods: List[Iterator[dict]] = []
        if self.vendor == "alpaca":
            if "alpaca_bars" in self.tasks:
                prods.append(alpaca_bars_units(self.edge, self.budget))
            if "alpaca_minute" in self.tasks:
                prods.append(alpaca_minute_units(self.edge, self.budget))
            if "alpaca_options_bars" in self.tasks:
                prods.append(alpaca_options_bars_units(self.edge, self.budget))
            if "alpaca_options_chain" in self.tasks:
                prods.append(alpaca_options_chain_units(self.edge, self.budget))
            if "alpaca_corporate_actions" in self.tasks:
                prods.append(alpaca_corporate_actions_units(self.edge, self.budget))
        elif self.vendor == "polygon":
            if "polygon_bars" in self.tasks:
                prods.append(polygon_bars_units(self.edge, self.budget))
        elif self.vendor == "finnhub":
            if "finnhub_options" in self.tasks:
                prods.append(finnhub_options_units(self.edge, self.budget))
            if "finnhub_daily" in self.tasks:
                prods.append(finnhub_daily_units(self.edge, self.budget))
        elif self.vendor == "fred":
            if "fred_treasury" in self.tasks:
                prods.append(fred_treasury_units(self.edge, self.budget))
        elif self.vendor == "av":
            if "av_corp_actions" in self.tasks:
                prods.append(av_corp_actions_units(self.edge, self.budget))
            if "av_options_hist" in self.tasks:
                prods.append(av_options_hist_units(self.edge, self.budget))
        elif self.vendor == "fmp":
            if "fmp_fundamentals" in self.tasks:
                prods.append(fmp_fundamentals_units(self.edge, self.budget))
        return prods

    def _can_submit(self, tokens: int) -> bool:
        now = time.time()
        if self._freeze_until and now < self._freeze_until:
            return False
        if self.budget and tokens > 0:
            return self.budget.try_consume(self.vendor, tokens)
        return True

    def _handle_result(self, unit: dict, status: str, rows: int, msg: str):
        if status == "ok":
            self._stats.completed_ok += 1
        elif status == "ratelimited":
            self._stats.ratelimited += 1
            self._freeze_until = time.time() + _vendor_freeze_seconds(self.vendor)
            logger.warning(f"{self.vendor}: rate-limited; freezing for {_vendor_freeze_seconds(self.vendor)}s")
        elif status == "error":
            self._stats.errors += 1
            logger.warning(f"{self.vendor}: unit error [{unit.get('desc')}]: {msg}")

    def _write_heartbeat(self, inflight: int):
        """Persist lightweight heartbeat for external monitors."""
        try:
            hb_path = Path(os.getenv("NODE_HEARTBEAT_PATH", "logs/edge_heartbeat.json"))
            hb_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "vendor": self.vendor,
                "submitted": self._stats.submitted,
                "ok": self._stats.completed_ok,
                "errors": self._stats.errors,
                "ratelimited": self._stats.ratelimited,
                "inflight": inflight,
                "timestamp": time.time(),
            }
            hb_path.write_text(json.dumps(payload, indent=2))
        except Exception:
            pass

    def _run_loop(self):
        cap = _vendor_cap(self.vendor)
        logger.info(f"{self.vendor}: runner starting with cap={cap}")

        producers = [p for p in self._producers_for_vendor() if p is not None]
        if not producers:
            logger.info(f"{self.vendor}: no runnable producers (connector missing or tasks not requested)")
            # No work; release lease promptly
            self._release_lease_now()
            return

        self._executor = ThreadPoolExecutor(max_workers=cap)
        rr = cycle(range(len(producers)))

        active: Dict[object, dict] = {}
        active_started: Dict[object, float] = {}
        last_heartbeat = time.time()

        def _try_schedule(max_slots: int) -> bool:
            """Attempt to schedule up to max_slots units; return True if any submitted."""
            if max_slots <= 0:
                return False
            progressed_local = False
            import random
            for _ in range(max_slots):
                scheduled = False
                for __ in range(len(producers)):
                    idx = next(rr)
                    it = producers[idx]
                    try:
                        unit = next(it)
                    except StopIteration:
                        producers[idx] = iter(())
                        continue
                    tokens = int(unit.get("tokens", 1))
                    if self._can_submit(tokens):
                        logger.info(f"{self.vendor}: submitting unit desc='{unit.get('desc')}' tokens={tokens}")
                        fut = self._executor.submit(unit["run"])  # returns (status, rows, msg)
                        active[fut] = unit
                        active_started[fut] = time.time()
                        self._stats.submitted += 1
                        progressed_local = True
                        scheduled = True
                        time.sleep(random.uniform(0.03, 0.12))
                        break
                if not scheduled:
                    break
            return progressed_local

        try:
            # Initial seeding
            progressed = _try_schedule(cap - len(active))
            if not progressed and not active:
                # If frozen, wait out the freeze window instead of exiting immediately
                now = time.time()
                if self._freeze_until and now < self._freeze_until:
                    sleep_for = min(5.0, self._freeze_until - now)
                    time.sleep(max(0.5, sleep_for))
                    progressed = _try_schedule(cap - len(active))
            if not progressed and not active:
                logger.info(f"{self.vendor}: no units scheduled; releasing lease")
                self._release_lease_now()
                return

            # Main loop: stay alive across cooldowns; only exit when no work and not frozen
            while not self._shutdown and not self.edge.shutdown_requested:
                # Harvest completions if any inflight
                if active:
                    done_set, _ = wait(list(active.keys()), timeout=2.0, return_when=FIRST_COMPLETED)
                else:
                    done_set = set()

                if not done_set:
                    now = time.time()
                    if now - last_heartbeat >= 30:
                        logger.info(
                            f"{self.vendor}: hb submitted={self._stats.submitted} ok={self._stats.completed_ok} "
                            f"err={self._stats.errors} rl={self._stats.ratelimited} inflight={len(active)} budget={self.budget.remaining(self.vendor) if self.budget else 'na'}"
                        )
                        self._write_heartbeat(len(active))
                        last_heartbeat = now
                else:
                    for fut in list(done_set):
                        unit = active.pop(fut, {})
                        try:
                            res = fut.result()
                        except Exception as e:
                            emsg = str(e)
                            status, rows, msg = ("ratelimited", 0, emsg) if ("429" in emsg or "rate" in emsg.lower()) else ("error", 0, emsg)
                        else:
                            if isinstance(res, tuple) and len(res) == 3:
                                status, rows, msg = res
                            elif isinstance(res, int):
                                status, rows, msg = ("ok", res, "")
                            else:
                                status, rows, msg = ("ok", 0, "")
                        self._handle_result(unit, status, rows, msg)

                # Try to keep the pipeline full
                progressed = _try_schedule(max(0, cap - len(active)))

                # If nothing to do and nothing inflight, decide to wait vs exit
                if not progressed and not active:
                    now = time.time()
                    if self._freeze_until and now < self._freeze_until:
                        # Still in cooldown; wait a bit then continue
                        sleep_for = min(5.0, self._freeze_until - now)
                        time.sleep(max(0.5, sleep_for))
                        continue
                    # No freeze and no progress: likely out of units/budget
                    logger.info(f"{self.vendor}: idle with no schedulable units; exiting runner")
                    break

            logger.info(f"{self.vendor}: runner complete. submitted={self._stats.submitted} ok={self._stats.completed_ok}")

        finally:
            try:
                if self._executor:
                    self._executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass


class VendorSupervisor:
    """Orchestrates per-vendor runners and their lifecycle."""

    def __init__(self, edge) -> None:
        self.edge = edge
        try:
            self.budget = edge._init_budget()
        except Exception:
            self.budget = None
        self.runners: List[VendorRunner] = []

    def _group_tasks_by_vendor(self, tasks: List[str]) -> Dict[str, List[str]]:
        mapping: Dict[str, List[str]] = {"alpaca": [], "polygon": [], "finnhub": [], "fred": [], "av": [], "fmp": []}
        for t in tasks:
            if t in ("alpaca_bars", "alpaca_minute"):
                mapping["alpaca"].append(t)
            elif t in ("alpaca_options_bars", "alpaca_options_chain", "alpaca_corporate_actions"):
                mapping["alpaca"].append(t)
            elif t == "polygon_bars":
                mapping["polygon"].append(t)
            elif t in ("finnhub_options", "finnhub_daily"):
                mapping["finnhub"].append(t)
            elif t == "fred_treasury":
                mapping["fred"].append(t)
            elif t in ("av_corp_actions", "av_options_hist"):
                mapping["av"].append(t)
            elif t == "fmp_fundamentals":
                mapping["fmp"].append(t)
        return {v: ts for v, ts in mapping.items() if ts}

    def run(self, tasks: List[str]) -> None:
        tasks_by_vendor = self._group_tasks_by_vendor(tasks)
        logger.info(f"Supervisor starting per-vendor runners for: {list(tasks_by_vendor.keys())}")

        for vendor, ts in tasks_by_vendor.items():
            if vendor not in self.edge.connectors:
                logger.info(f"Skipping {vendor}: connector not initialized")
                continue
            r = VendorRunner(self.edge, vendor=vendor, tasks=ts, budget_mgr=self.budget)
            r.start()
            self.runners.append(r)

        try:
            while self.runners and not self.edge.shutdown_requested:
                # Reap completed runners promptly (release leases early)
                for r in list(self.runners):
                    th = r._runner_thread
                    if th and not th.is_alive():
                        try:
                            r.stop(timeout=0.5)
                        except Exception:
                            pass
                        try:
                            self.runners.remove(r)
                        except Exception:
                            pass
                alive = any((r._runner_thread and r._runner_thread.is_alive()) for r in self.runners)
                if not alive:
                    break
                time.sleep(1.0)
        finally:
            for r in self.runners:
                r.stop(timeout=2.0)
