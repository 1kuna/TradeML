from __future__ import annotations

"""
Token-bucket style daily API budget manager for backfill workers.

Persists per-vendor remaining tokens and daily reset timestamp in S3 (if
configured) or local filesystem. Budgets are defined in configs/backfill.yml
under policy.daily_api_budget.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

from loguru import logger


DEFAULT_MANIFEST_KEY = "manifests/backfill_budget.json"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class VendorBudget:
    limit: int
    remaining: int
    reset_at: str  # ISO timestamp at midnight UTC


class BudgetManager:
    def __init__(self, initial_limits: Dict[str, int], s3_client: Optional[object] = None, manifest_key: str = DEFAULT_MANIFEST_KEY):
        self.s3 = s3_client
        self.manifest_key = manifest_key
        self.path_local = Path("data_layer") / manifest_key
        self.state: Dict[str, VendorBudget] = {}
        self._load(initial_limits)

    def _load(self, limits: Dict[str, int]):
        payload = None
        if self.s3:
            try:
                payload, _ = self.s3.get_json(self.manifest_key)
            except Exception:
                payload = None
        else:
            if self.path_local.exists():
                try:
                    payload = json.loads(self.path_local.read_text())
                except Exception:
                    payload = None

        if not payload:
            # Initialize fresh budgets
            self._reset_all(limits)
            self._persist()
            return

        # Load existing and ensure all vendors present
        try:
            now = _now_utc()
            for vendor, limit in limits.items():
                v = payload.get(vendor)
                if v is None:
                    self.state[vendor] = self._new_budget(limit)
                    continue
                reset_at = datetime.fromisoformat(v.get("reset_at"))
                if now >= reset_at:
                    self.state[vendor] = self._new_budget(limit)
                else:
                    self.state[vendor] = VendorBudget(limit=limit, remaining=int(v.get("remaining", limit)), reset_at=v.get("reset_at"))
        except Exception as e:
            logger.warning(f"Budget state invalid, resetting: {e}")
            self._reset_all(limits)
            self._persist()

    def _new_budget(self, limit: int) -> VendorBudget:
        # Reset at next UTC midnight
        tomorrow = (_now_utc() + timedelta(days=1)).date()
        reset_at = datetime(tomorrow.year, tomorrow.month, tomorrow.day, tzinfo=timezone.utc)
        return VendorBudget(limit=limit, remaining=limit, reset_at=reset_at.isoformat())

    def _reset_all(self, limits: Dict[str, int]):
        for vendor, limit in limits.items():
            self.state[vendor] = self._new_budget(limit)

    def _persist(self):
        payload = {k: {"limit": v.limit, "remaining": v.remaining, "reset_at": v.reset_at} for k, v in self.state.items()}
        if self.s3:
            try:
                self.s3.put_json(self.manifest_key, payload)
                return
            except Exception as e:
                logger.warning(f"Failed to persist budget to S3: {e}")
        # Local fallback
        self.path_local.parent.mkdir(parents=True, exist_ok=True)
        self.path_local.write_text(json.dumps(payload, indent=2))

    def _ensure_fresh(self):
        # Reset vendors that crossed reset time
        now = _now_utc()
        changed = False
        for vendor, vb in self.state.items():
            try:
                reset_at = datetime.fromisoformat(vb.reset_at)
            except Exception:
                reset_at = _now_utc()
            if now >= reset_at:
                self.state[vendor] = self._new_budget(vb.limit)
                changed = True
        if changed:
            self._persist()

    def remaining(self, vendor: str) -> int:
        self._ensure_fresh()
        return int(self.state.get(vendor, VendorBudget(limit=0, remaining=0, reset_at=_now_utc().isoformat())).remaining)

    def time_to_reset_seconds(self, vendor: str) -> int:
        self._ensure_fresh()
        vb = self.state.get(vendor)
        if not vb:
            return 0
        try:
            reset_at = datetime.fromisoformat(vb.reset_at)
        except Exception:
            return 0
        return max(0, int((reset_at - _now_utc()).total_seconds()))

    def try_consume(self, vendor: str, tokens: int) -> bool:
        """Attempt to consume tokens from vendor budget.

        Returns True if allowed, False if insufficient remaining tokens.
        """
        self._ensure_fresh()
        vb = self.state.get(vendor)
        if vb is None:
            logger.warning(f"Unknown vendor budget: {vendor}")
            return True  # do not block unknown vendors
        if vb.remaining >= tokens:
            vb.remaining -= tokens
            self.state[vendor] = vb
            self._persist()
            return True
        return False

