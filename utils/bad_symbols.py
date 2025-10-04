from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Set

from loguru import logger


@dataclass
class _Entry:
    reason: str
    added_at: str  # isoformat


class BadSymbolCache:
    """Persistent bad-symbols cache per vendor with TTL.

    Storage backends:
      - S3 (manifests/vendor_bad_symbols.json) when S3Client provided
      - Local file (data_layer/manifests/vendor_bad_symbols.json) otherwise
    """

    def __init__(self, s3_client=None, key: str = "manifests/vendor_bad_symbols.json", ttl_days: int = 7):
        self.s3 = s3_client
        self.key = key
        self.ttl_days = ttl_days
        self._data: Dict[str, Dict[str, _Entry]] = {}
        self._etag: Optional[str] = None
        self._path = Path("data_layer/manifests/vendor_bad_symbols.json")
        # In-memory strike tracking: require two strikes within 24h before persisting
        self._strikes: Dict[str, Dict[str, list[str]]] = {}
        self._load()

    def _expired(self, added_at_iso: str) -> bool:
        try:
            added = datetime.fromisoformat(added_at_iso)
        except Exception:
            return True
        return datetime.utcnow() - added > timedelta(days=self.ttl_days)

    def _load(self):
        try:
            if self.s3:
                data, etag = self.s3.get_json(self.key)
                self._etag = etag
            else:
                if not self._path.exists():
                    self._data = {}
                    return
                data = json.loads(self._path.read_text())
            # prune expired
            cleaned: Dict[str, Dict[str, _Entry]] = {}
            for vendor, d in data.items():
                kept = {}
                for sym, meta in d.items():
                    added_at = meta.get("added_at")
                    if not added_at or not self._expired(added_at):
                        kept[sym] = _Entry(reason=meta.get("reason", ""), added_at=added_at or datetime.utcnow().isoformat())
                if kept:
                    cleaned[vendor] = kept
            self._data = cleaned
        except Exception as e:
            logger.warning(f"BadSymbolCache load failed: {e}")
            self._data = {}

    def _save(self):
        try:
            payload = {v: {s: {"reason": e.reason, "added_at": e.added_at} for s, e in d.items()} for v, d in self._data.items()}
            if self.s3:
                try:
                    if self._etag:
                        self._etag = self.s3.put_json(self.key, payload, if_match=self._etag)
                    else:
                        self._etag = self.s3.put_json(self.key, payload)
                except Exception:
                    # retry without precondition
                    self._etag = self.s3.put_json(self.key, payload)
            else:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._path.write_text(json.dumps(payload, indent=2))
        except Exception as e:
            logger.warning(f"BadSymbolCache save failed: {e}")

    def add(self, vendor: str, symbol: str, reason: str):
        vendor = vendor.lower()
        sym = symbol.upper()
        ent = _Entry(reason=reason, added_at=datetime.utcnow().isoformat())
        self._data.setdefault(vendor, {})[sym] = ent
        self._save()
        logger.info(f"BadSymbolCache: marked {vendor}:{sym} ({reason})")

    def strike(self, vendor: str, symbol: str, reason: str, window_hours: int = 24, required: int = 2) -> bool:
        """Register a strike for a potentially invalid symbol.

        Persists to cache only after 'required' strikes within 'window_hours'.
        Returns True if persisted now; False otherwise.
        """
        vendor = vendor.lower()
        sym = symbol.upper()
        now = datetime.utcnow().isoformat()
        vmap = self._strikes.setdefault(vendor, {})
        arr = vmap.setdefault(sym, [])
        arr.append(now)
        # prune old
        try:
            cutoff = datetime.utcnow() - timedelta(hours=window_hours)
            arr2 = []
            for ts in arr:
                try:
                    if datetime.fromisoformat(ts) >= cutoff:
                        arr2.append(ts)
                except Exception:
                    continue
            vmap[sym] = arr2
        except Exception:
            pass
        if len(vmap.get(sym, [])) >= max(1, required):
            self.add(vendor, sym, reason)
            # reset strikes after persisting
            vmap[sym] = []
            return True
        return False

    def contains(self, vendor: str, symbol: str) -> bool:
        vendor = vendor.lower()
        sym = symbol.upper()
        return sym in self._data.get(vendor, {})

    def vendor_set(self, vendor: str) -> Set[str]:
        vendor = vendor.lower()
        return set(self._data.get(vendor, {}).keys())
