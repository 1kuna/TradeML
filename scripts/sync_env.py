#!/usr/bin/env python3
"""
Sync .env with .env.template non-destructively.

Behavior:
- Preserves existing values in .env for keys you already set (API keys, etc.).
- Adds any new keys from .env.template that aren't in your .env.
- Keeps template comments and ordering; appends your custom-only keys at the end.
- Creates a backup .env.bak unless --no-backup is passed.

Usage:
  python scripts/sync_env.py                     # sync .env from .env.template
  python scripts/sync_env.py --dry-run           # show a preview only
  python scripts/sync_env.py --template path     # custom template
  python scripts/sync_env.py --env path          # custom env path
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional


Line = Tuple[str, str, Optional[str]]  # (kind, text or key, value)


# Accept optional whitespace around '=' and ignore leading spaces
ASSIGN_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")


def parse_env_file(path: Path) -> List[Line]:
    lines: List[Line] = []
    if not path.exists():
        return lines
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw or raw.lstrip().startswith("#"):
            lines.append(("comment", raw, None))
            continue
        m = ASSIGN_RE.match(raw)
        if not m:
            # Keep unknown lines as comments to avoid dropping anything
            lines.append(("comment", raw, None))
            continue
        k, v = m.group(1), m.group(2)
        lines.append(("kv", k, v))
    return lines


def to_map(lines: List[Line]) -> Dict[str, str]:
    m: Dict[str, str] = {}
    for kind, a, b in lines:
        if kind == "kv" and a is not None and b is not None:
            m[a] = b
    return m


def keys_in_order(lines: List[Line]) -> List[str]:
    return [a for kind, a, _ in lines if kind == "kv" and a is not None]


def merge_env(template_lines: List[Line], env_lines: List[Line]) -> str:
    env_map = to_map(env_lines)
    tmpl_map = to_map(template_lines)
    out_lines: List[str] = []

    # Keys we never want to alter/relocate if present in current .env
    protected_re = re.compile(r"(.*(_API_KEY|_SECRET_KEY)$)|^(AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY)$")

    # Rebuild using template order and comments
    for kind, a, b in template_lines:
        if kind == "comment":
            out_lines.append(a)
        elif kind == "kv":
            key = a
            if key in env_map and protected_re.match(key):
                # Preserve current value for protected keys
                out_lines.append(f"{key}={env_map[key]}")
            elif key in env_map:
                out_lines.append(f"{key}={env_map[key]}")
            else:
                out_lines.append(f"{key}={b if b is not None else ''}")

    # Append custom-only keys from current .env (deduplicated, keep last value)
    extra_keys_raw = [k for k in keys_in_order(env_lines) if k not in tmpl_map]
    seen: Dict[str, bool] = {}
    extra_keys: List[str] = []
    # Deduplicate preserving the last occurrence
    for k in reversed(extra_keys_raw):
        if k not in seen:
            seen[k] = True
            extra_keys.insert(0, k)

    # Pin vendor keys into main body even if template doesn't declare them
    pinned_keys = [
        "ALPACA_API_KEY",
        "ALPACA_SECRET_KEY",
        "POLYGON_API_KEY",
        "FRED_API_KEY",
        "FINNHUB_API_KEY",
        "FMP_API_KEY",
    ]

    # Determine which keys are already present as KV in out_lines
    present_kv: Dict[str, int] = {}
    for idx, line in enumerate(out_lines):
        if not isinstance(line, str):
            continue
        m = ASSIGN_RE.match(line)
        if m:
            present_kv[m.group(1)] = idx

    # Insert pinned keys not already present
    for key in pinned_keys:
        if key not in env_map:
            continue
        if key in present_kv:
            # Ensure not duplicated in extras
            if key in extra_keys:
                extra_keys = [k for k in extra_keys if k != key]
            continue
        # Try to insert after a comment mentioning the key
        inserted = False
        for i, line in enumerate(out_lines):
            if isinstance(line, str) and line.strip().startswith('#') and key in line:
                out_lines.insert(i + 1, f"{key}={env_map[key]}")
                present_kv[key] = i + 1
                inserted = True
                break
        if not inserted:
            # Insert after a Data API Keys header if available
            anchor = None
            for i, line in enumerate(out_lines):
                if isinstance(line, str) and 'Data API Keys' in line:
                    anchor = i
                    break
            if anchor is not None:
                out_lines.insert(anchor + 1, f"{key}={env_map[key]}")
                present_kv[key] = anchor + 1
            else:
                # Fallback to appending near the end (before custom block)
                out_lines.append(f"{key}={env_map[key]}")
                present_kv[key] = len(out_lines) - 1
        # Remove from extras to avoid duplication
        if key in extra_keys:
            extra_keys = [k for k in extra_keys if k != key]
    if extra_keys:
        out_lines.append("")
        out_lines.append("# ===== Custom entries preserved from previous .env =====")
        for k in extra_keys:
            out_lines.append(f"{k}={env_map[k]}")

    return "\n".join(out_lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default=".env.template", help="Path to template file")
    ap.add_argument("--env", dest="env_path", default=".env", help="Path to .env file")
    ap.add_argument("--dry-run", action="store_true", help="Print merged content without writing")
    ap.add_argument("--no-backup", action="store_true", help="Do not create .env.bak backup")
    args = ap.parse_args()

    tpath = Path(args.template)
    epath = Path(args.env_path)

    tmpl = parse_env_file(tpath)
    cur = parse_env_file(epath)

    merged = merge_env(tmpl, cur)

    if args.dry_run:
        print(merged)
        return 0

    if not args.no_backup and epath.exists():
        bkp = epath.with_suffix(epath.suffix + ".bak")
        bkp.write_text(epath.read_text(encoding="utf-8"), encoding="utf-8")

    epath.write_text(merged, encoding="utf-8")
    print(f"Updated {epath} from {tpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
