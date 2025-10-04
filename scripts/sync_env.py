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

    # Rebuild using template order and comments
    for kind, a, b in template_lines:
        if kind == "comment":
            out_lines.append(a)
        elif kind == "kv":
            key = a
            if key in env_map:
                out_lines.append(f"{key}={env_map[key]}")
            else:
                out_lines.append(f"{key}={b if b is not None else ''}")

    # Append custom-only keys from current .env
    extra_keys = [k for k in keys_in_order(env_lines) if k not in tmpl_map]
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
