#!/usr/bin/env python3
"""
ATA CLI (SSOT): audit, backfill, curate, train, evaluate, promote.

Examples:
  python scripts/ata.py audit gaps --tables equities_eod
  python scripts/ata.py backfill run
  python scripts/ata.py curate
  python scripts/ata.py train if-ready --model equities_xs
"""

from __future__ import annotations

import argparse
from typing import List

from dotenv import load_dotenv
from loguru import logger

from ops.ssot import audit_scan, backfill_run, curate_incremental, train_if_ready, run_cpcv, promote_if_beat_champion


def _cmd_audit(args):
    if args.action == "gaps":
        audit_scan(args.tables)
    else:
        logger.error(f"Unsupported audit action: {args.action}")


def _cmd_backfill(args):
    if args.action in (None, "run"):
        budget = {}
        for kv in args.budget or []:
            k, v = kv.split(":", 1)
            budget[k] = int(v)
        backfill_run(budget or None)
    else:
        logger.error(f"Unsupported backfill action: {args.action}")


def _cmd_curate(args):
    curate_incremental()


def _cmd_train(args):
    if args.action == "if-ready":
        train_if_ready(args.model)
    else:
        logger.error(f"Unsupported train action: {args.action}")


def _cmd_evaluate(args):
    run_cpcv(args.model)


def _cmd_promote(args):
    promote_if_beat_champion(args.model)


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(prog="ata", description="Architecture SSOT CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # audit
    p_audit = sub.add_parser("audit", help="audit operations")
    p_audit_sub = p_audit.add_subparsers(dest="action", required=True)
    p_audit_gaps = p_audit_sub.add_parser("gaps", help="scan for data gaps and enqueue backfill")
    p_audit_gaps.add_argument("--tables", nargs="+", required=True)
    p_audit_gaps.set_defaults(func=_cmd_audit)

    # backfill
    p_bf = sub.add_parser("backfill", help="backfill operations")
    p_bf.add_argument("action", nargs="?", default="run", choices=["run"], help="run backfill")
    p_bf.add_argument("--budget", nargs="*", help="vendor budgets like vendor:count")
    p_bf.set_defaults(func=_cmd_backfill)

    # curate
    p_cur = sub.add_parser("curate", help="incremental curation for new raw")
    p_cur.set_defaults(func=_cmd_curate)

    # train
    p_tr = sub.add_parser("train", help="training operations")
    p_tr.add_argument("action", choices=["if-ready"]) 
    p_tr.add_argument("--model", required=True)
    p_tr.set_defaults(func=_cmd_train)

    # evaluate
    p_ev = sub.add_parser("evaluate", help="validation operations")
    p_ev.add_argument("--model", required=True)
    p_ev.set_defaults(func=_cmd_evaluate)

    # promote
    p_pr = sub.add_parser("promote", help="promotion operations")
    p_pr.add_argument("--model", required=True)
    p_pr.set_defaults(func=_cmd_promote)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

