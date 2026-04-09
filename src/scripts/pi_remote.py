"""Small CLI for reliable Pi operations over SSH/Tailscale."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trademl.pi_remote import (
    PiRemoteClient,
    RemoteTarget,
    dashboard_snapshot_script,
    python_here_doc,
    restart_node_script,
)


def main(argv: list[str] | None = None) -> int:
    """Dispatch Pi remote helper commands."""
    parser = argparse.ArgumentParser(description="TradeML Pi remote helper.")
    parser.add_argument("--host", default=None)
    parser.add_argument("--user", default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--port", type=int, default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    exec_parser = subparsers.add_parser("exec", help="Run an arbitrary remote shell command.")
    exec_parser.add_argument("shell_command")
    exec_parser.add_argument("--cwd", default=None)
    exec_parser.add_argument("--timeout", type=float, default=120.0)

    upload_parser = subparsers.add_parser("upload", help="Upload one or more files to the Pi.")
    upload_parser.add_argument("--remote-root", required=True)
    upload_parser.add_argument("files", nargs="+")

    snapshot_parser = subparsers.add_parser("snapshot", help="Collect a dashboard snapshot from the Pi.")
    snapshot_parser.add_argument("--workspace-root", default="/home/zach/trademl-node")
    snapshot_parser.add_argument("--config", default="/home/zach/trademl-node/node.yml")
    snapshot_parser.add_argument("--env-file", default="/home/zach/trademl-node/.env")

    restart_parser = subparsers.add_parser("restart-node", help="Restart the Pi worker through dashboard helpers.")
    restart_parser.add_argument("--workspace-root", default="/home/zach/trademl-node")
    restart_parser.add_argument("--config", default="/home/zach/trademl-node/node.yml")
    restart_parser.add_argument("--env-file", default="/home/zach/trademl-node/.env")

    args = parser.parse_args(argv)
    target = RemoteTarget.from_env(host=args.host, user=args.user, password=args.password, port=args.port)
    with PiRemoteClient(target) as client:
        if args.command == "exec":
            result = client.run(args.shell_command, cwd=args.cwd, timeout_seconds=args.timeout)
            print(json.dumps(result.__dict__, indent=2))
            return 0 if result.exit_status == 0 else result.exit_status
        if args.command == "upload":
            remote_root = Path(args.remote_root)
            for file_name in args.files:
                local = Path(file_name)
                client.upload_file(local, remote_root / local.name)
            print(json.dumps({"uploaded": len(args.files), "remote_root": str(remote_root)}, indent=2))
            return 0
        if args.command == "snapshot":
            result = client.run(
                python_here_doc(
                    dashboard_snapshot_script(
                        workspace_root=args.workspace_root,
                        config_path=args.config,
                        env_path=args.env_file,
                    ),
                    python_executable="/home/zach/TradeML/.venv/bin/python",
                ),
                cwd="/home/zach/TradeML",
                env={"PYTHONPATH": "src"},
            )
            if result.stdout.strip():
                print(result.stdout.strip())
            if result.stderr.strip():
                print(result.stderr.strip())
            return 0 if result.exit_status == 0 else result.exit_status
        if args.command == "restart-node":
            result = client.run(
                python_here_doc(
                    restart_node_script(
                        workspace_root=args.workspace_root,
                        config_path=args.config,
                        env_path=args.env_file,
                    ),
                    python_executable="/home/zach/TradeML/.venv/bin/python",
                ),
                cwd="/home/zach/TradeML",
                env={"PYTHONPATH": "src"},
            )
            if result.stdout.strip():
                print(result.stdout.strip())
            if result.stderr.strip():
                print(result.stderr.strip())
            return 0 if result.exit_status == 0 else result.exit_status
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
