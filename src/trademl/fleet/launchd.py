"""macOS launchd helpers for durable local research supervision."""

from __future__ import annotations

import os
import plistlib
import subprocess
from pathlib import Path
from typing import Any


def render_research_launch_agent(
    *,
    label: str,
    python_executable: str,
    repo_root: Path,
    program_path: Path,
    data_root: Path,
    local_state: Path,
    env_path: Path,
    poll_seconds: int,
    stdout_path: Path,
    stderr_path: Path,
) -> str:
    """Render a launchd plist for one foreground research supervisor."""
    payload: dict[str, Any] = {
        "Label": label,
        "ProgramArguments": [
            python_executable,
            "-m",
            "trademl.cli",
            "research",
            "--data-root",
            str(data_root),
            "--local-state",
            str(local_state),
            "--env-file",
            str(env_path),
            "start",
            "--program",
            str(program_path),
            "--poll-seconds",
            str(int(poll_seconds)),
        ],
        "WorkingDirectory": str(repo_root),
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": str(stdout_path),
        "StandardErrorPath": str(stderr_path),
        "EnvironmentVariables": {"PYTHONUNBUFFERED": "1"},
        "ProcessType": "Background",
    }
    return plistlib.dumps(payload, sort_keys=True).decode("utf-8")


def install_research_launch_agent(
    *,
    label: str,
    python_executable: str,
    repo_root: Path,
    program_path: Path,
    data_root: Path,
    local_state: Path,
    env_path: Path,
    poll_seconds: int,
    plist_path: Path | None = None,
    load: bool = False,
) -> dict[str, Any]:
    """Install a macOS LaunchAgent plist and optionally load it."""
    program_id = str(_read_program_id(program_path) or label.rsplit(".", 1)[-1])
    log_root = local_state / "research_programs" / program_id / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    resolved_plist = plist_path or (Path.home() / "Library" / "LaunchAgents" / f"{label}.plist")
    resolved_plist.parent.mkdir(parents=True, exist_ok=True)
    stdout_path = log_root / "launchd.out.log"
    stderr_path = log_root / "launchd.err.log"
    plist_text = render_research_launch_agent(
        label=label,
        python_executable=python_executable,
        repo_root=repo_root,
        program_path=program_path,
        data_root=data_root,
        local_state=local_state,
        env_path=env_path,
        poll_seconds=poll_seconds,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    resolved_plist.write_text(plist_text, encoding="utf-8")
    result: dict[str, Any] = {
        "label": label,
        "plist_path": str(resolved_plist),
        "loaded": False,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }
    if load:
        result["launchctl"] = load_launch_agent(resolved_plist)
        result["loaded"] = bool(result["launchctl"].get("ok"))
    return result


def load_launch_agent(plist_path: Path) -> dict[str, Any]:
    """Load a LaunchAgent plist for the current GUI user."""
    if os.uname().sysname != "Darwin":
        return {"ok": False, "reason": "launchd is only supported on macOS"}
    target = f"gui/{os.getuid()}"
    bootout = subprocess.run(["launchctl", "bootout", target, str(plist_path)], capture_output=True, text=True, check=False)
    bootstrap = subprocess.run(["launchctl", "bootstrap", target, str(plist_path)], capture_output=True, text=True, check=False)
    if bootstrap.returncode != 0:
        return {
            "ok": False,
            "bootout_returncode": bootout.returncode,
            "returncode": bootstrap.returncode,
            "stdout": bootstrap.stdout,
            "stderr": bootstrap.stderr,
        }
    enable = subprocess.run(["launchctl", "enable", f"{target}/{_label_from_plist(plist_path)}"], capture_output=True, text=True, check=False)
    kickstart = subprocess.run(["launchctl", "kickstart", "-k", f"{target}/{_label_from_plist(plist_path)}"], capture_output=True, text=True, check=False)
    return {
        "ok": enable.returncode == 0 and kickstart.returncode == 0,
        "bootout_returncode": bootout.returncode,
        "enable_returncode": enable.returncode,
        "kickstart_returncode": kickstart.returncode,
        "stdout": "\n".join(part for part in [bootstrap.stdout, enable.stdout, kickstart.stdout] if part),
        "stderr": "\n".join(part for part in [bootstrap.stderr, enable.stderr, kickstart.stderr] if part),
    }


def unload_launch_agent(label: str) -> dict[str, Any]:
    """Unload a LaunchAgent by label for the current GUI user."""
    if os.uname().sysname != "Darwin":
        return {"ok": False, "label": label, "reason": "launchd is only supported on macOS"}
    target = f"gui/{os.getuid()}/{label}"
    result = subprocess.run(["launchctl", "bootout", target], capture_output=True, text=True, check=False)
    return {
        "ok": result.returncode == 0,
        "label": label,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def kickstart_launch_agent(label: str) -> dict[str, Any]:
    """Restart a loaded LaunchAgent by label for the current GUI user."""
    if os.uname().sysname != "Darwin":
        return {"ok": False, "label": label, "reason": "launchd is only supported on macOS"}
    target = f"gui/{os.getuid()}/{label}"
    result = subprocess.run(["launchctl", "kickstart", "-k", target], capture_output=True, text=True, check=False)
    return {
        "ok": result.returncode == 0,
        "label": label,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def launch_agent_status(label: str) -> dict[str, Any]:
    """Return launchd status for one LaunchAgent label."""
    if os.uname().sysname != "Darwin":
        return {"ok": False, "label": label, "loaded": False, "reason": "launchd is only supported on macOS"}
    target = f"gui/{os.getuid()}/{label}"
    result = subprocess.run(["launchctl", "print", target], capture_output=True, text=True, check=False)
    payload: dict[str, Any] = {
        "ok": result.returncode == 0,
        "label": label,
        "loaded": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    if result.returncode == 0:
        payload.update(_parse_launchctl_print(result.stdout))
    return payload


def _parse_launchctl_print(output: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if line.startswith("state = ") and "state" not in parsed:
            parsed["state"] = line.partition(" = ")[2]
        elif line.startswith("pid = ") and "pid" not in parsed:
            try:
                parsed["pid"] = int(line.partition(" = ")[2])
            except ValueError:
                continue
        elif line.startswith("runs = ") and "runs" not in parsed:
            try:
                parsed["runs"] = int(line.partition(" = ")[2])
            except ValueError:
                continue
        elif line.startswith("path = ") and "plist_path" not in parsed:
            parsed["plist_path"] = line.partition(" = ")[2]
    return parsed


def _label_from_plist(plist_path: Path) -> str:
    payload = plistlib.loads(plist_path.read_bytes())
    return str(payload["Label"])


def _read_program_id(program_path: Path) -> str | None:
    try:
        import yaml

        payload = yaml.safe_load(program_path.read_text(encoding="utf-8")) or {}
        return str(payload.get("program_id")) if payload.get("program_id") else None
    except (OSError, yaml.YAMLError):
        return None
