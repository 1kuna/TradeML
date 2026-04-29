"""Shared remote command runner for fleet health checks."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True, frozen=True)
class RemoteCommandResult:
    """Serializable remote command result with secrets redacted."""

    returncode: int
    stdout: str
    stderr: str
    command: list[str]
    timed_out: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a dashboard-safe dictionary."""
        return {
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "command": self.command,
            "timed_out": self.timed_out,
        }


def run_password_ssh(
    target: dict[str, str],
    command: str,
    *,
    timeout_seconds: float = 30.0,
) -> RemoteCommandResult:
    """Run one SSH command using sshpass only when a password env var is configured."""
    host = target.get("host")
    user = target.get("user")
    password_env = target.get("password_env")
    if not host or not user:
        return RemoteCommandResult(
            returncode=2,
            stdout="",
            stderr="missing host/user",
            command=[],
        )
    env = dict(os.environ)
    prefix = ["ssh"]
    command_prefix: list[str] = ["ssh"]
    if password_env and env.get(password_env):
        env["SSHPASS"] = env[password_env]
        prefix = ["sshpass", "-e", "ssh"]
        command_prefix = ["sshpass", "-e", "ssh"]
    ssh = [
        *prefix,
        "-o",
        "PreferredAuthentications=password",
        "-o",
        "PubkeyAuthentication=no",
        "-o",
        "NumberOfPasswordPrompts=1",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ConnectTimeout=8",
        f"{user}@{host}",
        command,
    ]
    redacted = [
        *command_prefix,
        "-o",
        "PreferredAuthentications=password",
        "-o",
        "PubkeyAuthentication=no",
        "-o",
        "NumberOfPasswordPrompts=1",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ConnectTimeout=8",
        f"{user}@{host}",
        command,
    ]
    try:
        result = subprocess.run(
            ssh,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return RemoteCommandResult(
            returncode=124,
            stdout="",
            stderr=f"remote command timed out after {timeout_seconds:.1f}s",
            command=redacted,
            timed_out=True,
        )
    return RemoteCommandResult(
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        command=redacted,
    )
