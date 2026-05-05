"""Shared SSH/SCP runner for fleet and training control checks."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
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
    force_password_auth = bool(password_env and os.environ.get(password_env))
    return run_ssh_command(
        host=host,
        user=user,
        command=command,
        port=int(target.get("port") or 22),
        password_env=password_env,
        timeout_seconds=timeout_seconds,
        force_password_auth=force_password_auth,
        strict_host_key_checking=False,
    )


def run_ssh_command(
    *,
    host: str,
    user: str,
    command: str,
    port: int = 22,
    identity_file: str | Path | None = None,
    password_env: str | None = None,
    timeout_seconds: float = 30.0,
    connect_timeout_seconds: int = 8,
    server_alive_interval_seconds: int = 5,
    server_alive_count_max: int = 2,
    force_password_auth: bool = False,
    strict_host_key_checking: bool | None = None,
) -> RemoteCommandResult:
    """Run one SSH command and return a dashboard-safe result."""
    env, prefix = _command_prefix_and_env(password_env=password_env, binary="ssh")
    ssh = [
        *prefix,
        *_ssh_options(
            port=port,
            identity_file=identity_file,
            has_password=env is not None,
            connect_timeout_seconds=connect_timeout_seconds,
            server_alive_interval_seconds=server_alive_interval_seconds,
            server_alive_count_max=server_alive_count_max,
            force_password_auth=force_password_auth,
            strict_host_key_checking=strict_host_key_checking,
            scp=False,
        ),
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
            command=ssh,
            timed_out=True,
        )
    except FileNotFoundError as exc:
        return RemoteCommandResult(
            returncode=127,
            stdout="",
            stderr=f"missing remote command dependency: {exc.filename}",
            command=ssh,
        )
    return RemoteCommandResult(
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        command=ssh,
    )


def copy_file_to_remote(
    *,
    host: str,
    user: str,
    local_path: str | Path,
    remote_path: str | Path,
    port: int = 22,
    identity_file: str | Path | None = None,
    password_env: str | None = None,
    timeout_seconds: float = 30.0,
    connect_timeout_seconds: int = 8,
    server_alive_interval_seconds: int = 5,
    server_alive_count_max: int = 2,
) -> RemoteCommandResult:
    """Copy one local file to a remote host with the shared SSH option policy."""
    env, prefix = _command_prefix_and_env(password_env=password_env, binary="scp")
    scp = [
        *prefix,
        *_ssh_options(
            port=port,
            identity_file=identity_file,
            has_password=env is not None,
            connect_timeout_seconds=connect_timeout_seconds,
            server_alive_interval_seconds=server_alive_interval_seconds,
            server_alive_count_max=server_alive_count_max,
            scp=True,
        ),
        str(local_path),
        f"{user}@{host}:{remote_path}",
    ]
    try:
        result = subprocess.run(
            scp,
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
            stderr=f"remote copy timed out after {timeout_seconds:.1f}s",
            command=scp,
            timed_out=True,
        )
    except FileNotFoundError as exc:
        return RemoteCommandResult(
            returncode=127,
            stdout="",
            stderr=f"missing remote copy dependency: {exc.filename}",
            command=scp,
        )
    return RemoteCommandResult(
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        command=scp,
    )


def _command_prefix_and_env(*, password_env: str | None, binary: str) -> tuple[dict[str, str] | None, list[str]]:
    env = dict(os.environ)
    if password_env and env.get(password_env):
        env["SSHPASS"] = env[password_env]
        return env, ["sshpass", "-e", binary]
    return None, [binary]


def _ssh_options(
    *,
    port: int,
    identity_file: str | Path | None,
    has_password: bool,
    connect_timeout_seconds: int,
    server_alive_interval_seconds: int,
    server_alive_count_max: int,
    force_password_auth: bool = False,
    strict_host_key_checking: bool | None = None,
    scp: bool,
) -> list[str]:
    port_flag = "-P" if scp else "-p"
    options = [
        "-o",
        f"BatchMode={'no' if has_password else 'yes'}",
        "-o",
        f"ConnectTimeout={connect_timeout_seconds}",
        "-o",
        f"ServerAliveInterval={server_alive_interval_seconds}",
        "-o",
        f"ServerAliveCountMax={server_alive_count_max}",
        port_flag,
        str(port),
    ]
    if identity_file is not None:
        options.extend(["-i", str(identity_file)])
    if force_password_auth:
        options.extend(
            [
                "-o",
                "PreferredAuthentications=password",
                "-o",
                "PubkeyAuthentication=no",
                "-o",
                "NumberOfPasswordPrompts=1",
            ]
        )
    if strict_host_key_checking is not None:
        options.extend(["-o", f"StrictHostKeyChecking={'yes' if strict_host_key_checking else 'no'}"])
    return options
