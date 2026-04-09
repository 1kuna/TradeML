"""Utilities for reliable Pi remote operations over SSH/Tailscale."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import time
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    import paramiko


DEFAULT_TIMEOUT_SECONDS = 120.0


@dataclass(frozen=True)
class RemoteTarget:
    """Remote SSH target definition."""

    host: str
    user: str
    password: str
    port: int = 22

    @classmethod
    def from_env(
        cls,
        *,
        host: str | None = None,
        user: str | None = None,
        password: str | None = None,
        port: int | None = None,
    ) -> "RemoteTarget":
        """Resolve a target from explicit values or environment defaults."""
        resolved_host = host or os.getenv("TRADEML_PI_HOST") or "100.76.4.69"
        resolved_user = user or os.getenv("TRADEML_PI_USER") or "zach"
        resolved_password = password or os.getenv("TRADEML_PI_PASSWORD")
        if not resolved_password:
            raise ValueError("remote password required via argument or TRADEML_PI_PASSWORD")
        resolved_port = int(port or os.getenv("TRADEML_PI_PORT") or 22)
        return cls(
            host=resolved_host,
            user=resolved_user,
            password=resolved_password,
            port=resolved_port,
        )


@dataclass(frozen=True)
class RemoteResult:
    """Captured remote execution result."""

    command: str
    exit_status: int
    stdout: str
    stderr: str
    elapsed_seconds: float


def wrap_remote_command(
    command: str,
    *,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> str:
    """Wrap a command with optional cwd/env setup for remote execution."""
    segments: list[str] = []
    if cwd is not None:
        segments.append(f"cd {shlex.quote(str(cwd))}")
    rendered = command
    if env:
        prefix = " ".join(f"{key}={shlex.quote(str(value))}" for key, value in sorted(env.items()))
        rendered = f"env {prefix} {command}"
    segments.append(rendered)
    return " && ".join(segments)


def python_here_doc(script: str, *, python_executable: str = "python3") -> str:
    """Render a shell-safe Python heredoc command."""
    return f"{python_executable} - <<'PY'\n{script.rstrip()}\nPY"


def dashboard_snapshot_script(
    *,
    workspace_root: str | Path,
    config_path: str | Path,
    env_path: str | Path,
) -> str:
    """Return the remote Python snippet for a dashboard snapshot."""
    return f"""
from pathlib import Path
from trademl.dashboard.controller import collect_dashboard_snapshot, resolve_node_settings

settings = resolve_node_settings(
    workspace_root=Path({workspace_root!r}),
    config_path=Path({config_path!r}),
    env_path=Path({env_path!r}),
)
snapshot = collect_dashboard_snapshot(settings)
print(snapshot)
"""


def restart_node_script(
    *,
    workspace_root: str | Path,
    config_path: str | Path,
    env_path: str | Path,
) -> str:
    """Return the remote Python snippet for a clean node restart."""
    return f"""
from pathlib import Path
from trademl.dashboard.controller import resolve_node_settings, restart_node

settings = resolve_node_settings(
    workspace_root=Path({workspace_root!r}),
    config_path=Path({config_path!r}),
    env_path=Path({env_path!r}),
)
print(restart_node(settings))
"""


class PiRemoteClient:
    """Small Paramiko wrapper that streams output and avoids hanging black boxes."""

    def __init__(self, target: RemoteTarget, *, connect_timeout: float = 20.0) -> None:
        self.target = target
        self.connect_timeout = connect_timeout
        self._client: Any | None = None

    def connect(self) -> None:
        """Open the SSH connection if needed."""
        if self._client is not None:
            return
        import paramiko

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            self.target.host,
            port=self.target.port,
            username=self.target.user,
            password=self.target.password,
            timeout=self.connect_timeout,
        )
        self._client = client

    def close(self) -> None:
        """Close the SSH connection."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "PiRemoteClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def upload_file(self, local_path: str | Path, remote_path: str | Path) -> None:
        """Upload a single file to the Pi."""
        self.connect()
        assert self._client is not None
        remote = Path(str(remote_path))
        with self._client.open_sftp() as sftp:
            self._mkdir_p(sftp, remote.parent)
            sftp.put(str(local_path), str(remote))

    def run(
        self,
        command: str,
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        poll_interval_seconds: float = 0.05,
    ) -> RemoteResult:
        """Run a command and stream stdout/stderr until completion or timeout."""
        self.connect()
        assert self._client is not None
        wrapped = wrap_remote_command(command, cwd=cwd, env=env)
        start = time.monotonic()
        stdin, stdout, stderr = self._client.exec_command(wrapped, timeout=max(5.0, timeout_seconds))
        del stdin
        channel = stdout.channel
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []

        while True:
            activity = False
            while channel.recv_ready():
                stdout_chunks.append(channel.recv(65536).decode("utf-8", "replace"))
                activity = True
            while channel.recv_stderr_ready():
                stderr_chunks.append(channel.recv_stderr(65536).decode("utf-8", "replace"))
                activity = True
            if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready():
                break
            if (time.monotonic() - start) > timeout_seconds:
                channel.close()
                raise TimeoutError(f"remote command timed out after {timeout_seconds:.1f}s: {command}")
            if not activity:
                time.sleep(poll_interval_seconds)

        exit_status = channel.recv_exit_status()
        return RemoteResult(
            command=wrapped,
            exit_status=exit_status,
            stdout="".join(stdout_chunks),
            stderr="".join(stderr_chunks),
            elapsed_seconds=time.monotonic() - start,
        )

    @staticmethod
    def _mkdir_p(sftp: Any, path: Path) -> None:
        current = Path("/")
        for part in path.parts:
            if part == "/":
                current = Path("/")
                continue
            current /= part
            try:
                sftp.stat(str(current))
            except IOError:
                sftp.mkdir(str(current))
