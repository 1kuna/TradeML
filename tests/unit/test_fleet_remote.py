from __future__ import annotations

import subprocess
from typing import Any

from trademl.fleet import remote


def test_run_password_ssh_redacts_password_env(monkeypatch) -> None:
    observed: dict[str, Any] = {}
    monkeypatch.setenv("TRADEML_TEST_PASSWORD", "secret-value")

    def fake_run(command, **kwargs):  # noqa: ANN001
        observed["command"] = command
        observed["env"] = kwargs["env"]
        return subprocess.CompletedProcess(command, 0, "active\n", "")

    monkeypatch.setattr(remote.subprocess, "run", fake_run)

    result = remote.run_password_ssh(
        {
            "host": "100.0.0.1",
            "user": "zach",
            "password_env": "TRADEML_TEST_PASSWORD",
        },
        "systemctl --user is-active trademl-node.service",
    )

    payload = result.to_dict()
    assert observed["command"][:3] == ["sshpass", "-e", "ssh"]
    assert observed["env"]["SSHPASS"] == "secret-value"
    assert "secret-value" not in " ".join(payload["command"])
    assert payload["stdout"] == "active\n"


def test_run_password_ssh_reports_timeout(monkeypatch) -> None:
    def timeout_run(*_args, **_kwargs):  # noqa: ANN002, ANN003
        raise subprocess.TimeoutExpired(cmd="ssh", timeout=30)

    monkeypatch.setattr(remote.subprocess, "run", timeout_run)

    result = remote.run_password_ssh({"host": "100.0.0.1", "user": "zach"}, "date")

    assert result.returncode == 124
    assert result.timed_out is True
    assert "timed out" in result.stderr


def test_run_password_ssh_allows_key_auth_without_password_env(monkeypatch) -> None:
    observed: dict[str, Any] = {}

    def fake_run(command, **kwargs):  # noqa: ANN001
        observed["command"] = command
        observed["env"] = kwargs["env"]
        return subprocess.CompletedProcess(command, 0, "active\n", "")

    monkeypatch.setattr(remote.subprocess, "run", fake_run)

    result = remote.run_password_ssh(
        {
            "host": "100.0.0.1",
            "user": "zach",
            "password_env": "TRADEML_MISSING_PASSWORD",
        },
        "systemctl --user is-active trademl-node.service",
    )

    assert observed["command"][0] == "ssh"
    assert observed["env"] is None
    assert "PubkeyAuthentication=no" not in observed["command"]
    assert "BatchMode=yes" in observed["command"]
    assert result.returncode == 0


def test_run_password_ssh_reports_missing_sshpass(monkeypatch) -> None:
    monkeypatch.setenv("TRADEML_TEST_PASSWORD", "secret-value")

    def missing_binary(*_args, **_kwargs):  # noqa: ANN002, ANN003
        raise FileNotFoundError(2, "No such file or directory", "sshpass")

    monkeypatch.setattr(remote.subprocess, "run", missing_binary)

    result = remote.run_password_ssh(
        {"host": "100.0.0.1", "user": "zach", "password_env": "TRADEML_TEST_PASSWORD"},
        "date",
    )

    assert result.returncode == 127
    assert "missing remote command dependency: sshpass" in result.stderr
    assert "secret-value" not in " ".join(result.command)


def test_copy_file_to_remote_uses_scp_with_redacted_password(monkeypatch, tmp_path) -> None:
    observed: dict[str, Any] = {}
    monkeypatch.setenv("TRADEML_TEST_PASSWORD", "secret-value")

    def fake_run(command, **kwargs):  # noqa: ANN001
        observed["command"] = command
        observed["env"] = kwargs["env"]
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(remote.subprocess, "run", fake_run)

    local_path = tmp_path / "config.yml"
    local_path.write_text("ok: true\n")
    result = remote.copy_file_to_remote(
        host="100.0.0.1",
        user="zach",
        local_path=local_path,
        remote_path="/srv/trademl/control/config.yml",
        password_env="TRADEML_TEST_PASSWORD",
    )

    payload = result.to_dict()
    assert observed["command"][:3] == ["sshpass", "-e", "scp"]
    assert observed["env"]["SSHPASS"] == "secret-value"
    assert "secret-value" not in " ".join(payload["command"])
    assert "-P" in payload["command"]
    assert payload["returncode"] == 0


def test_copy_file_to_remote_reports_timeout(monkeypatch, tmp_path) -> None:
    def timeout_run(*_args, **_kwargs):  # noqa: ANN002, ANN003
        raise subprocess.TimeoutExpired(cmd="scp", timeout=30)

    monkeypatch.setattr(remote.subprocess, "run", timeout_run)

    local_path = tmp_path / "config.yml"
    local_path.write_text("ok: true\n")
    result = remote.copy_file_to_remote(
        host="100.0.0.1",
        user="zach",
        local_path=local_path,
        remote_path="/srv/trademl/control/config.yml",
    )

    assert result.returncode == 124
    assert result.timed_out is True
    assert "timed out" in result.stderr


def test_copy_file_to_remote_reports_missing_sshpass(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TRADEML_TEST_PASSWORD", "secret-value")

    def missing_binary(*_args, **_kwargs):  # noqa: ANN002, ANN003
        raise FileNotFoundError(2, "No such file or directory", "sshpass")

    monkeypatch.setattr(remote.subprocess, "run", missing_binary)

    local_path = tmp_path / "config.yml"
    local_path.write_text("ok: true\n")
    result = remote.copy_file_to_remote(
        host="100.0.0.1",
        user="zach",
        local_path=local_path,
        remote_path="/srv/trademl/control/config.yml",
        password_env="TRADEML_TEST_PASSWORD",
    )

    assert result.returncode == 127
    assert "missing remote copy dependency: sshpass" in result.stderr
    assert "secret-value" not in " ".join(result.command)
