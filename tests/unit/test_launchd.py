from __future__ import annotations

import plistlib
from pathlib import Path

from trademl.fleet import launchd
from trademl.fleet.launchd import install_research_launch_agent, render_research_launch_agent


def test_render_research_launch_agent_uses_foreground_research_start(tmp_path: Path) -> None:
    plist_text = render_research_launch_agent(
        label="com.trademl.research.perpetual-macmini",
        python_executable="/venv/bin/python",
        repo_root=tmp_path / "repo",
        program_path=tmp_path / "repo" / "configs" / "research.yml",
        data_root=tmp_path / "nas",
        local_state=tmp_path / "control",
        env_path=tmp_path / ".env",
        poll_seconds=60,
        stdout_path=tmp_path / "out.log",
        stderr_path=tmp_path / "err.log",
    )

    payload = plistlib.loads(plist_text.encode("utf-8"))

    assert payload["Label"] == "com.trademl.research.perpetual-macmini"
    assert payload["KeepAlive"] is True
    assert payload["RunAtLoad"] is True
    assert "--detach" not in payload["ProgramArguments"]
    assert payload["ProgramArguments"][:4] == ["/venv/bin/python", "-m", "trademl.cli", "research"]
    assert payload["WorkingDirectory"] == str(tmp_path / "repo")


def test_install_research_launch_agent_writes_plist_without_loading(tmp_path: Path) -> None:
    program_path = tmp_path / "program.yml"
    program_path.write_text("program_id: perpetual-macmini\nphase_order: [1]\n", encoding="utf-8")
    plist_path = tmp_path / "agent.plist"

    payload = install_research_launch_agent(
        label="com.trademl.research.perpetual-macmini",
        python_executable="/venv/bin/python",
        repo_root=tmp_path / "repo",
        program_path=program_path,
        data_root=tmp_path / "nas",
        local_state=tmp_path / "control",
        env_path=tmp_path / ".env",
        poll_seconds=60,
        plist_path=plist_path,
        load=False,
    )

    assert payload["loaded"] is False
    assert payload["plist_path"] == str(plist_path)
    assert plist_path.exists()
    assert Path(payload["stdout_path"]).parent == tmp_path / "control" / "research_programs" / "perpetual-macmini" / "logs"


def test_launch_agent_status_parses_launchctl_print(monkeypatch) -> None:
    monkeypatch.setattr(launchd.os, "uname", lambda: type("Uname", (), {"sysname": "Darwin"})())
    monkeypatch.setattr(launchd.os, "getuid", lambda: 501)

    def fake_run(command, **kwargs):
        assert command == ["launchctl", "print", "gui/501/com.trademl.research.perpetual"]
        return type(
            "Completed",
            (),
            {
                "returncode": 0,
                "stdout": "path = /tmp/agent.plist\nstate = running\nruns = 3\npid = 123\n",
                "stderr": "",
            },
        )()

    monkeypatch.setattr(launchd.subprocess, "run", fake_run)

    payload = launchd.launch_agent_status("com.trademl.research.perpetual")

    assert payload["loaded"] is True
    assert payload["state"] == "running"
    assert payload["runs"] == 3
    assert payload["pid"] == 123
    assert payload["plist_path"] == "/tmp/agent.plist"


def test_unload_launch_agent_bootouts_current_gui_label(monkeypatch) -> None:
    monkeypatch.setattr(launchd.os, "uname", lambda: type("Uname", (), {"sysname": "Darwin"})())
    monkeypatch.setattr(launchd.os, "getuid", lambda: 501)
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(launchd.subprocess, "run", fake_run)

    payload = launchd.unload_launch_agent("com.trademl.research.perpetual")

    assert payload["ok"] is True
    assert calls == [["launchctl", "bootout", "gui/501/com.trademl.research.perpetual"]]
