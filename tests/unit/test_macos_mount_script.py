from __future__ import annotations

from pathlib import Path


def test_macos_mount_script_uses_keychain_without_committed_password() -> None:
    script = Path("ops/macos_mount_trademl_nas.sh").read_text(encoding="utf-8")

    assert "security find-generic-password" in script
    assert "TRADEML_NAS_PASSWORD" in script
    assert "Asikuna" not in script
    assert "siejamgio" not in script
    assert "z20235" not in script


def test_macos_launchagent_installer_is_reproducible_without_committed_password() -> None:
    script = Path("ops/install_macos_nas_launchagent.sh").read_text(encoding="utf-8")

    assert "launchctl bootstrap" in script
    assert "launchctl kickstart" in script
    assert "security add-generic-password" in script
    assert "TRADEML_NAS_PASSWORD" in script
    assert "com.trademl.mount-nas" in script
    assert "Asikuna" not in script
    assert "siejamgio" not in script
    assert "z20235" not in script
