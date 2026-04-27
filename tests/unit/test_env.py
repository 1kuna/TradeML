from __future__ import annotations

import os
from pathlib import Path

from trademl.env import load_dotenv, read_env_file, write_env_file


def test_env_file_roundtrip_skips_comments_and_malformed_lines(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text(
        "\n".join(
            [
                "# comment",
                "ALPACA_API_KEY=abc",
                "MALFORMED",
                "ALPACA_API_SECRET=def=ghi",
                "",
            ]
        ),
        encoding="utf-8",
    )

    assert read_env_file(path) == {
        "ALPACA_API_KEY": "abc",
        "ALPACA_API_SECRET": "def=ghi",
    }


def test_write_env_file_sorts_by_default_and_can_preserve_order(tmp_path: Path) -> None:
    sorted_path = tmp_path / "sorted.env"
    preserved_path = tmp_path / "preserved.env"

    write_env_file(sorted_path, {"B": "2", "A": "1"})
    write_env_file(preserved_path, {"B": "2", "A": "1"}, sort_keys=False)

    assert sorted_path.read_text(encoding="utf-8") == "A=1\nB=2\n"
    assert preserved_path.read_text(encoding="utf-8") == "B=2\nA=1\n"


def test_load_dotenv_respects_existing_environment(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / ".env"
    path.write_text("TRADEML_TEST_ENV=from-file\nNEW_VALUE=loaded\n", encoding="utf-8")
    monkeypatch.setenv("TRADEML_TEST_ENV", "existing")
    monkeypatch.delenv("NEW_VALUE", raising=False)

    load_dotenv(path)

    assert os.environ["TRADEML_TEST_ENV"] == "existing"
    assert os.environ["NEW_VALUE"] == "loaded"
