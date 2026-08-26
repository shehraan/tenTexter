from pathlib import Path

import pytest

from ten_texter import __version__
from ten_texter.cli import main


def test_config_check_does_not_create_database(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database_path = tmp_path / "not_created.db"
    monkeypatch.setenv(
        "TENTEXTER_DATABASE_URL", f"sqlite+pysqlite:///{database_path}"
    )

    assert main(["config", "check"]) == 0
    assert capsys.readouterr().out == "configuration valid\n"
    assert not database_path.exists()


def test_db_upgrade_and_current(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database_path = tmp_path / "cli.db"
    monkeypatch.setenv(
        "TENTEXTER_DATABASE_URL", f"sqlite+pysqlite:///{database_path}"
    )

    assert main(["db", "upgrade"]) == 0
    assert capsys.readouterr().out == "database revision: 0001_phase1_baseline\n"
    assert main(["db", "current"]) == 0
    assert capsys.readouterr().out == "database revision: 0001_phase1_baseline\n"


def test_invalid_configuration_returns_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("TENTEXTER_DATABASE_URL", "postgresql://localhost/database")

    assert main(["config", "check"]) == 1
    assert "database_url must use SQLite" in capsys.readouterr().err


def test_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    assert "configuration operations" in capsys.readouterr().out


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])

    assert exc_info.value.code == 0
    assert capsys.readouterr().out == f"{__version__}\n"

