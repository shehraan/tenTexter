from pathlib import Path

from ten_texter.cli import main


def test_cli_migrates_and_checks_database(
    tmp_path: Path, monkeypatch,
) -> None:
    database_path = tmp_path / "app.db"
    monkeypatch.setenv("TEN_TEXTER_DATABASE_PATH", str(database_path))

    assert main(["db", "upgrade"]) == 0
    assert main(["check"]) == 0
    assert database_path.exists()

