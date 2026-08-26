from pathlib import Path

from sqlalchemy import text

from ten_texter.config import Settings
from ten_texter.db import create_sqlite_engine


def test_sqlite_initializes_required_pragmas(tmp_path: Path) -> None:
    settings = Settings(database_path=tmp_path / "app.db", sqlite_busy_timeout_ms=4321)
    engine = create_sqlite_engine(settings)

    with engine.connect() as connection:
        journal_mode = connection.execute(text("PRAGMA journal_mode")).scalar_one()
        busy_timeout = connection.execute(text("PRAGMA busy_timeout")).scalar_one()
        foreign_keys = connection.execute(text("PRAGMA foreign_keys")).scalar_one()

    assert journal_mode == "wal"
    assert busy_timeout == 4321
    assert foreign_keys == 1

