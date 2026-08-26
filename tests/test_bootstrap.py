from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import text

from ten_texter.app import Application
from ten_texter.config import Settings


def test_settings_and_sqlite_bootstrap(tmp_path: Path) -> None:
    database = tmp_path / "app.db"
    app = Application.bootstrap(Settings(database_url=f"sqlite:///{database}"))
    with app.engine.connect() as connection:
        assert connection.execute(text("PRAGMA foreign_keys")).scalar_one() == 1
        assert connection.execute(text("PRAGMA journal_mode")).scalar_one().lower() == "wal"


def test_empty_database_migrates_to_head(tmp_path: Path) -> None:
    database = tmp_path / "migrated.db"
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database}")
    command.upgrade(config, "head")
    app = Application.bootstrap(Settings(database_url=f"sqlite:///{database}"))
    with app.engine.connect() as connection:
        assert connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one() == ScriptDirectory.from_config(config).get_current_head()
