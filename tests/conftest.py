from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy.orm import Session

from ten_texter.config import Settings
from ten_texter.db import create_sqlite_engine, session_factory


@pytest.fixture
def db_session(tmp_path: Path) -> Iterator[Session]:
    database = tmp_path / "test.db"
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database}")
    command.upgrade(config, "head")
    engine = create_sqlite_engine(Settings(database_url=f"sqlite:///{database}").database_url)
    factory = session_factory(engine)
    with factory() as session:
        yield session
        session.rollback()
    engine.dispose()
