from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from ten_texter.config import Settings
from ten_texter.db import create_sqlite_engine, session_factory
from ten_texter.logging import configure_logging


@dataclass(slots=True)
class Application:
    settings: Settings
    engine: Engine
    sessions: sessionmaker[Session]

    @classmethod
    def bootstrap(cls, settings: Settings | None = None) -> "Application":
        chosen = settings or Settings.from_env()
        chosen.validate_runtime()
        configure_logging(chosen.log_level)
        engine = create_sqlite_engine(
            chosen.database_url, busy_timeout_ms=chosen.sqlite_busy_timeout_ms
        )
        return cls(chosen, engine, session_factory(engine))
