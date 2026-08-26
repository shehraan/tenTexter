"""Application bootstrap."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from ten_texter.config import Settings
from ten_texter.db import create_session_factory, create_sqlite_engine
from ten_texter.logging import configure_logging


@dataclass(frozen=True, slots=True)
class Application:
    settings: Settings
    engine: Engine
    sessions: sessionmaker[Session]


def bootstrap(settings: Settings | None = None) -> Application:
    resolved_settings = settings or Settings.from_env()
    configure_logging(resolved_settings.log_level)
    engine = create_sqlite_engine(resolved_settings)
    return Application(
        settings=resolved_settings,
        engine=engine,
        sessions=create_session_factory(engine),
    )

