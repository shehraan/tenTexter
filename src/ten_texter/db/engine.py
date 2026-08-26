from collections.abc import Callable
from dataclasses import dataclass
from sqlite3 import Connection as SQLiteConnection
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from ten_texter.config import Settings


@dataclass(frozen=True)
class DatabaseResources:
    engine: Engine
    session_factory: sessionmaker[Session]

    def dispose(self) -> None:
        self.engine.dispose()


def _connection_initializer(
    busy_timeout_ms: int,
) -> Callable[[Any, Any], None]:
    def initialize(dbapi_connection: Any, _connection_record: Any) -> None:
        if not isinstance(dbapi_connection, SQLiteConnection):
            raise RuntimeError("SQLite engine returned a non-SQLite connection")

        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute(f"PRAGMA busy_timeout={busy_timeout_ms:d}")

            cursor.execute("PRAGMA foreign_keys")
            foreign_keys = cursor.fetchone()[0]
            cursor.execute("PRAGMA busy_timeout")
            actual_busy_timeout = cursor.fetchone()[0]
        finally:
            cursor.close()

        if foreign_keys != 1:
            raise RuntimeError("failed to enable SQLite foreign key enforcement")
        if actual_busy_timeout != busy_timeout_ms:
            raise RuntimeError("failed to configure SQLite busy timeout")

    return initialize


def _establish_wal(engine: Engine) -> None:
    """Establish and verify WAL once as part of database bootstrap."""

    with engine.connect() as connection:
        established_mode = connection.exec_driver_sql(
            "PRAGMA journal_mode=WAL"
        ).scalar_one()
        verified_mode = connection.exec_driver_sql("PRAGMA journal_mode").scalar_one()

    if str(established_mode).lower() != "wal" or str(verified_mode).lower() != "wal":
        raise RuntimeError("failed to establish SQLite WAL journal mode")


def create_database_resources(settings: Settings) -> DatabaseResources:
    engine = create_engine(
        settings.database_url,
        connect_args={"timeout": settings.sqlite_busy_timeout_ms / 1_000},
    )
    event.listen(
        engine,
        "connect",
        _connection_initializer(settings.sqlite_busy_timeout_ms),
    )

    try:
        _establish_wal(engine)
    except Exception:
        engine.dispose()
        raise

    return DatabaseResources(
        engine=engine,
        session_factory=sessionmaker(
            bind=engine,
            class_=Session,
            autoflush=False,
            expire_on_commit=False,
        ),
    )
