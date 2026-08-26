from pathlib import Path

from ten_texter.config import Settings
from ten_texter.db.engine import create_database_resources


def _pragma_value(settings: Settings, pragma: str) -> object:
    resources = create_database_resources(settings)
    try:
        with resources.engine.connect() as connection:
            return connection.exec_driver_sql(f"PRAGMA {pragma}").scalar_one()
    finally:
        resources.dispose()


def test_database_bootstrap_creates_file_and_establishes_wal(
    settings: Settings, database_path: Path
) -> None:
    assert not database_path.exists()

    resources = create_database_resources(settings)
    try:
        with resources.engine.connect() as connection:
            assert connection.exec_driver_sql("SELECT 1").scalar_one() == 1
            journal_mode = connection.exec_driver_sql(
                "PRAGMA journal_mode"
            ).scalar_one()
            assert journal_mode == "wal"
    finally:
        resources.dispose()

    assert database_path.is_file()


def test_every_new_connection_configures_safety_pragmas(settings: Settings) -> None:
    assert _pragma_value(settings, "foreign_keys") == 1
    assert _pragma_value(settings, "busy_timeout") == 5_000

    overridden = Settings(
        database_url=settings.database_url,
        sqlite_busy_timeout_ms=1_234,
    )
    assert _pragma_value(overridden, "foreign_keys") == 1
    assert _pragma_value(overridden, "busy_timeout") == 1_234


def test_session_factory_is_bound_to_bootstrapped_engine(settings: Settings) -> None:
    resources = create_database_resources(settings)
    try:
        session = resources.session_factory()
        try:
            assert session.bind is resources.engine
        finally:
            session.close()
    finally:
        resources.dispose()
