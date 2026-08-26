import pytest
from pydantic import ValidationError

from ten_texter.config import LogLevel, Settings


def test_settings_defaults() -> None:
    settings = Settings()

    assert settings.database_url == "sqlite+pysqlite:///./ten_texter.db"
    assert settings.sqlite_busy_timeout_ms == 5_000
    assert settings.log_level is LogLevel.INFO


def test_settings_load_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TENTEXTER_DATABASE_URL", "sqlite:////tmp/overridden.db")
    monkeypatch.setenv("TENTEXTER_SQLITE_BUSY_TIMEOUT_MS", "2750")
    monkeypatch.setenv("TENTEXTER_LOG_LEVEL", "WARNING")

    settings = Settings()

    assert settings.database_url == "sqlite:////tmp/overridden.db"
    assert settings.sqlite_busy_timeout_ms == 2_750
    assert settings.log_level is LogLevel.WARNING


@pytest.mark.parametrize(
    "database_url",
    ["postgresql://localhost/ten_texter", "sqlite://", "sqlite:///:memory:", "bad"],
)
def test_settings_reject_non_file_sqlite_urls(database_url: str) -> None:
    with pytest.raises(ValidationError):
        Settings(database_url=database_url)


@pytest.mark.parametrize("timeout", [0, -1])
def test_settings_reject_non_positive_timeout(timeout: int) -> None:
    with pytest.raises(ValidationError):
        Settings(sqlite_busy_timeout_ms=timeout)


def test_settings_reject_unknown_log_level() -> None:
    with pytest.raises(ValidationError):
        Settings(log_level="VERBOSE")


def test_settings_are_immutable() -> None:
    settings = Settings()

    with pytest.raises(ValidationError):
        settings.log_level = LogLevel.DEBUG  # type: ignore[misc]

