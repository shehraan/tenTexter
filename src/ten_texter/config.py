from enum import StrEnum

from pydantic import PositiveInt, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    """Process configuration loaded from the environment at bootstrap."""

    model_config = SettingsConfigDict(
        env_prefix="TENTEXTER_",
        env_file=None,
        case_sensitive=False,
        extra="ignore",
        frozen=True,
    )

    database_url: str = "sqlite+pysqlite:///./ten_texter.db"
    sqlite_busy_timeout_ms: PositiveInt = 5_000
    log_level: LogLevel = LogLevel.INFO

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, value: str) -> str:
        try:
            url = make_url(value)
        except ArgumentError as exc:
            raise ValueError("database_url must be a valid SQLAlchemy URL") from exc

        if url.get_backend_name() != "sqlite":
            raise ValueError("database_url must use SQLite")
        if not url.database or url.database == ":memory:":
            raise ValueError(
                "database_url must reference a file-backed SQLite database"
            )
        return value
