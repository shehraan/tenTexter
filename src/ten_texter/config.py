"""Typed application configuration loaded from the environment."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Mapping


class ConfigurationError(ValueError):
    """Raised when environment configuration is invalid."""


@dataclass(frozen=True, slots=True)
class Settings:
    database_path: Path
    sqlite_busy_timeout_ms: int = 5_000
    log_level: str = "INFO"

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        cwd: Path | None = None,
    ) -> Settings:
        env = os.environ if environ is None else environ
        base_dir = Path.cwd() if cwd is None else cwd
        raw_path = env.get("TEN_TEXTER_DATABASE_PATH", "ten_texter.db")
        database_path = Path(raw_path).expanduser()
        if not database_path.is_absolute():
            database_path = base_dir / database_path

        raw_timeout = env.get("TEN_TEXTER_SQLITE_BUSY_TIMEOUT_MS", "5000")
        try:
            timeout = int(raw_timeout)
        except ValueError as exc:
            raise ConfigurationError(
                "TEN_TEXTER_SQLITE_BUSY_TIMEOUT_MS must be an integer"
            ) from exc
        if timeout <= 0:
            raise ConfigurationError(
                "TEN_TEXTER_SQLITE_BUSY_TIMEOUT_MS must be greater than zero"
            )

        log_level = env.get("TEN_TEXTER_LOG_LEVEL", "INFO").upper()
        valid_log_levels = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}
        if log_level not in valid_log_levels:
            raise ConfigurationError(
                "TEN_TEXTER_LOG_LEVEL must be one of "
                + ", ".join(sorted(valid_log_levels))
            )

        return cls(
            database_path=database_path.resolve(),
            sqlite_busy_timeout_ms=timeout,
            log_level=log_level,
        )

    @property
    def database_url(self) -> str:
        return f"sqlite+pysqlite:///{self.database_path.as_posix()}"

