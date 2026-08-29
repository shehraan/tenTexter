from __future__ import annotations

import os
from dataclasses import dataclass


def _optional_int(value: str | None) -> int | None:
    return int(value) if value else None


def _bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    database_url: str = "sqlite:///ten_texter.db"
    owner_id: int | None = None
    owner_chat_id: int | None = None
    log_level: str = "INFO"
    sqlite_busy_timeout_ms: int = 5_000
    real_transports_enabled: bool = False
    telegram_bot_token: str | None = None
    beeper_base_url: str = "http://127.0.0.1:23373"
    beeper_token: str | None = None
    primary_model_url: str = "http://127.0.0.1:8001"
    validator_model_url: str = "http://127.0.0.1:8002"
    owner_timezone: str = "UTC"

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ
        defaults = cls()
        return cls(
            database_url=env.get("TEN_TEXTER_DATABASE_URL", defaults.database_url),
            owner_id=_optional_int(env.get("TEN_TEXTER_OWNER_ID")),
            owner_chat_id=_optional_int(env.get("TEN_TEXTER_OWNER_CHAT_ID")),
            log_level=env.get("TEN_TEXTER_LOG_LEVEL", defaults.log_level).upper(),
            sqlite_busy_timeout_ms=int(env.get("TEN_TEXTER_SQLITE_BUSY_TIMEOUT_MS", "5000")),
            real_transports_enabled=_bool(env.get("TEN_TEXTER_REAL_TRANSPORTS_ENABLED")),
            telegram_bot_token=env.get("TEN_TEXTER_TELEGRAM_BOT_TOKEN") or None,
            beeper_base_url=env.get("TEN_TEXTER_BEEPER_BASE_URL", defaults.beeper_base_url),
            beeper_token=env.get("TEN_TEXTER_BEEPER_TOKEN") or None,
            primary_model_url=env.get("TEN_TEXTER_PRIMARY_MODEL_URL", defaults.primary_model_url),
            validator_model_url=env.get("TEN_TEXTER_VALIDATOR_MODEL_URL", defaults.validator_model_url),
            owner_timezone=env.get("TEN_TEXTER_OWNER_TIMEZONE", defaults.owner_timezone),
        )

    def validate_runtime(self) -> None:
        if self.real_transports_enabled:
            if not self.telegram_bot_token or not self.beeper_token:
                raise ValueError("real transports require Telegram and Beeper credentials")
