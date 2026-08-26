from pathlib import Path

import pytest

from ten_texter.config import ConfigurationError, Settings


def test_settings_load_typed_values_from_environment(tmp_path: Path) -> None:
    settings = Settings.from_env(
        {
            "TEN_TEXTER_DATABASE_PATH": "state/app.db",
            "TEN_TEXTER_SQLITE_BUSY_TIMEOUT_MS": "7500",
            "TEN_TEXTER_LOG_LEVEL": "debug",
        },
        cwd=tmp_path,
    )

    assert settings.database_path == (tmp_path / "state/app.db").resolve()
    assert settings.sqlite_busy_timeout_ms == 7500
    assert settings.log_level == "DEBUG"


@pytest.mark.parametrize("value", ["nope", "0", "-1"])
def test_settings_reject_invalid_busy_timeout(value: str, tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError):
        Settings.from_env(
            {"TEN_TEXTER_SQLITE_BUSY_TIMEOUT_MS": value}, cwd=tmp_path
        )

