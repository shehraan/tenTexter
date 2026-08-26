"""Programmatic Alembic migration entry points."""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config

from ten_texter.config import Settings


def alembic_config(settings: Settings) -> Config:
    project_root = Path(__file__).resolve().parents[2]
    config = Config(project_root / "alembic.ini")
    config.set_main_option("script_location", str(project_root / "migrations"))
    config.set_main_option("sqlalchemy.url", settings.database_url)
    config.attributes["sqlite_busy_timeout_ms"] = settings.sqlite_busy_timeout_ms
    return config


def upgrade_to_head(settings: Settings) -> None:
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    command.upgrade(alembic_config(settings), "head")
