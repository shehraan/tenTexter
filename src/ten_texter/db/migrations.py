from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext

from ten_texter.config import Settings
from ten_texter.db.engine import create_database_resources

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _alembic_config() -> Config:
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    return config


def upgrade_database(settings: Settings, revision: str = "head") -> str | None:
    resources = create_database_resources(settings)
    try:
        with resources.engine.begin() as connection:
            config = _alembic_config()
            config.attributes["connection"] = connection
            command.upgrade(config, revision)
            return MigrationContext.configure(connection).get_current_revision()
    finally:
        resources.dispose()


def current_database_revision(settings: Settings) -> str | None:
    resources = create_database_resources(settings)
    try:
        with resources.engine.connect() as connection:
            return MigrationContext.configure(connection).get_current_revision()
    finally:
        resources.dispose()
