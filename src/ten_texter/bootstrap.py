from dataclasses import dataclass

from ten_texter.config import Settings
from ten_texter.db.engine import DatabaseResources, create_database_resources
from ten_texter.logging import configure_logging


@dataclass(frozen=True)
class ApplicationBootstrap:
    settings: Settings
    database: DatabaseResources


def bootstrap_application(settings: Settings | None = None) -> ApplicationBootstrap:
    resolved_settings = settings or Settings()
    configure_logging(resolved_settings.log_level)
    database = create_database_resources(resolved_settings)
    return ApplicationBootstrap(settings=resolved_settings, database=database)

