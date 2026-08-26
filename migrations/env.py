from alembic import context

from ten_texter.config import Settings
from ten_texter.db.base import Base
from ten_texter.db.engine import create_database_resources

config = context.config
target_metadata = Base.metadata


def _configure(connection=None, url: str | None = None) -> None:
    options = {
        "target_metadata": target_metadata,
        "compare_type": True,
        "render_as_batch": True,
    }
    if connection is not None:
        context.configure(connection=connection, **options)
    else:
        context.configure(
            url=url,
            literal_binds=True,
            dialect_opts={"paramstyle": "named"},
            **options,
        )


def run_migrations_offline() -> None:
    settings = Settings()
    _configure(url=settings.database_url)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    supplied_connection = config.attributes.get("connection")
    if supplied_connection is not None:
        _configure(connection=supplied_connection)
        with context.begin_transaction():
            context.run_migrations()
        return

    resources = create_database_resources(Settings())
    try:
        with resources.engine.connect() as connection:
            _configure(connection=connection)
            with context.begin_transaction():
                context.run_migrations()
    finally:
        resources.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
