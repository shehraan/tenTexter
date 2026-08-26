from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Phase 1 deliberately has no domain metadata. The full v1 schema belongs to Phase 2.
target_metadata = None


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        connection.exec_driver_sql("PRAGMA journal_mode=WAL")
        busy_timeout_ms = config.attributes.get("sqlite_busy_timeout_ms", 5_000)
        connection.exec_driver_sql(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
        connection.exec_driver_sql("PRAGMA foreign_keys=ON")
        # End SQLAlchemy's implicit transaction before Alembic establishes its
        # own migration transaction boundary.
        connection.commit()
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
