from __future__ import annotations

import argparse
import time

from alembic import command
from alembic.config import Config

from ten_texter.app import Application
from ten_texter.config import Settings


def _run_outbox_worker(app: Application, *, once: bool) -> None:
    from sqlalchemy import select

    from ten_texter.beeper import BeeperDesktopAdapter
    from ten_texter.enums import OutboxStatus, Transport
    from ten_texter.model_clients import HTTPModelBackend
    from ten_texter.models import OutboxMessage
    from ten_texter.outbox import OutboxWorker
    from ten_texter.policy import PolicyRevalidator
    from ten_texter.telegram import TelegramBotAdapter
    from ten_texter.validator import (
        IndependentMessageValidator,
        MinimalValidatorContextProvider,
        OutboxValidatorGate,
    )

    if not app.settings.real_transports_enabled:
        raise ValueError("worker requires TEN_TEXTER_REAL_TRANSPORTS_ENABLED=true")
    validator = IndependentMessageValidator(
        HTTPModelBackend(app.settings.validator_model_url)
    )
    gate = OutboxValidatorGate(
        app.sessions,
        validator=validator,
        contexts=MinimalValidatorContextProvider(
            constraints=(
                "Do not make unsupported claims or unauthorized commitments.",
                "Respect the supplied message kind and disclosure scopes.",
            )
        ),
    )
    worker = OutboxWorker(
        app.sessions,
        revalidator=PolicyRevalidator(),
        validator=gate,
        adapters={
            Transport.BEEPER: BeeperDesktopAdapter(
                app.sessions,
                base_url=app.settings.beeper_base_url,
                access_token=app.settings.beeper_token,
                enabled=True,
            ),
            Transport.TELEGRAM: TelegramBotAdapter(
                token=app.settings.telegram_bot_token,
                enabled=True,
            ),
        },
    )
    while True:
        with app.sessions() as session:
            pending = list(
                session.scalars(
                    select(OutboxMessage.id)
                    .where(OutboxMessage.status == OutboxStatus.PENDING)
                    .order_by(OutboxMessage.id)
                )
            )
        for outbox_id in pending:
            worker.process(outbox_id)
        if once:
            return
        time.sleep(2)


def _alembic_config(database_url: str) -> Config:
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ten-texter")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check")
    subparsers.add_parser("migrate")
    worker = subparsers.add_parser("worker")
    worker.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    settings = Settings.from_env()
    if args.command == "migrate":
        command.upgrade(_alembic_config(settings.database_url), "head")
    elif args.command == "worker":
        _run_outbox_worker(Application.bootstrap(settings), once=args.once)
    else:
        app = Application.bootstrap(settings)
        with app.engine.connect() as connection:
            connection.exec_driver_sql("SELECT 1")
        print("tenTexter configuration and SQLite connection are healthy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
