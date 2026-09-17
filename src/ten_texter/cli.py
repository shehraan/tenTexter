from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

from alembic import command
from alembic.config import Config

from ten_texter.app import Application
from ten_texter.config import Settings


def _run_outbox_worker(app: Application, *, once: bool) -> None:
    from sqlalchemy import select

    from ten_texter.beeper import BeeperDesktopAdapter
    from ten_texter.domain import utc_now
    from ten_texter.enums import OutboxStatus, Transport
    from ten_texter.model_clients import HTTPModelBackend
    from ten_texter.models import OutboxMessage
    from ten_texter.outbox import OutboxWorker
    from ten_texter.policy import DatabaseContextProvider, PolicyRevalidator
    from ten_texter.telegram import TelegramBotAdapter
    from ten_texter.validator import (
        IndependentMessageValidator,
        DatabaseValidatorContextProvider,
        OutboxValidatorGate,
    )

    if not app.settings.real_transports_enabled:
        raise ValueError("worker requires TEN_TEXTER_REAL_TRANSPORTS_ENABLED=true")
    validator = IndependentMessageValidator(
        HTTPModelBackend(app.settings.validator_model_url)
    )
    facts = DatabaseContextProvider()
    gate = OutboxValidatorGate(
        app.sessions,
        validator=validator,
        contexts=DatabaseValidatorContextProvider(
            app.sessions,
            facts=facts,
            owner_chat_id=getattr(app.settings, "owner_chat_id", None),
        ),
        owner_chat_id=getattr(app.settings, "owner_chat_id", None),
    )
    worker = OutboxWorker(
        app.sessions,
        revalidator=PolicyRevalidator(
            facts=facts,
            owner_chat_id=getattr(app.settings, "owner_chat_id", None),
        ),
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
        raise_validation_errors=True,
        owner_chat_id=getattr(app.settings, "owner_chat_id", None),
    )
    while True:
        with app.sessions() as session:
            expired_sending = list(
                session.scalars(
                    select(OutboxMessage.id)
                    .where(
                        OutboxMessage.status == OutboxStatus.SENDING,
                        OutboxMessage.lease_expires_at <= utc_now(),
                    )
                    .order_by(OutboxMessage.id)
                )
            )
        for outbox_id in expired_sending:
            worker.reclaim_expired(outbox_id)
        with app.sessions() as session:
            reconciling = list(
                session.scalars(
                    select(OutboxMessage.id)
                    .where(OutboxMessage.status == OutboxStatus.RECONCILING)
                    .order_by(OutboxMessage.id)
                )
            )
            pending = list(
                session.scalars(
                    select(OutboxMessage.id)
                    .where(OutboxMessage.status == OutboxStatus.PENDING)
                    .order_by(OutboxMessage.id)
                )
            )
        for outbox_id in reconciling:
            worker.reconcile(outbox_id)
        for outbox_id in pending:
            worker.process(outbox_id)
        if once:
            return
        time.sleep(2)


def _alembic_config(database_url: str) -> Config:
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def _run_agent(app: Application, *, once: bool) -> None:
    from ten_texter.runtime import build_runtime

    runtime = build_runtime(app)
    if once:
        runtime.run_once()
    else:
        runtime.run_forever()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ten-texter")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check")
    subparsers.add_parser("migrate")
    worker = subparsers.add_parser("worker")
    worker.add_argument("--once", action="store_true")
    runtime = subparsers.add_parser("run")
    runtime.add_argument("--once", action="store_true")
    subparsers.add_parser("identities")
    live_tennis = subparsers.add_parser(
        "live-tennis-test",
        help="run one guarded real tennis flow against the alternate WhatsApp account",
    )
    live_tennis.add_argument(
        "--confirm-real-send",
        action="store_true",
        help="confirm that the test may send a real message to the hard-coded test account",
    )
    live_tennis.add_argument(
        "--resume-task-id",
        type=int,
        help="poll an existing allowlisted test task instead of creating a new send",
    )
    live_tennis.add_argument("--poll-rounds", type=int, default=1)
    live_tennis.add_argument("--poll-delay-seconds", type=float, default=0.0)
    link_identity = subparsers.add_parser("link-identity")
    link_identity.add_argument("--identity-id", type=int, required=True)
    link_identity.add_argument("--person-id", type=int, required=True)
    args = parser.parse_args(argv)
    settings = Settings.from_env()
    if args.command == "migrate":
        command.upgrade(_alembic_config(settings.database_url), "head")
    elif args.command == "worker":
        _run_outbox_worker(Application.bootstrap(settings), once=args.once)
    elif args.command == "run":
        _run_agent(Application.bootstrap(settings), once=args.once)
    elif args.command == "identities":
        from ten_texter.identity import IdentityLinkingService

        app = Application.bootstrap(settings)
        with app.sessions() as session:
            print(
                json.dumps(
                    [asdict(record) for record in IdentityLinkingService(session).inspect()],
                    indent=2,
                    sort_keys=True,
                )
            )
    elif args.command == "live-tennis-test":
        from ten_texter.live_tennis import (
            LiveTennisTestError,
            poll_live_tennis_test,
            run_live_tennis_test,
        )

        try:
            app = Application.bootstrap(settings)
            if args.resume_task_id is None:
                report = run_live_tennis_test(
                    app,
                    confirm_real_send=args.confirm_real_send,
                    poll_rounds=args.poll_rounds,
                    poll_delay_seconds=args.poll_delay_seconds,
                )
            else:
                report = poll_live_tennis_test(
                    app,
                    task_instance_id=args.resume_task_id,
                    confirm_real_send=args.confirm_real_send,
                    poll_rounds=args.poll_rounds,
                    poll_delay_seconds=args.poll_delay_seconds,
                )
        except LiveTennisTestError as exc:
            print(json.dumps({"ok": False, "error": str(exc)}, indent=2, sort_keys=True))
            return 1
        print(json.dumps(asdict(report), indent=2, sort_keys=True))
        return 0 if report.ok else 1
    elif args.command == "link-identity":
        from ten_texter.identity import IdentityLinkingService

        app = Application.bootstrap(settings)
        with app.sessions.begin() as session:
            result = IdentityLinkingService(session).link(
                identity_id=args.identity_id,
                target_person_id=args.person_id,
            )
            print(json.dumps(asdict(result), sort_keys=True))
    else:
        app = Application.bootstrap(settings)
        with app.engine.connect() as connection:
            connection.exec_driver_sql("SELECT 1")
        print("tenTexter configuration and SQLite connection are healthy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
