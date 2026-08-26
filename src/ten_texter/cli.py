import argparse
import sys
from collections.abc import Sequence

from alembic.util.exc import CommandError
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError

from ten_texter import __version__
from ten_texter.config import Settings
from ten_texter.db.migrations import current_database_revision, upgrade_database
from ten_texter.logging import configure_logging


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ten-texter")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    config_parser = commands.add_parser("config", help="configuration operations")
    config_commands = config_parser.add_subparsers(dest="config_command", required=True)
    config_commands.add_parser("check", help="validate configuration")

    db_parser = commands.add_parser("db", help="database operations")
    db_commands = db_parser.add_subparsers(dest="db_command", required=True)
    db_commands.add_parser("upgrade", help="upgrade the database to migration head")
    db_commands.add_parser("current", help="show the current database revision")
    return parser


def _run(args: argparse.Namespace) -> int:
    settings = Settings()
    configure_logging(settings.log_level)

    if args.command == "config":
        print("configuration valid")
        return 0

    if args.db_command == "upgrade":
        revision = upgrade_database(settings)
        print(f"database revision: {revision or 'base'}")
        return 0

    revision = current_database_revision(settings)
    print(f"database revision: {revision or 'base'}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return _run(args)
    except (
        CommandError,
        OSError,
        RuntimeError,
        SQLAlchemyError,
        ValidationError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
