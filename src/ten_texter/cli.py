"""Command-line interface for the agent application."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from sqlalchemy import text

from ten_texter import __version__
from ten_texter.app import bootstrap
from ten_texter.config import Settings
from ten_texter.migrations import upgrade_to_head


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ten-texter")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    db_parser = subparsers.add_parser("db", help="database operations")
    db_subparsers = db_parser.add_subparsers(dest="db_command", required=True)
    db_subparsers.add_parser("upgrade", help="migrate the database to head")
    subparsers.add_parser("check", help="check application database connectivity")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.from_env()
    if args.command == "db" and args.db_command == "upgrade":
        upgrade_to_head(settings)
        return 0
    if args.command == "check":
        app = bootstrap(settings)
        with app.engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        app.engine.dispose()
        return 0
    return 2

