"""Allow durable records for structurally unsupported Telegram updates.

Revision ID: 0010_nullable_telegram_rejection_metadata
Revises: 0009_beeper_sync_checkpoint
"""
from typing import Sequence

from alembic import op


revision: str = "0010_nullable_telegram_rejection_metadata"
down_revision: str | None = "0009_beeper_sync_checkpoint"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # SQLite's batch rewrite otherwise cannot drop a table referenced by
    # decision_request while foreign-key enforcement is enabled. Alembic's
    # SQLite migrations run with non-transactional DDL, so temporarily disable
    # enforcement for this atomic schema rewrite and restore it afterward.
    connection = op.get_bind()
    connection.exec_driver_sql("PRAGMA foreign_keys=OFF")
    try:
        with op.batch_alter_table("telegram_update") as batch_op:
            batch_op.alter_column("sender_user_id", nullable=True)
            batch_op.alter_column("chat_id", nullable=True)
    finally:
        connection.exec_driver_sql("PRAGMA foreign_keys=ON")


def downgrade() -> None:
    connection = op.get_bind()
    connection.exec_driver_sql("PRAGMA foreign_keys=OFF")
    try:
        with op.batch_alter_table("telegram_update") as batch_op:
            batch_op.alter_column("sender_user_id", nullable=False)
            batch_op.alter_column("chat_id", nullable=False)
    finally:
        connection.exec_driver_sql("PRAGMA foreign_keys=ON")
