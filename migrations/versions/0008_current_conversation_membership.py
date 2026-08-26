"""Distinguish current conversation membership from retained history.

Revision ID: 0008_current_conversation_membership
Revises: 0007_cross_row_integrity
"""
from typing import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0008_current_conversation_membership"
down_revision: str | None = "0007_cross_row_integrity"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("DROP TRIGGER task_participant_requires_pinned_identity_insert")
    op.execute("DROP TRIGGER task_participant_requires_pinned_identity_update")
    with op.batch_alter_table("conversation_participant") as batch:
        batch.add_column(sa.Column("is_current", sa.Boolean(), nullable=False, server_default=sa.true()))
        batch.add_column(sa.Column("left_at", sa.DateTime(timezone=True), nullable=True))
        batch.create_check_constraint(
            "ck_conversation_participant_current_left_at",
            "(is_current = 1 AND left_at IS NULL) OR (is_current = 0 AND left_at IS NOT NULL)",
        )
        batch.create_index("ix_conversation_participant_current", ["conversation_id", "is_current"])
    for operation, suffix in (("INSERT", "insert"), ("UPDATE OF person_id, conversation_id", "update")):
        op.execute(
            f"""
            CREATE TRIGGER task_participant_requires_pinned_identity_{suffix}
            BEFORE {operation} ON task_participant
            WHEN NOT EXISTS (
              SELECT 1 FROM conversation_participant cp
              JOIN identity i ON i.id = cp.identity_id
              WHERE cp.conversation_id = NEW.conversation_id
                AND cp.is_current = 1 AND i.person_id = NEW.person_id
            )
            BEGIN
              SELECT RAISE(ABORT, 'task participant person is not currently in pinned conversation');
            END
            """
        )


def downgrade() -> None:
    op.execute("DROP TRIGGER task_participant_requires_pinned_identity_insert")
    op.execute("DROP TRIGGER task_participant_requires_pinned_identity_update")
    with op.batch_alter_table("conversation_participant") as batch:
        batch.drop_index("ix_conversation_participant_current")
        batch.drop_constraint("ck_conversation_participant_current_left_at", type_="check")
        batch.drop_column("left_at")
        batch.drop_column("is_current")
    for operation, suffix in (("INSERT", "insert"), ("UPDATE OF person_id, conversation_id", "update")):
        op.execute(
            f"""
            CREATE TRIGGER task_participant_requires_pinned_identity_{suffix}
            BEFORE {operation} ON task_participant
            WHEN NOT EXISTS (
              SELECT 1 FROM conversation_participant cp
              JOIN identity i ON i.id = cp.identity_id
              WHERE cp.conversation_id = NEW.conversation_id AND i.person_id = NEW.person_id
            )
            BEGIN
              SELECT RAISE(ABORT, 'task participant person is not in pinned conversation');
            END
            """
        )
