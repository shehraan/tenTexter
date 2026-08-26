"""Enforce immutable v1 fields.

Revision ID: 0003_immutable_fields
Revises: 3017e5ac5b2e
"""
from typing import Sequence

from alembic import op

revision: str = "0003_immutable_fields"
down_revision: str | None = "3017e5ac5b2e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TRIGGER task_instance_occurrence_key_immutable
        BEFORE UPDATE OF occurrence_key, task_definition_id ON task_instance
        WHEN OLD.occurrence_key IS NOT NEW.occurrence_key
          OR OLD.task_definition_id IS NOT NEW.task_definition_id
        BEGIN
          SELECT RAISE(ABORT, 'task occurrence identity is immutable');
        END
        """
    )
    op.execute(
        """
        CREATE TRIGGER outbox_immutable_fields
        BEFORE UPDATE OF final_text, transport, destination_kind, idempotency_key,
                         parent_terminal_policy, corrects_outbox_message_id ON outbox_message
        WHEN OLD.final_text IS NOT NEW.final_text
          OR OLD.transport IS NOT NEW.transport
          OR OLD.destination_kind IS NOT NEW.destination_kind
          OR OLD.idempotency_key IS NOT NEW.idempotency_key
          OR OLD.parent_terminal_policy IS NOT NEW.parent_terminal_policy
          OR OLD.corrects_outbox_message_id IS NOT NEW.corrects_outbox_message_id
        BEGIN
          SELECT RAISE(ABORT, 'outbox immutable field changed');
        END
        """
    )
    op.execute(
        """
        CREATE TRIGGER decision_parent_policy_immutable
        BEFORE UPDATE OF parent_terminal_policy ON decision_request
        WHEN OLD.parent_terminal_policy IS NOT NEW.parent_terminal_policy
        BEGIN
          SELECT RAISE(ABORT, 'decision parent terminal policy is immutable');
        END
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER decision_parent_policy_immutable")
    op.execute("DROP TRIGGER outbox_immutable_fields")
    op.execute("DROP TRIGGER task_instance_occurrence_key_immutable")
