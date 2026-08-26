"""Protect exact inbound revision content.

Revision ID: 0004_revision_immutability
Revises: 0003_immutable_fields
"""
from typing import Sequence

from alembic import op

revision: str = "0004_revision_immutability"
down_revision: str | None = "0003_immutable_fields"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TRIGGER message_revision_exact_content_immutable
        BEFORE UPDATE OF message_id, provider_revision_key, provider_sequence,
                         provider_event_at, content_hash, is_deleted, text,
                         received_at, content_support ON message_revision
        WHEN OLD.message_id IS NOT NEW.message_id
          OR OLD.provider_revision_key IS NOT NEW.provider_revision_key
          OR OLD.provider_sequence IS NOT NEW.provider_sequence
          OR OLD.provider_event_at IS NOT NEW.provider_event_at
          OR OLD.content_hash IS NOT NEW.content_hash
          OR OLD.is_deleted IS NOT NEW.is_deleted
          OR OLD.text IS NOT NEW.text
          OR OLD.received_at IS NOT NEW.received_at
          OR OLD.content_support IS NOT NEW.content_support
        BEGIN
          SELECT RAISE(ABORT, 'message revision exact content is immutable');
        END
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER message_revision_exact_content_immutable")
