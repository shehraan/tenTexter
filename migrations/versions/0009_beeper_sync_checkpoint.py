"""Add durable Beeper synchronization checkpoints.

Revision ID: 0009_beeper_sync_checkpoint
Revises: 0008_current_conversation_membership
"""
from typing import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0009_beeper_sync_checkpoint"
down_revision: str | None = "0008_current_conversation_membership"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "beeper_sync_checkpoint",
        sa.Column("checkpoint_key", sa.String(length=500), nullable=False),
        sa.Column("scope", sa.String(length=30), nullable=False),
        sa.Column("conversation_id", sa.Integer(), nullable=True),
        sa.Column("newest_cursor", sa.Text(), nullable=True),
        sa.Column("backfill_cursor", sa.Text(), nullable=True),
        sa.Column("bootstrap_cutoff_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("bootstrap_complete", sa.Boolean(), nullable=False),
        sa.Column("reconciliation_cursor", sa.Text(), nullable=True),
        sa.Column("reconciliation_cutoff_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_reconciled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("provider_activity_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "(scope = 'CHAT_FEED' AND checkpoint_key = 'chat-feed' AND conversation_id IS NULL) OR "
            "(scope = 'CONVERSATION' AND checkpoint_key <> 'chat-feed' AND conversation_id IS NOT NULL)",
            name="ck_beeper_sync_checkpoint_scope",
        ),
        sa.CheckConstraint(
            "(reconciliation_cursor IS NULL AND reconciliation_cutoff_at IS NULL) OR "
            "(reconciliation_cursor IS NOT NULL AND reconciliation_cutoff_at IS NOT NULL)",
            name="ck_beeper_sync_reconciliation_pair",
        ),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversation.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("checkpoint_key"),
        sa.UniqueConstraint("conversation_id"),
    )
    op.create_index(
        "ix_beeper_sync_checkpoint_reconcile",
        "beeper_sync_checkpoint",
        ["scope", "bootstrap_complete", "last_reconciled_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_beeper_sync_checkpoint_reconcile",
        table_name="beeper_sync_checkpoint",
    )
    op.drop_table("beeper_sync_checkpoint")
