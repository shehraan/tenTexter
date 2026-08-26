"""Allow multiple prompts per decision request.

Revision ID: 0005_decision_prompt_cardinality
Revises: 0004_revision_immutability
"""
from typing import Sequence

from alembic import op
import sqlalchemy as sa

revision: str = "0005_decision_prompt_cardinality"
down_revision: str | None = "0004_revision_immutability"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "decision_request_prompt_new",
        sa.Column("decision_request_id", sa.Integer(), nullable=False),
        sa.Column("outbox_message_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["decision_request_id"], ["decision_request.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["outbox_message_id"], ["outbox_message.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("decision_request_id", "outbox_message_id"),
        sa.UniqueConstraint("outbox_message_id"),
    )
    op.execute(
        "INSERT INTO decision_request_prompt_new (decision_request_id, outbox_message_id) "
        "SELECT decision_request_id, outbox_message_id FROM decision_request_prompt"
    )
    op.drop_table("decision_request_prompt")
    op.rename_table("decision_request_prompt_new", "decision_request_prompt")


def downgrade() -> None:
    # Downgrade is intentionally lossless only when each decision has one prompt.
    op.create_table(
        "decision_request_prompt_old",
        sa.Column("decision_request_id", sa.Integer(), nullable=False),
        sa.Column("outbox_message_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["decision_request_id"], ["decision_request.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["outbox_message_id"], ["outbox_message.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("decision_request_id"),
        sa.UniqueConstraint("outbox_message_id"),
    )
    op.execute(
        "INSERT INTO decision_request_prompt_old (decision_request_id, outbox_message_id) "
        "SELECT decision_request_id, outbox_message_id FROM decision_request_prompt"
    )
    op.drop_table("decision_request_prompt")
    op.rename_table("decision_request_prompt_old", "decision_request_prompt")
