"""Enforce SQLite cross-row and terminal invariants.

Revision ID: 0007_cross_row_integrity
Revises: 0006_beeper_sort_key
"""
from typing import Sequence

from alembic import op

revision: str = "0007_cross_row_integrity"
down_revision: str | None = "0006_beeper_sort_key"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TRIGGER task_terminal_status_immutable
        BEFORE UPDATE OF status ON task_instance
        WHEN OLD.status <> 'ACTIVE' AND NEW.status <> OLD.status
        BEGIN
          SELECT RAISE(ABORT, 'terminal task status is immutable');
        END
        """
    )
    op.execute(
        """
        CREATE TRIGGER task_participant_requires_pinned_identity_insert
        BEFORE INSERT ON task_participant
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
    op.execute(
        """
        CREATE TRIGGER task_participant_requires_pinned_identity_update
        BEFORE UPDATE OF person_id, conversation_id ON task_participant
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
    op.execute(
        """
        CREATE TRIGGER contact_rule_core_immutable
        BEFORE UPDATE OF person_id, scope, type, value, source, strength, topic_key,
                         task_definition_id, task_instance_id, created_at, expires_at,
                         overrides_contact_rule_id ON contact_rule
        WHEN OLD.person_id IS NOT NEW.person_id
          OR OLD.scope IS NOT NEW.scope
          OR OLD.type IS NOT NEW.type
          OR OLD.value IS NOT NEW.value
          OR OLD.source IS NOT NEW.source
          OR OLD.strength IS NOT NEW.strength
          OR OLD.topic_key IS NOT NEW.topic_key
          OR OLD.task_definition_id IS NOT NEW.task_definition_id
          OR OLD.task_instance_id IS NOT NEW.task_instance_id
          OR OLD.created_at IS NOT NEW.created_at
          OR OLD.expires_at IS NOT NEW.expires_at
          OR OLD.overrides_contact_rule_id IS NOT NEW.overrides_contact_rule_id
        BEGIN
          SELECT RAISE(ABORT, 'contact rule core is immutable; revoke and replace');
        END
        """
    )
    op.execute(
        """
        CREATE TRIGGER contact_rule_override_integrity_insert
        BEFORE INSERT ON contact_rule
        WHEN NEW.overrides_contact_rule_id IS NOT NULL AND (
          NOT EXISTS (
            SELECT 1 FROM contact_rule original
            WHERE original.id = NEW.overrides_contact_rule_id
              AND original.person_id = NEW.person_id
          )
          OR NEW.type <> 'ALLOW'
          OR NEW.source <> 'USER_CONFIGURED'
          OR (CASE NEW.scope WHEN 'GLOBAL' THEN 0 WHEN 'TOPIC' THEN 1
                WHEN 'TASK_DEFINITION' THEN 2 WHEN 'TASK_INSTANCE' THEN 3 END)
             <= (SELECT CASE original.scope WHEN 'GLOBAL' THEN 0 WHEN 'TOPIC' THEN 1
                    WHEN 'TASK_DEFINITION' THEN 2 WHEN 'TASK_INSTANCE' THEN 3 END
                 FROM contact_rule original WHERE original.id = NEW.overrides_contact_rule_id)
          OR EXISTS (
            SELECT 1 FROM contact_rule original
            WHERE original.id = NEW.overrides_contact_rule_id
              AND original.scope IN ('TOPIC', 'TASK_DEFINITION')
              AND NEW.scope <> 'TASK_INSTANCE'
          )
          OR EXISTS (
            SELECT 1 FROM contact_rule original
            JOIN task_instance ti ON ti.id = NEW.task_instance_id
            WHERE original.id = NEW.overrides_contact_rule_id
              AND original.scope = 'TOPIC'
              AND ti.topic_key <> original.topic_key
          )
          OR EXISTS (
            SELECT 1 FROM contact_rule original
            JOIN task_instance ti ON ti.id = NEW.task_instance_id
            WHERE original.id = NEW.overrides_contact_rule_id
              AND original.scope = 'TASK_DEFINITION'
              AND ti.task_definition_id IS NOT original.task_definition_id
          )
        )
        BEGIN
          SELECT RAISE(ABORT, 'invalid contact rule override');
        END
        """
    )
    op.execute(
        """
        CREATE TRIGGER beeper_destination_matches_transport
        BEFORE INSERT ON beeper_outbox_destination
        WHEN NOT EXISTS (
          SELECT 1 FROM outbox_message o
          WHERE o.id = NEW.outbox_message_id
            AND o.transport = 'BEEPER' AND o.destination_kind = 'PARTICIPANT'
        ) OR EXISTS (
          SELECT 1 FROM telegram_outbox_destination t
          WHERE t.outbox_message_id = NEW.outbox_message_id
        )
        BEGIN
          SELECT RAISE(ABORT, 'Beeper destination does not match Outbox transport');
        END
        """
    )
    op.execute(
        """
        CREATE TRIGGER telegram_destination_matches_transport
        BEFORE INSERT ON telegram_outbox_destination
        WHEN NOT EXISTS (
          SELECT 1 FROM outbox_message o
          WHERE o.id = NEW.outbox_message_id
            AND o.transport = 'TELEGRAM' AND o.destination_kind = 'OWNER'
        ) OR EXISTS (
          SELECT 1 FROM beeper_outbox_destination b
          WHERE b.outbox_message_id = NEW.outbox_message_id
        )
        BEGIN
          SELECT RAISE(ABORT, 'Telegram destination does not match Outbox transport');
        END
        """
    )
    op.execute(
        """
        CREATE TRIGGER outbox_participant_matches_task_destination
        BEFORE INSERT ON outbox_message_participant
        WHEN NOT EXISTS (
          SELECT 1
          FROM outbox_message o
          JOIN beeper_outbox_destination b ON b.outbox_message_id = o.id
          JOIN task_participant tp ON tp.id = NEW.task_participant_id
          WHERE o.id = NEW.outbox_message_id
            AND o.transport = 'BEEPER'
            AND o.task_instance_id = tp.task_instance_id
            AND b.conversation_id = tp.conversation_id
        )
        BEGIN
          SELECT RAISE(ABORT, 'Outbox participant does not match task destination');
        END
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER outbox_participant_matches_task_destination")
    op.execute("DROP TRIGGER telegram_destination_matches_transport")
    op.execute("DROP TRIGGER beeper_destination_matches_transport")
    op.execute("DROP TRIGGER contact_rule_override_integrity_insert")
    op.execute("DROP TRIGGER contact_rule_core_immutable")
    op.execute("DROP TRIGGER task_participant_requires_pinned_identity_update")
    op.execute("DROP TRIGGER task_participant_requires_pinned_identity_insert")
    op.execute("DROP TRIGGER task_terminal_status_immutable")
