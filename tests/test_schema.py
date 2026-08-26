from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ten_texter.enums import (
    AvailabilityEvidence,
    AvailabilityStatus,
    ContentSupport,
    ConversationKind,
    DestinationKind,
    MessageKind,
    OutboxStatus,
    ParentTerminalPolicy,
    ProcessingStatus,
    TaskStatus,
    Transport,
)
from ten_texter.models import (
    Conversation,
    ConversationParticipant,
    Identity,
    Message,
    MessageRevision,
    OutboxDeliveryAttempt,
    OutboxMessage,
    Person,
    TaskInstance,
    TaskParticipant,
)


NOW = datetime(2026, 8, 26, 15, 0, tzinfo=UTC)


def seed_core(session: Session) -> dict[str, object]:
    person = Person(display_name="Alex", metadata_json={})
    session.add(person)
    session.flush()
    identity = Identity(person_id=person.id, beeper_user_id="beeper:alex", network="discord", metadata_json={})
    conversation = Conversation(
        beeper_conversation_id="conv:alex",
        network="discord",
        kind=ConversationKind.DIRECT,
        counterparty_person_id=person.id,
        metadata_json={},
    )
    session.add_all([identity, conversation])
    session.flush()
    session.add(ConversationParticipant(conversation_id=conversation.id, identity_id=identity.id))
    task = TaskInstance(
        scheduled_at=NOW + timedelta(days=1),
        duration_minutes=60,
        topic_key="tennis",
        status=TaskStatus.ACTIVE,
    )
    session.add(task)
    session.flush()
    participant = TaskParticipant(
        task_instance_id=task.id,
        person_id=person.id,
        conversation_id=conversation.id,
        availability_status=AvailabilityStatus.UNKNOWN,
    )
    session.add(participant)
    session.flush()
    message = Message(
        conversation_id=conversation.id,
        provider_message_id="m1",
        sender_identity_id=identity.id,
        created_at=NOW,
    )
    session.add(message)
    session.flush()
    revision = MessageRevision(
        message_id=message.id,
        provider_revision_key="r1",
        provider_sequence=1,
        content_hash="a" * 64,
        text="yes",
        content_support=ContentSupport.SUPPORTED,
        processing_status=ProcessingStatus.PENDING,
    )
    session.add(revision)
    session.flush()
    message.current_revision_id = revision.id
    session.flush()
    return {
        "person": person,
        "identity": identity,
        "conversation": conversation,
        "task": task,
        "participant": participant,
        "message": message,
        "revision": revision,
    }


def test_full_schema_round_trip(db_session: Session) -> None:
    rows = seed_core(db_session)
    db_session.commit()
    assert db_session.get(Person, rows["person"].id).display_name == "Alex"  # type: ignore[union-attr]
    names = set(inspect(db_session.bind).get_table_names())
    assert {
        "task_definition", "task_instance", "message_revision", "decision_request",
        "contact_rule", "disclosure_grant", "trigger_execution", "outbox_message",
        "telegram_update",
    } <= names


@pytest.mark.parametrize(
    "statement",
    [
        "INSERT INTO task_instance (scheduled_at,duration_minutes,topic_key,coordination_close_offset_minutes,status,created_at,updated_at) VALUES ('2026-08-27',60,'x',60,'ACTIVE','2026-08-26','2026-08-26')",
        "INSERT INTO task_instance (task_definition_id,occurrence_key,scheduled_at,duration_minutes,topic_key,coordination_close_offset_minutes,status,created_at,updated_at) VALUES (NULL,'slot','2026-08-27',60,'x',60,'ACTIVE','2026-08-26','2026-08-26')",
    ],
)
def test_task_instance_constraints(db_session: Session, statement: str) -> None:
    if "NULL,'slot'" not in statement:
        db_session.execute(text(statement))
        db_session.rollback()
        return
    with pytest.raises(IntegrityError):
        db_session.execute(text(statement))


def test_availability_provenance_constraint(db_session: Session) -> None:
    core = seed_core(db_session)
    with pytest.raises(IntegrityError):
        db_session.execute(
            text("UPDATE task_participant SET availability_status='AVAILABLE' WHERE id=:id"),
            {"id": core["participant"].id},  # type: ignore[union-attr]
        )


def test_transport_destination_and_lifecycle_constraints(db_session: Session) -> None:
    with pytest.raises(IntegrityError):
        db_session.add(
            OutboxMessage(
                transport=Transport.TELEGRAM,
                destination_kind=DestinationKind.PARTICIPANT,
                final_text="bad pair",
                message_kind=MessageKind.NOTIFICATION,
                status=OutboxStatus.PENDING,
                idempotency_key="bad",
                parent_terminal_policy=ParentTerminalPolicy.SURVIVE,
            )
        )
        db_session.flush()
    db_session.rollback()

    message = OutboxMessage(
        transport=Transport.TELEGRAM,
        destination_kind=DestinationKind.OWNER,
        final_text="hello",
        message_kind=MessageKind.NOTIFICATION,
        status=OutboxStatus.PENDING,
        idempotency_key="owner-1",
        parent_terminal_policy=ParentTerminalPolicy.SURVIVE,
    )
    db_session.add(message)
    db_session.flush()
    db_session.add_all([
        OutboxDeliveryAttempt(outbox_message_id=message.id),
        OutboxDeliveryAttempt(outbox_message_id=message.id),
    ])
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_composite_current_revision_parent_constraint(db_session: Session) -> None:
    core = seed_core(db_session)
    other = Message(
        conversation_id=core["conversation"].id,  # type: ignore[union-attr]
        provider_message_id="m2",
        sender_identity_id=core["identity"].id,  # type: ignore[union-attr]
        created_at=NOW,
    )
    db_session.add(other)
    db_session.flush()
    with pytest.raises(IntegrityError):
        other.current_revision_id = core["revision"].id  # type: ignore[union-attr]
        db_session.flush()


def test_availability_valid_non_unknown_round_trip(db_session: Session) -> None:
    core = seed_core(db_session)
    participant = core["participant"]
    participant.availability_status = AvailabilityStatus.AVAILABLE  # type: ignore[union-attr]
    participant.availability_evidence = AvailabilityEvidence.FIRST_PARTY  # type: ignore[union-attr]
    participant.availability_source_revision_id = core["revision"].id  # type: ignore[union-attr]
    db_session.commit()
