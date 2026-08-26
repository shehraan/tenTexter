from __future__ import annotations

from datetime import timedelta

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from ten_texter.enums import ContentSupport, ProcessingStatus
from ten_texter.inbound import InboundEvent, MessageIngestor, RevisionProcessor
from ten_texter.models import DecisionRequest, Message, MessageProcessingAttempt, MessageRevision, TaskEvent
from tests.test_schema import NOW, seed_core


def event(core: dict[str, object], **changes: object) -> InboundEvent:
    values: dict[str, object] = {
        "conversation_id": core["conversation"].id,
        "provider_message_id": "provider:new",
        "sender_identity_id": core["identity"].id,
        "provider_revision_key": "revision:1",
        "created_at": NOW,
        "received_at": NOW,
        "text": "yes",
        "provider_sequence": 10,
    }
    values.update(changes)
    return InboundEvent(**values)  # type: ignore[arg-type]


def processor(session: Session) -> RevisionProcessor:
    return RevisionProcessor(sessionmaker(bind=session.bind, expire_on_commit=False, autoflush=False))


def test_duplicate_delivery_is_idempotent(db_session: Session) -> None:
    core = seed_core(db_session)
    first = MessageIngestor(db_session).ingest(event(core))
    second = MessageIngestor(db_session).ingest(event(core))
    assert not first.duplicate
    assert second.duplicate
    assert first.revision_id == second.revision_id
    assert db_session.scalar(
        select(func.count(MessageRevision.id)).where(MessageRevision.message_id == first.message_id)
    ) == 1


def test_edit_creates_revision_and_advances_current(db_session: Session) -> None:
    core = seed_core(db_session)
    ingestor = MessageIngestor(db_session)
    first = ingestor.ingest(event(core))
    second = ingestor.ingest(
        event(
            core,
            provider_revision_key="revision:2",
            provider_sequence=11,
            text="actually no",
        )
    )
    message = db_session.get(Message, first.message_id)
    assert first.revision_id != second.revision_id
    assert message.current_revision_id == second.revision_id
    assert db_session.scalar(
        select(func.count(MessageRevision.id)).where(MessageRevision.message_id == message.id)
    ) == 2


def test_conflicting_same_order_is_preserved_and_requires_reconciliation(db_session: Session) -> None:
    core = seed_core(db_session)
    ingestor = MessageIngestor(db_session)
    first = ingestor.ingest(event(core))
    conflict = ingestor.ingest(
        event(
            core,
            provider_revision_key="revision:conflict",
            provider_sequence=10,
            text="no",
        )
    )
    assert conflict.ordering_conflict
    assert db_session.get(Message, first.message_id).current_revision_id == first.revision_id
    assert db_session.scalar(
        select(func.count(MessageRevision.id)).where(MessageRevision.message_id == first.message_id)
    ) == 2
    decision = db_session.scalar(
        select(DecisionRequest).where(DecisionRequest.message_revision_id == conflict.revision_id)
    )
    assert decision is not None


def test_stale_revision_processing_succeeds_without_semantic_mutation(db_session: Session) -> None:
    core = seed_core(db_session)
    ingestor = MessageIngestor(db_session)
    first = ingestor.ingest(event(core))
    ingestor.ingest(
        event(core, provider_revision_key="revision:2", provider_sequence=11, text="new")
    )
    db_session.commit()
    worker = processor(db_session)
    claim = worker.claim(first.revision_id, now=NOW)
    assert claim is not None
    applied = False

    def apply(session: Session, revision: MessageRevision) -> None:
        nonlocal applied
        applied = True
        session.add(TaskEvent(task_instance_id=core["task"].id, source_message_revision_id=revision.id, event_type="BAD", payload_json={}))

    assert not worker.commit(claim, apply)
    assert not applied
    db_session.expire_all()
    assert db_session.get(MessageRevision, first.revision_id).processing_status is ProcessingStatus.PROCESSED


def test_reclaimed_revision_fences_stale_worker(db_session: Session) -> None:
    core = seed_core(db_session)
    result = MessageIngestor(db_session).ingest(event(core))
    db_session.commit()
    worker = processor(db_session)
    old = worker.claim(result.revision_id, now=NOW)
    assert old is not None
    new = worker.claim(result.revision_id, now=NOW + timedelta(minutes=1))
    assert new is not None
    assert not worker.commit(old, lambda *_: None)
    assert worker.commit(new, lambda *_: None)
    db_session.expire_all()
    attempts = list(
        db_session.scalars(
            select(MessageProcessingAttempt).where(
                MessageProcessingAttempt.message_revision_id == result.revision_id
            )
        )
    )
    assert len(attempts) == 2
    assert all(attempt.finished_at is not None for attempt in attempts)


def test_unsupported_content_and_deletion_tombstone(db_session: Session) -> None:
    core = seed_core(db_session)
    unsupported = MessageIngestor(db_session).ingest(
        event(core, content_support=ContentSupport.UNSUPPORTED, text="[sticker]")
    )
    assert db_session.get(MessageRevision, unsupported.revision_id).processing_status is ProcessingStatus.PROCESSED
    db_session.commit()
    assert processor(db_session).claim(unsupported.revision_id, now=NOW) is None

    tombstone = MessageIngestor(db_session).ingest(
        event(
            core,
            provider_revision_key="revision:deleted",
            provider_sequence=11,
            text=None,
            is_deleted=True,
        )
    )
    stored = db_session.get(MessageRevision, tombstone.revision_id)
    assert stored.is_deleted and stored.text is None


def test_exact_revision_content_is_database_immutable(db_session: Session) -> None:
    core = seed_core(db_session)
    result = MessageIngestor(db_session).ingest(event(core))
    revision = db_session.get(MessageRevision, result.revision_id)
    revision.text = "tampered"
    with pytest.raises(IntegrityError):
        db_session.flush()
