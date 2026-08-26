from __future__ import annotations

from datetime import timedelta

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from ten_texter.enums import AvailabilityEvidence, AvailabilityStatus, ContentSupport, ProcessingStatus
from ten_texter.inbound import InboundEvent, MessageIngestor, RevisionProcessor
from ten_texter.models import (
    AwaitedResponse,
    DecisionRequest,
    DecisionRequestPrompt,
    Message,
    MessageProcessingAttempt,
    MessageRevision,
    TaskEvent,
)
from ten_texter.correlation import Classification
from ten_texter.control import ProductionOwnerCommandHandler
from ten_texter.enums import AwaitedResponseStatus
from ten_texter.telegram import TelegramControlGateway
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


class AvailabilityClassifier:
    def classify(self, *_: object) -> Classification:
        return Classification(kind="AVAILABILITY", availability=AvailabilityStatus.UNAVAILABLE)


def test_ordering_conflict_owner_selects_exact_revision_then_only_current_mutates(db_session: Session) -> None:
    core = seed_core(db_session)
    awaited = AwaitedResponse(
        task_participant_id=core["participant"].id,
        expected_response_type="availability",
        status=AwaitedResponseStatus.OPEN,
    )
    db_session.add(awaited)
    db_session.flush()
    ingestor = MessageIngestor(db_session, owner_chat_id=99)
    first = ingestor.ingest(event(core, text="yes"))
    conflict = ingestor.ingest(
        event(core, provider_revision_key="revision:conflict", provider_sequence=10, text="no")
    )
    decision = db_session.scalar(
        select(DecisionRequest).where(DecisionRequest.message_revision_id == conflict.revision_id)
    )
    assert decision is not None and decision.type == "MESSAGE_ORDERING_CONFLICT"
    assert db_session.scalar(
        select(DecisionRequestPrompt).where(DecisionRequestPrompt.decision_request_id == decision.id)
    ) is not None
    db_session.commit()
    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    assert RevisionProcessor(factory).claim(first.revision_id, now=NOW) is None
    assert RevisionProcessor(factory).claim(conflict.revision_id, now=NOW) is None
    handler = ProductionOwnerCommandHandler(
        factory,
        owner_chat_id=99,
        resolver=object(),  # type: ignore[arg-type]
        generation=object(),  # type: ignore[arg-type]
        classifier=AvailabilityClassifier(),
    )
    control = TelegramControlGateway(factory, owner_id=7, parser=object(), handler=handler)  # type: ignore[arg-type]
    raw = {
        "update_id": 900,
        "callback_query": {
            "id": "ordering",
            "from": {"id": 7},
            "message": {"message_id": 1, "chat": {"id": 99, "type": "private"}},
            "data": f"decision:{decision.id}:message_revision:{conflict.revision_id}",
        },
    }
    received = control.receive(raw)
    control.process(received.telegram_update_row_id)
    duplicate = control.receive(raw)
    assert duplicate.outcome == "DUPLICATE"
    control.process(duplicate.telegram_update_row_id)
    repeated_raw = dict(raw)
    repeated_raw["update_id"] = 901
    repeated = control.receive(repeated_raw)
    control.process(repeated.telegram_update_row_id)
    worker = RevisionProcessor(factory)
    selected_claim = worker.claim(conflict.revision_id, now=NOW)
    stale_claim = worker.claim(first.revision_id, now=NOW)
    assert selected_claim is not None and stale_claim is not None

    def semantics(session: Session, revision: MessageRevision) -> None:
        participant = session.get(type(core["participant"]), core["participant"].id)
        participant.availability_status = AvailabilityStatus.UNAVAILABLE
        participant.availability_evidence = AvailabilityEvidence.FIRST_PARTY
        participant.availability_source_revision_id = revision.id

    assert worker.commit(selected_claim, semantics)
    assert not worker.commit(stale_claim, semantics)
    with factory() as session:
        assert session.get(Message, first.message_id).current_revision_id == conflict.revision_id
        participant = session.get(type(core["participant"]), core["participant"].id)
        assert participant.availability_source_revision_id == conflict.revision_id


def test_ordering_conflict_rejects_non_candidate_and_stale_answer_has_no_effect(db_session: Session) -> None:
    core = seed_core(db_session)
    ingestor = MessageIngestor(db_session, owner_chat_id=99)
    first = ingestor.ingest(event(core))
    conflict = ingestor.ingest(
        event(core, provider_revision_key="revision:conflict", provider_sequence=10, text="no")
    )
    unrelated = core["revision"]
    decision = db_session.scalar(
        select(DecisionRequest).where(DecisionRequest.message_revision_id == conflict.revision_id)
    )
    db_session.commit()
    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    handler = ProductionOwnerCommandHandler(
        factory, owner_chat_id=99, resolver=object(), generation=object()
    )  # type: ignore[arg-type]
    update = type("Update", (), {})()
    with pytest.raises(Exception, match="candidate"):
        handler.prepare_decision(
            decision.id,
            {"callback_query": {"data": f"decision:{decision.id}:message_revision:{unrelated.id}"}},
            update,  # type: ignore[arg-type]
        )
    with factory.begin() as session:
        message = session.get(Message, first.message_id)
        message.current_revision_id = conflict.revision_id
    prepared = handler.prepare_decision(
        decision.id,
        {"callback_query": {"data": f"decision:{decision.id}:message_revision:{first.revision_id}"}},
        update,  # type: ignore[arg-type]
    )
    with factory.begin() as session:
        handler.apply_decision(session, decision.id, prepared, update)  # type: ignore[arg-type]
    with factory() as session:
        stored = session.get(DecisionRequest, decision.id)
        assert stored.close_reason.value == "SUBJECT_RESOLVED"
        assert session.get(Message, first.message_id).current_revision_id == conflict.revision_id


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
