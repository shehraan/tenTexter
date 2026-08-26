from __future__ import annotations

from datetime import timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from ten_texter.correlation import Classification, CorrelationOrchestrator
from ten_texter.domain import distinct_response_count
from ten_texter.enums import (
    AttemptResult,
    AvailabilityStatus,
    AwaitedResponseStatus,
    ContentSupport,
    ConversationKind,
    MessageKind,
    OutboxStatus,
    ParentTerminalPolicy,
    ProcessingStatus,
)
from ten_texter.models import (
    AwaitedResponse,
    AwaitedResponsePrompt,
    BeeperOutboxDestination,
    Conversation,
    ConversationParticipant,
    DecisionRequestAwaitedResponseCandidate,
    Identity,
    Message,
    MessageRevision,
    OutboxDeliveryAttempt,
    OutboxMessage,
    Proposal,
    DecisionRequest,
    DecisionRequestPrompt,
)
from ten_texter.control import ProductionOwnerCommandHandler
from ten_texter.telegram import TelegramControlGateway
from ten_texter.outbox import OutboxService
from tests.test_schema import NOW, seed_core


class Semantic:
    def __init__(self, choice: int | None = None):
        self.choice = choice

    def choose(self, *_: object) -> int | None:
        return self.choice


class Classifier:
    def __init__(self, classification: Classification):
        self.classification = classification

    def classify(self, *_: object) -> Classification:
        return self.classification


def orchestrator(session: Session, classification: Classification, choice: int | None = None) -> CorrelationOrchestrator:
    return CorrelationOrchestrator(
        session,
        semantic=Semantic(choice),
        classifier=Classifier(classification),
    )


def awaited(session: Session, core: dict[str, object], *, created_at=NOW - timedelta(minutes=5)) -> AwaitedResponse:
    row = AwaitedResponse(
        task_participant_id=core["participant"].id,
        expected_response_type="availability",
        status=AwaitedResponseStatus.OPEN,
        created_at=created_at,
        expires_at=NOW + timedelta(hours=1),
    )
    session.add(row)
    session.flush()
    return row


def test_exact_conversation_single_open_response_correlates(db_session: Session) -> None:
    core = seed_core(db_session)
    response = awaited(db_session, core)
    result = orchestrator(
        db_session,
        Classification(kind="AVAILABILITY", availability=AvailabilityStatus.AVAILABLE),
    ).process(core["revision"].id)
    assert result.awaited_response_id == response.id
    assert response.status is AwaitedResponseStatus.SATISFIED
    assert core["revision"].awaited_response_id == response.id
    assert core["participant"].availability_status is AvailabilityStatus.AVAILABLE


def test_correlation_ambiguity_is_not_guessed(db_session: Session) -> None:
    core = seed_core(db_session)
    first = awaited(db_session, core, created_at=NOW - timedelta(minutes=10))
    second = awaited(db_session, core, created_at=NOW - timedelta(minutes=5))
    result = orchestrator(db_session, Classification(kind="AMBIGUOUS")).process(core["revision"].id)
    assert result.outcome == "CORRELATION_AMBIGUITY"
    assert core["revision"].awaited_response_id is None
    assert first.status is AwaitedResponseStatus.OPEN
    assert second.status is AwaitedResponseStatus.OPEN
    candidate_count = db_session.scalar(
        select(func.count(DecisionRequestAwaitedResponseCandidate.awaited_response_id)).where(
            DecisionRequestAwaitedResponseCandidate.decision_request_id == result.decision_request_id
        )
    )
    assert candidate_count == 2


def test_ambiguous_message_owner_selection_continues_semantic_processing(db_session: Session) -> None:
    core = seed_core(db_session)
    first = awaited(db_session, core, created_at=NOW - timedelta(minutes=10))
    second = awaited(db_session, core, created_at=NOW - timedelta(minutes=5))
    result = CorrelationOrchestrator(
        db_session,
        semantic=Semantic(),
        classifier=Classifier(Classification(kind="AMBIGUOUS")),
        owner_chat_id=99,
    ).process(core["revision"].id)
    prompt = db_session.scalar(
        select(DecisionRequestPrompt).where(
            DecisionRequestPrompt.decision_request_id == result.decision_request_id
        )
    )
    assert prompt is not None
    db_session.commit()
    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    handler = ProductionOwnerCommandHandler(
        factory,
        owner_chat_id=99,
        resolver=object(),  # type: ignore[arg-type]
        generation=object(),  # type: ignore[arg-type]
        classifier=Classifier(
            Classification(kind="AVAILABILITY", availability=AvailabilityStatus.AVAILABLE)
        ),
    )
    control = TelegramControlGateway(factory, owner_id=7, parser=object(), handler=handler)  # type: ignore[arg-type]
    raw = {
        "update_id": 700,
        "callback_query": {
            "id": "selection",
            "from": {"id": 7},
            "message": {"message_id": 1, "chat": {"id": 99, "type": "private"}},
            "data": f"decision:{result.decision_request_id}:awaited_response:{second.id}",
        },
    }
    received = control.receive(raw)
    control.process(received.telegram_update_row_id)
    with factory() as session:
        revision = session.get(MessageRevision, core["revision"].id)
        assert revision.awaited_response_id == second.id
        assert session.get(AwaitedResponse, second.id).status is AwaitedResponseStatus.SATISFIED
        assert session.get(AwaitedResponse, first.id).status is AwaitedResponseStatus.OPEN
        assert session.get(type(core["participant"]), core["participant"].id).availability_status is AvailabilityStatus.AVAILABLE
        decision = session.get(DecisionRequest, result.decision_request_id)
        assert decision.status.value == "CLOSED"


def test_stale_and_repeated_owner_correlation_answers_have_no_second_effect(db_session: Session) -> None:
    core = seed_core(db_session)
    first = awaited(db_session, core)
    second = awaited(db_session, core, created_at=NOW - timedelta(minutes=4))
    result = CorrelationOrchestrator(
        db_session, semantic=Semantic(), classifier=Classifier(Classification(kind="AMBIGUOUS")), owner_chat_id=99
    ).process(core["revision"].id)
    core["revision"].awaited_response_id = first.id
    db_session.commit()
    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    handler = ProductionOwnerCommandHandler(
        factory, owner_chat_id=99, resolver=object(), generation=object(),
        classifier=Classifier(Classification(kind="AVAILABILITY", availability=AvailabilityStatus.UNAVAILABLE)),
    )  # type: ignore[arg-type]
    control = TelegramControlGateway(factory, owner_id=7, parser=object(), handler=handler)  # type: ignore[arg-type]
    def callback(update_id: int) -> dict[str, object]:
        return {"update_id": update_id, "callback_query": {"id": str(update_id), "from": {"id": 7}, "message": {"message_id": 1, "chat": {"id": 99, "type": "private"}}, "data": f"decision:{result.decision_request_id}:awaited_response:{second.id}"}}
    first_update = control.receive(callback(701))
    control.process(first_update.telegram_update_row_id)
    repeated = control.receive(callback(702))
    control.process(repeated.telegram_update_row_id)
    with factory() as session:
        decision = session.get(DecisionRequest, result.decision_request_id)
        assert decision.close_reason.value == "SUBJECT_RESOLVED"
        assert session.get(MessageRevision, core["revision"].id).awaited_response_id == first.id


def test_known_correlation_interpretation_ambiguity_marks_awaited(db_session: Session) -> None:
    core = seed_core(db_session)
    response = awaited(db_session, core)
    result = orchestrator(db_session, Classification(kind="AMBIGUOUS")).process(core["revision"].id)
    assert result.outcome == "INTERPRETATION_AMBIGUOUS"
    assert core["revision"].awaited_response_id == response.id
    assert response.status is AwaitedResponseStatus.AMBIGUOUS


def test_response_count_is_distinct_participants_not_awaited_rows(db_session: Session) -> None:
    core = seed_core(db_session)
    first = awaited(db_session, core)
    second = awaited(db_session, core)
    first.status = AwaitedResponseStatus.SATISFIED
    second.status = AwaitedResponseStatus.AMBIGUOUS
    db_session.flush()
    assert distinct_response_count(db_session, core["task"].id) == 1


def test_cross_conversation_same_person_asks_owner(db_session: Session) -> None:
    core = seed_core(db_session)
    response = awaited(db_session, core)
    second_identity = Identity(
        person_id=core["person"].id,
        beeper_user_id="beeper:alex:instagram",
        network="instagram",
        metadata_json={},
    )
    other_conversation = Conversation(
        beeper_conversation_id="conv:alex:instagram",
        network="instagram",
        kind=ConversationKind.DIRECT,
        counterparty_person_id=core["person"].id,
        metadata_json={},
    )
    db_session.add_all([second_identity, other_conversation])
    db_session.flush()
    db_session.add(
        ConversationParticipant(conversation_id=other_conversation.id, identity_id=second_identity.id)
    )
    db_session.flush()
    message = Message(
        conversation_id=other_conversation.id,
        provider_message_id="cross-message",
        sender_identity_id=second_identity.id,
        created_at=NOW,
    )
    db_session.add(message)
    db_session.flush()
    revision = MessageRevision(
        message_id=message.id,
        provider_revision_key="cross-r1",
        provider_sequence=1,
        content_hash="c" * 64,
        text="yes",
        content_support=ContentSupport.SUPPORTED,
        processing_status=ProcessingStatus.PENDING,
    )
    db_session.add(revision)
    db_session.flush()
    message.current_revision_id = revision.id
    db_session.flush()
    result = orchestrator(
        db_session,
        Classification(kind="AVAILABILITY", availability=AvailabilityStatus.AVAILABLE),
    ).process(revision.id)
    assert result.outcome == "CROSS_CONVERSATION_RESPONSE"
    assert revision.awaited_response_id is None
    assert response.status is AwaitedResponseStatus.OPEN


def test_provider_reply_linkage_precedes_ambiguous_conversation(db_session: Session) -> None:
    core = seed_core(db_session)
    chosen = awaited(db_session, core)
    awaited(db_session, core, created_at=NOW - timedelta(minutes=2))
    sent = OutboxService(db_session).create_beeper(
        task_instance_id=core["task"].id,
        conversation_id=core["conversation"].id,
        participant_ids=[core["participant"].id],
        final_text="question",
        message_kind=MessageKind.INITIAL,
        idempotency_key="prompt",
    )
    sent.status = OutboxStatus.SENT
    attempt = OutboxDeliveryAttempt(
        outbox_message_id=sent.id,
        started_at=NOW,
        finished_at=NOW,
        result=AttemptResult.SUCCESS,
        provider_message_id="provider:prompt",
    )
    db_session.add_all([attempt, AwaitedResponsePrompt(awaited_response_id=chosen.id, outbox_message_id=sent.id)])
    core["message"].provider_reply_to_message_id = "provider:prompt"
    db_session.flush()
    result = orchestrator(
        db_session,
        Classification(kind="AVAILABILITY", availability=AvailabilityStatus.AVAILABLE),
    ).process(core["revision"].id)
    assert result.awaited_response_id == chosen.id
    assert chosen.status is AwaitedResponseStatus.SATISFIED
