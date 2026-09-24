from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from ten_texter.control import ProductionOwnerCommandHandler
from ten_texter.decision_prompt_context import authorize_decision_prompt
from ten_texter.enums import (
    ConversationKind,
    DecisionCloseReason,
    DecisionStatus,
    MessageKind,
    OutboxCancelReason,
    OutboxStatus,
    TaskStatus,
    Transport,
)
from ten_texter.mass_contact import MASS_CONTACT_DECISION_TYPE, MASS_CONTACT_THRESHOLD
from ten_texter.models import (
    Conversation,
    ConversationParticipant,
    DecisionRequest,
    DecisionRequestPrompt,
    Identity,
    OutboxMessage,
    Person,
    TaskInstance,
)
from ten_texter.outbox import (
    DeliveryResult,
    OutboxService,
    OutboxWorker,
    PreSendDecision,
)
from ten_texter.policy import DatabaseContextProvider, PolicyRevalidator
from ten_texter.validator import DatabaseValidatorContextProvider
from ten_texter.workflows import CoordinationWorkflow, ParticipantSendPlan


class AlwaysValid:
    def validate(self, **_: object) -> bool:
        return True


class RecordingTransport:
    def __init__(self) -> None:
        self.calls = 0

    def send(self, _request: object) -> DeliveryResult:
        self.calls += 1
        return DeliveryResult(True, True, provider_message_id=f"sent-{self.calls}")

    def reconcile(self, _request: object, **_: object) -> str | None:
        return None


def _start_task(
    session: Session,
    participant_count: int,
    *,
    key: str,
    group: bool = False,
) -> tuple[int, tuple[int, ...]]:
    plans: list[ParticipantSendPlan] = []
    for index in range(participant_count):
        person = Person(display_name=f"Person {key} {index}", metadata_json={})
        session.add(person)
        session.flush()
        identity = Identity(
            person_id=person.id,
            beeper_user_id=f"beeper:{key}:{index}",
            network="discord",
            metadata_json={},
        )
        conversation = Conversation(
            beeper_conversation_id=f"conversation:{key}:{index}",
            network="discord",
            kind=ConversationKind.GROUP if group else ConversationKind.DIRECT,
            counterparty_person_id=None if group else person.id,
            metadata_json={},
        )
        session.add_all([identity, conversation])
        session.flush()
        session.add(
            ConversationParticipant(
                conversation_id=conversation.id,
                identity_id=identity.id,
            )
        )
        plans.append(
            ParticipantSendPlan(
                person_id=person.id,
                conversation_id=conversation.id,
                final_text=f"Are you available, Person {key} {index}?",
            )
        )
    started = CoordinationWorkflow(session, owner_chat_id=99).start(
        scheduled_at=datetime(2026, 10, 1, 17, tzinfo=UTC),
        duration_minutes=60,
        location=None,
        topic_key="tennis",
        participant_sends=plans,
    )
    return started.task_id, started.outbox_ids


def _factory(session: Session) -> sessionmaker[Session]:
    return sessionmaker(bind=session.bind, expire_on_commit=False, autoflush=False)


def _handler(factory: sessionmaker[Session]) -> ProductionOwnerCommandHandler:
    return ProductionOwnerCommandHandler(
        factory,
        owner_chat_id=99,
        resolver=object(),  # type: ignore[arg-type]
        generation=object(),  # type: ignore[arg-type]
    )


def test_twenty_five_distinct_participants_is_not_mass_contact(
    db_session: Session,
) -> None:
    _task_id, outbox_ids = _start_task(
        db_session, MASS_CONTACT_THRESHOLD, key="at-threshold"
    )

    assert (
        PolicyRevalidator(owner_chat_id=99).check(
            db_session, db_session.get(OutboxMessage, outbox_ids[0])
        )
        is PreSendDecision.READY
    )
    assert (
        db_session.scalar(
            select(func.count(DecisionRequest.id)).where(
                DecisionRequest.type == MASS_CONTACT_DECISION_TYPE
            )
        )
        == 0
    )


def test_mass_contact_is_held_once_across_ticks_and_process_restart(
    db_session: Session,
) -> None:
    task_id, outbox_ids = _start_task(
        db_session, MASS_CONTACT_THRESHOLD + 1, key="held"
    )
    db_session.commit()
    factory = _factory(db_session)
    transport = RecordingTransport()

    first_worker = OutboxWorker(
        factory,
        revalidator=PolicyRevalidator(owner_chat_id=99),
        validator=AlwaysValid(),
        adapters={Transport.BEEPER: transport},
    )
    assert first_worker.process(outbox_ids[0]) is OutboxStatus.PENDING
    assert first_worker.process(outbox_ids[1]) is OutboxStatus.PENDING
    assert transport.calls == 0

    restarted_worker = OutboxWorker(
        factory,
        revalidator=PolicyRevalidator(owner_chat_id=99),
        validator=AlwaysValid(),
        adapters={Transport.BEEPER: transport},
    )
    assert restarted_worker.process(outbox_ids[2]) is OutboxStatus.PENDING
    assert transport.calls == 0

    with factory() as session:
        decisions = list(
            session.scalars(
                select(DecisionRequest).where(
                    DecisionRequest.task_instance_id == task_id,
                    DecisionRequest.type == MASS_CONTACT_DECISION_TYPE,
                )
            )
        )
        assert len(decisions) == 1
        assert decisions[0].status is DecisionStatus.PENDING
        assert (
            session.scalar(
                select(func.count(DecisionRequestPrompt.outbox_message_id)).where(
                    DecisionRequestPrompt.decision_request_id == decisions[0].id
                )
            )
            == 1
        )


def test_mass_contact_prompt_has_exact_database_derived_validator_context(
    db_session: Session,
) -> None:
    task_id, outbox_ids = _start_task(
        db_session, MASS_CONTACT_THRESHOLD + 1, key="validator"
    )
    target = db_session.get(OutboxMessage, outbox_ids[0])
    assert (
        PolicyRevalidator(owner_chat_id=99).check(db_session, target)
        is PreSendDecision.AWAITING_OWNER
    )
    decision = db_session.scalar(
        select(DecisionRequest).where(
            DecisionRequest.task_instance_id == task_id,
            DecisionRequest.type == MASS_CONTACT_DECISION_TYPE,
        )
    )
    prompt = db_session.scalar(
        select(OutboxMessage)
        .join(DecisionRequestPrompt)
        .where(DecisionRequestPrompt.decision_request_id == decision.id)
    )
    authorization = authorize_decision_prompt(
        db_session, prompt, owner_chat_id=99
    )
    assert authorization is not None
    assert authorization.expected_text == prompt.final_text
    assert authorization.allowed_claims == (
        f"Task {task_id} targets 26 distinct participants.",
        "The mass-contact threshold is 25 distinct participants.",
        "No participant messages for this task may be sent before owner approval.",
        "The owner may approve this task's mass contact or cancel this task by replying to this Telegram message.",
    )

    db_session.commit()
    context = DatabaseValidatorContextProvider(
        _factory(db_session),
        facts=DatabaseContextProvider(),
        owner_chat_id=99,
    ).context_for(prompt.id, prompt.message_kind)
    assert context.allowed_claims == authorization.allowed_claims
    assert "approve mass contact" in prompt.final_text
    assert "Reply to this Telegram message" in prompt.final_text
    assert (
        PolicyRevalidator(owner_chat_id=99).check(db_session, prompt)
        is PreSendDecision.READY
    )

    with _factory(db_session).begin() as session:
        spoof = OutboxService(session).create_owner(
            telegram_chat_id=99,
            final_text=prompt.final_text
            + " Alex has a private medical appointment.",
            message_kind=MessageKind.NOTIFICATION,
            idempotency_key=f"decision:{decision.id}:spoof",
            task_instance_id=task_id,
        )
        spoof_id = spoof.id
    with _factory(db_session)() as session:
        malformed = session.get(OutboxMessage, spoof_id)
        assert authorize_decision_prompt(session, malformed, owner_chat_id=99) is None


def test_incidental_group_members_do_not_count_as_logical_targets(
    db_session: Session,
) -> None:
    _task_id, outbox_ids = _start_task(
        db_session, 1, key="group-target", group=True
    )
    target = db_session.get(OutboxMessage, outbox_ids[0])
    participant = target and db_session.scalar(
        select(ConversationParticipant.conversation_id)
        .join(Identity, Identity.id == ConversationParticipant.identity_id)
        .where(Identity.beeper_user_id == "beeper:group-target:0")
    )
    assert participant is not None
    for index in range(MASS_CONTACT_THRESHOLD + 5):
        person = Person(display_name=f"Incidental {index}", metadata_json={})
        db_session.add(person)
        db_session.flush()
        identity = Identity(
            person_id=person.id,
            beeper_user_id=f"beeper:incidental:{index}",
            network="discord",
            metadata_json={},
        )
        db_session.add(identity)
        db_session.flush()
        db_session.add(
            ConversationParticipant(
                conversation_id=participant,
                identity_id=identity.id,
            )
        )

    assert (
        PolicyRevalidator(owner_chat_id=99).check(db_session, target)
        is PreSendDecision.READY
    )
    assert (
        db_session.scalar(
            select(func.count(DecisionRequest.id)).where(
                DecisionRequest.type == MASS_CONTACT_DECISION_TYPE
            )
        )
        == 0
    )


def test_owner_approval_is_durable_and_scoped_to_one_task(
    db_session: Session,
) -> None:
    first_task_id, first_outbox_ids = _start_task(
        db_session, MASS_CONTACT_THRESHOLD + 1, key="approved"
    )
    second_task_id, second_outbox_ids = _start_task(
        db_session, MASS_CONTACT_THRESHOLD + 1, key="other-occurrence"
    )
    revalidator = PolicyRevalidator(owner_chat_id=99)
    assert (
        revalidator.check(db_session, db_session.get(OutboxMessage, first_outbox_ids[0]))
        is PreSendDecision.AWAITING_OWNER
    )
    assert (
        revalidator.check(db_session, db_session.get(OutboxMessage, second_outbox_ids[0]))
        is PreSendDecision.AWAITING_OWNER
    )
    first_decision = db_session.scalar(
        select(DecisionRequest).where(
            DecisionRequest.task_instance_id == first_task_id,
            DecisionRequest.type == MASS_CONTACT_DECISION_TYPE,
        )
    )
    db_session.commit()
    factory = _factory(db_session)
    handler = _handler(factory)
    payload = {"message": {"text": "approve mass contact"}}
    prepared = handler.prepare_decision(first_decision.id, payload, object())  # type: ignore[arg-type]
    with factory.begin() as session:
        handler.apply_decision(
            session, first_decision.id, prepared, object()  # type: ignore[arg-type]
        )

    transport = RecordingTransport()
    restarted_worker = OutboxWorker(
        factory,
        revalidator=PolicyRevalidator(owner_chat_id=99),
        validator=AlwaysValid(),
        adapters={Transport.BEEPER: transport},
    )
    assert restarted_worker.process(first_outbox_ids[0]) is OutboxStatus.SENT
    assert restarted_worker.process(second_outbox_ids[0]) is OutboxStatus.PENDING
    assert transport.calls == 1

    with factory() as session:
        approved = session.get(DecisionRequest, first_decision.id)
        assert approved.close_reason is DecisionCloseReason.ANSWERED
        assert approved.resolution_json == {
            "action": "approve_mass_contact",
            "participant_count": 26,
            "threshold": 25,
        }
        assert session.get(TaskInstance, second_task_id).status is TaskStatus.ACTIVE


def test_cancel_mass_contact_terminalizes_task_and_cancels_pending_sends(
    db_session: Session,
) -> None:
    task_id, outbox_ids = _start_task(
        db_session, MASS_CONTACT_THRESHOLD + 1, key="cancelled"
    )
    assert (
        PolicyRevalidator(owner_chat_id=99).check(
            db_session, db_session.get(OutboxMessage, outbox_ids[0])
        )
        is PreSendDecision.AWAITING_OWNER
    )
    decision = db_session.scalar(
        select(DecisionRequest).where(
            DecisionRequest.task_instance_id == task_id,
            DecisionRequest.type == MASS_CONTACT_DECISION_TYPE,
        )
    )
    db_session.commit()
    factory = _factory(db_session)
    handler = _handler(factory)
    payload = {"message": {"text": "cancel task"}}
    prepared = handler.prepare_decision(decision.id, payload, object())  # type: ignore[arg-type]
    with factory.begin() as session:
        handler.apply_decision(
            session, decision.id, prepared, object()  # type: ignore[arg-type]
        )

    with factory() as session:
        assert session.get(TaskInstance, task_id).status is TaskStatus.CANCELLED
        stored_decision = session.get(DecisionRequest, decision.id)
        assert stored_decision.close_reason is DecisionCloseReason.ANSWERED
        assert stored_decision.resolution_json["action"] == "cancel_task"
        participant_messages = list(
            session.scalars(
                select(OutboxMessage).where(OutboxMessage.id.in_(outbox_ids))
            )
        )
        assert all(
            message.status is OutboxStatus.CANCELLED
            and message.cancel_reason is OutboxCancelReason.PARENT_TERMINAL
            for message in participant_messages
        )
