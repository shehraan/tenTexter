from __future__ import annotations

from datetime import timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from ten_texter.correlation import Classification, CorrelationOrchestrator
from ten_texter.domain import TaskService
from ten_texter.enums import (
    AvailabilityEvidence,
    AvailabilityStatus,
    AwaitedResponseStatus,
    MessageKind,
    OutboxCancelReason,
    OutboxStatus,
    ParentTerminalPolicy,
    TaskStatus,
    Transport,
    ValidatorCategory,
)
from ten_texter.inbound import InboundEvent, MessageIngestor
from ten_texter.models import AwaitedResponse, OutboxMessage, TaskParticipant
from ten_texter.nobody_available import (
    NobodyAvailableNotifier,
    authorize_nobody_available_notification,
    nobody_available_text,
)
from ten_texter.outbox import DeliveryResult, OutboxService, OutboxWorker
from ten_texter.policy import DatabaseContextProvider, PolicyRevalidator
from ten_texter.runtime import AgentRuntime
from ten_texter.validator import (
    DatabaseValidatorContextProvider,
    IndependentMessageValidator,
    OutboxValidatorGate,
)
from ten_texter.workflows import CoordinationWorkflow, ParticipantSendPlan
from tests.test_schema import NOW, seed_core


def _sessions(session: Session) -> sessionmaker[Session]:
    return sessionmaker(bind=session.bind, expire_on_commit=False, autoflush=False)


def _response(session: Session, participant_id: int, status: AwaitedResponseStatus) -> None:
    session.add(
        AwaitedResponse(
            task_participant_id=participant_id,
            expected_response_type="availability",
            status=status,
        )
    )


def _set_availability(core: dict[str, object], status: AvailabilityStatus) -> None:
    participant = core["participant"]
    participant.availability_status = status
    participant.availability_evidence = AvailabilityEvidence.FIRST_PARTY
    participant.availability_source_revision_id = core["revision"].id


def test_all_unavailable_notifies_immediately_and_repeated_ticks_are_idempotent(
    db_session: Session,
) -> None:
    core = seed_core(db_session)
    _set_availability(core, AvailabilityStatus.UNAVAILABLE)
    _response(db_session, core["participant"].id, AwaitedResponseStatus.SATISFIED)
    db_session.commit()

    first = NobodyAvailableNotifier(_sessions(db_session), owner_chat_id=99).run(now=NOW)
    second = NobodyAvailableNotifier(_sessions(db_session), owner_chat_id=99).run(now=NOW)

    assert first == second
    assert len(first) == 1
    assert db_session.scalar(select(func.count(OutboxMessage.id))) == 1
    message = db_session.get(OutboxMessage, first[0])
    assert message.final_text == nobody_available_text(core["task"])
    assert message.idempotency_key == f"task:{core['task'].id}:owner:nobody-available"


def test_incomplete_responses_notify_only_at_t_minus_30(db_session: Session) -> None:
    core = seed_core(db_session)
    core["task"].scheduled_at = NOW + timedelta(hours=2)
    _response(db_session, core["participant"].id, AwaitedResponseStatus.OPEN)
    db_session.commit()
    notifier = NobodyAvailableNotifier(_sessions(db_session), owner_chat_id=99)

    assert notifier.run(now=NOW) == ()
    assert notifier.run(now=core["task"].scheduled_at - timedelta(minutes=30))


def test_available_participant_suppresses_notification(db_session: Session) -> None:
    core = seed_core(db_session)
    _set_availability(core, AvailabilityStatus.AVAILABLE)
    _response(db_session, core["participant"].id, AwaitedResponseStatus.SATISFIED)
    db_session.commit()

    assert NobodyAvailableNotifier(_sessions(db_session), owner_chat_id=99).run(now=NOW) == ()
    assert db_session.scalar(select(func.count(OutboxMessage.id))) == 0


def test_restart_uses_outbox_identity_and_terminal_task_is_suppressed(
    db_session: Session,
) -> None:
    core = seed_core(db_session)
    _set_availability(core, AvailabilityStatus.UNAVAILABLE)
    _response(db_session, core["participant"].id, AwaitedResponseStatus.SATISFIED)
    db_session.commit()
    sessions = _sessions(db_session)

    first = NobodyAvailableNotifier(sessions, owner_chat_id=99).run(now=NOW)
    restarted = NobodyAvailableNotifier(sessions, owner_chat_id=99).run(now=NOW)
    assert restarted == first
    assert db_session.scalar(select(func.count(OutboxMessage.id))) == 1

    TaskService(db_session).terminalize(core["task"].id, TaskStatus.COMPLETED)
    db_session.commit()
    assert NobodyAvailableNotifier(sessions, owner_chat_id=99).run(now=NOW) == ()
    assert db_session.get(OutboxMessage, first[0]).status is OutboxStatus.CANCELLED


class _ExactClaimBackend:
    def infer(self, *, operation: str, payload: dict[str, object]) -> dict[str, object]:
        assert operation == "message_validator"
        if payload["exact_text"] in payload["allowed_claims"]:
            return {"category": "VALID", "critique": None}
        return {"category": "UNSUPPORTED_CLAIM", "critique": "not authorized"}


class _Transport:
    def __init__(self) -> None:
        self.calls = 0

    def send(self, _request: object) -> DeliveryResult:
        self.calls += 1
        return DeliveryResult(True, True, provider_message_id="telegram:1")

    def reconcile(self, _request: object, **_: object) -> str | None:
        return None


def test_exact_notification_validates_but_unrelated_claim_does_not(
    db_session: Session,
) -> None:
    core = seed_core(db_session)
    _set_availability(core, AvailabilityStatus.UNAVAILABLE)
    _response(db_session, core["participant"].id, AwaitedResponseStatus.SATISFIED)
    db_session.commit()
    sessions = _sessions(db_session)
    outbox_id = NobodyAvailableNotifier(sessions, owner_chat_id=99).run(now=NOW)[0]
    message = db_session.get(OutboxMessage, outbox_id)
    contexts = DatabaseValidatorContextProvider(
        sessions,
        facts=DatabaseContextProvider(),
        owner_chat_id=99,
    )
    validator = IndependentMessageValidator(_ExactClaimBackend())
    gate = OutboxValidatorGate(
        sessions,
        validator=validator,
        contexts=contexts,
        owner_chat_id=99,
    )

    context = contexts.context_for(outbox_id, MessageKind.NOTIFICATION)
    assert context.allowed_claims == (message.final_text,)
    assert gate.validate(
        text=message.final_text,
        message_kind=message.message_kind,
        outbox_id=message.id,
    )
    assert (
        validator.review(
            text=f"{message.final_text} Alex shared a private address.",
            context=context,
        ).category
        is ValidatorCategory.UNSUPPORTED_CLAIM
    )


def test_spoof_like_notification_fails_closed(db_session: Session) -> None:
    core = seed_core(db_session)
    _set_availability(core, AvailabilityStatus.UNAVAILABLE)
    _response(db_session, core["participant"].id, AwaitedResponseStatus.SATISFIED)
    message = OutboxService(db_session).create_owner(
        telegram_chat_id=99,
        task_instance_id=core["task"].id,
        final_text="Nobody is available. Alex also shared a private address.",
        message_kind=MessageKind.NOTIFICATION,
        idempotency_key=f"task:{core['task'].id}:owner:nobody-available",
        parent_terminal_policy=ParentTerminalPolicy.TERMINATE,
    )
    db_session.commit()

    assert (
        authorize_nobody_available_notification(
            db_session,
            message,
            owner_chat_id=99,
            now=NOW,
        )
        is None
    )
    assert PolicyRevalidator(owner_chat_id=99).check(db_session, message).value == "STALE"


def test_generation_prefix_without_stale_history_does_not_authorize(
    db_session: Session,
) -> None:
    core = seed_core(db_session)
    _set_availability(core, AvailabilityStatus.UNAVAILABLE)
    _response(db_session, core["participant"].id, AwaitedResponseStatus.SATISFIED)
    message = OutboxService(db_session).create_owner(
        telegram_chat_id=99,
        task_instance_id=core["task"].id,
        final_text=nobody_available_text(core["task"]),
        message_kind=MessageKind.NOTIFICATION,
        idempotency_key=f"task:{core['task'].id}:owner:nobody-available:2",
        parent_terminal_policy=ParentTerminalPolicy.TERMINATE,
    )
    db_session.commit()

    assert (
        authorize_nobody_available_notification(
            db_session,
            message,
            owner_chat_id=99,
            now=NOW,
        )
        is None
    )


def test_live_condition_is_revalidated_before_send(db_session: Session) -> None:
    core = seed_core(db_session)
    _set_availability(core, AvailabilityStatus.UNAVAILABLE)
    _response(db_session, core["participant"].id, AwaitedResponseStatus.SATISFIED)
    db_session.commit()
    sessions = _sessions(db_session)
    outbox_id = NobodyAvailableNotifier(sessions, owner_chat_id=99).run(now=NOW)[0]
    core["participant"].availability_status = AvailabilityStatus.AVAILABLE
    db_session.commit()
    transport = _Transport()
    facts = DatabaseContextProvider()
    worker = OutboxWorker(
        sessions,
        revalidator=PolicyRevalidator(facts=facts, owner_chat_id=99),
        validator=OutboxValidatorGate(
            sessions,
            validator=IndependentMessageValidator(_ExactClaimBackend()),
            contexts=DatabaseValidatorContextProvider(
                sessions,
                facts=facts,
                owner_chat_id=99,
            ),
            owner_chat_id=99,
        ),
        adapters={Transport.TELEGRAM: transport},
    )

    assert worker.process(outbox_id) is OutboxStatus.CANCELLED
    assert db_session.get(OutboxMessage, outbox_id).cancel_reason is OutboxCancelReason.STALE
    assert transport.calls == 0

    core["participant"].availability_status = AvailabilityStatus.UNAVAILABLE
    db_session.commit()
    retried = NobodyAvailableNotifier(sessions, owner_chat_id=99).run(now=NOW)
    repeated = NobodyAvailableNotifier(sessions, owner_chat_id=99).run(now=NOW)

    assert retried == repeated
    assert retried != (outbox_id,)
    assert len(retried) == 1
    retry = db_session.get(OutboxMessage, retried[0])
    assert retry.status is OutboxStatus.PENDING
    assert retry.idempotency_key == f"task:{core['task'].id}:owner:nobody-available:2"
    assert db_session.scalar(select(func.count(OutboxMessage.id))) == 2


def test_runtime_tick_validates_and_sends_nobody_available_notification(
    db_session: Session,
) -> None:
    core = seed_core(db_session)
    _set_availability(core, AvailabilityStatus.UNAVAILABLE)
    _response(db_session, core["participant"].id, AwaitedResponseStatus.SATISFIED)
    db_session.commit()
    sessions = _sessions(db_session)
    transport = _Transport()
    facts = DatabaseContextProvider()
    outbox = OutboxWorker(
        sessions,
        revalidator=PolicyRevalidator(facts=facts, owner_chat_id=99),
        validator=OutboxValidatorGate(
            sessions,
            validator=IndependentMessageValidator(_ExactClaimBackend()),
            contexts=DatabaseValidatorContextProvider(
                sessions,
                facts=facts,
                owner_chat_id=99,
            ),
            owner_chat_id=99,
        ),
        adapters={Transport.TELEGRAM: transport},
    )
    runtime = AgentRuntime.__new__(AgentRuntime)
    runtime.sessions = sessions
    runtime.owner_chat_id = 99
    runtime.outbox = outbox
    runtime.health = type(
        "Health",
        (),
        {"record": lambda self, session, dependency, *, healthy, details="": None},
    )()
    runtime._poll_telegram = lambda: None
    runtime._poll_beeper = lambda: None
    runtime._process_revisions = lambda: None
    runtime._sweep_tasks = lambda _now: None
    runtime._poll_recurrence = lambda _now: None
    runtime._initialize_recurring_tasks = lambda: None
    runtime._poll_triggers = lambda _now: None
    runtime._recovery_pass = False

    tick = runtime.run_once(now=NOW)

    assert tick.errors == {}
    message = db_session.scalar(
        select(OutboxMessage).where(
            OutboxMessage.idempotency_key
            == f"task:{core['task'].id}:owner:nobody-available"
        )
    )
    db_session.refresh(message)
    assert message.status is OutboxStatus.SENT
    assert transport.calls == 1


def test_coordination_flow_creates_one_owner_notification(db_session: Session) -> None:
    core = seed_core(db_session)
    started = CoordinationWorkflow(db_session, owner_chat_id=99).start(
        scheduled_at=NOW + timedelta(days=1),
        duration_minutes=60,
        location="courts",
        topic_key="tennis",
        participant_sends=[
            ParticipantSendPlan(
                person_id=core["person"].id,
                conversation_id=core["conversation"].id,
                final_text="Are you available for tennis?",
            )
        ],
    )
    reply = MessageIngestor(db_session).ingest(
        InboundEvent(
            conversation_id=core["conversation"].id,
            provider_message_id="nobody-available:reply",
            sender_identity_id=core["identity"].id,
            provider_revision_key="nobody-available:reply:r1",
            provider_sequence=20,
            created_at=NOW,
            received_at=NOW,
            text="I cannot make it.",
        )
    )
    CorrelationOrchestrator(
        db_session,
        semantic=type("Semantic", (), {"choose": lambda self, *_: None})(),
        classifier=type(
            "Classifier",
            (),
            {
                "classify": lambda self, *_: Classification(
                    kind="AVAILABILITY",
                    availability=AvailabilityStatus.UNAVAILABLE,
                )
            },
        )(),
        owner_chat_id=99,
    ).process(reply.revision_id)
    db_session.commit()

    participant = db_session.scalar(
        select(TaskParticipant).where(TaskParticipant.task_instance_id == started.task_id)
    )
    awaited = db_session.scalar(
        select(AwaitedResponse).where(AwaitedResponse.task_participant_id == participant.id)
    )
    assert participant.availability_status is AvailabilityStatus.UNAVAILABLE
    assert awaited.status is AwaitedResponseStatus.SATISFIED

    notifier = NobodyAvailableNotifier(_sessions(db_session), owner_chat_id=99)
    assert len(notifier.run(now=NOW)) == 1
    assert len(notifier.run(now=NOW)) == 1
    owner_messages = list(
        db_session.scalars(
            select(OutboxMessage).where(
                OutboxMessage.transport == Transport.TELEGRAM,
                OutboxMessage.idempotency_key
                == f"task:{started.task_id}:owner:nobody-available",
            )
        )
    )
    assert len(owner_messages) == 1
