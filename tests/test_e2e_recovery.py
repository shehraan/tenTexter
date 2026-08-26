from __future__ import annotations

from datetime import timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from ten_texter.correlation import AtomicProposal, Classification, CorrelationOrchestrator
from ten_texter.domain import AvailabilityService, ContactRuleService, DecisionService, TaskService
from ten_texter.enums import (
    AvailabilityEvidence,
    AvailabilityStatus,
    AwaitedResponseStatus,
    ContactRuleScope,
    ContactRuleSource,
    ContentSupport,
    MessageKind,
    OutboxStatus,
    ProcessingFailureType,
    ProcessingStatus,
    ProposalStatus,
    RuleStrength,
    TargetSelector,
    TaskStatus,
    Transport,
    TriggerActionType,
    TriggerExecutionStatus,
    TriggerStatus,
)
from ten_texter.inbound import InboundEvent, MessageIngestor, RevisionProcessor
from ten_texter.models import (
    AwaitedResponse,
    DecisionRequest,
    MessageRevision,
    OutboxMessage,
    Proposal,
    TaskParticipant,
    TaskTrigger,
)
from ten_texter.outbox import (
    DeliveryRequest,
    DeliveryResult,
    OutboxService,
    OutboxWorker,
)
from ten_texter.policy import PolicyRevalidator
from ten_texter.triggers import TriggerWorker
from ten_texter.workflows import CoordinationWorkflow, ParticipantSendPlan
from tests.test_schema import NOW, seed_core


class Semantic:
    def choose(self, *_: object) -> int | None:
        return None


class Classifier:
    def __init__(self, result: Classification):
        self.result = result

    def classify(self, *_: object) -> Classification:
        return self.result


class Validator:
    def __init__(self, available: bool = True):
        self.available = available

    def validate(self, **_: object) -> bool:
        if not self.available:
            raise RuntimeError("validator unavailable")
        return True


class TransportAdapter:
    def __init__(self, results: list[DeliveryResult] | None = None):
        self.results = results or []
        self.sent: list[DeliveryRequest] = []

    def send(self, request: DeliveryRequest) -> DeliveryResult:
        self.sent.append(request)
        if self.results:
            return self.results.pop(0)
        return DeliveryResult(True, True, provider_message_id=f"provider:{request.outbox_id}")

    def reconcile(self, request: DeliveryRequest, **_: object) -> str | None:
        return f"provider:{request.outbox_id}"


def factory(session: Session) -> sessionmaker[Session]:
    return sessionmaker(bind=session.bind, expire_on_commit=False, autoflush=False)


def worker(session: Session, transport: TransportAdapter, validator: Validator | None = None) -> OutboxWorker:
    return OutboxWorker(
        factory(session),
        revalidator=PolicyRevalidator(),
        validator=validator or Validator(),
        adapters={Transport.BEEPER: transport, Transport.TELEGRAM: transport},
    )


def test_owner_plan_to_participant_reply_to_owner_notification(db_session: Session) -> None:
    core = seed_core(db_session)
    workflow = CoordinationWorkflow(db_session, owner_chat_id=99)
    started = workflow.start(
        scheduled_at=NOW + timedelta(days=1),
        duration_minutes=60,
        location="courts",
        topic_key="tennis",
        participant_sends=[
            ParticipantSendPlan(
                person_id=core["person"].id,
                conversation_id=core["conversation"].id,
                final_text="Are you available for tennis tomorrow?",
            )
        ],
    )
    db_session.commit()
    transport = TransportAdapter()
    outbox_worker = worker(db_session, transport)
    assert outbox_worker.process(started.outbox_ids[0]) is OutboxStatus.SENT

    reply = MessageIngestor(db_session).ingest(
        InboundEvent(
            conversation_id=core["conversation"].id,
            provider_message_id="reply:yes",
            sender_identity_id=core["identity"].id,
            provider_revision_key="reply:yes:r1",
            provider_sequence=20,
            created_at=NOW,
            received_at=NOW,
            text="Yes, I am available.",
            provider_reply_to_message_id=f"provider:{started.outbox_ids[0]}",
        )
    )
    db_session.commit()
    revision_worker = RevisionProcessor(factory(db_session))
    claim = revision_worker.claim(reply.revision_id, now=NOW)
    assert claim is not None

    def apply(session: Session, revision: MessageRevision) -> None:
        CorrelationOrchestrator(
            session,
            semantic=Semantic(),
            classifier=Classifier(
                Classification(kind="AVAILABILITY", availability=AvailabilityStatus.AVAILABLE)
            ),
            owner_chat_id=99,
        ).process(revision.id)

    assert revision_worker.commit(claim, apply)
    db_session.expire_all()
    participant = db_session.scalar(
        select(TaskParticipant).where(TaskParticipant.task_instance_id == started.task_id)
    )
    assert participant.availability_status is AvailabilityStatus.AVAILABLE
    owner_message_id = CoordinationWorkflow(db_session, owner_chat_id=99).notify_owner(
        started.task_id,
        "Alex is available.",
        key="availability-summary",
    )
    db_session.commit()
    assert outbox_worker.process(owner_message_id) is OutboxStatus.SENT
    assert [request.transport for request in transport.sent] == [Transport.BEEPER, Transport.TELEGRAM]


def test_counterproposal_creates_owner_decision_and_approval_revalidates(db_session: Session) -> None:
    core = seed_core(db_session)
    response = AwaitedResponse(
        task_participant_id=core["participant"].id,
        expected_response_type="scheduling",
        status=AwaitedResponseStatus.OPEN,
    )
    db_session.add(response)
    db_session.flush()
    result = CorrelationOrchestrator(
        db_session,
        semantic=Semantic(),
        classifier=Classifier(
            Classification(
                kind="COUNTERPROPOSAL",
                proposals=(AtomicProposal("location", "SET", None, "park"),),
            )
        ),
        owner_chat_id=99,
    ).process(core["revision"].id)
    assert result.outcome == "CORRELATED"
    proposal = db_session.scalar(select(Proposal).where(Proposal.task_instance_id == core["task"].id))
    decision = db_session.scalar(select(DecisionRequest).where(DecisionRequest.proposal_id == proposal.id))
    CoordinationWorkflow(db_session, owner_chat_id=99).approve_proposal(decision.id)
    assert proposal.status is ProposalStatus.ACCEPTED
    assert core["task"].location == "park"


def test_message_edit_reverses_prior_availability_via_revision_lineage(db_session: Session) -> None:
    core = seed_core(db_session)
    response = AwaitedResponse(
        task_participant_id=core["participant"].id,
        expected_response_type="availability",
        status=AwaitedResponseStatus.OPEN,
    )
    db_session.add(response)
    db_session.flush()
    CorrelationOrchestrator(
        db_session,
        semantic=Semantic(),
        classifier=Classifier(
            Classification(kind="AVAILABILITY", availability=AvailabilityStatus.AVAILABLE)
        ),
    ).process(core["revision"].id)
    edit = MessageRevision(
        message_id=core["message"].id,
        provider_revision_key="edit:r2",
        provider_sequence=2,
        content_hash="d" * 64,
        text="Actually, I cannot make it.",
        content_support=ContentSupport.SUPPORTED,
        processing_status=ProcessingStatus.PENDING,
    )
    db_session.add(edit)
    db_session.flush()
    core["message"].current_revision_id = edit.id
    db_session.flush()
    result = CorrelationOrchestrator(
        db_session,
        semantic=Semantic(),
        classifier=Classifier(
            Classification(kind="AVAILABILITY", availability=AvailabilityStatus.UNAVAILABLE)
        ),
    ).process(edit.id)
    assert result.awaited_response_id == response.id
    assert core["participant"].availability_status is AvailabilityStatus.UNAVAILABLE
    assert core["participant"].availability_source_revision_id == edit.id


class TriggerGenerator:
    def generate(self, **_: object) -> str:
        return "Checking in because nobody is available yet."


class TriggerValidator:
    def validate(self, **_: object) -> bool:
        return True


def test_nobody_available_trigger_and_terminalization_race(db_session: Session) -> None:
    core = seed_core(db_session)
    trigger = TaskTrigger(
        task_instance_id=core["task"].id,
        condition_json={"kind": "NOBODY_AVAILABLE"},
        target_selector=TargetSelector.ALL_PARTICIPANTS,
        action_type=TriggerActionType.SEND_MESSAGE,
        action_payload_json={"goal": "follow up"},
        next_run_at=NOW,
        status=TriggerStatus.ACTIVE,
    )
    db_session.add(trigger)
    db_session.commit()
    trigger_worker = TriggerWorker(
        factory(db_session),
        generator=TriggerGenerator(),
        validator=TriggerValidator(),
        owner_chat_id=99,
    )
    claim = trigger_worker.claim(trigger.id, "nobody:1", now=NOW)
    assert claim is not None
    assert trigger_worker.run_claim(claim) is TriggerExecutionStatus.COMPLETED
    assert db_session.scalar(select(func.count(OutboxMessage.id))) == 1

    second = TaskTrigger(
        task_instance_id=core["task"].id,
        condition_json={"kind": "ALWAYS"},
        target_selector=TargetSelector.ALL_PARTICIPANTS,
        action_type=TriggerActionType.SEND_MESSAGE,
        action_payload_json={"goal": "race"},
        next_run_at=NOW,
        status=TriggerStatus.ACTIVE,
    )
    db_session.add(second)
    db_session.commit()
    race_claim = trigger_worker.claim(second.id, "race:1", now=NOW)
    assert race_claim is not None
    TaskService(db_session).terminalize(core["task"].id, TaskStatus.CANCELLED)
    db_session.commit()
    assert trigger_worker.run_claim(race_claim) is TriggerExecutionStatus.CANCELLED
    assert db_session.scalar(select(func.count(OutboxMessage.id))) == 1


def test_contact_boundary_blocks_send_before_transport(db_session: Session) -> None:
    core = seed_core(db_session)
    ContactRuleService(db_session).create(
        person_id=core["person"].id,
        scope=ContactRuleScope.GLOBAL,
        type="DO_NOT_CONTACT",
        value="stop",
        source=ContactRuleSource.PARTICIPANT_REQUESTED,
        strength=RuleStrength.STRONG,
    )
    message = OutboxService(db_session).create_beeper(
        task_instance_id=core["task"].id,
        conversation_id=core["conversation"].id,
        participant_ids=[core["participant"].id],
        final_text="blocked",
        message_kind=MessageKind.INITIAL,
        idempotency_key="boundary",
    )
    db_session.commit()
    transport = TransportAdapter()
    assert worker(db_session, transport).process(message.id) is OutboxStatus.CANCELLED
    assert transport.sent == []


def test_model_validator_and_transport_outages_recover_without_duplicate_send(db_session: Session) -> None:
    core = seed_core(db_session)
    ingested = MessageIngestor(db_session).ingest(
        InboundEvent(
            conversation_id=core["conversation"].id,
            provider_message_id="outage-reply",
            sender_identity_id=core["identity"].id,
            provider_revision_key="outage-r1",
            provider_sequence=30,
            created_at=NOW,
            received_at=NOW,
            text="yes",
        )
    )
    db_session.commit()
    revisions = RevisionProcessor(factory(db_session))
    first_claim = revisions.claim(ingested.revision_id, now=NOW)
    assert first_claim is not None
    assert revisions.fail(
        first_claim,
        ProcessingFailureType.MODEL_ERROR,
        "primary model offline",
        retryable=True,
    )
    recovered_claim = revisions.claim(ingested.revision_id, now=NOW + timedelta(seconds=1))
    assert recovered_claim is not None
    assert revisions.commit(recovered_claim, lambda *_: None)

    message = OutboxService(db_session).create_beeper(
        task_instance_id=core["task"].id,
        conversation_id=core["conversation"].id,
        participant_ids=[core["participant"].id],
        final_text="retry safely",
        message_kind=MessageKind.INITIAL,
        idempotency_key="outage-send",
    )
    db_session.commit()
    validator = Validator(available=False)
    transport = TransportAdapter(
        [
            DeliveryResult(False, False, definitely_not_sent=True, error="Beeper offline"),
            DeliveryResult(True, True, provider_message_id="provider:recovered"),
        ]
    )
    outbox_worker = worker(db_session, transport, validator)
    assert outbox_worker.process(message.id) is OutboxStatus.PENDING
    assert transport.sent == []
    validator.available = True
    assert outbox_worker.process(message.id) is OutboxStatus.PENDING
    assert outbox_worker.process(message.id) is OutboxStatus.SENT
    assert len(transport.sent) == 2
