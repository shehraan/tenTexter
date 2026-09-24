from __future__ import annotations

from datetime import timedelta

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from ten_texter.domain import DomainError, TaskService, utc_now
from ten_texter.enums import (
    MessageKind,
    OutboxStatus,
    ParentTerminalPolicy,
    TargetSelector,
    TaskStatus,
    TriggerActionType,
    TriggerExecutionStatus,
    TriggerStatus,
    Transport,
)
from ten_texter.models import OutboxMessage, TaskTrigger, TriggerExecution
from ten_texter.outbox import (
    AllowingRevalidator,
    DeliveryRequest,
    DeliveryResult,
    OutboxService,
    OutboxWorker,
)
from tests.test_schema import seed_core


class Validator:
    def __init__(self, valid: bool = True):
        self.valid = valid
        self.calls = 0

    def validate(self, **_: object) -> bool:
        self.calls += 1
        return self.valid


class FakeTransport:
    def __init__(self, results: list[DeliveryResult], reconciled_id: str | None = None):
        self.results = results
        self.requests: list[DeliveryRequest] = []
        self.reconciled_id = reconciled_id

    def send(self, request: DeliveryRequest) -> DeliveryResult:
        self.requests.append(request)
        return self.results.pop(0)

    def reconcile(self, request: DeliveryRequest, **_: object) -> str | None:
        self.requests.append(request)
        return self.reconciled_id


def worker_for(session: Session, adapter: FakeTransport, validator: Validator | None = None) -> OutboxWorker:
    factory = sessionmaker(bind=session.bind, expire_on_commit=False, autoflush=False)
    return OutboxWorker(
        factory,
        revalidator=AllowingRevalidator(),
        validator=validator or Validator(),
        adapters={Transport.BEEPER: adapter, Transport.TELEGRAM: adapter},
    )


def create_send(session: Session, core: dict[str, object], *, key: str = "send-1") -> OutboxMessage:
    message = OutboxService(session).create_beeper(
        task_instance_id=core["task"].id,
        conversation_id=core["conversation"].id,
        participant_ids=[core["participant"].id],
        final_text="Are you available?",
        message_kind=MessageKind.INITIAL,
        idempotency_key=key,
    )
    session.commit()
    return message


def test_duplicate_logical_send_collapses(db_session: Session) -> None:
    core = seed_core(db_session)
    service = OutboxService(db_session)
    one = service.create_beeper(
        task_instance_id=core["task"].id,
        conversation_id=core["conversation"].id,
        participant_ids=[core["participant"].id],
        final_text="hello",
        message_kind=MessageKind.INITIAL,
        idempotency_key="same",
    )
    two = service.create_beeper(
        task_instance_id=core["task"].id,
        conversation_id=core["conversation"].id,
        participant_ids=[core["participant"].id],
        final_text="hello",
        message_kind=MessageKind.INITIAL,
        idempotency_key="same",
    )
    assert one.id == two.id
    with pytest.raises(DomainError, match="idempotency key reused for a different logical send"):
        service.create_beeper(
            task_instance_id=core["task"].id,
            conversation_id=core["conversation"].id,
            participant_ids=[core["participant"].id],
            final_text="different",
            message_kind=MessageKind.INITIAL,
            idempotency_key="same",
        )


def test_trigger_execution_can_own_multiple_distinct_sends(db_session: Session) -> None:
    core = seed_core(db_session)
    trigger = TaskTrigger(
        task_instance_id=core["task"].id,
        condition_json={"kind": "AT_TIME"},
        target_selector=TargetSelector.ALL_PARTICIPANTS,
        action_type=TriggerActionType.SEND_MESSAGE,
        action_payload_json={},
        status=TriggerStatus.ACTIVE,
    )
    db_session.add(trigger)
    db_session.flush()
    execution = TriggerExecution(
        task_trigger_id=trigger.id,
        fire_key="slot",
        status=TriggerExecutionStatus.COMPLETED,
    )
    db_session.add(execution)
    db_session.flush()
    service = OutboxService(db_session)
    sends = [
        service.create_beeper(
            task_instance_id=core["task"].id,
            conversation_id=core["conversation"].id,
            participant_ids=[core["participant"].id],
            final_text=f"message {index}",
            message_kind=MessageKind.REMINDER,
            idempotency_key=f"execution:{execution.id}:target:{index}",
            trigger_execution_id=execution.id,
        )
        for index in range(2)
    ]
    assert sends[0].id != sends[1].id


def test_successful_send_and_validator_gate(db_session: Session) -> None:
    core = seed_core(db_session)
    message = create_send(db_session, core)
    validator = Validator(valid=False)
    adapter = FakeTransport([DeliveryResult(True, True, provider_message_id="provider:1")])
    worker = worker_for(db_session, adapter, validator)
    assert worker.process(message.id) is OutboxStatus.PENDING
    assert adapter.requests == []
    validator.valid = True
    assert worker.process(message.id) is OutboxStatus.SENT
    assert len(adapter.requests) == 1


def test_uncertain_boundary_never_blindly_retries(db_session: Session) -> None:
    core = seed_core(db_session)
    message = create_send(db_session, core)
    adapter = FakeTransport([DeliveryResult(False, True, pending_provider_id="pending:1")], reconciled_id="final:1")
    worker = worker_for(db_session, adapter)
    assert worker.process(message.id) is OutboxStatus.RECONCILING
    assert worker.process(message.id) is OutboxStatus.RECONCILING
    assert len(adapter.requests) == 1
    assert worker.reconcile(message.id) is OutboxStatus.SENT


def test_definitely_not_sent_can_retry(db_session: Session) -> None:
    core = seed_core(db_session)
    message = create_send(db_session, core)
    adapter = FakeTransport([
        DeliveryResult(False, False, definitely_not_sent=True),
        DeliveryResult(True, True, provider_message_id="provider:2"),
    ])
    worker = worker_for(db_session, adapter)
    assert worker.process(message.id) is OutboxStatus.PENDING
    assert worker.process(message.id) is OutboxStatus.SENT
    assert len(adapter.requests) == 2


def test_terminal_parent_cancels_before_send(db_session: Session) -> None:
    core = seed_core(db_session)
    message = create_send(db_session, core)
    TaskService(db_session).terminalize(core["task"].id, TaskStatus.CANCELLED)
    db_session.commit()
    adapter = FakeTransport([])
    assert worker_for(db_session, adapter).process(message.id) is OutboxStatus.CANCELLED
    assert adapter.requests == []


def test_expired_worker_with_unfinished_attempt_reconciles(db_session: Session) -> None:
    core = seed_core(db_session)
    message = create_send(db_session, core)
    # Simulate the durable state immediately before the transport call.
    from ten_texter.models import OutboxDeliveryAttempt

    message = db_session.get(OutboxMessage, message.id)
    message.status = OutboxStatus.SENDING
    message.lease_expires_at = utc_now() - timedelta(seconds=1)
    db_session.add(OutboxDeliveryAttempt(outbox_message_id=message.id))
    db_session.commit()
    assert worker_for(db_session, FakeTransport([])).reclaim_expired(message.id) is OutboxStatus.RECONCILING


def test_correction_requires_owner_approval_and_linear_same_destination(db_session: Session) -> None:
    core = seed_core(db_session)
    original = create_send(db_session, core)
    original = db_session.get(OutboxMessage, original.id)
    original.status = OutboxStatus.SENT
    db_session.commit()
    service = OutboxService(db_session)
    with pytest.raises(DomainError):
        service.create_beeper(
            task_instance_id=core["task"].id,
            conversation_id=core["conversation"].id,
            participant_ids=[core["participant"].id],
            final_text="Correction",
            message_kind=MessageKind.CORRECTION,
            idempotency_key="correction:no-approval",
            corrects_outbox_message_id=original.id,
        )
    correction = service.create_beeper(
        task_instance_id=core["task"].id,
        conversation_id=core["conversation"].id,
        participant_ids=[core["participant"].id],
        final_text="Correction",
        message_kind=MessageKind.CORRECTION,
        idempotency_key="correction:approved",
        corrects_outbox_message_id=original.id,
        owner_approved_correction=True,
    )
    assert correction.corrects_outbox_message_id == original.id
    with pytest.raises(DomainError):
        service.create_beeper(
            task_instance_id=core["task"].id,
            conversation_id=core["conversation"].id,
            participant_ids=[core["participant"].id],
            final_text="Second direct correction",
            message_kind=MessageKind.CORRECTION,
            idempotency_key="correction:duplicate",
            corrects_outbox_message_id=original.id,
            owner_approved_correction=True,
        )


def test_database_rejects_final_text_mutation(db_session: Session) -> None:
    core = seed_core(db_session)
    message = create_send(db_session, core)
    stored = db_session.get(OutboxMessage, message.id)
    stored.final_text = "mutated"
    with pytest.raises(IntegrityError):
        db_session.flush()
