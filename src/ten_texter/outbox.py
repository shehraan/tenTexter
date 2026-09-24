from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from ten_texter.decision_prompts import uncertain_delivery_prompt
from ten_texter.domain import DecisionService, DomainError, utc_now
from ten_texter.enums import (
    AttemptResult,
    DecisionCloseReason,
    DecisionStatus,
    DestinationKind,
    MessageKind,
    OutboxCancelReason,
    OutboxStatus,
    ParentTerminalPolicy,
    TaskStatus,
    Transport,
)
from ten_texter.models import (
    BeeperDeliveryAttemptDetail,
    BeeperOutboxDestination,
    OutboxDeliveryAttempt,
    OutboxMessage,
    OutboxMessageParticipant,
    TaskInstance,
    TaskParticipant,
    TelegramOutboxDestination,
)


class PreSendDecision(str, Enum):
    READY = "READY"
    STALE = "STALE"
    POLICY_BLOCKED = "POLICY_BLOCKED"
    UNAVAILABLE = "UNAVAILABLE"
    AWAITING_OWNER = "AWAITING_OWNER"


class PreSendRevalidator(Protocol):
    def check(self, session: Session, message: OutboxMessage) -> PreSendDecision: ...


class IndependentTextValidator(Protocol):
    def validate(self, *, text: str, message_kind: MessageKind, outbox_id: int) -> bool: ...


@dataclass(frozen=True, slots=True)
class DeliveryRequest:
    outbox_id: int
    transport: Transport
    destination: str
    text: str
    idempotency_key: str


@dataclass(frozen=True, slots=True)
class DeliveryResult:
    success: bool
    boundary_crossed: bool
    definitely_not_sent: bool = False
    provider_message_id: str | None = None
    pending_provider_id: str | None = None
    error: str | None = None


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


class TransportAdapter(Protocol):
    def send(self, request: DeliveryRequest) -> DeliveryResult: ...

    def reconcile(self, request: DeliveryRequest, *, provider_message_id: str | None, pending_provider_id: str | None) -> str | None: ...


class AllowingRevalidator:
    """Useful only for tests and composition after real policy hooks are installed."""

    def check(self, session: Session, message: OutboxMessage) -> PreSendDecision:
        if message.task_instance_id is not None:
            task = session.get(TaskInstance, message.task_instance_id)
            if task is None:
                return PreSendDecision.STALE
            if task.status is not TaskStatus.ACTIVE and (
                message.transport is Transport.BEEPER
                or message.parent_terminal_policy is ParentTerminalPolicy.TERMINATE
            ):
                return PreSendDecision.STALE
        return PreSendDecision.READY


class OutboxService:
    def __init__(self, session: Session):
        self.session = session

    def create_beeper(
        self,
        *,
        task_instance_id: int,
        conversation_id: int,
        participant_ids: list[int],
        final_text: str,
        message_kind: MessageKind,
        idempotency_key: str,
        parent_terminal_policy: ParentTerminalPolicy = ParentTerminalPolicy.TERMINATE,
        trigger_execution_id: int | None = None,
        corrects_outbox_message_id: int | None = None,
        owner_approved_correction: bool = False,
    ) -> OutboxMessage:
        if not participant_ids:
            raise DomainError("participant send requires logical targets")
        participants = list(
            self.session.scalars(select(TaskParticipant).where(TaskParticipant.id.in_(participant_ids)))
        )
        if len(participants) != len(set(participant_ids)):
            raise DomainError("unknown participant target")
        if any(
            participant.task_instance_id != task_instance_id or participant.conversation_id != conversation_id
            for participant in participants
        ):
            raise DomainError("all targets must be pinned to the exact task destination")
        existing = self._existing(Transport.BEEPER, idempotency_key)
        if existing is not None:
            self._assert_same(existing, final_text, message_kind, task_instance_id)
            return existing
        if message_kind is MessageKind.CORRECTION:
            self._validate_correction(
                corrects_outbox_message_id,
                task_instance_id=task_instance_id,
                transport=Transport.BEEPER,
                destination=str(conversation_id),
                owner_approved=owner_approved_correction,
            )
        message = OutboxMessage(
            task_instance_id=task_instance_id,
            transport=Transport.BEEPER,
            destination_kind=DestinationKind.PARTICIPANT,
            final_text=final_text,
            message_kind=message_kind,
            status=OutboxStatus.PENDING,
            idempotency_key=idempotency_key,
            parent_terminal_policy=parent_terminal_policy,
            trigger_execution_id=trigger_execution_id,
            corrects_outbox_message_id=corrects_outbox_message_id,
        )
        self.session.add(message)
        self.session.flush()
        self.session.add(BeeperOutboxDestination(outbox_message_id=message.id, conversation_id=conversation_id))
        self.session.add_all(
            [OutboxMessageParticipant(outbox_message_id=message.id, task_participant_id=value) for value in participant_ids]
        )
        self.session.flush()
        return message

    def create_owner(
        self,
        *,
        telegram_chat_id: int,
        final_text: str,
        message_kind: MessageKind,
        idempotency_key: str,
        task_instance_id: int | None = None,
        parent_terminal_policy: ParentTerminalPolicy = ParentTerminalPolicy.SURVIVE,
        trigger_execution_id: int | None = None,
        corrects_outbox_message_id: int | None = None,
        owner_approved_correction: bool = False,
    ) -> OutboxMessage:
        existing = self._existing(Transport.TELEGRAM, idempotency_key)
        if existing is not None:
            self._assert_same(existing, final_text, message_kind, task_instance_id)
            return existing
        if message_kind is MessageKind.CORRECTION:
            self._validate_correction(
                corrects_outbox_message_id,
                task_instance_id=task_instance_id,
                transport=Transport.TELEGRAM,
                destination=str(telegram_chat_id),
                owner_approved=owner_approved_correction,
            )
        message = OutboxMessage(
            task_instance_id=task_instance_id,
            transport=Transport.TELEGRAM,
            destination_kind=DestinationKind.OWNER,
            final_text=final_text,
            message_kind=message_kind,
            status=OutboxStatus.PENDING,
            idempotency_key=idempotency_key,
            parent_terminal_policy=parent_terminal_policy,
            trigger_execution_id=trigger_execution_id,
            corrects_outbox_message_id=corrects_outbox_message_id,
        )
        self.session.add(message)
        self.session.flush()
        self.session.add(TelegramOutboxDestination(outbox_message_id=message.id, telegram_chat_id=telegram_chat_id))
        self.session.flush()
        return message

    def _existing(self, transport: Transport, idempotency_key: str) -> OutboxMessage | None:
        return self.session.scalar(
            select(OutboxMessage).where(
                OutboxMessage.transport == transport,
                OutboxMessage.idempotency_key == idempotency_key,
            )
        )

    @staticmethod
    def _assert_same(existing: OutboxMessage, final_text: str, kind: MessageKind, task_id: int | None) -> None:
        if (existing.final_text, existing.message_kind, existing.task_instance_id) != (final_text, kind, task_id):
            raise DomainError("idempotency key reused for a different logical send")

    def _validate_correction(
        self,
        target_id: int | None,
        *,
        task_instance_id: int | None,
        transport: Transport,
        destination: str,
        owner_approved: bool,
    ) -> None:
        if not owner_approved:
            raise DomainError("v1 corrections require owner approval")
        target = self.session.get(OutboxMessage, target_id) if target_id is not None else None
        if target is None or target.status is not OutboxStatus.SENT:
            raise DomainError("correction target must be SENT")
        if target.task_instance_id != task_instance_id or target.transport is not transport:
            raise DomainError("correction target must share task and transport")
        if self._destination(target) != destination:
            raise DomainError("correction target must share exact destination")
        if self.session.scalar(
            select(OutboxMessage.id).where(OutboxMessage.corrects_outbox_message_id == target.id)
        ) is not None:
            raise DomainError("message has already been directly corrected")
        seen: set[int] = set()
        cursor = target
        while cursor.corrects_outbox_message_id is not None:
            if cursor.id in seen:
                raise DomainError("correction cycle")
            seen.add(cursor.id)
            cursor = self.session.get(OutboxMessage, cursor.corrects_outbox_message_id)
            if cursor is None:
                raise DomainError("broken correction chain")

    def _destination(self, message: OutboxMessage) -> str:
        if message.transport is Transport.BEEPER:
            child = self.session.get(BeeperOutboxDestination, message.id)
            if child is None:
                raise DomainError("missing Beeper destination")
            return str(child.conversation_id)
        child = self.session.get(TelegramOutboxDestination, message.id)
        if child is None:
            raise DomainError("missing Telegram destination")
        return str(child.telegram_chat_id)


class OutboxWorker:
    def __init__(
        self,
        sessions: sessionmaker[Session],
        *,
        revalidator: PreSendRevalidator,
        validator: IndependentTextValidator,
        adapters: dict[Transport, TransportAdapter],
        lease_duration: timedelta = timedelta(seconds=30),
        pending_reconciliation_grace: timedelta = timedelta(minutes=5),
        raise_validation_errors: bool = False,
        owner_chat_id: int | None = None,
    ):
        self.sessions = sessions
        self.revalidator = revalidator
        self.validator = validator
        self.adapters = adapters
        self.lease_duration = lease_duration
        self.pending_reconciliation_grace = pending_reconciliation_grace
        self.raise_validation_errors = raise_validation_errors
        self.owner_chat_id = owner_chat_id

    def process(self, outbox_id: int) -> OutboxStatus:
        snapshot = self._snapshot(outbox_id)
        if snapshot is None:
            raise DomainError("outbox message not found")
        if snapshot["status"] is not OutboxStatus.PENDING:
            return snapshot["status"]
        policy_token: object | None = None
        token_builder = getattr(self.revalidator, "context_token", None)
        with self.sessions.begin() as session:
            message = session.get(OutboxMessage, outbox_id)
            if message is None or message.status is not OutboxStatus.PENDING:
                return message.status if message is not None else OutboxStatus.CANCELLED
            initial_decision = self.revalidator.check(session, message)
            if initial_decision in {PreSendDecision.UNAVAILABLE, PreSendDecision.AWAITING_OWNER}:
                return OutboxStatus.PENDING
            if initial_decision in {
                PreSendDecision.STALE,
                PreSendDecision.POLICY_BLOCKED,
            }:
                message.status = OutboxStatus.CANCELLED
                message.cancel_reason = (
                    OutboxCancelReason.STALE
                    if initial_decision is PreSendDecision.STALE
                    else OutboxCancelReason.POLICY_BLOCKED
                )
                return message.status
            if token_builder is not None:
                policy_token = token_builder(session, message)
        try:
            valid = self.validator.validate(
                text=snapshot["text"], message_kind=snapshot["kind"], outbox_id=outbox_id
            )
        except Exception:
            if self.raise_validation_errors:
                raise
            return OutboxStatus.PENDING
        if not valid:
            return OutboxStatus.PENDING

        with self.sessions.begin() as session:
            message = session.get(OutboxMessage, outbox_id)
            assert message is not None
            if message.status is not OutboxStatus.PENDING or message.final_text != snapshot["text"]:
                return message.status
            decision = self.revalidator.check(session, message)
            if (
                decision is PreSendDecision.READY
                and token_builder is not None
                and token_builder(session, message) != policy_token
            ):
                decision = PreSendDecision.POLICY_BLOCKED
            if decision in {PreSendDecision.UNAVAILABLE, PreSendDecision.AWAITING_OWNER}:
                return OutboxStatus.PENDING
            if decision in {PreSendDecision.STALE, PreSendDecision.POLICY_BLOCKED}:
                message.status = OutboxStatus.CANCELLED
                message.cancel_reason = (
                    OutboxCancelReason.STALE
                    if decision is PreSendDecision.STALE
                    else OutboxCancelReason.POLICY_BLOCKED
                )
                return message.status
            message.status = OutboxStatus.SENDING
            message.lease_expires_at = utc_now() + self.lease_duration
            attempt = OutboxDeliveryAttempt(outbox_message_id=message.id)
            session.add(attempt)
            session.flush()
            request = self._request(session, message)
            attempt_id = attempt.id

        adapter = self.adapters.get(request.transport)
        if adapter is None:
            result = DeliveryResult(False, False, definitely_not_sent=True, error="adapter unavailable")
        else:
            try:
                result = adapter.send(request)
            except Exception as exc:
                result = DeliveryResult(False, True, error=str(exc))
        return self._finish(outbox_id, attempt_id, result)

    def reclaim_expired(self, outbox_id: int, *, at: datetime | None = None) -> OutboxStatus:
        timestamp = at or utc_now()
        with self.sessions.begin() as session:
            message = session.get(OutboxMessage, outbox_id)
            if message is None:
                raise DomainError("outbox message not found")
            if message.status is not OutboxStatus.SENDING or message.lease_expires_at is None:
                return message.status
            lease = message.lease_expires_at
            if lease.tzinfo is None:
                lease = lease.replace(tzinfo=timestamp.tzinfo)
            if lease > timestamp:
                return message.status
            attempt = session.scalar(
                select(OutboxDeliveryAttempt).where(
                    OutboxDeliveryAttempt.outbox_message_id == message.id,
                    OutboxDeliveryAttempt.finished_at.is_(None),
                )
            )
            message.lease_expires_at = None
            if attempt is None:
                message.status = OutboxStatus.PENDING
            else:
                attempt.finished_at = timestamp
                attempt.result = AttemptResult.ABANDONED
                attempt.failure_type = "WORKER_LOST"
                message.status = OutboxStatus.RECONCILING
            return message.status

    def reconcile(self, outbox_id: int, *, at: datetime | None = None) -> OutboxStatus:
        with self.sessions() as read_session:
            message = read_session.get(OutboxMessage, outbox_id)
            if message is None:
                raise DomainError("outbox message not found")
            if message.status is not OutboxStatus.RECONCILING:
                return message.status
            request = self._request(read_session, message)
            attempt = read_session.scalar(
                select(OutboxDeliveryAttempt)
                .where(OutboxDeliveryAttempt.outbox_message_id == outbox_id)
                .order_by(OutboxDeliveryAttempt.id.desc())
            )
            assert attempt is not None
            provider_message_id = attempt.provider_message_id
            attempt_started_at = _aware(attempt.started_at)
            detail = read_session.get(BeeperDeliveryAttemptDetail, attempt.id)
            pending_id = detail.pending_provider_id if detail else None
        adapter = self.adapters.get(request.transport)
        found = adapter.reconcile(
            request,
            provider_message_id=provider_message_id,
            pending_provider_id=pending_id,
        ) if adapter else None
        with self.sessions.begin() as session:
            message = session.get(OutboxMessage, outbox_id)
            assert message is not None
            if message.status is not OutboxStatus.RECONCILING:
                return message.status
            if found is not None:
                message.status = OutboxStatus.SENT
                latest = session.scalar(
                    select(OutboxDeliveryAttempt)
                    .where(OutboxDeliveryAttempt.outbox_message_id == outbox_id)
                    .order_by(OutboxDeliveryAttempt.id.desc())
                )
                assert latest is not None
                latest.provider_message_id = found
                pending_decisions = list(
                    session.scalars(
                        select(DecisionRequestPlaceholder).where(
                            DecisionRequestPlaceholder.outbox_message_id == outbox_id,
                            DecisionRequestPlaceholder.type == "UNCERTAIN_DELIVERY",
                            DecisionRequestPlaceholder.status == DecisionStatus.PENDING,
                        )
                    )
                )
                for decision in pending_decisions:
                    DecisionService(session).close(
                        decision.id,
                        DecisionCloseReason.SUBJECT_RESOLVED,
                    )
                return message.status
            timestamp = at or utc_now()
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=attempt_started_at.tzinfo)
            if (
                pending_id is not None
                and timestamp < attempt_started_at + self.pending_reconciliation_grace
            ):
                return message.status
            existing = session.scalar(
                select(DecisionRequestPlaceholder.id).where(DecisionRequestPlaceholder.outbox_message_id == outbox_id)
            )
            if existing is None:
                # Import avoids coupling worker construction to decision orchestration.
                from ten_texter.models import DecisionRequest

                decision = DecisionService(session).create(
                    decision_type="UNCERTAIN_DELIVERY",
                    subject_kind="outbox_message",
                    subject_id=outbox_id,
                    context={"message": "Delivery outcome could not be reconciled; do not retry blindly."},
                    task_instance_id=message.task_instance_id,
                    parent_terminal_policy=ParentTerminalPolicy.SURVIVE,
                )
                if self.owner_chat_id is not None:
                    from ten_texter.models import DecisionRequestPrompt

                    prompt = OutboxService(session).create_owner(
                        telegram_chat_id=self.owner_chat_id,
                        final_text=uncertain_delivery_prompt(outbox_id),
                        message_kind=MessageKind.NOTIFICATION,
                        idempotency_key=f"decision:{decision.id}:owner-prompt",
                        task_instance_id=message.task_instance_id,
                        parent_terminal_policy=ParentTerminalPolicy.SURVIVE,
                    )
                    session.add(DecisionRequestPrompt(decision_request_id=decision.id, outbox_message_id=prompt.id))
            return message.status

    def _snapshot(self, outbox_id: int) -> dict[str, object] | None:
        with self.sessions() as session:
            message = session.get(OutboxMessage, outbox_id)
            if message is None:
                return None
            return {"status": message.status, "text": message.final_text, "kind": message.message_kind}

    def _request(self, session: Session, message: OutboxMessage) -> DeliveryRequest:
        destination = OutboxService(session)._destination(message)
        return DeliveryRequest(
            outbox_id=message.id,
            transport=message.transport,
            destination=destination,
            text=message.final_text,
            idempotency_key=message.idempotency_key,
        )

    def _finish(self, outbox_id: int, attempt_id: int, result: DeliveryResult) -> OutboxStatus:
        with self.sessions.begin() as session:
            message = session.get(OutboxMessage, outbox_id)
            attempt = session.get(OutboxDeliveryAttempt, attempt_id)
            if message is None or attempt is None:
                raise DomainError("delivery state missing")
            if attempt.finished_at is not None or message.status is not OutboxStatus.SENDING:
                return message.status
            timestamp = utc_now()
            attempt.finished_at = timestamp
            attempt.error_details = result.error
            attempt.provider_message_id = result.provider_message_id
            message.lease_expires_at = None
            if result.pending_provider_id is not None:
                session.add(
                    BeeperDeliveryAttemptDetail(
                        outbox_delivery_attempt_id=attempt.id,
                        pending_provider_id=result.pending_provider_id,
                    )
                )
            if result.success:
                attempt.result = AttemptResult.SUCCESS
                message.status = OutboxStatus.SENT
            elif not result.boundary_crossed or result.definitely_not_sent:
                attempt.result = AttemptResult.FAILED
                attempt.failure_type = "DEFINITELY_NOT_SENT"
                message.status = OutboxStatus.PENDING
            else:
                attempt.result = AttemptResult.ABANDONED
                attempt.failure_type = "UNCERTAIN"
                message.status = OutboxStatus.RECONCILING
            return message.status


# Only used to make the existence query type-safe without loading presentation data.
from ten_texter.models import DecisionRequest as DecisionRequestPlaceholder
