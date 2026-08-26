from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from ten_texter.domain import DomainError, distinct_response_count, utc_now
from ten_texter.enums import (
    AvailabilityStatus,
    MessageKind,
    ParentTerminalPolicy,
    TargetSelector,
    TaskStatus,
    TriggerExecutionStatus,
    TriggerInactiveReason,
    TriggerStatus,
)
from ten_texter.models import (
    TaskInstance,
    TaskParticipant,
    TaskTrigger,
    TriggerExecution,
)
from ten_texter.outbox import OutboxService


class TriggerGenerator(Protocol):
    def generate(self, *, trigger: TaskTrigger, participant: TaskParticipant) -> str: ...


class GeneratedTextValidator(Protocol):
    def validate(self, *, text: str, trigger: TaskTrigger, participant: TaskParticipant) -> bool: ...


@dataclass(frozen=True, slots=True)
class ExecutionClaim:
    execution_id: int
    trigger_id: int
    lease_expires_at: datetime


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def evaluate_condition(session: Session, trigger: TaskTrigger, condition: dict[str, object] | None) -> bool:
    if condition is None:
        return False
    kind = condition.get("kind")
    if kind == "ALWAYS":
        return True
    if kind == "NOBODY_AVAILABLE":
        available = session.scalar(
            select(func.count(TaskParticipant.id)).where(
                TaskParticipant.task_instance_id == trigger.task_instance_id,
                TaskParticipant.availability_status == AvailabilityStatus.AVAILABLE,
            )
        )
        return int(available or 0) == 0
    if kind == "ALL_UNAVAILABLE":
        total = session.scalar(
            select(func.count(TaskParticipant.id)).where(TaskParticipant.task_instance_id == trigger.task_instance_id)
        )
        unavailable = session.scalar(
            select(func.count(TaskParticipant.id)).where(
                TaskParticipant.task_instance_id == trigger.task_instance_id,
                TaskParticipant.availability_status == AvailabilityStatus.UNAVAILABLE,
            )
        )
        return int(total or 0) > 0 and total == unavailable
    if kind == "RESPONSE_COUNT_AT_LEAST":
        threshold = condition.get("count")
        if not isinstance(threshold, int) or threshold < 0:
            raise DomainError("invalid response-count trigger")
        return distinct_response_count(session, trigger.task_instance_id) >= threshold
    if kind == "AT_TIME":
        return True
    raise DomainError("unsupported bounded trigger condition")


def select_targets(session: Session, trigger: TaskTrigger) -> list[TaskParticipant]:
    query = select(TaskParticipant).where(TaskParticipant.task_instance_id == trigger.task_instance_id)
    if trigger.target_selector is TargetSelector.SPECIFIC_PARTICIPANT:
        query = query.where(TaskParticipant.id == trigger.target_task_participant_id)
    elif trigger.target_selector is TargetSelector.AVAILABLE:
        query = query.where(TaskParticipant.availability_status == AvailabilityStatus.AVAILABLE)
    elif trigger.target_selector is TargetSelector.UNAVAILABLE:
        query = query.where(TaskParticipant.availability_status == AvailabilityStatus.UNAVAILABLE)
    elif trigger.target_selector is TargetSelector.UNCERTAIN:
        query = query.where(TaskParticipant.availability_status == AvailabilityStatus.UNCERTAIN)
    elif trigger.target_selector is TargetSelector.NO_RESPONSE:
        query = query.where(TaskParticipant.availability_status == AvailabilityStatus.UNKNOWN)
    return list(session.scalars(query.order_by(TaskParticipant.id)))


class TriggerWorker:
    def __init__(
        self,
        sessions: sessionmaker[Session],
        *,
        generator: TriggerGenerator,
        validator: GeneratedTextValidator,
        owner_chat_id: int,
        lease_duration: timedelta = timedelta(seconds=30),
    ):
        self.sessions = sessions
        self.generator = generator
        self.validator = validator
        self.owner_chat_id = owner_chat_id
        self.lease_duration = lease_duration

    def claim(self, trigger_id: int, fire_key: str, *, now: datetime | None = None) -> ExecutionClaim | None:
        timestamp = now or utc_now()
        with self.sessions.begin() as session:
            trigger = session.get(TaskTrigger, trigger_id)
            if trigger is None or trigger.status is not TriggerStatus.ACTIVE:
                return None
            task = session.get(TaskInstance, trigger.task_instance_id)
            if task is None or task.status is not TaskStatus.ACTIVE:
                return None
            if evaluate_condition(session, trigger, trigger.stop_condition_json):
                trigger.status = TriggerStatus.INACTIVE
                trigger.inactive_reason = TriggerInactiveReason.STOP_CONDITION_MET
                trigger.next_run_at = None
                return None
            if not evaluate_condition(session, trigger, trigger.condition_json):
                return None
            execution = session.scalar(
                select(TriggerExecution).where(
                    TriggerExecution.task_trigger_id == trigger.id,
                    TriggerExecution.fire_key == fire_key,
                )
            )
            if execution is None:
                execution = TriggerExecution(
                    task_trigger_id=trigger.id,
                    fire_key=fire_key,
                    scheduled_for=trigger.next_run_at,
                    status=TriggerExecutionStatus.PENDING,
                )
                session.add(execution)
                session.flush()
            if execution.status is TriggerExecutionStatus.PROCESSING:
                lease = _aware(execution.lease_expires_at)  # type: ignore[arg-type]
                if lease > _aware(timestamp):
                    return None
            elif execution.status is not TriggerExecutionStatus.PENDING:
                return None
            new_lease = _aware(timestamp) + self.lease_duration
            execution.status = TriggerExecutionStatus.PROCESSING
            execution.lease_expires_at = new_lease
            session.flush()
            return ExecutionClaim(execution.id, trigger.id, new_lease)

    def run_claim(self, claim: ExecutionClaim) -> TriggerExecutionStatus:
        with self.sessions() as session:
            execution = session.get(TriggerExecution, claim.execution_id)
            trigger = session.get(TaskTrigger, claim.trigger_id)
            if execution is None or trigger is None:
                raise DomainError("trigger execution disappeared")
            targets = select_targets(session, trigger)
            generated: list[tuple[int, str]] = []
            try:
                for participant in targets:
                    text = self.generator.generate(trigger=trigger, participant=participant)
                    if not self.validator.validate(text=text, trigger=trigger, participant=participant):
                        raise DomainError("generated trigger text failed validation")
                    generated.append((participant.id, text))
            except Exception as exc:
                return self._permanent_failure(claim, str(exc))
        return self._commit(claim, generated)

    def _commit(self, claim: ExecutionClaim, generated: list[tuple[int, str]]) -> TriggerExecutionStatus:
        with self.sessions.begin() as session:
            execution = session.get(TriggerExecution, claim.execution_id)
            trigger = session.get(TaskTrigger, claim.trigger_id)
            if execution is None or trigger is None:
                raise DomainError("trigger execution disappeared")
            if not self._claim_matches(execution, claim):
                return execution.status
            task = session.get(TaskInstance, trigger.task_instance_id)
            if task is None or task.status is not TaskStatus.ACTIVE:
                execution.status = TriggerExecutionStatus.CANCELLED
                execution.lease_expires_at = None
                return execution.status
            outbox = OutboxService(session)
            for participant_id, text in generated:
                participant = session.get(TaskParticipant, participant_id)
                if participant is None or participant.task_instance_id != task.id:
                    execution.status = TriggerExecutionStatus.CANCELLED
                    execution.lease_expires_at = None
                    return execution.status
                outbox.create_beeper(
                    task_instance_id=task.id,
                    conversation_id=participant.conversation_id,
                    participant_ids=[participant.id],
                    final_text=text,
                    message_kind=MessageKind.REMINDER,
                    idempotency_key=f"trigger:{execution.id}:participant:{participant.id}",
                    trigger_execution_id=execution.id,
                )
            execution.status = TriggerExecutionStatus.COMPLETED
            execution.lease_expires_at = None
            self._advance_or_deactivate(trigger)
            return execution.status

    def _permanent_failure(self, claim: ExecutionClaim, details: str) -> TriggerExecutionStatus:
        with self.sessions.begin() as session:
            execution = session.get(TriggerExecution, claim.execution_id)
            trigger = session.get(TaskTrigger, claim.trigger_id)
            if execution is None or trigger is None:
                raise DomainError("trigger execution disappeared")
            if not self._claim_matches(execution, claim):
                return execution.status
            execution.status = TriggerExecutionStatus.FAILED
            execution.lease_expires_at = None
            OutboxService(session).create_owner(
                telegram_chat_id=self.owner_chat_id,
                task_instance_id=trigger.task_instance_id,
                final_text=f"Trigger execution failed permanently: {details}",
                message_kind=MessageKind.NOTIFICATION,
                idempotency_key=f"trigger:{execution.id}:permanent-failure",
                parent_terminal_policy=ParentTerminalPolicy.SURVIVE,
                trigger_execution_id=execution.id,
            )
            self._advance_or_deactivate(trigger)
            return execution.status

    @staticmethod
    def _claim_matches(execution: TriggerExecution, claim: ExecutionClaim) -> bool:
        if execution.status is not TriggerExecutionStatus.PROCESSING or execution.lease_expires_at is None:
            return False
        return _aware(execution.lease_expires_at) == _aware(claim.lease_expires_at)

    @staticmethod
    def _advance_or_deactivate(trigger: TaskTrigger) -> None:
        repeat_seconds = trigger.condition_json.get("repeat_seconds")
        if isinstance(repeat_seconds, int) and repeat_seconds > 0 and trigger.next_run_at is not None:
            trigger.next_run_at = _aware(trigger.next_run_at) + timedelta(seconds=repeat_seconds)
        else:
            trigger.status = TriggerStatus.INACTIVE
            trigger.inactive_reason = TriggerInactiveReason.FIRED
            trigger.next_run_at = None
