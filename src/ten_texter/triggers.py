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
    AwaitedResponseStatus,
    MessageKind,
    ParentTerminalPolicy,
    TargetSelector,
    TaskStatus,
    TriggerExecutionStatus,
    TriggerInactiveReason,
    TriggerStatus,
)
from ten_texter.models import (
    AwaitedResponse,
    TaskInstance,
    TaskParticipant,
    TaskTrigger,
    TriggerExecution,
)
from ten_texter.model_clients import ModelUnavailable
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


def _metric(
    session: Session,
    trigger: TaskTrigger,
    name: str,
    *,
    now: datetime,
) -> int | float:
    if name == "RESPONSE_COUNT":
        return distinct_response_count(session, trigger.task_instance_id)
    if name == "TIME_UNTIL_EVENT_MINUTES":
        task = session.get(TaskInstance, trigger.task_instance_id)
        if task is None:
            raise DomainError("trigger task not found")
        return (_aware(task.scheduled_at) - _aware(now)).total_seconds() / 60
    if name == "NO_RESPONSE_COUNT":
        value = session.scalar(
            select(func.count(func.distinct(TaskParticipant.id)))
            .join(AwaitedResponse, AwaitedResponse.task_participant_id == TaskParticipant.id)
            .where(
                TaskParticipant.task_instance_id == trigger.task_instance_id,
                AwaitedResponse.status == AwaitedResponseStatus.OPEN,
            )
        )
        return int(value or 0)
    statuses = {
        "AVAILABLE_COUNT": AvailabilityStatus.AVAILABLE,
        "UNAVAILABLE_COUNT": AvailabilityStatus.UNAVAILABLE,
        "UNCERTAIN_COUNT": AvailabilityStatus.UNCERTAIN,
    }
    if name == "TOTAL_PARTICIPANTS":
        value = session.scalar(
            select(func.count(TaskParticipant.id)).where(
                TaskParticipant.task_instance_id == trigger.task_instance_id
            )
        )
        return int(value or 0)
    status = statuses.get(name)
    if status is None:
        raise DomainError("unsupported trigger metric")
    value = session.scalar(
        select(func.count(TaskParticipant.id)).where(
            TaskParticipant.task_instance_id == trigger.task_instance_id,
            TaskParticipant.availability_status == status,
        )
    )
    return int(value or 0)


def evaluate_condition(
    session: Session,
    trigger: TaskTrigger,
    condition: dict[str, object] | None,
    *,
    now: datetime | None = None,
    _depth: int = 0,
) -> bool:
    if condition is None:
        return False
    if _depth > 8:
        raise DomainError("trigger condition nesting exceeds v1 bound")
    timestamp = _aware(now or utc_now())
    kind = condition.get("kind")
    if kind == "ALWAYS":
        return True
    if kind in {"AND", "OR"}:
        children = condition.get("conditions")
        if not isinstance(children, list) or not children or len(children) > 16:
            raise DomainError("boolean trigger requires 1-16 conditions")
        if any(not isinstance(child, dict) for child in children):
            raise DomainError("trigger child condition must be an object")
        results = [
            evaluate_condition(
                session,
                trigger,
                child,
                now=timestamp,
                _depth=_depth + 1,
            )
            for child in children
        ]
        return all(results) if kind == "AND" else any(results)
    if kind == "NOT":
        child = condition.get("condition")
        if not isinstance(child, dict):
            raise DomainError("NOT trigger requires one condition")
        return not evaluate_condition(
            session,
            trigger,
            child,
            now=timestamp,
            _depth=_depth + 1,
        )
    if kind == "COMPARE":
        metric_name = condition.get("metric")
        operator = condition.get("operator")
        if not isinstance(metric_name, str) or operator not in {"EQ", "NE", "GT", "GTE", "LT", "LTE"}:
            raise DomainError("invalid trigger comparison")
        left = _metric(session, trigger, metric_name, now=timestamp)
        value_metric = condition.get("value_metric")
        if value_metric is not None:
            if not isinstance(value_metric, str) or "value" in condition:
                raise DomainError("comparison must use one right-hand value")
            right: int | float = _metric(session, trigger, value_metric, now=timestamp)
        else:
            value = condition.get("value")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise DomainError("comparison value must be numeric")
            right = value
        comparisons = {
            "EQ": left == right,
            "NE": left != right,
            "GT": left > right,
            "GTE": left >= right,
            "LT": left < right,
            "LTE": left <= right,
        }
        return comparisons[operator]
    if kind == "NOBODY_AVAILABLE":
        return _metric(session, trigger, "AVAILABLE_COUNT", now=timestamp) == 0
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
        return trigger.next_run_at is not None and _aware(trigger.next_run_at) <= timestamp
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
        query = query.where(
            select(AwaitedResponse.id)
            .where(
                AwaitedResponse.task_participant_id == TaskParticipant.id,
                AwaitedResponse.status == AwaitedResponseStatus.OPEN,
            )
            .exists()
        )
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
            if evaluate_condition(session, trigger, trigger.stop_condition_json, now=timestamp):
                trigger.status = TriggerStatus.INACTIVE
                trigger.inactive_reason = TriggerInactiveReason.STOP_CONDITION_MET
                trigger.next_run_at = None
                return None
            if not evaluate_condition(session, trigger, trigger.condition_json, now=timestamp):
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
            except ModelUnavailable:
                self._retryable_failure(claim)
                raise
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

    def _retryable_failure(self, claim: ExecutionClaim) -> TriggerExecutionStatus:
        with self.sessions.begin() as session:
            execution = session.get(TriggerExecution, claim.execution_id)
            if execution is None:
                raise DomainError("trigger execution disappeared")
            if not self._claim_matches(execution, claim):
                return execution.status
            execution.status = TriggerExecutionStatus.PENDING
            execution.lease_expires_at = None
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
