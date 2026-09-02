from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from ten_texter.domain import distinct_response_count, utc_now
from ten_texter.enums import (
    AvailabilityStatus,
    DestinationKind,
    MessageKind,
    OutboxCancelReason,
    OutboxStatus,
    ParentTerminalPolicy,
    TaskStatus,
    Transport,
)
from ten_texter.models import (
    OutboxMessage,
    TaskInstance,
    TaskParticipant,
    TelegramOutboxDestination,
)
from ten_texter.outbox import OutboxService


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def nobody_available_text(task: TaskInstance) -> str:
    return (
        f"No participant is currently recorded as available for task {task.id} "
        f"({task.topic_key})."
    )


def nobody_available_idempotency_key(task_id: int, generation: int = 1) -> str:
    if generation < 1:
        raise ValueError("nobody-available generation must be positive")
    base = f"task:{task_id}:owner:nobody-available"
    return base if generation == 1 else f"{base}:{generation}"


def _generation_from_key(message: OutboxMessage) -> int | None:
    if message.task_instance_id is None:
        return None
    base = nobody_available_idempotency_key(message.task_instance_id)
    if message.idempotency_key == base:
        return 1
    prefix = f"{base}:"
    if not message.idempotency_key.startswith(prefix):
        return None
    suffix = message.idempotency_key[len(prefix) :]
    if not suffix.isascii() or not suffix.isdigit():
        return None
    generation = int(suffix)
    return generation if generation >= 2 and suffix == str(generation) else None


def nobody_available_condition(
    session: Session,
    task: TaskInstance,
    *,
    now: datetime | None = None,
) -> bool:
    if task.status is not TaskStatus.ACTIVE:
        return False
    total = int(
        session.scalar(
            select(func.count(TaskParticipant.id)).where(
                TaskParticipant.task_instance_id == task.id
            )
        )
        or 0
    )
    if total == 0:
        return False
    available = int(
        session.scalar(
            select(func.count(TaskParticipant.id)).where(
                TaskParticipant.task_instance_id == task.id,
                TaskParticipant.availability_status == AvailabilityStatus.AVAILABLE,
            )
        )
        or 0
    )
    if available != 0:
        return False
    all_responded = distinct_response_count(session, task.id) == total
    timestamp = _aware(now or utc_now())
    within_thirty_minutes = (
        _aware(task.scheduled_at) - timestamp
    ).total_seconds() <= 30 * 60
    return all_responded or within_thirty_minutes


def is_nobody_available_notification(message: OutboxMessage) -> bool:
    return _generation_from_key(message) is not None


def _represented_generation(
    session: Session,
    message: OutboxMessage,
    *,
    owner_chat_id: int | None,
) -> int | None:
    generation = _generation_from_key(message)
    if (
        owner_chat_id is None
        or generation is None
        or message.transport is not Transport.TELEGRAM
        or message.destination_kind is not DestinationKind.OWNER
        or message.message_kind is not MessageKind.NOTIFICATION
        or message.parent_terminal_policy is not ParentTerminalPolicy.TERMINATE
        or message.trigger_execution_id is not None
        or message.corrects_outbox_message_id is not None
    ):
        return None
    destination = session.get(TelegramOutboxDestination, message.id)
    task = session.get(TaskInstance, message.task_instance_id)
    if (
        destination is None
        or destination.telegram_chat_id != owner_chat_id
        or task is None
        or message.final_text != nobody_available_text(task)
    ):
        return None
    return generation


def _generation_row(
    session: Session,
    task_id: int,
    generation: int,
) -> OutboxMessage | None:
    return session.scalar(
        select(OutboxMessage).where(
            OutboxMessage.transport == Transport.TELEGRAM,
            OutboxMessage.idempotency_key
            == nobody_available_idempotency_key(task_id, generation),
        )
    )


def _prior_generations_are_stale(
    session: Session,
    message: OutboxMessage,
    generation: int,
    *,
    owner_chat_id: int,
) -> bool:
    assert message.task_instance_id is not None
    for prior_generation in range(1, generation):
        prior = _generation_row(
            session,
            message.task_instance_id,
            prior_generation,
        )
        if (
            prior is None
            or _represented_generation(
                session,
                prior,
                owner_chat_id=owner_chat_id,
            )
            != prior_generation
            or prior.status is not OutboxStatus.CANCELLED
            or prior.cancel_reason is not OutboxCancelReason.STALE
        ):
            return False
    return True


def authorize_nobody_available_notification(
    session: Session,
    message: OutboxMessage,
    *,
    owner_chat_id: int | None,
    now: datetime | None = None,
) -> str | None:
    if owner_chat_id is None:
        return None
    generation = _represented_generation(
        session,
        message,
        owner_chat_id=owner_chat_id,
    )
    if generation is None or message.task_instance_id is None:
        return None
    task = session.get(TaskInstance, message.task_instance_id)
    if (
        task is None
        or not _prior_generations_are_stale(
            session,
            message,
            generation,
            owner_chat_id=owner_chat_id,
        )
        or not nobody_available_condition(session, task, now=now)
    ):
        return None
    return message.final_text


class NobodyAvailableNotifier:
    """Creates the default v1 owner alert from current authoritative task state."""

    def __init__(self, sessions: sessionmaker[Session], *, owner_chat_id: int):
        self.sessions = sessions
        self.owner_chat_id = owner_chat_id

    def run(self, *, now: datetime | None = None) -> tuple[int, ...]:
        timestamp = _aware(now or utc_now())
        with self.sessions.begin() as session:
            tasks = list(
                session.scalars(
                    select(TaskInstance)
                    .where(TaskInstance.status == TaskStatus.ACTIVE)
                    .order_by(TaskInstance.id)
                )
            )
            message_ids: list[int] = []
            for task in tasks:
                if not nobody_available_condition(session, task, now=timestamp):
                    continue
                message = self._current_or_next(session, task)
                if message is not None:
                    message_ids.append(message.id)
            return tuple(message_ids)

    def _current_or_next(
        self,
        session: Session,
        task: TaskInstance,
    ) -> OutboxMessage | None:
        generation = 1
        while True:
            message = _generation_row(session, task.id, generation)
            if message is None:
                return OutboxService(session).create_owner(
                    telegram_chat_id=self.owner_chat_id,
                    task_instance_id=task.id,
                    final_text=nobody_available_text(task),
                    message_kind=MessageKind.NOTIFICATION,
                    idempotency_key=nobody_available_idempotency_key(
                        task.id,
                        generation,
                    ),
                    parent_terminal_policy=ParentTerminalPolicy.TERMINATE,
                )
            if (
                _represented_generation(
                    session,
                    message,
                    owner_chat_id=self.owner_chat_id,
                )
                != generation
            ):
                return None
            if (
                message.status is not OutboxStatus.CANCELLED
                or message.cancel_reason is not OutboxCancelReason.STALE
            ):
                return message
            generation += 1
