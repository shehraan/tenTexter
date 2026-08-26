from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol
from zoneinfo import ZoneInfo

from dateutil.rrule import rrulestr
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from ten_texter.domain import DomainError, TaskService
from ten_texter.enums import MessageKind, ParentTerminalPolicy
from ten_texter.models import TaskDefinition, TaskDefinitionParticipant, TaskInstance
from ten_texter.outbox import OutboxService


class SpawnRouter(Protocol):
    def conversation_for(self, session: Session, *, task_definition_id: int, person_id: int) -> int | None: ...


@dataclass(frozen=True, slots=True)
class SchedulerResult:
    outcome: str
    task_instance_id: int | None = None


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def next_recurrence(definition: TaskDefinition, current: datetime) -> datetime | None:
    zone = ZoneInfo(definition.timezone)
    local_current = _aware(current).astimezone(zone)
    rule = rrulestr(definition.recurrence_rule, dtstart=local_current)
    following = rule.after(local_current, inc=False)
    return following.astimezone(UTC) if following is not None else None


class RecurrenceScheduler:
    def __init__(self, *, router: SpawnRouter, owner_chat_id: int):
        self.router = router
        self.owner_chat_id = owner_chat_id

    def poll_definition(
        self,
        session: Session,
        definition_id: int,
        *,
        now: datetime,
        horizon: timedelta = timedelta(hours=24),
        missed_grace: timedelta = timedelta(minutes=5),
    ) -> SchedulerResult:
        definition = session.get(TaskDefinition, definition_id)
        if definition is None or definition.archived_at is not None or definition.next_occurrence_at is None:
            return SchedulerResult("NOT_DUE")
        cursor = _aware(definition.next_occurrence_at)
        now = _aware(now)
        if cursor > now + horizon:
            return SchedulerResult("NOT_DUE")
        following = next_recurrence(definition, cursor)
        if cursor < now - missed_grace:
            if not self._advance_cursor(session, definition.id, definition.next_occurrence_at, following):
                return SchedulerResult("LOST_CAS")
            OutboxService(session).create_owner(
                telegram_chat_id=self.owner_chat_id,
                final_text=f"Skipped missed recurring occurrence '{definition.name}' at {cursor.isoformat()}.",
                message_kind=MessageKind.NOTIFICATION,
                idempotency_key=f"recurrence:{definition.id}:{cursor.isoformat()}:missed",
                parent_terminal_policy=ParentTerminalPolicy.SURVIVE,
            )
            return SchedulerResult("MISSED_SKIPPED")

        people = list(
            session.scalars(
                select(TaskDefinitionParticipant).where(
                    TaskDefinitionParticipant.task_definition_id == definition.id
                )
            )
        )
        routes: list[tuple[int, int]] = []
        for membership in people:
            conversation_id = self.router.conversation_for(
                session,
                task_definition_id=definition.id,
                person_id=membership.person_id,
            )
            if conversation_id is None:
                if not self._advance_cursor(session, definition.id, definition.next_occurrence_at, following):
                    return SchedulerResult("LOST_CAS")
                OutboxService(session).create_owner(
                    telegram_chat_id=self.owner_chat_id,
                    final_text=(
                        f"Skipped recurring occurrence '{definition.name}' at {cursor.isoformat()}: "
                        "participant routing was ambiguous or unavailable."
                    ),
                    message_kind=MessageKind.NOTIFICATION,
                    idempotency_key=f"recurrence:{definition.id}:{cursor.isoformat()}:unroutable",
                    parent_terminal_policy=ParentTerminalPolicy.SURVIVE,
                )
                return SchedulerResult("UNROUTABLE_SKIPPED")
            routes.append((membership.person_id, conversation_id))

        if not self._advance_cursor(session, definition.id, definition.next_occurrence_at, following):
            return SchedulerResult("LOST_CAS")
        task = TaskService(session).create(
            task_definition_id=definition.id,
            occurrence_key=cursor.isoformat(),
            scheduled_at=cursor,
            duration_minutes=definition.default_duration_minutes,
            location=definition.default_location,
            topic_key=definition.default_topic_key,
            participants=routes,
        )
        return SchedulerResult("CREATED", task.id)

    @staticmethod
    def _advance_cursor(
        session: Session,
        definition_id: int,
        expected: datetime,
        following: datetime | None,
    ) -> bool:
        result = session.execute(
            update(TaskDefinition)
            .where(
                TaskDefinition.id == definition_id,
                TaskDefinition.next_occurrence_at == expected,
            )
            .values(next_occurrence_at=following)
        )
        return result.rowcount == 1
