from __future__ import annotations

import re

from sqlalchemy import select
from sqlalchemy.orm import Session

from ten_texter.enums import DestinationKind, MessageKind, ParentTerminalPolicy, Transport
from ten_texter.models import OutboxMessage, TelegramOutboxDestination, TelegramUpdate
from ten_texter.outbox import OutboxService


_HEALTH_KEY = re.compile(
    r"health:(?P<dependency>[a-z][a-z0-9-]*):unhealthy:(?P<generation>[1-9][0-9]*)"
)
_HEALTH_DETAIL_TOKEN = r"[A-Za-z][A-Za-z0-9_-]*(?:\.[A-Za-z][A-Za-z0-9_-]*)*"
_HEALTH_DETAIL = re.compile(rf"{_HEALTH_DETAIL_TOKEN}(?: {_HEALTH_DETAIL_TOKEN})*")


def owner_health_claim(
    session: Session,
    message: OutboxMessage,
    *,
    owner_chat_id: int | None,
) -> str | None:
    """Return the exact persisted claim only for a canonical owner health notification."""
    if (
        owner_chat_id is None
        or message.transport is not Transport.TELEGRAM
        or message.destination_kind is not DestinationKind.OWNER
        or message.message_kind is not MessageKind.NOTIFICATION
        or message.task_instance_id is not None
        or message.trigger_execution_id is not None
        or message.corrects_outbox_message_id is not None
        or message.parent_terminal_policy is not ParentTerminalPolicy.SURVIVE
    ):
        return None
    key = _HEALTH_KEY.fullmatch(message.idempotency_key)
    if key is None:
        return None
    destination = session.get(TelegramOutboxDestination, message.id)
    if destination is None or destination.telegram_chat_id != owner_chat_id:
        return None
    dependency = key.group("dependency")
    prefix = f"{dependency} became unhealthy."
    if message.final_text == prefix:
        return message.final_text
    detail_prefix = f"{prefix} "
    if not message.final_text.startswith(detail_prefix):
        return None
    detail = message.final_text.removeprefix(detail_prefix)
    return message.final_text if _HEALTH_DETAIL.fullmatch(detail) else None


def owner_telegram_update_failure_claim(
    session: Session,
    message: OutboxMessage,
    *,
    owner_chat_id: int | None,
) -> str | None:
    """Authorize only the canonical owner notice for one still-pending update."""
    prefix = "telegram-update:"
    suffix = ":processing-failed"
    if (
        owner_chat_id is None
        or message.transport is not Transport.TELEGRAM
        or message.destination_kind is not DestinationKind.OWNER
        or message.message_kind is not MessageKind.NOTIFICATION
        or message.task_instance_id is not None
        or message.trigger_execution_id is not None
        or message.corrects_outbox_message_id is not None
        or message.parent_terminal_policy is not ParentTerminalPolicy.SURVIVE
        or not message.idempotency_key.startswith(prefix)
        or not message.idempotency_key.endswith(suffix)
    ):
        return None
    row_id = message.idempotency_key[len(prefix) : -len(suffix)]
    if not row_id.isdigit():
        return None
    destination = session.get(TelegramOutboxDestination, message.id)
    update = session.get(TelegramUpdate, int(row_id))
    if destination is None or destination.telegram_chat_id != owner_chat_id or update is None:
        return None
    expected = (
        f"Telegram command {update.telegram_update_id} could not be processed and remains pending for retry."
    )
    return expected if message.final_text == expected else None


class HealthMonitor:
    """Single-process transition detector; durable notifications remain in Outbox."""

    def __init__(self, *, owner_chat_id: int):
        self.owner_chat_id = owner_chat_id
        self._healthy: dict[str, bool] = {}
        self._failure_generation: dict[str, int] = {}

    def record(self, session: Session, dependency: str, *, healthy: bool, details: str = "") -> None:
        observed_in_process = dependency in self._healthy
        previous = self._healthy.get(dependency, True)
        self._healthy[dependency] = healthy
        if previous and not healthy:
            final_text = f"{dependency} became unhealthy. {details}".strip()
            local_generation = self._failure_generation.get(dependency)
            durable_generation, latest_text = self._latest_failure(session, dependency)
            if local_generation is not None or observed_in_process or latest_text != final_text:
                generation = max(local_generation or 0, durable_generation) + 1
            else:
                generation = durable_generation
            self._failure_generation[dependency] = generation
            OutboxService(session).create_owner(
                telegram_chat_id=self.owner_chat_id,
                final_text=final_text,
                message_kind=MessageKind.NOTIFICATION,
                idempotency_key=f"health:{dependency}:unhealthy:{generation}",
                parent_terminal_policy=ParentTerminalPolicy.SURVIVE,
            )

    @staticmethod
    def _latest_failure(session: Session, dependency: str) -> tuple[int, str | None]:
        prefix = f"health:{dependency}:unhealthy:"
        latest_generation = 0
        latest_text: str | None = None
        alerts = session.execute(
            select(OutboxMessage.idempotency_key, OutboxMessage.final_text).where(
                OutboxMessage.transport == Transport.TELEGRAM,
                OutboxMessage.idempotency_key.startswith("health:"),
            )
        )
        for key, text in alerts:
            if not key.startswith(prefix):
                continue
            suffix = key.removeprefix(prefix)
            if not suffix.isdigit():
                continue
            generation = int(suffix)
            if generation > latest_generation:
                latest_generation = generation
                latest_text = text
        return latest_generation, latest_text
