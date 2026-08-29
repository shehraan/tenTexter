from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from ten_texter.enums import MessageKind, ParentTerminalPolicy, Transport
from ten_texter.models import OutboxMessage
from ten_texter.outbox import OutboxService


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
