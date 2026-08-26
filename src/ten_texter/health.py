from __future__ import annotations

from sqlalchemy.orm import Session

from ten_texter.enums import MessageKind, ParentTerminalPolicy
from ten_texter.outbox import OutboxService


class HealthMonitor:
    """Single-process transition detector; durable notifications remain in Outbox."""

    def __init__(self, *, owner_chat_id: int):
        self.owner_chat_id = owner_chat_id
        self._healthy: dict[str, bool] = {}
        self._failure_generation: dict[str, int] = {}

    def record(self, session: Session, dependency: str, *, healthy: bool, details: str = "") -> None:
        previous = self._healthy.get(dependency, True)
        self._healthy[dependency] = healthy
        if previous and not healthy:
            generation = self._failure_generation.get(dependency, 0) + 1
            self._failure_generation[dependency] = generation
            OutboxService(session).create_owner(
                telegram_chat_id=self.owner_chat_id,
                final_text=f"{dependency} became unhealthy. {details}".strip(),
                message_kind=MessageKind.NOTIFICATION,
                idempotency_key=f"health:{dependency}:unhealthy:{generation}",
                parent_terminal_policy=ParentTerminalPolicy.SURVIVE,
            )
