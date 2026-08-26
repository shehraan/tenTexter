from __future__ import annotations

from sqlalchemy import func, select

from ten_texter.health import HealthMonitor
from ten_texter.models import OutboxMessage


def test_health_transition_creates_one_durable_owner_notification(db_session) -> None:
    monitor = HealthMonitor(owner_chat_id=99)
    monitor.record(db_session, "validator", healthy=True)
    monitor.record(db_session, "validator", healthy=False, details="connection refused")
    monitor.record(db_session, "validator", healthy=False, details="still unavailable")
    assert db_session.scalar(select(func.count(OutboxMessage.id))) == 1
    monitor.record(db_session, "validator", healthy=True)
    monitor.record(db_session, "validator", healthy=False, details="timed out")
    assert db_session.scalar(select(func.count(OutboxMessage.id))) == 2
