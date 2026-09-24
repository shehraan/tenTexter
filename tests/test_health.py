from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

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


def test_health_restart_deduplicates_same_ongoing_failure(db_session: Session) -> None:
    HealthMonitor(owner_chat_id=99).record(
        db_session,
        "beeper",
        healthy=False,
        details="connection refused",
    )
    db_session.commit()

    with Session(db_session.bind) as restarted_session:
        HealthMonitor(owner_chat_id=99).record(
            restarted_session,
            "beeper",
            healthy=False,
            details="connection refused",
        )
        restarted_session.commit()

    assert db_session.scalar(select(func.count(OutboxMessage.id))) == 1


def test_health_restart_changed_details_creates_distinct_alert(db_session: Session) -> None:
    HealthMonitor(owner_chat_id=99).record(
        db_session,
        "beeper",
        healthy=False,
        details="connection refused",
    )
    db_session.commit()

    with Session(db_session.bind) as restarted_session:
        HealthMonitor(owner_chat_id=99).record(
            restarted_session,
            "beeper",
            healthy=False,
            details="request timed out",
        )
        restarted_session.commit()

    alerts = list(db_session.scalars(select(OutboxMessage).order_by(OutboxMessage.id)))
    assert [alert.idempotency_key for alert in alerts] == [
        "health:beeper:unhealthy:1",
        "health:beeper:unhealthy:2",
    ]
    assert [alert.final_text for alert in alerts] == [
        "beeper became unhealthy. connection refused",
        "beeper became unhealthy. request timed out",
    ]


def test_health_restart_after_observed_recovery_reports_same_failure_again(
    db_session: Session,
) -> None:
    HealthMonitor(owner_chat_id=99).record(
        db_session,
        "beeper",
        healthy=False,
        details="connection refused",
    )
    db_session.commit()

    with Session(db_session.bind) as restarted_session:
        restarted = HealthMonitor(owner_chat_id=99)
        restarted.record(restarted_session, "beeper", healthy=True)
        restarted.record(
            restarted_session,
            "beeper",
            healthy=False,
            details="connection refused",
        )
        restarted_session.commit()

    alerts = list(db_session.scalars(select(OutboxMessage).order_by(OutboxMessage.id)))
    assert [alert.idempotency_key for alert in alerts] == [
        "health:beeper:unhealthy:1",
        "health:beeper:unhealthy:2",
    ]
