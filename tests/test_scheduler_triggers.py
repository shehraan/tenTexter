from __future__ import annotations

from datetime import timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from ten_texter.enums import (
    TargetSelector,
    TriggerActionType,
    TriggerExecutionStatus,
    TriggerInactiveReason,
    TriggerStatus,
)
from ten_texter.models import (
    OutboxMessage,
    TaskDefinition,
    TaskDefinitionParticipant,
    TaskInstance,
    TaskTrigger,
    TriggerExecution,
)
from ten_texter.scheduler import RecurrenceScheduler
from ten_texter.triggers import TriggerWorker
from tests.test_schema import NOW, seed_core


class Router:
    def __init__(self, routes: dict[int, int]):
        self.routes = routes

    def conversation_for(self, _session: Session, *, person_id: int, **_: object) -> int | None:
        return self.routes.get(person_id)


def definition_for(session: Session, core: dict[str, object], *, cursor=NOW) -> TaskDefinition:
    definition = TaskDefinition(
        name="Weekly tennis",
        recurrence_rule="FREQ=DAILY",
        default_time="15:00",
        timezone="America/Toronto",
        default_duration_minutes=60,
        default_location="courts",
        default_topic_key="tennis",
        next_occurrence_at=cursor,
    )
    session.add(definition)
    session.flush()
    session.add(
        TaskDefinitionParticipant(
            task_definition_id=definition.id,
            person_id=core["person"].id,
        )
    )
    session.flush()
    return definition


def test_repeated_scheduler_poll_does_not_duplicate_occurrence(db_session: Session) -> None:
    core = seed_core(db_session)
    definition = definition_for(db_session, core, cursor=NOW + timedelta(hours=1))
    scheduler = RecurrenceScheduler(
        router=Router({core["person"].id: core["conversation"].id}),
        owner_chat_id=123,
    )
    first = scheduler.poll_definition(db_session, definition.id, now=NOW)
    second = scheduler.poll_definition(db_session, definition.id, now=NOW)
    assert first.outcome == "CREATED"
    assert second.outcome == "NOT_DUE"
    count = db_session.scalar(
        select(func.count(TaskInstance.id)).where(TaskInstance.task_definition_id == definition.id)
    )
    assert count == 1
    occurrence = db_session.get(TaskInstance, first.task_instance_id)
    assert occurrence.occurrence_key == (NOW + timedelta(hours=1)).isoformat()


def test_missed_occurrence_is_skipped_with_owner_alert(db_session: Session) -> None:
    core = seed_core(db_session)
    definition = definition_for(db_session, core, cursor=NOW - timedelta(days=2))
    result = RecurrenceScheduler(
        router=Router({core["person"].id: core["conversation"].id}),
        owner_chat_id=123,
    ).poll_definition(db_session, definition.id, now=NOW)
    assert result.outcome == "MISSED_SKIPPED"
    assert db_session.scalar(
        select(func.count(TaskInstance.id)).where(TaskInstance.task_definition_id == definition.id)
    ) == 0
    assert db_session.scalar(select(func.count(OutboxMessage.id))) == 1


def test_unroutable_occurrence_has_no_partial_instance(db_session: Session) -> None:
    core = seed_core(db_session)
    definition = definition_for(db_session, core, cursor=NOW + timedelta(hours=1))
    result = RecurrenceScheduler(router=Router({}), owner_chat_id=123).poll_definition(
        db_session, definition.id, now=NOW
    )
    assert result.outcome == "UNROUTABLE_SKIPPED"
    assert db_session.scalar(
        select(func.count(TaskInstance.id)).where(TaskInstance.task_definition_id == definition.id)
    ) == 0
    assert db_session.scalar(select(func.count(OutboxMessage.id))) == 1


class Generator:
    def __init__(self, fail: bool = False):
        self.fail = fail

    def generate(self, **_: object) -> str:
        if self.fail:
            raise RuntimeError("model unavailable")
        return "Reminder"


class Validator:
    def validate(self, **_: object) -> bool:
        return True


def trigger_for(session: Session, core: dict[str, object], *, recurring: bool = False) -> TaskTrigger:
    condition: dict[str, object] = {"kind": "ALWAYS"}
    if recurring:
        condition["repeat_seconds"] = 3600
    trigger = TaskTrigger(
        task_instance_id=core["task"].id,
        condition_json=condition,
        target_selector=TargetSelector.ALL_PARTICIPANTS,
        action_type=TriggerActionType.SEND_MESSAGE,
        action_payload_json={"goal": "remind"},
        next_run_at=NOW,
        status=TriggerStatus.ACTIVE,
    )
    session.add(trigger)
    session.flush()
    return trigger


def trigger_worker(session: Session, generator: Generator) -> TriggerWorker:
    factory = sessionmaker(bind=session.bind, expire_on_commit=False, autoflush=False)
    return TriggerWorker(factory, generator=generator, validator=Validator(), owner_chat_id=123)


def test_trigger_claim_generate_commit_and_distinct_outbox(db_session: Session) -> None:
    core = seed_core(db_session)
    trigger = trigger_for(db_session, core)
    db_session.commit()
    worker = trigger_worker(db_session, Generator())
    claim = worker.claim(trigger.id, "fire:1", now=NOW)
    assert claim is not None
    assert worker.run_claim(claim) is TriggerExecutionStatus.COMPLETED
    with db_session.bind.connect() as connection:
        assert connection.execute(select(func.count(OutboxMessage.id))).scalar_one() == 1
    db_session.expire_all()
    stored = db_session.get(TaskTrigger, trigger.id)
    assert (stored.status, stored.inactive_reason) == (TriggerStatus.INACTIVE, TriggerInactiveReason.FIRED)


def test_stale_reclaimed_trigger_worker_cannot_commit(db_session: Session) -> None:
    core = seed_core(db_session)
    trigger = trigger_for(db_session, core, recurring=True)
    db_session.commit()
    worker = trigger_worker(db_session, Generator())
    old_claim = worker.claim(trigger.id, "fire:1", now=NOW)
    assert old_claim is not None
    new_claim = worker.claim(trigger.id, "fire:1", now=NOW + timedelta(minutes=1))
    assert new_claim is not None
    assert worker.run_claim(old_claim) is TriggerExecutionStatus.PROCESSING
    with db_session.bind.connect() as connection:
        assert connection.execute(select(func.count(OutboxMessage.id))).scalar_one() == 0
    assert worker.run_claim(new_claim) is TriggerExecutionStatus.COMPLETED


def test_permanent_recurring_failure_alerts_and_advances(db_session: Session) -> None:
    core = seed_core(db_session)
    trigger = trigger_for(db_session, core, recurring=True)
    db_session.commit()
    worker = trigger_worker(db_session, Generator(fail=True))
    claim = worker.claim(trigger.id, "fire:failed", now=NOW)
    assert claim is not None
    assert worker.run_claim(claim) is TriggerExecutionStatus.FAILED
    db_session.expire_all()
    stored = db_session.get(TaskTrigger, trigger.id)
    assert stored.status is TriggerStatus.ACTIVE
    assert stored.next_run_at.replace(tzinfo=NOW.tzinfo) == NOW + timedelta(hours=1)
    assert db_session.scalar(select(func.count(OutboxMessage.id))) == 1


def test_permanent_one_shot_failure_alerts_and_deactivates(db_session: Session) -> None:
    core = seed_core(db_session)
    trigger = trigger_for(db_session, core)
    db_session.commit()
    worker = trigger_worker(db_session, Generator(fail=True))
    claim = worker.claim(trigger.id, "fire:failed", now=NOW)
    assert claim is not None
    assert worker.run_claim(claim) is TriggerExecutionStatus.FAILED
    db_session.expire_all()
    stored = db_session.get(TaskTrigger, trigger.id)
    assert (stored.status, stored.inactive_reason) == (TriggerStatus.INACTIVE, TriggerInactiveReason.FIRED)
