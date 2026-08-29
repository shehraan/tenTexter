from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from ten_texter import cli
from ten_texter.correlation import AtomicProposal, Classification, CorrelationOrchestrator
from ten_texter.domain import TaskService
from ten_texter.enums import (
    AvailabilityStatus,
    AwaitedResponseStatus,
    ContentSupport,
    DecisionCloseReason,
    DecisionStatus,
    MessageKind,
    OutboxStatus,
    ProcessingStatus,
    ProposalStatus,
    TargetSelector,
    Transport,
    TriggerActionType,
    TriggerStatus,
)
from ten_texter.models import (
    AwaitedResponse,
    DecisionRequest,
    MessageRevision,
    OutboxMessage,
    Proposal,
    TaskDefinition,
    TaskDefinitionParticipant,
    TaskInstance,
    TaskTrigger,
    TelegramUpdate,
)
from ten_texter.outbox import OutboxService
from ten_texter.outbox import DeliveryResult
from ten_texter.runtime import build_runtime
from ten_texter.triggers import evaluate_condition, select_targets
from tests.test_schema import NOW, seed_core


class Semantic:
    def choose(self, *_: object) -> int | None:
        return None


class Classifier:
    def __init__(self, result: Classification):
        self.result = result

    def classify(self, *_: object) -> Classification:
        return self.result


def _orchestrator(session: Session, result: Classification) -> CorrelationOrchestrator:
    return CorrelationOrchestrator(
        session,
        semantic=Semantic(),
        classifier=Classifier(result),
        owner_chat_id=99,
    )


def _edit(session: Session, core: dict[str, object], *, text: str, sequence: int) -> MessageRevision:
    revision = MessageRevision(
        message_id=core["message"].id,
        provider_revision_key=f"edit:{sequence}",
        provider_sequence=sequence,
        content_hash=str(sequence) * 64,
        text=text,
        content_support=ContentSupport.SUPPORTED,
        processing_status=ProcessingStatus.PENDING,
    )
    session.add(revision)
    session.flush()
    core["message"].current_revision_id = revision.id
    session.flush()
    return revision


def test_edit_to_ambiguous_retracts_prior_availability(db_session: Session) -> None:
    core = seed_core(db_session)
    awaited = AwaitedResponse(
        task_participant_id=core["participant"].id,
        expected_response_type="availability",
        status=AwaitedResponseStatus.OPEN,
    )
    db_session.add(awaited)
    db_session.flush()
    _orchestrator(
        db_session,
        Classification(kind="AVAILABILITY", availability=AvailabilityStatus.AVAILABLE),
    ).process(core["revision"].id)

    edit = _edit(db_session, core, text="maybe", sequence=2)
    _orchestrator(db_session, Classification(kind="AMBIGUOUS")).process(edit.id)

    assert core["participant"].availability_status is AvailabilityStatus.UNKNOWN
    assert core["participant"].availability_evidence is None
    assert core["participant"].availability_source_revision_id is None
    assert awaited.status is AwaitedResponseStatus.AMBIGUOUS


def test_edit_supersedes_obsolete_pending_proposal_and_decision(db_session: Session) -> None:
    core = seed_core(db_session)
    awaited = AwaitedResponse(
        task_participant_id=core["participant"].id,
        expected_response_type="scheduling",
        status=AwaitedResponseStatus.OPEN,
    )
    db_session.add(awaited)
    db_session.flush()
    _orchestrator(
        db_session,
        Classification(
            kind="COUNTERPROPOSAL",
            proposals=(AtomicProposal("location", "SET", None, "park"),),
        ),
    ).process(core["revision"].id)
    proposal = db_session.scalar(select(Proposal).where(Proposal.task_instance_id == core["task"].id))
    decision = db_session.scalar(select(DecisionRequest).where(DecisionRequest.proposal_id == proposal.id))

    edit = _edit(db_session, core, text="Never mind, the original location is fine.", sequence=2)
    _orchestrator(db_session, Classification(kind="AMBIGUOUS")).process(edit.id)

    assert proposal.status is ProposalStatus.SUPERSEDED
    assert decision.status is DecisionStatus.CLOSED
    assert decision.close_reason is DecisionCloseReason.SUBJECT_RESOLVED


def test_no_response_targets_awaited_response_state_not_availability(db_session: Session) -> None:
    core = seed_core(db_session)
    db_session.add(
        AwaitedResponse(
            task_participant_id=core["participant"].id,
            expected_response_type="availability",
            status=AwaitedResponseStatus.AMBIGUOUS,
        )
    )
    trigger = TaskTrigger(
        task_instance_id=core["task"].id,
        condition_json={"kind": "ALWAYS"},
        target_selector=TargetSelector.NO_RESPONSE,
        action_type=TriggerActionType.SEND_MESSAGE,
        action_payload_json={"goal": "remind"},
        next_run_at=NOW,
        status=TriggerStatus.ACTIVE,
    )
    db_session.add(trigger)
    db_session.flush()

    assert core["participant"].availability_status is AvailabilityStatus.UNKNOWN
    assert select_targets(db_session, trigger) == []


def test_trigger_dsl_composes_boolean_metrics_and_time_comparisons(db_session: Session) -> None:
    core = seed_core(db_session)
    trigger = TaskTrigger(
        task_instance_id=core["task"].id,
        condition_json={"kind": "ALWAYS"},
        target_selector=TargetSelector.ALL_PARTICIPANTS,
        action_type=TriggerActionType.SEND_MESSAGE,
        action_payload_json={"goal": "notify"},
        next_run_at=NOW,
        status=TriggerStatus.ACTIVE,
    )
    db_session.add(trigger)
    db_session.flush()
    condition = {
        "kind": "AND",
        "conditions": [
            {"kind": "COMPARE", "metric": "AVAILABLE_COUNT", "operator": "EQ", "value": 0},
            {
                "kind": "OR",
                "conditions": [
                    {
                        "kind": "COMPARE",
                        "metric": "RESPONSE_COUNT",
                        "operator": "EQ",
                        "value_metric": "TOTAL_PARTICIPANTS",
                    },
                    {
                        "kind": "COMPARE",
                        "metric": "TIME_UNTIL_EVENT_MINUTES",
                        "operator": "LTE",
                        "value": 30,
                    },
                ],
            },
        ],
    }

    assert evaluate_condition(db_session, trigger, condition, now=core["task"].scheduled_at - timedelta(minutes=20))


def test_cli_worker_operates_all_recovery_states(db_session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    core = seed_core(db_session)
    pending = OutboxService(db_session).create_beeper(
        task_instance_id=core["task"].id,
        conversation_id=core["conversation"].id,
        participant_ids=[core["participant"].id],
        final_text="pending",
        message_kind=MessageKind.INITIAL,
        idempotency_key="pump:pending",
    )
    sending = OutboxService(db_session).create_beeper(
        task_instance_id=core["task"].id,
        conversation_id=core["conversation"].id,
        participant_ids=[core["participant"].id],
        final_text="sending",
        message_kind=MessageKind.INITIAL,
        idempotency_key="pump:sending",
    )
    sending.status = OutboxStatus.SENDING
    sending.lease_expires_at = NOW - timedelta(seconds=1)
    reconciling = OutboxService(db_session).create_beeper(
        task_instance_id=core["task"].id,
        conversation_id=core["conversation"].id,
        participant_ids=[core["participant"].id],
        final_text="reconciling",
        message_kind=MessageKind.INITIAL,
        idempotency_key="pump:reconciling",
    )
    reconciling.status = OutboxStatus.RECONCILING
    db_session.commit()

    calls: dict[str, list[int]] = {"process": [], "reclaim": [], "reconcile": []}

    class FakeWorker:
        def __init__(self, *_: object, **__: object):
            pass

        def process(self, outbox_id: int) -> None:
            calls["process"].append(outbox_id)

        def reclaim_expired(self, outbox_id: int) -> None:
            calls["reclaim"].append(outbox_id)

        def reconcile(self, outbox_id: int) -> None:
            calls["reconcile"].append(outbox_id)

    monkeypatch.setattr("ten_texter.outbox.OutboxWorker", FakeWorker)
    settings = SimpleNamespace(
        real_transports_enabled=True,
        validator_model_url="http://validator.invalid",
        beeper_base_url="http://beeper.invalid",
        beeper_token="token",
        telegram_bot_token="token",
    )
    app = SimpleNamespace(settings=settings, sessions=sessionmaker(bind=db_session.bind, expire_on_commit=False))

    cli._run_outbox_worker(app, once=True)

    assert calls == {
        "process": [pending.id],
        "reclaim": [sending.id],
        "reconcile": [reconciling.id],
    }


def test_production_context_provider_supplies_destination_aware_validator_facts() -> None:
    from ten_texter.policy import DatabaseContextProvider
    from ten_texter.validator import DatabaseValidatorContextProvider

    assert DatabaseContextProvider is not None
    assert DatabaseValidatorContextProvider is not None


def test_cli_exposes_full_agent_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[bool] = []

    def fake_run(_app: object, *, once: bool) -> None:
        called.append(once)

    monkeypatch.setattr(cli, "_run_agent", fake_run, raising=False)
    monkeypatch.setenv("TEN_TEXTER_REAL_TRANSPORTS_ENABLED", "false")
    assert cli.main(["run", "--once"]) == 0
    assert called == [True]


def test_production_runtime_composes_owner_command_through_durable_send(
    db_session: Session,
) -> None:
    core = seed_core(db_session)
    definition = TaskDefinition(
        name="Weekly tennis",
        recurrence_rule="FREQ=DAILY",
        default_time="16:00",
        timezone="America/Toronto",
        default_duration_minutes=60,
        default_location="courts",
        default_topic_key="tennis",
        next_occurrence_at=NOW + timedelta(hours=1),
    )
    db_session.add(definition)
    db_session.flush()
    db_session.add(
        TaskDefinitionParticipant(
            task_definition_id=definition.id,
            person_id=core["person"].id,
        )
    )
    trigger = TaskTrigger(
        task_instance_id=core["task"].id,
        condition_json={"kind": "ALWAYS"},
        target_selector=TargetSelector.ALL_PARTICIPANTS,
        action_type=TriggerActionType.SEND_MESSAGE,
        action_payload_json={"goal": "remind"},
        next_run_at=NOW,
        status=TriggerStatus.ACTIVE,
    )
    db_session.add(trigger)
    overdue = TaskService(db_session).create(
        scheduled_at=NOW - timedelta(hours=2),
        duration_minutes=60,
        topic_key="overdue",
    )
    db_session.commit()
    sessions = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)

    class PrimaryBackend:
        def infer(self, *, operation: str, payload: dict[str, object]) -> dict[str, object]:
            if operation == "task_parser":
                return {
                    "scheduled_at": (NOW + timedelta(days=2)).isoformat(),
                    "duration_minutes": 60,
                    "location": "courts",
                    "topic_key": "tennis",
                    "participant_references": ["Alex"],
                    "recurrence_rule": None,
                    "timezone": None,
                }
            if operation == "message_generator":
                return {"text": "Are you available for tennis at the courts?"}
            raise AssertionError(f"unexpected primary operation: {operation}")

    class ValidatorBackend:
        def infer(self, *, operation: str, payload: dict[str, object]) -> dict[str, object]:
            assert operation == "message_validator"
            assert payload["allowed_claims"]
            return {"category": "VALID", "critique": None}

    class Telegram:
        def __init__(self):
            self.polled = False

        def poll(self, **_: object) -> list[dict[str, object]]:
            if self.polled:
                return []
            self.polled = True
            return [
                {
                    "update_id": 100,
                    "message": {
                        "from": {"id": 7},
                        "chat": {"id": 9, "type": "private"},
                        "text": "Set up tennis with Alex at 5 PM for 60 minutes.",
                    },
                }
            ]

        def send(self, _request: object) -> DeliveryResult:
            return DeliveryResult(True, True, provider_message_id="telegram:sent")

        def reconcile(self, _request: object, **_: object) -> str | None:
            return None

    class Beeper:
        def __init__(self):
            self.sent: list[object] = []

        def poll_inbound(self) -> list[int]:
            return []

        def send(self, request: object) -> DeliveryResult:
            self.sent.append(request)
            return DeliveryResult(True, True, provider_message_id="beeper:sent")

        def reconcile(self, _request: object, **_: object) -> str | None:
            return None

    telegram = Telegram()
    beeper = Beeper()
    settings = SimpleNamespace(
        real_transports_enabled=True,
        owner_id=7,
        owner_chat_id=9,
        telegram_bot_token="token",
        beeper_token="token",
        beeper_base_url="http://beeper.invalid",
        primary_model_url="http://primary.invalid",
        validator_model_url="http://validator.invalid",
    )
    runtime = build_runtime(
        SimpleNamespace(settings=settings, sessions=sessions),
        primary_backend=PrimaryBackend(),
        validator_backend=ValidatorBackend(),
        telegram=telegram,
        beeper=beeper,
        task_parser_clock=lambda: NOW,
    )

    tick = runtime.run_once(now=NOW)

    assert tick.errors == {}
    with sessions() as session:
        assert session.query(TaskTrigger).count() == 1
        assert session.get(TaskTrigger, trigger.id).status.value == "INACTIVE"
        assert session.query(TaskInstance).count() == 4
        assert session.get(TaskInstance, overdue.id).status.value == "LAPSED"
        sent = session.scalars(select(OutboxMessage)).all()
        assert len(sent) == 3
        assert all(message.status is OutboxStatus.SENT for message in sent)
        update = session.scalar(select(TelegramUpdate))
        assert update.status.value == "PROCESSED"
    assert len(beeper.sent) == 3
