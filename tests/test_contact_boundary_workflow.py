from __future__ import annotations

from datetime import timedelta

import pytest
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from ten_texter.contact_boundaries import (
    BoundaryContext,
    ContactBoundary,
    apply_contact_boundary,
    boundary_context,
)
from ten_texter.control import PreparedOwnerDecision, ProductionOwnerCommandHandler
from ten_texter.correlation import Classification, CorrelationOrchestrator
from ten_texter.domain import ContactRuleService, DecisionService, TaskService, utc_now
from ten_texter.enums import (
    AvailabilityStatus,
    AwaitedResponseStatus,
    ContactRuleScope,
    ContactRuleSource,
    MessageKind,
    OutboxStatus,
    PolicyOutcome,
    RuleStrength,
    DecisionCloseReason,
    DecisionStatus,
    ParentTerminalPolicy,
    Transport,
)
from ten_texter.model_clients import (
    ContactBoundaryOutput,
    ModelContactBoundaryClassifier,
    ModelOutputError,
    _operation_json_schema,
)
from ten_texter.models import (
    AwaitedResponse,
    ContactRule,
    ConversationParticipant,
    DecisionRequest,
    DecisionRequestPrompt,
    Identity,
    OutboxMessage,
    Person,
    TaskParticipant,
)
from ten_texter.decision_prompt_context import authorize_decision_prompt
from ten_texter.outbox import DeliveryResult, OutboxService, OutboxWorker, PreSendDecision
from ten_texter.policy import ContactRuleResolver, PolicyRevalidator
from tests.test_schema import NOW, seed_core


class Semantic:
    def choose(self, *_: object) -> None:
        return None


class AvailabilityClassifier:
    def classify(self, *_: object) -> Classification:
        return Classification("AVAILABILITY", AvailabilityStatus.AVAILABLE)


class BoundaryClassifier:
    def __init__(self, result: ContactBoundary):
        self.result = result

    def classify(self, *_: object) -> ContactBoundary:
        return self.result


def _awaited(session: Session, core: dict[str, object]) -> AwaitedResponse:
    row = AwaitedResponse(
        task_participant_id=core["participant"].id,
        expected_response_type="availability",
        status=AwaitedResponseStatus.OPEN,
        created_at=NOW,
    )
    session.add(row)
    session.flush()
    return row


def test_boundary_classifier_contract_is_separate_and_candidate_bounded(db_session: Session) -> None:
    core = seed_core(db_session)
    context = boundary_context(db_session, core["revision"])

    class Backend:
        def infer(self, *, operation: str, payload: dict[str, object]) -> dict[str, object]:
            assert operation == "contact_boundary_classifier"
            assert payload["untrusted_participant_text"] == core["revision"].text
            assert payload["scope_candidates"] == [
                {"task_instance_id": core["task"].id, "topic_key": "tennis"}
            ]
            return {"kind": "BOUNDARY", "scope": "TOPIC", "topic_key": "tennis", "task_instance_id": None}

    result = ModelContactBoundaryClassifier(Backend()).classify(core["revision"], context)
    assert result == ContactBoundary("BOUNDARY", ContactRuleScope.TOPIC, "tennis")
    schema = _operation_json_schema("contact_boundary_classifier")
    assert len(schema["oneOf"]) == 5


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        (
            {"kind": "NONE", "scope": None, "topic_key": None, "task_instance_id": None},
            ContactBoundary("NONE"),
        ),
        (
            {"kind": "AMBIGUOUS", "scope": None, "topic_key": None, "task_instance_id": None},
            ContactBoundary("AMBIGUOUS"),
        ),
        (
            {"kind": "BOUNDARY", "scope": "GLOBAL", "topic_key": None, "task_instance_id": None},
            ContactBoundary("BOUNDARY", ContactRuleScope.GLOBAL),
        ),
        (
            {"kind": "BOUNDARY", "scope": "TOPIC", "topic_key": "tennis", "task_instance_id": None},
            ContactBoundary("BOUNDARY", ContactRuleScope.TOPIC, "tennis"),
        ),
    ],
)
def test_boundary_classifier_accepts_each_non_task_shape(
    db_session: Session,
    output: dict[str, object],
    expected: ContactBoundary,
) -> None:
    core = seed_core(db_session)

    class Backend:
        def infer(self, **_: object) -> dict[str, object]:
            return output

    assert ModelContactBoundaryClassifier(Backend()).classify(
        core["revision"], boundary_context(db_session, core["revision"])
    ) == expected


def test_boundary_classifier_accepts_bounded_task_candidate(
    db_session: Session,
) -> None:
    core = seed_core(db_session)

    class Backend:
        def infer(self, **_: object) -> dict[str, object]:
            return {
                "kind": "BOUNDARY",
                "scope": "TASK_INSTANCE",
                "topic_key": None,
                "task_instance_id": core["task"].id,
            }

    assert ModelContactBoundaryClassifier(Backend()).classify(
        core["revision"], boundary_context(db_session, core["revision"])
    ) == ContactBoundary(
        "BOUNDARY",
        ContactRuleScope.TASK_INSTANCE,
        task_instance_id=core["task"].id,
    )


def test_boundary_schema_and_pydantic_reject_mismatched_shapes() -> None:
    schema = _operation_json_schema("contact_boundary_classifier")
    assert all(
        branch["required"] == ["kind", "scope", "topic_key", "task_instance_id"]
        and branch["additionalProperties"] is False
        for branch in schema["oneOf"]
    )
    with pytest.raises(ValidationError):
        ContactBoundaryOutput.model_validate(
            {
                "kind": "NONE",
                "scope": "GLOBAL",
                "topic_key": None,
                "task_instance_id": None,
            }
        )
    with pytest.raises(ValidationError):
        ContactBoundaryOutput.model_validate(
            {
                "kind": "BOUNDARY",
                "scope": "TASK_INSTANCE",
                "topic_key": None,
                "task_instance_id": None,
            }
        )


def test_availability_and_contact_boundary_apply_from_one_revision(db_session: Session) -> None:
    core = seed_core(db_session)
    awaited = _awaited(db_session, core)
    plan = CorrelationOrchestrator(
        db_session,
        semantic=Semantic(),
        classifier=AvailabilityClassifier(),
        boundary_classifier=BoundaryClassifier(ContactBoundary("BOUNDARY", ContactRuleScope.TOPIC, "tennis")),
    ).prepare(core["revision"].id)
    CorrelationOrchestrator(
        db_session,
        semantic=Semantic(),
        classifier=AvailabilityClassifier(),
    ).apply_prepared(core["revision"].id, plan)
    assert awaited.status is AwaitedResponseStatus.SATISFIED
    rule = db_session.scalar(select(ContactRule))
    assert rule is not None
    assert (rule.person_id, rule.scope, rule.type, rule.source, rule.strength) == (
        core["person"].id,
        ContactRuleScope.TOPIC,
        "DO_NOT_CONTACT",
        ContactRuleSource.PARTICIPANT_REQUESTED,
        RuleStrength.STRONG,
    )
    context = boundary_context(db_session, core["revision"])
    apply_contact_boundary(
        db_session,
        core["revision"],
        ContactBoundary("BOUNDARY", ContactRuleScope.TOPIC, "tennis"),
        context,
        owner_chat_id=None,
    )
    assert db_session.scalar(select(func.count(ContactRule.id))) == 1


def test_ambiguous_boundary_creates_one_owner_decision_and_holds_send(db_session: Session) -> None:
    core = seed_core(db_session)
    context = boundary_context(db_session, core["revision"])
    apply_contact_boundary(
        db_session,
        core["revision"],
        ContactBoundary("AMBIGUOUS"),
        context,
        owner_chat_id=99,
    )
    apply_contact_boundary(db_session, core["revision"], ContactBoundary("AMBIGUOUS"), context, owner_chat_id=99)
    message = OutboxService(db_session).create_beeper(
        task_instance_id=core["task"].id,
        conversation_id=core["conversation"].id,
        participant_ids=[core["participant"].id],
        final_text="follow up",
        message_kind=MessageKind.REMINDER,
        idempotency_key="held-by-boundary",
    )
    assert db_session.scalar(select(func.count(DecisionRequest.id))) == 1
    assert PolicyRevalidator(owner_chat_id=99).check(db_session, message) is PreSendDecision.AWAITING_OWNER
    prompt = db_session.scalar(select(OutboxMessage).join(DecisionRequestPrompt).where(OutboxMessage.transport == "TELEGRAM"))
    assert prompt is not None
    authorization = authorize_decision_prompt(db_session, prompt, owner_chat_id=99)
    assert authorization is not None
    assert authorization.untrusted_data == (f"participant text={core['revision'].text!r}",)


def test_known_correlation_scopes_ambiguous_hold_despite_another_active_task(db_session: Session) -> None:
    core = seed_core(db_session)
    _awaited(db_session, core)
    other = TaskService(db_session).create(
        scheduled_at=NOW,
        duration_minutes=30,
        topic_key="dinner",
    )
    db_session.add(
        TaskParticipant(
            task_instance_id=other.id,
            person_id=core["person"].id,
            conversation_id=core["conversation"].id,
        )
    )
    db_session.flush()
    orchestrator = CorrelationOrchestrator(
        db_session,
        semantic=Semantic(),
        classifier=AvailabilityClassifier(),
        boundary_classifier=BoundaryClassifier(ContactBoundary("AMBIGUOUS")),
        owner_chat_id=99,
    )
    plan = orchestrator.prepare(core["revision"].id)
    orchestrator.apply_prepared(core["revision"].id, plan)
    decision = db_session.scalar(
        select(DecisionRequest).where(
            DecisionRequest.type == "CONTACT_BOUNDARY_AMBIGUITY"
        )
    )
    assert decision is not None
    assert decision.task_instance_id == core["task"].id


@pytest.mark.parametrize(
    ("action", "selected_task", "expected_scope", "expected_topic", "cancelled"),
    [
        ("set_global", None, ContactRuleScope.GLOBAL, None, True),
        ("set_topic:pickleball", None, ContactRuleScope.TOPIC, "pickleball", False),
        ("set_task", "current", ContactRuleScope.TASK_INSTANCE, None, True),
        ("dismiss", None, None, None, False),
    ],
)
def test_owner_can_resolve_each_boundary_ambiguity_action(
    db_session: Session,
    action: str,
    selected_task: str | None,
    expected_scope: ContactRuleScope | None,
    expected_topic: str | None,
    cancelled: bool,
) -> None:
    core = seed_core(db_session)
    decision = apply_contact_boundary(
        db_session,
        core["revision"],
        ContactBoundary("AMBIGUOUS"),
        boundary_context(db_session, core["revision"]),
        owner_chat_id=99,
    )
    assert decision is not None
    pending = OutboxService(db_session).create_beeper(
        task_instance_id=core["task"].id,
        conversation_id=core["conversation"].id,
        participant_ids=[core["participant"].id],
        final_text="follow up",
        message_kind=MessageKind.REMINDER,
        idempotency_key=f"ambiguity-action:{action}",
    )
    handler = ProductionOwnerCommandHandler(
        sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False),
        owner_chat_id=99,
        resolver=object(),  # type: ignore[arg-type]
        generation=object(),  # type: ignore[arg-type]
    )
    handler.apply_decision(
        db_session,
        decision.id,
        PreparedOwnerDecision(
            action,
            message_revision_id=(core["task"].id if selected_task else None),
        ),
        object(),  # type: ignore[arg-type]
    )

    assert decision.status is DecisionStatus.CLOSED
    rule = db_session.scalar(
        select(ContactRule).where(
            ContactRule.source == ContactRuleSource.PARTICIPANT_REQUESTED
        )
    )
    if expected_scope is None:
        assert rule is None
        assert decision.close_reason is DecisionCloseReason.DISMISSED
    else:
        assert rule is not None
        assert rule.scope is expected_scope
        assert rule.topic_key == expected_topic
        if expected_scope is ContactRuleScope.TASK_INSTANCE:
            assert rule.task_instance_id == core["task"].id
    assert (pending.status is OutboxStatus.CANCELLED) is cancelled


def test_topic_boundary_is_isolated_but_global_boundary_reaches_other_task(
    db_session: Session,
) -> None:
    core = seed_core(db_session)
    other = TaskService(db_session).create(
        scheduled_at=NOW,
        duration_minutes=30,
        topic_key="dinner",
    )
    other_participant = TaskParticipant(
        task_instance_id=other.id,
        person_id=core["person"].id,
        conversation_id=core["conversation"].id,
    )
    db_session.add(other_participant)
    db_session.flush()
    outbox = OutboxService(db_session).create_beeper(
        task_instance_id=other.id,
        conversation_id=core["conversation"].id,
        participant_ids=[other_participant.id],
        final_text="dinner question",
        message_kind=MessageKind.INITIAL,
        idempotency_key="topic-isolation",
    )
    ContactRuleService(db_session).create(
        person_id=core["person"].id,
        scope=ContactRuleScope.TOPIC,
        topic_key="tennis",
        type="DO_NOT_CONTACT",
        value="no tennis",
        source=ContactRuleSource.PARTICIPANT_REQUESTED,
        strength=RuleStrength.STRONG,
    )
    revalidator = PolicyRevalidator(owner_chat_id=99)
    assert revalidator.check(db_session, outbox) is PreSendDecision.READY
    ContactRuleService(db_session).create(
        person_id=core["person"].id,
        scope=ContactRuleScope.GLOBAL,
        type="DO_NOT_CONTACT",
        value="no messages",
        source=ContactRuleSource.PARTICIPANT_REQUESTED,
        strength=RuleStrength.STRONG,
    )
    assert revalidator.check(db_session, outbox) is PreSendDecision.AWAITING_OWNER


def test_incidental_group_member_boundary_does_not_block_logical_target(
    db_session: Session,
) -> None:
    core = seed_core(db_session)
    incidental = Person(display_name="Incidental", metadata_json={})
    db_session.add(incidental)
    db_session.flush()
    incidental_identity = Identity(
        person_id=incidental.id,
        beeper_user_id="beeper:incidental",
        network="discord",
        metadata_json={},
    )
    db_session.add(incidental_identity)
    db_session.flush()
    db_session.add(
        ConversationParticipant(
            conversation_id=core["conversation"].id,
            identity_id=incidental_identity.id,
            is_current=True,
        )
    )
    ContactRuleService(db_session).create(
        person_id=incidental.id,
        scope=ContactRuleScope.GLOBAL,
        type="DO_NOT_CONTACT",
        value="do not contact me",
        source=ContactRuleSource.PARTICIPANT_REQUESTED,
        strength=RuleStrength.STRONG,
    )
    outbox = OutboxService(db_session).create_beeper(
        task_instance_id=core["task"].id,
        conversation_id=core["conversation"].id,
        participant_ids=[core["participant"].id],
        final_text="question for the logical target",
        message_kind=MessageKind.INITIAL,
        idempotency_key="incidental-member",
    )
    assert PolicyRevalidator(owner_chat_id=99).check(
        db_session, outbox
    ) is PreSendDecision.READY


def test_task_boundary_is_absolute_and_broader_boundary_can_get_narrow_override(db_session: Session) -> None:
    core = seed_core(db_session)
    absolute = ContactRuleService(db_session).create(
        person_id=core["person"].id,
        scope=ContactRuleScope.TASK_INSTANCE,
        task_instance_id=core["task"].id,
        type="DO_NOT_CONTACT",
        value="not this task",
        source=ContactRuleSource.PARTICIPANT_REQUESTED,
        strength=RuleStrength.STRONG,
    )
    assessment = ContactRuleResolver().assess(
        db_session, person_id=core["person"].id, task=core["task"], topic_key="tennis"
    )
    assert assessment.outcome is PolicyOutcome.DO_NOT_ACT
    ContactRuleService(db_session).revoke(absolute.id)

    broad = ContactRuleService(db_session).create(
        person_id=core["person"].id,
        scope=ContactRuleScope.GLOBAL,
        type="DO_NOT_CONTACT",
        value="stop",
        source=ContactRuleSource.PARTICIPANT_REQUESTED,
        strength=RuleStrength.STRONG,
    )
    outbox = OutboxService(db_session).create_beeper(
        task_instance_id=core["task"].id,
        conversation_id=core["conversation"].id,
        participant_ids=[core["participant"].id],
        final_text="question",
        message_kind=MessageKind.INITIAL,
        idempotency_key="needs-exception",
    )
    assert PolicyRevalidator(owner_chat_id=99).check(db_session, outbox) is PreSendDecision.AWAITING_OWNER
    decision = db_session.scalar(select(DecisionRequest).where(DecisionRequest.contact_rule_id == broad.id))
    assert decision is not None
    prompt = db_session.scalar(select(OutboxMessage).join(DecisionRequestPrompt).where(
        DecisionRequestPrompt.decision_request_id == decision.id
    ))
    assert prompt is not None
    assert authorize_decision_prompt(db_session, prompt, owner_chat_id=99) is not None
    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    handler = ProductionOwnerCommandHandler(factory, owner_chat_id=99, resolver=object(), generation=object())  # type: ignore[arg-type]
    handler.apply_decision(db_session, decision.id, PreparedOwnerDecision("allow_this"), object())  # type: ignore[arg-type]
    override = db_session.scalar(select(ContactRule).where(ContactRule.overrides_contact_rule_id == broad.id))
    assert override is not None
    assert (override.scope, override.task_instance_id, override.type) == (
        ContactRuleScope.TASK_INSTANCE,
        core["task"].id,
        "ALLOW",
    )
    assert PolicyRevalidator(owner_chat_id=99).check(db_session, outbox) is PreSendDecision.READY


def test_worker_holds_for_owner_without_validation_or_transport(db_session: Session) -> None:
    core = seed_core(db_session)
    ContactRuleService(db_session).create(
        person_id=core["person"].id,
        scope=ContactRuleScope.GLOBAL,
        type="DO_NOT_CONTACT",
        value="stop",
        source=ContactRuleSource.PARTICIPANT_REQUESTED,
        strength=RuleStrength.STRONG,
    )
    outbox = OutboxService(db_session).create_beeper(
        task_instance_id=core["task"].id,
        conversation_id=core["conversation"].id,
        participant_ids=[core["participant"].id],
        final_text="question",
        message_kind=MessageKind.INITIAL,
        idempotency_key="worker-awaiting-owner",
    )
    db_session.commit()
    calls = {"validator": 0, "transport": 0}

    class Validator:
        def validate(self, **_: object) -> bool:
            calls["validator"] += 1
            return True

    class Adapter:
        def send(self, _request: object) -> DeliveryResult:
            calls["transport"] += 1
            return DeliveryResult(True, True, provider_message_id="sent")

        def reconcile(self, _request: object, **_: object) -> None:
            return None

    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    worker = OutboxWorker(
        factory,
        revalidator=PolicyRevalidator(owner_chat_id=99),
        validator=Validator(),
        adapters={Transport.BEEPER: Adapter()},
    )
    assert worker.process(outbox.id) is OutboxStatus.PENDING
    assert calls == {"validator": 0, "transport": 0}


def test_respect_boundary_cancels_pending_task_sends_without_repeat_decision(db_session: Session) -> None:
    core = seed_core(db_session)
    rule = ContactRuleService(db_session).create(
        person_id=core["person"].id, scope=ContactRuleScope.TOPIC, topic_key="tennis",
        type="DO_NOT_CONTACT", value="no tennis", source=ContactRuleSource.PARTICIPANT_REQUESTED,
        strength=RuleStrength.STRONG,
    )
    first = OutboxService(db_session).create_beeper(
        task_instance_id=core["task"].id, conversation_id=core["conversation"].id,
        participant_ids=[core["participant"].id], final_text="one", message_kind=MessageKind.INITIAL,
        idempotency_key="respect-one",
    )
    second = OutboxService(db_session).create_beeper(
        task_instance_id=core["task"].id, conversation_id=core["conversation"].id,
        participant_ids=[core["participant"].id], final_text="two", message_kind=MessageKind.REMINDER,
        idempotency_key="respect-two",
    )
    revalidator = PolicyRevalidator(owner_chat_id=99)
    assert revalidator.check(db_session, first) is PreSendDecision.AWAITING_OWNER
    decision = db_session.scalar(select(DecisionRequest).where(DecisionRequest.contact_rule_id == rule.id))
    assert decision is not None
    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    handler = ProductionOwnerCommandHandler(factory, owner_chat_id=99, resolver=object(), generation=object())  # type: ignore[arg-type]
    handler.apply_decision(db_session, decision.id, PreparedOwnerDecision("respect_boundary"), object())  # type: ignore[arg-type]
    assert first.status is OutboxStatus.CANCELLED
    assert second.status is OutboxStatus.CANCELLED
    assert db_session.scalar(
        select(func.count(DecisionRequest.id)).where(
            DecisionRequest.contact_rule_id == rule.id
        )
    ) == 1


def test_multiple_boundaries_are_decided_one_at_a_time(db_session: Session) -> None:
    core = seed_core(db_session)
    first_rule = ContactRuleService(db_session).create(
        person_id=core["person"].id,
        scope=ContactRuleScope.GLOBAL,
        type="DO_NOT_CONTACT",
        value="global boundary",
        source=ContactRuleSource.PARTICIPANT_REQUESTED,
        strength=RuleStrength.STRONG,
    )
    second_rule = ContactRuleService(db_session).create(
        person_id=core["person"].id,
        scope=ContactRuleScope.TOPIC,
        topic_key="tennis",
        type="DO_NOT_CONTACT",
        value="topic boundary",
        source=ContactRuleSource.PARTICIPANT_REQUESTED,
        strength=RuleStrength.STRONG,
    )
    outbox = OutboxService(db_session).create_beeper(
        task_instance_id=core["task"].id,
        conversation_id=core["conversation"].id,
        participant_ids=[core["participant"].id],
        final_text="question",
        message_kind=MessageKind.INITIAL,
        idempotency_key="sequential-boundaries",
    )
    revalidator = PolicyRevalidator(owner_chat_id=99)
    assert revalidator.check(db_session, outbox) is PreSendDecision.AWAITING_OWNER
    first_decision = db_session.scalar(
        select(DecisionRequest).where(DecisionRequest.contact_rule_id == first_rule.id)
    )
    assert first_decision is not None
    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    handler = ProductionOwnerCommandHandler(factory, owner_chat_id=99, resolver=object(), generation=object())  # type: ignore[arg-type]
    handler.apply_decision(db_session, first_decision.id, PreparedOwnerDecision("allow_this"), object())  # type: ignore[arg-type]
    assert revalidator.check(db_session, outbox) is PreSendDecision.AWAITING_OWNER
    second_decision = db_session.scalar(
        select(DecisionRequest).where(DecisionRequest.contact_rule_id == second_rule.id)
    )
    assert second_decision is not None


def test_non_candidate_model_scope_fails_closed(db_session: Session) -> None:
    core = seed_core(db_session)

    class Backend:
        def infer(self, **_: object) -> dict[str, object]:
            return {"kind": "BOUNDARY", "scope": "TASK_INSTANCE", "topic_key": None, "task_instance_id": 999999}

    with pytest.raises(ModelOutputError):
        ModelContactBoundaryClassifier(Backend()).classify(
            core["revision"], boundary_context(db_session, core["revision"])
        )


def test_expired_equivalent_boundary_is_not_reused(db_session: Session) -> None:
    core = seed_core(db_session)
    ContactRuleService(db_session).create(
        person_id=core["person"].id,
        scope=ContactRuleScope.TOPIC,
        topic_key="tennis",
        type="DO_NOT_CONTACT",
        value="expired",
        source=ContactRuleSource.PARTICIPANT_REQUESTED,
        strength=RuleStrength.STRONG,
        expires_at=utc_now() - timedelta(minutes=1),
    )
    apply_contact_boundary(
        db_session,
        core["revision"],
        ContactBoundary("BOUNDARY", ContactRuleScope.TOPIC, "tennis"),
        boundary_context(db_session, core["revision"]),
        owner_chat_id=None,
    )
    assert db_session.scalar(select(func.count(ContactRule.id))) == 2


@pytest.mark.parametrize(
    "replacement",
    [
        ContactBoundary("NONE"),
        ContactBoundary("BOUNDARY", ContactRuleScope.GLOBAL),
    ],
)
def test_reclassification_clears_current_ambiguity_hold(
    db_session: Session,
    replacement: ContactBoundary,
) -> None:
    core = seed_core(db_session)
    context = boundary_context(db_session, core["revision"])
    decision = apply_contact_boundary(
        db_session,
        core["revision"],
        ContactBoundary("AMBIGUOUS"),
        context,
        owner_chat_id=99,
    )
    assert decision is not None
    apply_contact_boundary(
        db_session,
        core["revision"],
        replacement,
        context,
        owner_chat_id=99,
    )
    assert decision.status is DecisionStatus.CLOSED
    assert decision.close_reason is DecisionCloseReason.SUBJECT_RESOLVED


def test_malformed_contact_rule_decision_cannot_override_or_cancel(db_session: Session) -> None:
    core = seed_core(db_session)
    rule = ContactRuleService(db_session).create(
        person_id=core["person"].id,
        scope=ContactRuleScope.GLOBAL,
        type="DO_NOT_CONTACT",
        value="owner rule",
        source=ContactRuleSource.USER_CONFIGURED,
        strength=RuleStrength.STRONG,
    )
    outbox = OutboxService(db_session).create_beeper(
        task_instance_id=core["task"].id,
        conversation_id=core["conversation"].id,
        participant_ids=[core["participant"].id],
        final_text="still pending",
        message_kind=MessageKind.INITIAL,
        idempotency_key="spoof-rule-decision",
    )
    decision = DecisionService(db_session).create(
        decision_type="CONTACT_RULE_EXCEPTION",
        subject_kind="contact_rule",
        subject_id=rule.id,
        context={},
        task_instance_id=core["task"].id,
        parent_terminal_policy=ParentTerminalPolicy.TERMINATE,
    )
    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    handler = ProductionOwnerCommandHandler(factory, owner_chat_id=99, resolver=object(), generation=object())  # type: ignore[arg-type]
    handler.apply_decision(
        db_session,
        decision.id,
        PreparedOwnerDecision("respect_boundary"),
        object(),  # type: ignore[arg-type]
    )
    assert decision.close_reason is DecisionCloseReason.SUBJECT_RESOLVED
    assert outbox.status is OutboxStatus.PENDING
    assert db_session.scalar(
        select(func.count(ContactRule.id)).where(
            ContactRule.overrides_contact_rule_id == rule.id
        )
    ) == 0


def test_aggregate_logical_targets_block_without_stale_exception_prompt(db_session: Session) -> None:
    core = seed_core(db_session)
    other_person = Person(display_name="Blair", metadata_json={})
    db_session.add(other_person)
    db_session.flush()
    other_identity = Identity(
        person_id=other_person.id,
        beeper_user_id="beeper:blair",
        network="discord",
        metadata_json={},
    )
    db_session.add(other_identity)
    db_session.flush()
    db_session.add(
        ConversationParticipant(
            conversation_id=core["conversation"].id,
            identity_id=other_identity.id,
            is_current=True,
        )
    )
    other_participant = TaskParticipant(
        task_instance_id=core["task"].id,
        person_id=other_person.id,
        conversation_id=core["conversation"].id,
    )
    db_session.add(other_participant)
    db_session.flush()
    ContactRuleService(db_session).create(
        person_id=core["person"].id,
        scope=ContactRuleScope.GLOBAL,
        type="DO_NOT_CONTACT",
        value="ask owner",
        source=ContactRuleSource.PARTICIPANT_REQUESTED,
        strength=RuleStrength.STRONG,
    )
    ContactRuleService(db_session).create(
        person_id=other_person.id,
        scope=ContactRuleScope.TASK_INSTANCE,
        task_instance_id=core["task"].id,
        type="DO_NOT_CONTACT",
        value="absolute",
        source=ContactRuleSource.PARTICIPANT_REQUESTED,
        strength=RuleStrength.STRONG,
    )
    outbox = OutboxService(db_session).create_beeper(
        task_instance_id=core["task"].id,
        conversation_id=core["conversation"].id,
        participant_ids=[core["participant"].id, other_participant.id],
        final_text="group question",
        message_kind=MessageKind.INITIAL,
        idempotency_key="aggregate-policy",
    )
    assert PolicyRevalidator(owner_chat_id=99).check(db_session, outbox) is PreSendDecision.POLICY_BLOCKED
    assert db_session.scalar(select(func.count(DecisionRequest.id))) == 0
