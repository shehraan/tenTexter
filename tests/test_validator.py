from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from ten_texter.enums import MessageKind, OutboxCancelReason, OutboxStatus, Transport, ValidatorCategory
from ten_texter.control import ProductionOwnerCommandHandler
from ten_texter.health import HealthMonitor
from ten_texter.model_clients import MessageGenerator, ModelOutputError, ModelUnavailable
from ten_texter.models import DecisionRequest, DecisionRequestPrompt, OutboxMessage
from ten_texter.outbox import (
    AllowingRevalidator,
    DeliveryRequest,
    DeliveryResult,
    OutboxService,
    OutboxWorker,
)
from ten_texter.validator import (
    DatabaseValidatorContextProvider,
    GenerationOutcome,
    IndependentMessageValidator,
    MinimalValidatorContextProvider,
    OutboxValidatorGate,
    ValidatedGenerationPipeline,
    ValidatorContext,
)
from ten_texter.policy import DatabaseContextProvider
from ten_texter.telegram import TelegramControlGateway
from tests.test_schema import seed_core


class Backend:
    def __init__(self, outputs: list[dict[str, object]], unavailable: bool = False):
        self.outputs = outputs
        self.unavailable = unavailable
        self.payloads: list[dict[str, object]] = []

    def infer(self, *, operation: str, payload: dict[str, object]) -> dict[str, object]:
        self.payloads.append(payload)
        if self.unavailable:
            raise ModelUnavailable("offline")
        return self.outputs.pop(0)


class ExactClaimBackend:
    def __init__(self):
        self.payloads: list[dict[str, object]] = []

    def infer(self, *, operation: str, payload: dict[str, object]) -> dict[str, object]:
        self.payloads.append(payload)
        if payload["exact_text"] in payload["allowed_claims"]:
            return {"category": "VALID", "critique": None}
        return {
            "category": "UNSUPPORTED_CLAIM",
            "critique": "The exact text is not one of the allowed claims.",
        }


def context() -> ValidatorContext:
    return ValidatorContext(
        message_kind=MessageKind.INITIAL,
        allowed_claims=("The event is at 5 PM.",),
        constraints=("Ask availability only.",),
        allowed_disclosure_scopes=("SCHEDULING",),
    )


def test_validator_output_is_strict_and_minimal() -> None:
    backend = Backend([{"category": "VALID", "critique": None}])
    result = IndependentMessageValidator(backend).review(text="Are you free at 5?", context=context())
    assert result.category is ValidatorCategory.VALID
    payload = backend.payloads[0]
    assert set(payload) == {
        "trusted_instructions",
        "exact_text",
        "message_kind",
        "allowed_claims",
        "constraints",
        "allowed_disclosure_scopes",
    }


def test_health_notification_gets_exact_owner_only_validator_claim(db_session) -> None:
    HealthMonitor(owner_chat_id=99).record(
        db_session,
        "telegram",
        healthy=False,
        details="ModelUnavailable",
    )
    message = db_session.scalar(select(OutboxMessage))
    assert message is not None
    db_session.commit()
    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    contexts = DatabaseValidatorContextProvider(
        factory,
        facts=DatabaseContextProvider(),
        owner_chat_id=99,
    )

    health_context = contexts.context_for(message.id, message.message_kind)

    assert health_context.allowed_claims == ("telegram became unhealthy. ModelUnavailable",)
    assert health_context.allowed_disclosure_scopes == ()
    assert "Do not make commitments on the owner's behalf." in health_context.constraints
    assert (
        "This owner-only notification may report exactly the enumerated health-status claim; "
        "it does not authorize any other fact or commitment."
        in health_context.constraints
    )


def test_health_notification_still_passes_through_independent_validator_gate(db_session) -> None:
    HealthMonitor(owner_chat_id=99).record(
        db_session,
        "telegram",
        healthy=False,
        details="ModelUnavailable",
    )
    message = db_session.scalar(select(OutboxMessage))
    assert message is not None
    db_session.commit()
    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    backend = ExactClaimBackend()
    contexts = DatabaseValidatorContextProvider(
        factory,
        facts=DatabaseContextProvider(),
        owner_chat_id=99,
    )
    gate = OutboxValidatorGate(
        factory,
        validator=IndependentMessageValidator(backend),
        contexts=contexts,
    )

    assert gate.validate(
        text=message.final_text,
        message_kind=message.message_kind,
        outbox_id=message.id,
    )
    assert len(backend.payloads) == 1

    result = IndependentMessageValidator(backend).review(
        text=f"{message.final_text} Alex is at a private location.",
        context=contexts.context_for(message.id, message.message_kind),
    )
    assert result.category is ValidatorCategory.UNSUPPORTED_CLAIM


def test_health_notification_without_configured_owner_fails_closed(db_session) -> None:
    HealthMonitor(owner_chat_id=99).record(
        db_session,
        "telegram",
        healthy=False,
        details="ModelUnavailable",
    )
    message = db_session.scalar(select(OutboxMessage))
    assert message is not None
    db_session.commit()
    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)

    context_without_owner = DatabaseValidatorContextProvider(
        factory,
        facts=DatabaseContextProvider(),
        owner_chat_id=None,
    ).context_for(message.id, message.message_kind)

    assert context_without_owner.allowed_claims == ()


def test_ordinary_owner_notification_gets_no_health_authorization(db_session) -> None:
    message = OutboxService(db_session).create_owner(
        telegram_chat_id=99,
        final_text="Routine owner notification.",
        message_kind=MessageKind.NOTIFICATION,
        idempotency_key="owner:routine:1",
    )
    db_session.commit()
    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)

    ordinary = DatabaseValidatorContextProvider(
        factory,
        facts=DatabaseContextProvider(),
        owner_chat_id=99,
    ).context_for(message.id, message.message_kind)

    assert ordinary.allowed_claims == ()


def test_beeper_message_never_inherits_health_authorization(db_session) -> None:
    core = seed_core(db_session)
    message = OutboxService(db_session).create_beeper(
        task_instance_id=core["task"].id,
        conversation_id=core["conversation"].id,
        participant_ids=[core["participant"].id],
        final_text="telegram became unhealthy. ModelUnavailable",
        message_kind=MessageKind.NOTIFICATION,
        idempotency_key="health:telegram:unhealthy:1",
    )
    db_session.commit()
    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)

    participant = DatabaseValidatorContextProvider(
        factory,
        facts=DatabaseContextProvider(),
        owner_chat_id=99,
    ).context_for(message.id, message.message_kind)

    assert message.final_text not in participant.allowed_claims


def test_task_linked_owner_message_cannot_spoof_health_authorization(db_session) -> None:
    core = seed_core(db_session)
    message = OutboxService(db_session).create_owner(
        telegram_chat_id=99,
        task_instance_id=core["task"].id,
        final_text="telegram became unhealthy. ModelUnavailable",
        message_kind=MessageKind.NOTIFICATION,
        idempotency_key="health:telegram:unhealthy:1",
    )
    db_session.commit()
    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)

    task_linked = DatabaseValidatorContextProvider(
        factory,
        facts=DatabaseContextProvider(),
        owner_chat_id=99,
    ).context_for(message.id, message.message_kind)

    assert message.final_text not in task_linked.allowed_claims


@pytest.mark.parametrize(
    ("telegram_chat_id", "key", "text"),
    [
        (99, "health:telegram:unhealthy:0", "telegram became unhealthy. ModelUnavailable"),
        (99, "health:telegram:unhealthy:1:extra", "telegram became unhealthy. ModelUnavailable"),
        (99, "health:telegram:unhealthy:1", "Alex is at a private location."),
        (
            99,
            "health:telegram:unhealthy:1",
            "telegram became unhealthy. ModelUnavailable Alex is at a private location.",
        ),
        (100, "health:telegram:unhealthy:1", "telegram became unhealthy. ModelUnavailable"),
    ],
)
def test_malformed_or_spoof_like_health_notification_fails_closed(
    db_session,
    telegram_chat_id: int,
    key: str,
    text: str,
) -> None:
    message = OutboxService(db_session).create_owner(
        telegram_chat_id=telegram_chat_id,
        final_text=text,
        message_kind=MessageKind.NOTIFICATION,
        idempotency_key=key,
    )
    db_session.commit()
    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)

    malformed = DatabaseValidatorContextProvider(
        factory,
        facts=DatabaseContextProvider(),
        owner_chat_id=99,
    ).context_for(message.id, message.message_kind)

    assert malformed.allowed_claims == ()


@pytest.mark.parametrize(
    "output",
    [
        {"category": "VALID", "critique": "No issues."},
        {"category": "UNSUPPORTED_CLAIM", "critique": None},
        {"category": "UNSUPPORTED_CLAIM", "critique": ""},
    ],
)
def test_validator_rejects_inconsistent_category_critique(output: dict[str, object]) -> None:
    with pytest.raises(ModelOutputError):
        IndependentMessageValidator(Backend([output])).review(
            text="Are you free at 5?",
            context=context(),
        )


@pytest.mark.parametrize(
    "output",
    [
        {"category": "VALID", "critique": None},
        {"category": "RULE_VIOLATION", "critique": "A current rule blocks this message."},
    ],
)
def test_validator_accepts_consistent_category_critique(output: dict[str, object]) -> None:
    result = IndependentMessageValidator(Backend([output])).review(
        text="Are you free at 5?",
        context=context(),
    )
    assert result.category.value == output["category"]


def test_clarity_failure_repairs_with_bounded_critique() -> None:
    generator_backend = Backend(
        [
            {"text": "Free?"},
            {"text": "Are you available for tennis at 5 PM?"},
        ]
    )
    validator_backend = Backend(
        [
            {"category": "UNCLEAR_OR_AMBIGUOUS", "critique": "Name the activity and time."},
            {"category": "VALID", "critique": None},
        ]
    )
    result = ValidatedGenerationPipeline(
        generator=MessageGenerator(generator_backend),
        validator=IndependentMessageValidator(validator_backend),
        max_repairs=2,
    ).run(
        goal="ask availability",
        facts=[],
        constraints=[],
        context=context(),
    )
    assert result.outcome is GenerationOutcome.READY
    assert result.text == "Are you available for tennis at 5 PM?"
    assert generator_backend.payloads[1]["validator_critique"] == "Name the activity and time."


def test_authority_violation_asks_owner_without_repair() -> None:
    generator_backend = Backend([{"text": "I booked the court for us."}])
    validator_backend = Backend(
        [
            {
                "category": "UNAUTHORIZED_COMMITMENT",
                "critique": "The owner did not authorize booking.",
            }
        ]
    )
    result = ValidatedGenerationPipeline(
        generator=MessageGenerator(generator_backend),
        validator=IndependentMessageValidator(validator_backend),
    ).run(goal="ask", facts=[], constraints=[], context=context())
    assert result.outcome is GenerationOutcome.ASK_ME
    assert result.text is None
    assert len(generator_backend.payloads) == 1


def test_repair_cap_exhaustion_asks_owner() -> None:
    generator_backend = Backend([{"text": "Free?"}, {"text": "When?"}])
    validator_backend = Backend(
        [
            {"category": "UNCLEAR_OR_AMBIGUOUS", "critique": "Unclear."},
            {"category": "WRONG_MESSAGE_KIND", "critique": "Still wrong."},
        ]
    )
    result = ValidatedGenerationPipeline(
        generator=MessageGenerator(generator_backend),
        validator=IndependentMessageValidator(validator_backend),
        max_repairs=1,
    ).run(goal="ask", facts=[], constraints=[], context=context())
    assert result.outcome is GenerationOutcome.ASK_ME
    assert len(generator_backend.payloads) == 2


class FakeTransport:
    def __init__(self):
        self.calls = 0

    def send(self, request: DeliveryRequest) -> DeliveryResult:
        self.calls += 1
        return DeliveryResult(True, True, provider_message_id="sent")

    def reconcile(self, *_: object, **__: object) -> str | None:
        return None


def test_invalid_outbox_text_cannot_send_and_creates_owner_decision(db_session) -> None:
    core = seed_core(db_session)
    message = OutboxService(db_session).create_beeper(
        task_instance_id=core["task"].id,
        conversation_id=core["conversation"].id,
        participant_ids=[core["participant"].id],
        final_text="I booked it for us.",
        message_kind=MessageKind.INITIAL,
        idempotency_key="validator-block",
    )
    db_session.commit()
    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    backend = Backend(
        [
            {
                "category": "UNAUTHORIZED_COMMITMENT",
                "critique": "No authority to book.",
            }
        ]
    )
    gate = OutboxValidatorGate(
        factory,
        validator=IndependentMessageValidator(backend),
        contexts=MinimalValidatorContextProvider(constraints=("No commitments.",)),
        owner_chat_id=99,
    )
    transport = FakeTransport()
    worker = OutboxWorker(
        factory,
        revalidator=AllowingRevalidator(),
        validator=gate,
        adapters={Transport.BEEPER: transport},
    )
    assert worker.process(message.id) is OutboxStatus.PENDING
    assert transport.calls == 0
    db_session.expire_all()
    decision = db_session.scalar(
        select(DecisionRequest).where(DecisionRequest.outbox_message_id == message.id)
    )
    assert decision is not None
    assert decision.type == "VALIDATOR_AUTHORITY_VIOLATION"
    assert db_session.scalar(
        select(DecisionRequestPrompt).where(DecisionRequestPrompt.decision_request_id == decision.id)
    ) is not None
    assert db_session.get(OutboxMessage, message.id).final_text == "I booked it for us."
    assert worker.process(message.id) is OutboxStatus.PENDING
    assert transport.calls == 0


def test_keep_blocked_durably_cancels_exact_immutable_message(db_session) -> None:
    core = seed_core(db_session)
    original_text = "I booked it for us."
    message = OutboxService(db_session).create_beeper(
        task_instance_id=core["task"].id,
        conversation_id=core["conversation"].id,
        participant_ids=[core["participant"].id],
        final_text=original_text,
        message_kind=MessageKind.INITIAL,
        idempotency_key="validator-durable-block",
    )
    db_session.commit()
    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    backend = Backend(
        [
            {"category": "UNAUTHORIZED_COMMITMENT", "critique": "No authority."},
            {"category": "VALID", "critique": None},
        ]
    )
    gate = OutboxValidatorGate(
        factory,
        validator=IndependentMessageValidator(backend),
        contexts=MinimalValidatorContextProvider(),
        owner_chat_id=99,
    )
    transport = FakeTransport()
    worker = OutboxWorker(
        factory,
        revalidator=AllowingRevalidator(),
        validator=gate,
        adapters={Transport.BEEPER: transport},
    )
    assert worker.process(message.id) is OutboxStatus.PENDING
    with factory() as session:
        decision = session.scalar(
            select(DecisionRequest).where(DecisionRequest.outbox_message_id == message.id)
        )
        assert decision is not None
        decision_id = decision.id
    handler = ProductionOwnerCommandHandler(
        factory, owner_chat_id=99, resolver=object(), generation=object()
    )  # type: ignore[arg-type]
    control = TelegramControlGateway(factory, owner_id=7, parser=object(), handler=handler)  # type: ignore[arg-type]

    def callback(update_id: int) -> dict[str, object]:
        return {
            "update_id": update_id,
            "callback_query": {
                "id": str(update_id),
                "from": {"id": 7},
                "message": {"message_id": 1, "chat": {"id": 99, "type": "private"}},
                "data": f"decision:{decision_id}:keep_blocked",
            },
        }

    received = control.receive(callback(910))
    control.process(received.telegram_update_row_id)
    repeated = control.receive(callback(911))
    control.process(repeated.telegram_update_row_id)
    restarted = OutboxWorker(
        factory,
        revalidator=AllowingRevalidator(),
        validator=gate,
        adapters={Transport.BEEPER: transport},
    )
    assert restarted.process(message.id) is OutboxStatus.CANCELLED
    assert transport.calls == 0
    with factory() as session:
        stored = session.get(OutboxMessage, message.id)
        assert stored.cancel_reason is OutboxCancelReason.POLICY_BLOCKED
        assert stored.final_text == original_text


def test_validator_unavailable_keeps_outbox_pending(db_session) -> None:
    core = seed_core(db_session)
    message = OutboxService(db_session).create_beeper(
        task_instance_id=core["task"].id,
        conversation_id=core["conversation"].id,
        participant_ids=[core["participant"].id],
        final_text="Are you free?",
        message_kind=MessageKind.INITIAL,
        idempotency_key="validator-offline",
    )
    db_session.commit()
    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    gate = OutboxValidatorGate(
        factory,
        validator=IndependentMessageValidator(Backend([], unavailable=True)),
        contexts=MinimalValidatorContextProvider(),
    )
    transport = FakeTransport()
    worker = OutboxWorker(
        factory,
        revalidator=AllowingRevalidator(),
        validator=gate,
        adapters={Transport.BEEPER: transport},
    )
    assert worker.process(message.id) is OutboxStatus.PENDING
    assert transport.calls == 0
