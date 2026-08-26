from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from ten_texter.enums import MessageKind, OutboxStatus, Transport, ValidatorCategory
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
    GenerationOutcome,
    IndependentMessageValidator,
    MinimalValidatorContextProvider,
    OutboxValidatorGate,
    ValidatedGenerationPipeline,
    ValidatorContext,
)
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
