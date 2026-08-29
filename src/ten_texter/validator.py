from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from pydantic import Field, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from ten_texter.domain import DecisionService
from ten_texter.enums import (
    MessageKind,
    ParentTerminalPolicy,
    ValidatorCategory,
)
from ten_texter.model_clients import MessageGenerator, ModelBackend, StrictOutput, _validate
from ten_texter.health import owner_health_claim, owner_telegram_update_failure_claim
from ten_texter.models import DecisionRequest, DecisionRequestPrompt, OutboxMessage
from ten_texter.outbox import OutboxService
from ten_texter.policy import DatabaseContextProvider


VALIDATOR_CRITIQUE_MAX_LENGTH = 1000


class ValidatorOutput(StrictOutput):
    category: ValidatorCategory
    critique: str | None = Field(default=None, max_length=VALIDATOR_CRITIQUE_MAX_LENGTH)

    @model_validator(mode="after")
    def critique_consistency(self) -> "ValidatorOutput":
        if self.category is ValidatorCategory.VALID and self.critique is not None:
            raise ValueError("VALID output cannot include critique")
        if self.category is not ValidatorCategory.VALID and not self.critique:
            raise ValueError("invalid output requires bounded critique")
        return self


def validator_output_json_schema() -> dict[str, Any]:
    """llama.cpp-compatible discriminated schema for constrained validation output."""
    invalid_categories = [
        category.value for category in ValidatorCategory if category is not ValidatorCategory.VALID
    ]
    return {
        "oneOf": [
            {
                "type": "object",
                "properties": {
                    "category": {"const": ValidatorCategory.VALID.value},
                    "critique": {"type": "null"},
                },
                "required": ["category", "critique"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {
                    "category": {"enum": invalid_categories},
                    "critique": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": VALIDATOR_CRITIQUE_MAX_LENGTH,
                    },
                },
                "required": ["category", "critique"],
                "additionalProperties": False,
            },
        ]
    }


@dataclass(frozen=True, slots=True)
class ValidatorContext:
    message_kind: MessageKind
    allowed_claims: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    allowed_disclosure_scopes: tuple[str, ...] = ()


class IndependentMessageValidator:
    def __init__(self, backend: ModelBackend):
        self.backend = backend

    def review(self, *, text: str, context: ValidatorContext) -> ValidatorOutput:
        output = self.backend.infer(
            operation="message_validator",
            payload={
                "trusted_instructions": (
                    "Independently validate the exact immutable outbound text. Return only a bounded "
                    "category and critique. Do not repair text and do not request side effects."
                ),
                "exact_text": text,
                "message_kind": context.message_kind.value,
                "allowed_claims": list(context.allowed_claims),
                "constraints": list(context.constraints),
                "allowed_disclosure_scopes": list(context.allowed_disclosure_scopes),
            },
        )
        return _validate(ValidatorOutput, output)


class ValidatorContextProvider(Protocol):
    def context_for(self, outbox_id: int, message_kind: MessageKind) -> ValidatorContext: ...


class MinimalValidatorContextProvider:
    def __init__(
        self,
        *,
        allowed_claims: tuple[str, ...] = (),
        constraints: tuple[str, ...] = (),
        allowed_disclosure_scopes: tuple[str, ...] = (),
    ):
        self.allowed_claims = allowed_claims
        self.constraints = constraints
        self.allowed_disclosure_scopes = allowed_disclosure_scopes

    def context_for(self, outbox_id: int, message_kind: MessageKind) -> ValidatorContext:
        return ValidatorContext(
            message_kind=message_kind,
            allowed_claims=self.allowed_claims,
            constraints=self.constraints,
            allowed_disclosure_scopes=self.allowed_disclosure_scopes,
        )


class DatabaseValidatorContextProvider:
    """Recomputes the validator's exact allowed claims at pre-send time."""

    def __init__(
        self,
        sessions: sessionmaker[Session],
        *,
        facts: DatabaseContextProvider,
        owner_chat_id: int | None = None,
    ):
        self.sessions = sessions
        self.facts = facts
        self.owner_chat_id = owner_chat_id

    def context_for(self, outbox_id: int, message_kind: MessageKind) -> ValidatorContext:
        with self.sessions() as session:
            message = session.get(OutboxMessage, outbox_id)
            if message is None:
                return ValidatorContext(
                    message_kind=message_kind,
                    constraints=("Outbox message no longer exists; reject.",),
                )
            health_claim = (
                owner_health_claim(session, message, owner_chat_id=self.owner_chat_id)
                if message_kind is message.message_kind
                else None
            )
            update_failure_claim = (
                owner_telegram_update_failure_claim(
                    session, message, owner_chat_id=self.owner_chat_id
                )
                if message_kind is message.message_kind
                else None
            )
            operational_claim = health_claim or update_failure_claim
            if operational_claim is not None:
                claim_kind = "health-status" if health_claim is not None else "operational-status"
                return ValidatorContext(
                    message_kind=message_kind,
                    allowed_claims=(operational_claim,),
                    constraints=(
                        "Use only the enumerated allowed claims.",
                        "Do not make commitments on the owner's behalf.",
                        "Reject any private fact not present in allowed_claims.",
                        f"This owner-only notification may report exactly the enumerated {claim_kind} "
                        "claim; it does not authorize any other fact or commitment.",
                    ),
                )
            disclosed = self.facts.facts_for(session, message)
            allowed_claims = self.facts.task_claims(session, message) + tuple(
                str(fact.value) for fact in disclosed
            )
            scopes = tuple(sorted({fact.scope.value for fact in disclosed}))
            return ValidatorContext(
                message_kind=message_kind,
                allowed_claims=allowed_claims,
                constraints=(
                    "Use only the enumerated allowed claims.",
                    "Do not make commitments on the owner's behalf.",
                    "Reject any private fact not present in allowed_claims.",
                ),
                allowed_disclosure_scopes=scopes,
            )


class OutboxValidatorGate:
    """Adapter used by OutboxWorker; invalid immutable text is escalated, never repaired there."""

    AUTHORITY_CATEGORIES = {
        ValidatorCategory.UNSUPPORTED_CLAIM,
        ValidatorCategory.UNAUTHORIZED_COMMITMENT,
        ValidatorCategory.RULE_VIOLATION,
    }

    def __init__(
        self,
        sessions: sessionmaker[Session],
        *,
        validator: IndependentMessageValidator,
        contexts: ValidatorContextProvider,
        owner_chat_id: int | None = None,
    ):
        self.sessions = sessions
        self.validator = validator
        self.contexts = contexts
        self.owner_chat_id = owner_chat_id

    def validate(self, *, text: str, message_kind: MessageKind, outbox_id: int) -> bool:
        result = self.validator.review(
            text=text,
            context=self.contexts.context_for(outbox_id, message_kind),
        )
        if result.category is ValidatorCategory.VALID:
            return True
        decision_type = (
            "VALIDATOR_AUTHORITY_VIOLATION"
            if result.category in self.AUTHORITY_CATEGORIES
            else "VALIDATOR_REPAIR_REQUIRED"
        )
        with self.sessions.begin() as session:
            message = session.get(OutboxMessage, outbox_id)
            if message is None or message.final_text != text:
                return False
            existing = session.scalar(
                select(DecisionRequest).where(
                    DecisionRequest.outbox_message_id == outbox_id,
                    DecisionRequest.type == decision_type,
                )
            )
            if existing is None:
                decision = DecisionService(session).create(
                    decision_type=decision_type,
                    subject_kind="outbox_message",
                    subject_id=outbox_id,
                    context={
                        "validator_category": result.category.value,
                        "critique": result.critique,
                    },
                    task_instance_id=message.task_instance_id,
                    parent_terminal_policy=ParentTerminalPolicy.SURVIVE,
                )
                if self.owner_chat_id is not None:
                    prompt = OutboxService(session).create_owner(
                        telegram_chat_id=self.owner_chat_id,
                        final_text=(
                            f"Outbox {outbox_id} is blocked by validator category {result.category.value}. "
                            "Its immutable text will not be sent or regenerated by the worker. "
                            "Reply `keep blocked` to acknowledge."
                        ),
                        message_kind=MessageKind.NOTIFICATION,
                        idempotency_key=f"decision:{decision.id}:owner-prompt",
                        task_instance_id=message.task_instance_id,
                        parent_terminal_policy=ParentTerminalPolicy.SURVIVE,
                    )
                    session.add(DecisionRequestPrompt(decision_request_id=decision.id, outbox_message_id=prompt.id))
        return False


class GenerationOutcome(str, Enum):
    READY = "READY"
    ASK_ME = "ASK_ME"


@dataclass(frozen=True, slots=True)
class GenerationResult:
    outcome: GenerationOutcome
    text: str | None
    category: ValidatorCategory
    critique: str | None = None


class ValidatedGenerationPipeline:
    AUTHORITY_CATEGORIES = OutboxValidatorGate.AUTHORITY_CATEGORIES
    REPAIRABLE_CATEGORIES = {
        ValidatorCategory.WRONG_MESSAGE_KIND,
        ValidatorCategory.UNCLEAR_OR_AMBIGUOUS,
    }

    def __init__(
        self,
        *,
        generator: MessageGenerator,
        validator: IndependentMessageValidator,
        max_repairs: int = 2,
    ):
        if max_repairs < 0:
            raise ValueError("max_repairs must be non-negative")
        self.generator = generator
        self.validator = validator
        self.max_repairs = max_repairs

    def run(
        self,
        *,
        goal: str,
        facts: list[dict[str, Any]],
        constraints: list[str],
        context: ValidatorContext,
        untrusted_text: str | None = None,
    ) -> GenerationResult:
        critique: str | None = None
        for attempt in range(self.max_repairs + 1):
            text = self.generator.generate(
                goal=goal,
                facts=facts,
                constraints=constraints,
                untrusted_text=untrusted_text,
                critique=critique,
            )
            result = self.validator.review(text=text, context=context)
            if result.category is ValidatorCategory.VALID:
                return GenerationResult(GenerationOutcome.READY, text, result.category)
            if result.category in self.AUTHORITY_CATEGORIES:
                return GenerationResult(
                    GenerationOutcome.ASK_ME,
                    None,
                    result.category,
                    result.critique,
                )
            if result.category not in self.REPAIRABLE_CATEGORIES or attempt == self.max_repairs:
                return GenerationResult(
                    GenerationOutcome.ASK_ME,
                    None,
                    result.category,
                    result.critique,
                )
            critique = result.critique
        raise AssertionError("bounded generation loop exhausted unexpectedly")
