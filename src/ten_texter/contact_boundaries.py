from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from ten_texter.decision_prompts import contact_boundary_ambiguity_prompt
from ten_texter.domain import ContactRuleService, DecisionService, DomainError, utc_now
from ten_texter.enums import (
    ContactRuleScope,
    ContactRuleSource,
    DecisionCloseReason,
    DecisionStatus,
    MessageKind,
    ParentTerminalPolicy,
    RuleStrength,
    TaskStatus,
    OutboxStatus,
)
from ten_texter.models import (
    DecisionRequest,
    DecisionRequestPrompt,
    ContactRule,
    Identity,
    Message,
    MessageRevision,
    TaskInstance,
    TaskParticipant,
    TaskDefinition,
    OutboxMessage,
    OutboxMessageParticipant,
    Person,
)
from ten_texter.outbox import OutboxService


MAX_BOUNDARY_CANDIDATES = 100


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class BoundaryCandidate:
    task_instance_id: int
    topic_key: str


@dataclass(frozen=True, slots=True)
class BoundaryContext:
    person_id: int
    candidates: tuple[BoundaryCandidate, ...]
    complete: bool
    attributable_task_id: int | None = None

    @property
    def task_ids(self) -> frozenset[int]:
        return frozenset(item.task_instance_id for item in self.candidates) if self.complete else frozenset()

    @property
    def topic_keys(self) -> frozenset[str]:
        return frozenset(item.topic_key for item in self.candidates) if self.complete else frozenset()


@dataclass(frozen=True, slots=True)
class ContactBoundary:
    kind: str
    scope: ContactRuleScope | None = None
    topic_key: str | None = None
    task_instance_id: int | None = None


class ContactBoundaryClassifier(Protocol):
    def classify(self, revision: MessageRevision, context: BoundaryContext) -> ContactBoundary: ...


class NoContactBoundaryClassifier:
    def classify(self, revision: MessageRevision, context: BoundaryContext) -> ContactBoundary:
        return ContactBoundary("NONE")


def boundary_context(
    session: Session,
    revision: MessageRevision,
    *,
    attributable_task_id: int | None = None,
) -> BoundaryContext:
    message = session.get(Message, revision.message_id)
    identity = session.get(Identity, message.sender_identity_id) if message is not None else None
    if identity is None:
        raise DomainError("contact-boundary sender is unavailable")
    rows = list(
        session.execute(
            select(TaskInstance.id, TaskInstance.topic_key)
            .join(TaskParticipant, TaskParticipant.task_instance_id == TaskInstance.id)
            .where(
                TaskParticipant.person_id == identity.person_id,
                TaskInstance.status == TaskStatus.ACTIVE,
            )
            .order_by(TaskInstance.id)
            .limit(MAX_BOUNDARY_CANDIDATES + 1)
        )
    )
    return BoundaryContext(
        identity.person_id,
        tuple(BoundaryCandidate(task_id, topic) for task_id, topic in rows[:MAX_BOUNDARY_CANDIDATES]),
        len(rows) <= MAX_BOUNDARY_CANDIDATES,
        attributable_task_id,
    )


def apply_contact_boundary(
    session: Session,
    revision: MessageRevision,
    boundary: ContactBoundary,
    context: BoundaryContext,
    *,
    owner_chat_id: int | None,
) -> DecisionRequest | None:
    message = session.get(Message, revision.message_id)
    if message is None or message.current_revision_id != revision.id:
        raise DomainError("contact-boundary revision is not current")
    stale_decisions = list(
        session.scalars(
            select(DecisionRequest)
            .join(
                MessageRevision,
                MessageRevision.id == DecisionRequest.message_revision_id,
            )
            .where(
                MessageRevision.message_id == revision.message_id,
                DecisionRequest.type == "CONTACT_BOUNDARY_AMBIGUITY",
                DecisionRequest.status == DecisionStatus.PENDING,
                DecisionRequest.message_revision_id != revision.id,
            )
        )
    )
    for stale in stale_decisions:
        DecisionService(session).close(stale.id, DecisionCloseReason.SUBJECT_RESOLVED)
    current_ambiguity = session.scalar(
        select(DecisionRequest).where(
            DecisionRequest.message_revision_id == revision.id,
            DecisionRequest.type == "CONTACT_BOUNDARY_AMBIGUITY",
            DecisionRequest.status == DecisionStatus.PENDING,
        )
    )
    if boundary.kind != "AMBIGUOUS" and current_ambiguity is not None:
        DecisionService(session).close(
            current_ambiguity.id,
            DecisionCloseReason.SUBJECT_RESOLVED,
        )
    if boundary.kind == "NONE":
        return None
    if boundary.kind == "BOUNDARY":
        if boundary.scope is ContactRuleScope.GLOBAL:
            targets = {}
        elif boundary.scope is ContactRuleScope.TOPIC and boundary.topic_key in context.topic_keys:
            targets = {"topic_key": boundary.topic_key}
        elif boundary.scope is ContactRuleScope.TASK_INSTANCE and boundary.task_instance_id in context.task_ids:
            targets = {"task_instance_id": boundary.task_instance_id}
        else:
            raise DomainError("contact-boundary classifier selected a non-candidate scope")
        existing = session.scalar(select(ContactRule).where(
            ContactRule.person_id == context.person_id,
            ContactRule.scope == boundary.scope,
            ContactRule.type == "DO_NOT_CONTACT",
            ContactRule.source == ContactRuleSource.PARTICIPANT_REQUESTED,
            ContactRule.strength == RuleStrength.STRONG,
            ContactRule.revoked_at.is_(None),
            or_(ContactRule.expires_at.is_(None), ContactRule.expires_at > utc_now()),
            *[getattr(ContactRule, key) == value for key, value in targets.items()],
        ))
        if existing is None:
            ContactRuleService(session).create(
                person_id=context.person_id,
                scope=boundary.scope,
                type="DO_NOT_CONTACT",
                value=revision.text or "<deleted>",
                source=ContactRuleSource.PARTICIPANT_REQUESTED,
                strength=RuleStrength.STRONG,
                **targets,
            )
        return None
    if boundary.kind != "AMBIGUOUS":
        raise DomainError("unsupported contact-boundary classification")
    decision = session.scalar(
        select(DecisionRequest).where(
            DecisionRequest.message_revision_id == revision.id,
            DecisionRequest.type == "CONTACT_BOUNDARY_AMBIGUITY",
        )
    )
    if decision is None:
        unique_task = context.attributable_task_id
        if unique_task is None and context.complete and len(context.candidates) == 1:
            unique_task = context.candidates[0].task_instance_id
        decision = DecisionService(session).create(
            decision_type="CONTACT_BOUNDARY_AMBIGUITY",
            subject_kind="message_revision",
            subject_id=revision.id,
            context={},
            task_instance_id=unique_task,
            parent_terminal_policy=(
                ParentTerminalPolicy.TERMINATE
                if unique_task is not None
                else ParentTerminalPolicy.SURVIVE
            ),
        )
    if owner_chat_id is not None:
        prompt = OutboxService(session).create_owner(
            telegram_chat_id=owner_chat_id,
            final_text=contact_boundary_ambiguity_prompt(decision.id, context.candidates),
            message_kind=MessageKind.NOTIFICATION,
            idempotency_key=f"decision:{decision.id}:owner-prompt",
            task_instance_id=decision.task_instance_id,
            parent_terminal_policy=decision.parent_terminal_policy,
        )
        if session.get(DecisionRequestPrompt, {"decision_request_id": decision.id, "outbox_message_id": prompt.id}) is None:
            session.add(DecisionRequestPrompt(decision_request_id=decision.id, outbox_message_id=prompt.id))
            session.flush()
    return decision


def pending_boundary_hold(session: Session, *, person_id: int, task_id: int) -> bool:
    decisions = session.scalars(
        select(DecisionRequest)
        .join(MessageRevision, MessageRevision.id == DecisionRequest.message_revision_id)
        .join(Message, Message.id == MessageRevision.message_id)
        .join(Identity, Identity.id == Message.sender_identity_id)
        .where(
            Identity.person_id == person_id,
            DecisionRequest.type == "CONTACT_BOUNDARY_AMBIGUITY",
            DecisionRequest.status == DecisionStatus.PENDING,
            Message.current_revision_id == MessageRevision.id,
        )
    )
    return any(decision.task_instance_id is None or decision.task_instance_id == task_id for decision in decisions)


@dataclass(frozen=True, slots=True)
class ContactRuleExceptionSubject:
    rule: ContactRule
    task: TaskInstance
    person: Person
    pending_outbox_ids: tuple[int, ...]


def contact_rule_exception_subject(
    session: Session,
    *,
    rule_id: int,
    task_id: int,
) -> ContactRuleExceptionSubject | None:
    rule = session.get(ContactRule, rule_id)
    task = session.get(TaskInstance, task_id)
    person = session.get(Person, rule.person_id) if rule is not None else None
    if (
        rule is None
        or task is None
        or person is None
        or task.status is not TaskStatus.ACTIVE
        or task.archived_at is not None
        or rule.revoked_at is not None
        or (rule.expires_at is not None and _aware(rule.expires_at) <= utc_now())
        or rule.source is not ContactRuleSource.PARTICIPANT_REQUESTED
        or rule.strength is not RuleStrength.STRONG
        or rule.type != "DO_NOT_CONTACT"
        or rule.scope is ContactRuleScope.TASK_INSTANCE
    ):
        return None
    if rule.scope is ContactRuleScope.TOPIC and rule.topic_key != task.topic_key:
        return None
    if rule.scope is ContactRuleScope.TASK_DEFINITION:
        definition = session.get(TaskDefinition, rule.task_definition_id)
        if (
            definition is None
            or definition.archived_at is not None
            or rule.task_definition_id != task.task_definition_id
        ):
            return None
    override = session.scalar(
        select(ContactRule.id).where(
            ContactRule.overrides_contact_rule_id == rule.id,
            ContactRule.scope == ContactRuleScope.TASK_INSTANCE,
            ContactRule.task_instance_id == task.id,
            ContactRule.revoked_at.is_(None),
            or_(ContactRule.expires_at.is_(None), ContactRule.expires_at > utc_now()),
        )
    )
    if override is not None:
        return None
    pending_ids = tuple(
        session.scalars(
            select(OutboxMessage.id)
            .join(
                OutboxMessageParticipant,
                OutboxMessageParticipant.outbox_message_id == OutboxMessage.id,
            )
            .join(
                TaskParticipant,
                TaskParticipant.id == OutboxMessageParticipant.task_participant_id,
            )
            .where(
                OutboxMessage.task_instance_id == task.id,
                OutboxMessage.status == OutboxStatus.PENDING,
                TaskParticipant.person_id == rule.person_id,
            )
            .order_by(OutboxMessage.id)
        )
    )
    if not pending_ids:
        return None
    return ContactRuleExceptionSubject(rule, task, person, pending_ids)
