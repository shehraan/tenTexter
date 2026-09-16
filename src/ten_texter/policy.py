from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from ten_texter.decision_prompt_context import (
    authorize_decision_prompt,
    decision_for_prompt,
)
from ten_texter.domain import DecisionService, normalize_topic_key, utc_now
from ten_texter.enums import (
    ContactRuleScope,
    ContactRuleSource,
    DisclosureGrantStatus,
    DisclosureScope,
    PolicyOutcome,
    RuleStrength,
    TaskStatus,
    Transport,
)
from ten_texter.models import (
    BeeperOutboxDestination,
    ConversationParticipant,
    ContactRule,
    DecisionRequest,
    DecisionRequestPrompt,
    DisclosureGrant,
    DisclosureGrantScope,
    Identity,
    OutboxMessage,
    OutboxMessageParticipant,
    Message,
    MessageRevision,
    Person,
    TaskDefinition,
    TaskInstance,
    TaskParticipant,
)
from ten_texter.contact_boundaries import (
    contact_rule_exception_subject,
    pending_boundary_hold,
)
from ten_texter.decision_prompts import contact_rule_exception_prompt
from ten_texter.enums import DecisionStatus, MessageKind, ParentTerminalPolicy
from ten_texter.outbox import OutboxService
from ten_texter.nobody_available import (
    authorize_nobody_available_notification,
    is_nobody_available_notification,
)
from ten_texter.mass_contact import (
    MASS_CONTACT_THRESHOLD,
    ensure_mass_contact_decision,
    mass_contact_is_approved,
    mass_contact_participant_count,
)
from ten_texter.outbox import PreSendDecision, PreSendRevalidator


_SPECIFICITY = {
    ContactRuleScope.GLOBAL: 0,
    ContactRuleScope.TOPIC: 1,
    ContactRuleScope.TASK_DEFINITION: 2,
    ContactRuleScope.TASK_INSTANCE: 3,
}
_STRENGTH = {RuleStrength.WEAK: 0, RuleStrength.MEDIUM: 1, RuleStrength.STRONG: 2}
_TYPE_OUTCOME = {
    "ALLOW": PolicyOutcome.AUTO,
    "AUTO": PolicyOutcome.AUTO,
    "ASK_PARTICIPANT": PolicyOutcome.ASK_PARTICIPANT,
    "ASK_ME": PolicyOutcome.ASK_ME,
    "DO_NOT_CONTACT": PolicyOutcome.DO_NOT_ACT,
    "DO_NOT_ACT": PolicyOutcome.DO_NOT_ACT,
}


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


class ContactRuleResolver:
    @dataclass(frozen=True, slots=True)
    class Assessment:
        outcome: PolicyOutcome
        blocking_rule_ids: tuple[int, ...] = ()

    def resolve(
        self,
        session: Session,
        *,
        person_id: int,
        task: TaskInstance | None,
        topic_key: str | None,
        task_definition_id: int | None = None,
        at: datetime | None = None,
    ) -> PolicyOutcome:
        return self.assess(session, person_id=person_id, task=task, topic_key=topic_key,
                           task_definition_id=task_definition_id, at=at).outcome

    def assess(
        self, session: Session, *, person_id: int, task: TaskInstance | None,
        topic_key: str | None, task_definition_id: int | None = None,
        at: datetime | None = None,
    ) -> Assessment:
        timestamp = _aware(at or utc_now())
        rules = list(
            session.scalars(
                select(ContactRule).where(
                    ContactRule.person_id == person_id,
                    ContactRule.revoked_at.is_(None),
                )
            )
        )
        if topic_key is None and any(
            rule.scope is ContactRuleScope.TOPIC and self._live(session, rule, timestamp)
            for rule in rules
        ):
            return self.Assessment(PolicyOutcome.ASK_ME)
        normalized_topic = normalize_topic_key(topic_key) if topic_key is not None else None
        applicable = [
            rule
            for rule in rules
            if self._live(session, rule, timestamp)
            and self._matches(
                rule,
                task=task,
                topic_key=normalized_topic,
                task_definition_id=(
                    task.task_definition_id if task is not None else task_definition_id
                ),
            )
        ]
        overridden = {
            rule.overrides_contact_rule_id
            for rule in applicable
            if rule.overrides_contact_rule_id is not None
            and rule.type == "ALLOW"
            and rule.source is ContactRuleSource.USER_CONFIGURED
        }
        effective = [rule for rule in applicable if rule.id not in overridden]
        strong_boundaries = [
            rule
            for rule in effective
            if rule.source is ContactRuleSource.PARTICIPANT_REQUESTED
            and rule.strength is RuleStrength.STRONG
            and _TYPE_OUTCOME.get(rule.type, PolicyOutcome.ASK_ME) is not PolicyOutcome.AUTO
        ]
        if strong_boundaries:
            task_boundaries = [rule for rule in strong_boundaries if rule.scope is ContactRuleScope.TASK_INSTANCE]
            if task_boundaries:
                return self.Assessment(PolicyOutcome.DO_NOT_ACT, tuple(sorted(rule.id for rule in task_boundaries)))
            return self.Assessment(PolicyOutcome.ASK_ME, tuple(sorted(rule.id for rule in strong_boundaries)))
        if not effective:
            return self.Assessment(PolicyOutcome.AUTO)
        specificity = max(_SPECIFICITY[rule.scope] for rule in effective)
        scoped = [rule for rule in effective if _SPECIFICITY[rule.scope] == specificity]
        strength = max(_STRENGTH[rule.strength] for rule in scoped)
        strongest = [rule for rule in scoped if _STRENGTH[rule.strength] == strength]
        outcomes = {_TYPE_OUTCOME.get(rule.type, PolicyOutcome.ASK_ME) for rule in strongest}
        outcome = outcomes.pop() if len(outcomes) == 1 else PolicyOutcome.ASK_ME
        blocking = tuple(sorted(rule.id for rule in strongest)) if outcome is not PolicyOutcome.AUTO else ()
        return self.Assessment(outcome, blocking)

    @staticmethod
    def _live(session: Session, rule: ContactRule, at: datetime) -> bool:
        if rule.expires_at is not None and _aware(rule.expires_at) <= at:
            return False
        if rule.scope is ContactRuleScope.TASK_DEFINITION:
            definition = session.get(TaskDefinition, rule.task_definition_id)
            return definition is not None and definition.archived_at is None
        if rule.scope is ContactRuleScope.TASK_INSTANCE:
            task = session.get(TaskInstance, rule.task_instance_id)
            return task is not None and task.archived_at is None
        return True

    @staticmethod
    def _matches(
        rule: ContactRule,
        *,
        task: TaskInstance | None,
        topic_key: str | None,
        task_definition_id: int | None,
    ) -> bool:
        if rule.scope is ContactRuleScope.GLOBAL:
            return True
        if rule.scope is ContactRuleScope.TOPIC:
            return rule.topic_key == topic_key
        if rule.scope is ContactRuleScope.TASK_DEFINITION:
            return task_definition_id == rule.task_definition_id
        return task is not None and task.id == rule.task_instance_id


class DisclosurePolicy:
    def allows(
        self,
        session: Session,
        *,
        source_person_id: int,
        source_conversation_id: int,
        destination_conversation_id: int,
        task_instance_id: int,
        scope: DisclosureScope,
        at: datetime | None = None,
    ) -> bool:
        if source_conversation_id == destination_conversation_id:
            return True
        timestamp = _aware(at or utc_now())
        grants = list(
            session.scalars(
                select(DisclosureGrant)
                .join(
                    DisclosureGrantScope,
                    DisclosureGrantScope.disclosure_grant_id == DisclosureGrant.id,
                )
                .where(
                    DisclosureGrant.source_person_id == source_person_id,
                    DisclosureGrant.source_conversation_id == source_conversation_id,
                    DisclosureGrant.destination_conversation_id == destination_conversation_id,
                    DisclosureGrant.task_instance_id == task_instance_id,
                    DisclosureGrant.status == DisclosureGrantStatus.ACTIVE,
                    DisclosureGrantScope.scope == scope,
                )
            )
        )
        return any(_aware(grant.expires_at) > timestamp for grant in grants)


@dataclass(frozen=True, slots=True)
class ContextFact:
    source_person_id: int
    source_conversation_id: int
    scope: DisclosureScope
    value: object


class ContextBuilder:
    def __init__(self, disclosure: DisclosurePolicy):
        self.disclosure = disclosure

    def build(
        self,
        session: Session,
        *,
        facts: list[ContextFact],
        destination_conversation_id: int,
        task_instance_id: int,
        at: datetime | None = None,
    ) -> list[ContextFact]:
        return [
            fact
            for fact in facts
            if self.disclosure.allows(
                session,
                source_person_id=fact.source_person_id,
                source_conversation_id=fact.source_conversation_id,
                destination_conversation_id=destination_conversation_id,
                task_instance_id=task_instance_id,
                scope=fact.scope,
                at=at,
            )
        ]


class ContextFactProvider(Protocol):
    def facts_for(self, session: Session, message: OutboxMessage) -> list[ContextFact]: ...


class EmptyContextFactProvider:
    def facts_for(self, session: Session, message: OutboxMessage) -> list[ContextFact]:
        return []


class DatabaseContextProvider:
    """Build fresh task facts and filter them for the exact Outbox destination."""

    def __init__(
        self,
        *,
        disclosure: DisclosurePolicy | None = None,
    ):
        self.disclosure = disclosure or DisclosurePolicy()
        self.builder = ContextBuilder(self.disclosure)

    def candidate_facts(self, session: Session, message: OutboxMessage) -> list[ContextFact]:
        if message.task_instance_id is None:
            return []
        participants = list(
            session.scalars(
                select(TaskParticipant).where(
                    TaskParticipant.task_instance_id == message.task_instance_id,
                    TaskParticipant.availability_status != "UNKNOWN",
                    TaskParticipant.availability_source_revision_id.is_not(None),
                )
            )
        )
        facts: list[ContextFact] = []
        for participant in participants:
            revision = session.get(MessageRevision, participant.availability_source_revision_id)
            source_message = session.get(Message, revision.message_id) if revision is not None else None
            person = session.get(Person, participant.person_id)
            if source_message is None or person is None:
                continue
            facts.append(
                ContextFact(
                    source_person_id=participant.person_id,
                    source_conversation_id=source_message.conversation_id,
                    scope=DisclosureScope.AVAILABILITY,
                    value=f"{person.display_name} availability is {participant.availability_status.value.lower()}",
                )
            )
        return facts

    def facts_for(self, session: Session, message: OutboxMessage) -> list[ContextFact]:
        if message.transport is Transport.TELEGRAM or message.task_instance_id is None:
            return []
        destination = session.get(BeeperOutboxDestination, message.id)
        if destination is None:
            return []
        return self.builder.build(
            session,
            facts=self.candidate_facts(session, message),
            destination_conversation_id=destination.conversation_id,
            task_instance_id=message.task_instance_id,
        )

    @staticmethod
    def task_claims(session: Session, message: OutboxMessage) -> tuple[str, ...]:
        task = session.get(TaskInstance, message.task_instance_id) if message.task_instance_id else None
        if task is None:
            return ()
        claims = [
            f"topic: {task.topic_key}",
            f"scheduled_at: {task.scheduled_at.isoformat()}",
            f"duration_minutes: {task.duration_minutes}",
        ]
        if task.location:
            claims.append(f"location: {task.location}")
        return tuple(claims)


class PolicyRevalidator(PreSendRevalidator):
    def __init__(
        self,
        *,
        rules: ContactRuleResolver | None = None,
        disclosure: DisclosurePolicy | None = None,
        facts: ContextFactProvider | None = None,
        owner_chat_id: int | None = None,
    ):
        self.rules = rules or ContactRuleResolver()
        self.disclosure = disclosure or DisclosurePolicy()
        self.facts = facts or EmptyContextFactProvider()
        self.owner_chat_id = owner_chat_id

    def check(self, session: Session, message: OutboxMessage) -> PreSendDecision:
        task = session.get(TaskInstance, message.task_instance_id) if message.task_instance_id else None
        if task is not None and task.status is not TaskStatus.ACTIVE and (
            message.transport is Transport.BEEPER
            or message.parent_terminal_policy.value == "TERMINATE"
        ):
            return PreSendDecision.STALE
        if message.transport is Transport.TELEGRAM:
            if is_nobody_available_notification(message) and (
                authorize_nobody_available_notification(
                    session,
                    message,
                    owner_chat_id=self.owner_chat_id,
                )
                is None
            ):
                return PreSendDecision.STALE
            decision = decision_for_prompt(session, message)
            if decision is not None and authorize_decision_prompt(
                session,
                message,
                owner_chat_id=self.owner_chat_id,
            ) is None:
                return PreSendDecision.STALE
            return PreSendDecision.READY
        destination = session.get(BeeperOutboxDestination, message.id)
        if destination is None or task is None:
            return PreSendDecision.STALE
        participant_ids = list(
            session.scalars(
                select(OutboxMessageParticipant.task_participant_id).where(
                    OutboxMessageParticipant.outbox_message_id == message.id
                )
            )
        )
        participants = [session.get(TaskParticipant, value) for value in participant_ids]
        if any(participant is None for participant in participants):
            return PreSendDecision.STALE
        boundary_hold = False
        exception_requests: list[int] = []
        for participant in participants:
            assert participant is not None
            if participant.task_instance_id != task.id or participant.conversation_id != destination.conversation_id:
                return PreSendDecision.STALE
            if not self._current_identity_ids(
                session,
                person_id=participant.person_id,
                conversation_id=destination.conversation_id,
            ):
                return PreSendDecision.STALE
            boundary_hold = boundary_hold or pending_boundary_hold(
                session,
                person_id=participant.person_id,
                task_id=task.id,
            )
            assessment = self.rules.assess(
                session,
                person_id=participant.person_id,
                task=task,
                topic_key=task.topic_key,
            )
            if assessment.outcome is PolicyOutcome.ASK_ME and assessment.blocking_rule_ids:
                respected = session.scalar(select(DecisionRequest.id).where(
                    DecisionRequest.contact_rule_id == assessment.blocking_rule_ids[0],
                    DecisionRequest.task_instance_id == task.id,
                    DecisionRequest.type == "CONTACT_RULE_EXCEPTION",
                    DecisionRequest.status == DecisionStatus.CLOSED,
                    DecisionRequest.resolution_json == {"action": "respect_boundary"},
                ).limit(1))
                if respected is not None:
                    return PreSendDecision.POLICY_BLOCKED
                exception_requests.append(assessment.blocking_rule_ids[0])
            if assessment.outcome is not PolicyOutcome.AUTO:
                if assessment.outcome is not PolicyOutcome.ASK_ME:
                    return PreSendDecision.POLICY_BLOCKED
        for fact in self.facts.facts_for(session, message):
            if not self.disclosure.allows(
                session,
                source_person_id=fact.source_person_id,
                source_conversation_id=fact.source_conversation_id,
                destination_conversation_id=destination.conversation_id,
                task_instance_id=task.id,
                scope=fact.scope,
            ):
                return PreSendDecision.POLICY_BLOCKED
        if boundary_hold:
            return PreSendDecision.AWAITING_OWNER
        if exception_requests:
            if self.owner_chat_id is None:
                return PreSendDecision.POLICY_BLOCKED
            self._ensure_contact_rule_decision(
                session,
                min(exception_requests),
                task.id,
            )
            return PreSendDecision.AWAITING_OWNER
        participant_count = mass_contact_participant_count(session, task.id)
        if participant_count > MASS_CONTACT_THRESHOLD:
            if mass_contact_is_approved(session, task.id):
                return PreSendDecision.READY
            if self.owner_chat_id is None:
                return PreSendDecision.POLICY_BLOCKED
            if ensure_mass_contact_decision(
                session,
                task_id=task.id,
                owner_chat_id=self.owner_chat_id,
            ) is None:
                return PreSendDecision.POLICY_BLOCKED
            return PreSendDecision.AWAITING_OWNER
        return PreSendDecision.READY

    def _ensure_contact_rule_decision(self, session: Session, rule_id: int, task_id: int) -> None:
        subject = contact_rule_exception_subject(
            session,
            rule_id=rule_id,
            task_id=task_id,
        )
        if subject is None:
            return
        rule = subject.rule
        person = subject.person
        existing = session.scalar(select(DecisionRequest).where(
            DecisionRequest.contact_rule_id == rule_id,
            DecisionRequest.task_instance_id == task_id,
            DecisionRequest.type == "CONTACT_RULE_EXCEPTION",
            DecisionRequest.status == DecisionStatus.PENDING,
        ))
        if existing is None:
            existing = DecisionService(session).create(
                decision_type="CONTACT_RULE_EXCEPTION", subject_kind="contact_rule",
                subject_id=rule_id, context={}, task_instance_id=task_id,
                parent_terminal_policy=ParentTerminalPolicy.TERMINATE,
            )
        if self.owner_chat_id is not None:
            prompt = OutboxService(session).create_owner(
                telegram_chat_id=self.owner_chat_id,
                final_text=contact_rule_exception_prompt(
                    rule_id,
                    task_id,
                    person.display_name,
                    self._rule_scope_description(rule),
                    rule.value,
                ),
                message_kind=MessageKind.NOTIFICATION,
                idempotency_key=f"decision:{existing.id}:owner-prompt",
                task_instance_id=task_id,
                parent_terminal_policy=ParentTerminalPolicy.TERMINATE,
            )
            if session.get(DecisionRequestPrompt, {"decision_request_id": existing.id, "outbox_message_id": prompt.id}) is None:
                session.add(DecisionRequestPrompt(decision_request_id=existing.id, outbox_message_id=prompt.id))
                session.flush()

    @staticmethod
    def _rule_scope_description(rule: ContactRule) -> str:
        if rule.scope is ContactRuleScope.GLOBAL:
            return "global"
        if rule.scope is ContactRuleScope.TOPIC:
            return f"topic {rule.topic_key}"
        if rule.scope is ContactRuleScope.TASK_DEFINITION:
            return f"task definition {rule.task_definition_id}"
        return f"task {rule.task_instance_id}"

    def context_token(self, session: Session, message: OutboxMessage) -> tuple[object, ...]:
        """Stable pre-send snapshot used to detect policy/context changes during validation."""
        task = session.get(TaskInstance, message.task_instance_id) if message.task_instance_id else None
        task_token: tuple[object, ...] = ()
        if task is not None:
            task_token = (
                task.id,
                task.status.value,
                task.topic_key,
                task.scheduled_at.isoformat(),
                task.duration_minutes,
                task.location,
            )
        if message.transport is Transport.TELEGRAM:
            nobody_available_token: tuple[object, ...] = ()
            if is_nobody_available_notification(message):
                nobody_available_token = (
                    "NOBODY_AVAILABLE",
                    authorize_nobody_available_notification(
                        session,
                        message,
                        owner_chat_id=self.owner_chat_id,
                    ),
                )
            return (
                "TELEGRAM",
            ) + task_token + nobody_available_token + self._decision_prompt_token(session, message)
        destination = session.get(BeeperOutboxDestination, message.id)
        participant_ids = tuple(
            session.scalars(
                select(OutboxMessageParticipant.task_participant_id)
                .where(OutboxMessageParticipant.outbox_message_id == message.id)
                .order_by(OutboxMessageParticipant.task_participant_id)
            )
        )
        rule_outcomes: list[tuple[object, ...]] = []
        membership_tokens: list[tuple[int, tuple[int, ...]]] = []
        for participant_id in participant_ids:
            participant = session.get(TaskParticipant, participant_id)
            if participant is None:
                rule_outcomes.append((participant_id, "MISSING"))
                membership_tokens.append((participant_id, ()))
                continue
            membership_tokens.append(
                (
                    participant_id,
                    self._current_identity_ids(
                        session,
                        person_id=participant.person_id,
                        conversation_id=(
                            destination.conversation_id
                            if destination is not None
                            else participant.conversation_id
                        ),
                    ),
                )
            )
            assessment = self.rules.assess(
                session,
                person_id=participant.person_id,
                task=task,
                topic_key=task.topic_key if task is not None else None,
            )
            hold = pending_boundary_hold(session, person_id=participant.person_id, task_id=task.id) if task else False
            decision_states = tuple(session.execute(select(
                DecisionRequest.id, DecisionRequest.status
            ).where(
                DecisionRequest.contact_rule_id.in_(assessment.blocking_rule_ids),
                DecisionRequest.task_instance_id == (task.id if task else None),
                DecisionRequest.type == "CONTACT_RULE_EXCEPTION",
            ).order_by(DecisionRequest.id))) if assessment.blocking_rule_ids else ()
            rule_outcomes.append((participant_id, assessment.outcome.value,
                                  assessment.blocking_rule_ids, hold,
                                  tuple((row[0], row[1].value) for row in decision_states)))
        facts = tuple(
            sorted(
                (
                    fact.source_person_id,
                    fact.source_conversation_id,
                    fact.scope.value,
                    repr(fact.value),
                )
                for fact in self.facts.facts_for(session, message)
            )
        )
        return (
            "BEEPER",
            destination.conversation_id if destination is not None else None,
            task_token,
            participant_ids,
            tuple(membership_tokens),
            tuple(rule_outcomes),
            facts,
            (
                "MASS_CONTACT",
                mass_contact_participant_count(session, task.id),
                MASS_CONTACT_THRESHOLD,
                mass_contact_is_approved(session, task.id),
            )
            if task is not None
            else (),
        )

    def _decision_prompt_token(
        self, session: Session, message: OutboxMessage
    ) -> tuple[object, ...]:
        decision = decision_for_prompt(session, message)
        if decision is None:
            return ("NO_DECISION_PROMPT",)
        authorization = authorize_decision_prompt(
            session,
            message,
            owner_chat_id=self.owner_chat_id,
        )
        if authorization is None:
            return (
                "INVALID_DECISION_PROMPT",
                decision.id,
                decision.type,
            )
        return (
            "DECISION_PROMPT",
            decision.id,
            decision.type,
            *authorization.applicability_token,
        )

    @staticmethod
    def _current_identity_ids(
        session: Session, *, person_id: int, conversation_id: int
    ) -> tuple[int, ...]:
        return tuple(
            session.scalars(
                select(Identity.id)
                .join(
                    ConversationParticipant,
                    ConversationParticipant.identity_id == Identity.id,
                )
                .where(
                    Identity.person_id == person_id,
                    ConversationParticipant.conversation_id == conversation_id,
                    ConversationParticipant.is_current.is_(True),
                )
                .order_by(Identity.id)
            )
        )
