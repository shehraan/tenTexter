from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from ten_texter.domain import normalize_topic_key, utc_now
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
    ContactRule,
    DisclosureGrant,
    DisclosureGrantScope,
    OutboxMessage,
    OutboxMessageParticipant,
    TaskDefinition,
    TaskInstance,
    TaskParticipant,
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
    def resolve(
        self,
        session: Session,
        *,
        person_id: int,
        task: TaskInstance | None,
        topic_key: str | None,
        at: datetime | None = None,
    ) -> PolicyOutcome:
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
            return PolicyOutcome.ASK_ME
        normalized_topic = normalize_topic_key(topic_key) if topic_key is not None else None
        applicable = [
            rule
            for rule in rules
            if self._live(session, rule, timestamp)
            and self._matches(rule, task=task, topic_key=normalized_topic)
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
            return PolicyOutcome.ASK_ME
        if not effective:
            return PolicyOutcome.AUTO
        specificity = max(_SPECIFICITY[rule.scope] for rule in effective)
        scoped = [rule for rule in effective if _SPECIFICITY[rule.scope] == specificity]
        strength = max(_STRENGTH[rule.strength] for rule in scoped)
        strongest = [rule for rule in scoped if _STRENGTH[rule.strength] == strength]
        outcomes = {_TYPE_OUTCOME.get(rule.type, PolicyOutcome.ASK_ME) for rule in strongest}
        return outcomes.pop() if len(outcomes) == 1 else PolicyOutcome.ASK_ME

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
    def _matches(rule: ContactRule, *, task: TaskInstance | None, topic_key: str | None) -> bool:
        if rule.scope is ContactRuleScope.GLOBAL:
            return True
        if rule.scope is ContactRuleScope.TOPIC:
            return rule.topic_key == topic_key
        if task is None:
            return False
        if rule.scope is ContactRuleScope.TASK_DEFINITION:
            return task.task_definition_id == rule.task_definition_id
        return task.id == rule.task_instance_id


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


class PolicyRevalidator(PreSendRevalidator):
    def __init__(
        self,
        *,
        rules: ContactRuleResolver | None = None,
        disclosure: DisclosurePolicy | None = None,
        facts: ContextFactProvider | None = None,
    ):
        self.rules = rules or ContactRuleResolver()
        self.disclosure = disclosure or DisclosurePolicy()
        self.facts = facts or EmptyContextFactProvider()

    def check(self, session: Session, message: OutboxMessage) -> PreSendDecision:
        task = session.get(TaskInstance, message.task_instance_id) if message.task_instance_id else None
        if task is not None and task.status is not TaskStatus.ACTIVE and (
            message.transport is Transport.BEEPER
            or message.parent_terminal_policy.value == "TERMINATE"
        ):
            return PreSendDecision.STALE
        if message.transport is Transport.TELEGRAM:
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
        for participant in participants:
            assert participant is not None
            if participant.task_instance_id != task.id or participant.conversation_id != destination.conversation_id:
                return PreSendDecision.STALE
            if self.rules.resolve(
                session,
                person_id=participant.person_id,
                task=task,
                topic_key=task.topic_key,
            ) is not PolicyOutcome.AUTO:
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
        return PreSendDecision.READY
