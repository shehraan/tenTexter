from __future__ import annotations

from datetime import timedelta

from sqlalchemy.orm import Session

from ten_texter.domain import ContactRuleService, DisclosureGrantService, TaskService
from ten_texter.enums import (
    ContactRuleScope,
    ContactRuleSource,
    DisclosureScope,
    MessageKind,
    PolicyOutcome,
    RuleStrength,
)
from ten_texter.models import ContactRule, DisclosureGrant
from ten_texter.outbox import OutboxService, PreSendDecision
from ten_texter.policy import (
    ContactRuleResolver,
    ContextBuilder,
    ContextFact,
    DisclosurePolicy,
    PolicyRevalidator,
)
from tests.test_domain import _second_conversation
from tests.test_schema import NOW, seed_core


def add_rule(session: Session, core: dict[str, object], **changes: object) -> ContactRule:
    values: dict[str, object] = {
        "person_id": core["person"].id,
        "scope": ContactRuleScope.GLOBAL,
        "type": "DO_NOT_CONTACT",
        "value": "boundary",
        "source": ContactRuleSource.USER_CONFIGURED,
        "strength": RuleStrength.MEDIUM,
    }
    values.update(changes)
    return ContactRuleService(session).create(**values)


def test_strong_participant_boundary_asks_owner(db_session: Session) -> None:
    core = seed_core(db_session)
    add_rule(
        db_session,
        core,
        source=ContactRuleSource.PARTICIPANT_REQUESTED,
        strength=RuleStrength.STRONG,
    )
    outcome = ContactRuleResolver().resolve(
        db_session,
        person_id=core["person"].id,
        task=core["task"],
        topic_key="tennis",
        at=NOW,
    )
    assert outcome is PolicyOutcome.ASK_ME


def test_explicit_narrow_exception_overrides_strong_boundary(db_session: Session) -> None:
    core = seed_core(db_session)
    original = add_rule(
        db_session,
        core,
        source=ContactRuleSource.PARTICIPANT_REQUESTED,
        strength=RuleStrength.STRONG,
        scope=ContactRuleScope.TOPIC,
        topic_key="  Tennis  ",
    )
    add_rule(
        db_session,
        core,
        scope=ContactRuleScope.TASK_INSTANCE,
        task_instance_id=core["task"].id,
        type="ALLOW",
        value="owner-approved once",
        source=ContactRuleSource.USER_CONFIGURED,
        strength=RuleStrength.STRONG,
        overrides_contact_rule_id=original.id,
    )
    assert ContactRuleResolver().resolve(
        db_session,
        person_id=core["person"].id,
        task=core["task"],
        topic_key="Tennis",
        at=NOW,
    ) is PolicyOutcome.AUTO


def test_specific_scope_wins_and_tie_conflict_fails_closed(db_session: Session) -> None:
    core = seed_core(db_session)
    add_rule(db_session, core)
    add_rule(
        db_session,
        core,
        scope=ContactRuleScope.TOPIC,
        topic_key="tennis",
        type="ALLOW",
        strength=RuleStrength.WEAK,
    )
    resolver = ContactRuleResolver()
    assert resolver.resolve(
        db_session,
        person_id=core["person"].id,
        task=core["task"],
        topic_key="tennis",
        at=NOW,
    ) is PolicyOutcome.AUTO
    add_rule(
        db_session,
        core,
        scope=ContactRuleScope.TOPIC,
        topic_key="tennis",
        type="ASK_ME",
        strength=RuleStrength.WEAK,
    )
    assert resolver.resolve(
        db_session,
        person_id=core["person"].id,
        task=core["task"],
        topic_key="tennis",
        at=NOW,
    ) is PolicyOutcome.ASK_ME


def test_revoked_and_expired_rules_stop_applying_without_deletion(db_session: Session) -> None:
    core = seed_core(db_session)
    expired = add_rule(db_session, core, expires_at=NOW - timedelta(seconds=1))
    revoked = add_rule(db_session, core, type="ASK_ME")
    ContactRuleService(db_session).revoke(revoked.id)
    assert ContactRuleResolver().resolve(
        db_session,
        person_id=core["person"].id,
        task=core["task"],
        topic_key="tennis",
        at=NOW,
    ) is PolicyOutcome.AUTO
    assert db_session.get(ContactRule, expired.id) is not None
    assert db_session.get(ContactRule, revoked.id).revoked_at is not None


def test_uncertain_topic_with_live_topic_rule_asks_owner(db_session: Session) -> None:
    core = seed_core(db_session)
    add_rule(
        db_session,
        core,
        scope=ContactRuleScope.TOPIC,
        topic_key="tennis",
    )
    assert ContactRuleResolver().resolve(
        db_session,
        person_id=core["person"].id,
        task=core["task"],
        topic_key=None,
        at=NOW,
    ) is PolicyOutcome.ASK_ME


def test_destination_aware_context_requires_exact_grant(db_session: Session) -> None:
    core = seed_core(db_session)
    destination = _second_conversation(db_session, core)
    fact = ContextFact(
        source_person_id=core["person"].id,
        source_conversation_id=core["conversation"].id,
        scope=DisclosureScope.AVAILABILITY,
        value="available",
    )
    builder = ContextBuilder(DisclosurePolicy())
    assert builder.build(
        db_session,
        facts=[fact],
        destination_conversation_id=destination,
        task_instance_id=core["task"].id,
        at=NOW,
    ) == []
    DisclosureGrantService(db_session).create(
        scopes=[DisclosureScope.AVAILABILITY],
        source_person_id=core["person"].id,
        source_conversation_id=core["conversation"].id,
        destination_conversation_id=destination,
        task_instance_id=core["task"].id,
        expires_at=NOW + timedelta(hours=1),
    )
    assert builder.build(
        db_session,
        facts=[fact],
        destination_conversation_id=destination,
        task_instance_id=core["task"].id,
        at=NOW,
    ) == [fact]


def test_reschedule_recomputes_task_relative_grant_expiry(db_session: Session) -> None:
    core = seed_core(db_session)
    destination = _second_conversation(db_session, core)
    grant = DisclosureGrantService(db_session).create(
        scopes=[DisclosureScope.SCHEDULING],
        source_person_id=core["person"].id,
        source_conversation_id=core["conversation"].id,
        destination_conversation_id=destination,
        task_instance_id=core["task"].id,
        expires_at=NOW + timedelta(hours=1),
    )
    new_time = NOW + timedelta(days=2)
    TaskService(db_session).reschedule(core["task"].id, new_time)
    db_session.expire(grant)
    expected = new_time + timedelta(minutes=core["task"].coordination_close_offset_minutes)
    assert grant.expires_at.replace(tzinfo=expected.tzinfo) == expected


class Facts:
    def __init__(self, fact: ContextFact):
        self.fact = fact

    def facts_for(self, _session: Session, _message: object) -> list[ContextFact]:
        return [self.fact]


def test_outbox_policy_revalidation_blocks_contact_and_disclosure(db_session: Session) -> None:
    core = seed_core(db_session)
    message = OutboxService(db_session).create_beeper(
        task_instance_id=core["task"].id,
        conversation_id=core["conversation"].id,
        participant_ids=[core["participant"].id],
        final_text="message",
        message_kind=MessageKind.INITIAL,
        idempotency_key="policy-check",
    )
    add_rule(db_session, core)
    assert PolicyRevalidator().check(db_session, message) is PreSendDecision.POLICY_BLOCKED
    ContactRuleService(db_session).revoke(
        db_session.query(ContactRule).filter(ContactRule.person_id == core["person"].id).one().id
    )
    fact = ContextFact(
        source_person_id=core["person"].id,
        source_conversation_id=_second_conversation(db_session, core),
        scope=DisclosureScope.LOCATION,
        value="private place",
    )
    assert PolicyRevalidator(facts=Facts(fact)).check(db_session, message) is PreSendDecision.POLICY_BLOCKED
