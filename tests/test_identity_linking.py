from __future__ import annotations

import json
from datetime import timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from ten_texter.beeper import BeeperSyncService
from ten_texter import cli
from ten_texter.control import ProductionOwnerCommandHandler
from ten_texter.domain import ContactRuleService, DomainError, TaskService
from ten_texter.enums import (
    ContactRuleScope,
    ContactRuleSource,
    MessageKind,
    RuleStrength,
)
from ten_texter.identity import IdentityLinkingService
from ten_texter.models import (
    ContactRule,
    ConversationParticipant,
    Identity,
    Message,
    Person,
    TaskParticipant,
)
from ten_texter.outbox import OutboxService, PreSendDecision
from ten_texter.policy import PolicyRevalidator
from tests.test_schema import NOW


def chat(chat_id: str, user_id: str, *, name: str = "Same Name") -> dict[str, object]:
    return {
        "id": chat_id,
        "network": "Discord",
        "type": "single",
        "title": name,
        "participants": {
            "hasMore": False,
            "items": [
                {"id": "@self:beeper", "isSelf": True},
                {"id": user_id, "fullName": name, "username": name.casefold().replace(" ", "")},
            ],
        },
    }


def seed_two_identities(session: Session) -> dict[str, object]:
    sync = BeeperSyncService(session)
    first_conversation = sync.sync_chat(chat("!first", "@first:beeper"))
    second_conversation = sync.sync_chat(chat("!second", "@second:beeper"))
    first_identity = session.scalar(select(Identity).where(Identity.beeper_user_id == "@first:beeper"))
    second_identity = session.scalar(select(Identity).where(Identity.beeper_user_id == "@second:beeper"))
    return {
        "sync": sync,
        "first_conversation": first_conversation,
        "second_conversation": second_conversation,
        "first_identity": first_identity,
        "second_identity": second_identity,
        "first_person": session.get(Person, first_identity.person_id),
        "second_person": session.get(Person, second_identity.person_id),
    }


def test_explicit_identity_link_preserves_provider_and_history_and_unifies_routing(db_session: Session) -> None:
    seeded = seed_two_identities(db_session)
    first_identity = seeded["first_identity"]
    second_identity = seeded["second_identity"]
    assert first_identity.person_id != second_identity.person_id
    assert db_session.scalar(select(Person).where(Person.display_name == "Same Name")) is not None
    assert len(list(db_session.scalars(select(Person).where(Person.display_name == "Same Name")))) == 2

    result = seeded["sync"].ingest_message(
        {
            "id": "historical-message",
            "chatID": "!second",
            "senderID": "@second:beeper",
            "timestamp": "2026-08-26T15:00:00Z",
            "type": "TEXT",
            "text": "hello",
        }
    )
    provider_id = second_identity.beeper_user_id
    membership_key = {
        "conversation_id": seeded["second_conversation"].id,
        "identity_id": second_identity.id,
    }
    records = IdentityLinkingService(db_session).inspect()
    assert {record.identity_id for record in records} == {first_identity.id, second_identity.id}
    assert records[0].person_id != records[1].person_id

    linked = IdentityLinkingService(db_session).link(
        identity_id=second_identity.id,
        target_person_id=first_identity.person_id,
    )
    repeated = IdentityLinkingService(db_session).link(
        identity_id=second_identity.id,
        target_person_id=first_identity.person_id,
    )
    assert linked.changed
    assert not repeated.changed
    seeded["sync"].sync_chat(chat("!second", "@second:beeper"))
    assert second_identity.beeper_user_id == provider_id
    assert second_identity.person_id == first_identity.person_id
    assert db_session.get(Message, result.message_id).sender_identity_id == second_identity.id
    assert db_session.get(ConversationParticipant, membership_key) is not None

    candidates = ProductionOwnerCommandHandler._route_candidates(db_session)
    linked_candidates = [row for row in candidates if row["person_id"] == first_identity.person_id]
    assert {row["conversation_id"] for row in linked_candidates} == {
        seeded["first_conversation"].id,
        seeded["second_conversation"].id,
    }


def test_linked_identity_uses_target_person_rules_and_prevents_duplicate_task_people(db_session: Session) -> None:
    seeded = seed_two_identities(db_session)
    first_identity = seeded["first_identity"]
    second_identity = seeded["second_identity"]
    IdentityLinkingService(db_session).link(
        identity_id=second_identity.id,
        target_person_id=first_identity.person_id,
    )
    ContactRuleService(db_session).create(
        person_id=first_identity.person_id,
        scope=ContactRuleScope.GLOBAL,
        type="DO_NOT_CONTACT",
        value="owner boundary",
        source=ContactRuleSource.USER_CONFIGURED,
        strength=RuleStrength.STRONG,
    )
    task = TaskService(db_session).create(
        scheduled_at=NOW + timedelta(days=1),
        duration_minutes=30,
        topic_key="tennis",
        participants=[(first_identity.person_id, seeded["second_conversation"].id)],
    )
    participant = db_session.scalar(
        select(TaskParticipant).where(TaskParticipant.task_instance_id == task.id)
    )
    message = OutboxService(db_session).create_beeper(
        task_instance_id=task.id,
        conversation_id=seeded["second_conversation"].id,
        participant_ids=[participant.id],
        final_text="hello",
        message_kind=MessageKind.INITIAL,
        idempotency_key="linked-rule",
    )
    assert PolicyRevalidator().check(db_session, message) is PreSendDecision.POLICY_BLOCKED

    with pytest.raises(DomainError, match="duplicate canonical person"):
        TaskService(db_session).create(
            scheduled_at=NOW + timedelta(days=2),
            duration_minutes=30,
            topic_key="tennis",
            participants=[
                (first_identity.person_id, seeded["first_conversation"].id),
                (first_identity.person_id, seeded["second_conversation"].id),
            ],
        )


def test_link_with_person_scoped_state_fails_closed(db_session: Session) -> None:
    seeded = seed_two_identities(db_session)
    source_identity = seeded["second_identity"]
    target_person = seeded["first_person"]
    db_session.add(
        ContactRule(
            person_id=source_identity.person_id,
            scope=ContactRuleScope.GLOBAL,
            type="DO_NOT_CONTACT",
            value="existing state",
            source=ContactRuleSource.USER_CONFIGURED,
            strength=RuleStrength.STRONG,
        )
    )
    db_session.flush()
    with pytest.raises(DomainError, match="Person-scoped state"):
        IdentityLinkingService(db_session).link(
            identity_id=source_identity.id,
            target_person_id=target_person.id,
        )
    assert source_identity.person_id == seeded["second_person"].id


def test_owner_cli_inspects_and_links_exact_ids(
    db_session: Session, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seeded = seed_two_identities(db_session)
    source_identity = seeded["second_identity"]
    target_person = seeded["first_person"]
    db_session.commit()
    monkeypatch.setenv("TEN_TEXTER_DATABASE_URL", str(db_session.bind.url))

    assert cli.main(["identities"]) == 0
    inspected = json.loads(capsys.readouterr().out)
    selected = next(row for row in inspected if row["identity_id"] == source_identity.id)
    assert selected["beeper_user_id"] == "@second:beeper"
    assert selected["memberships"][0]["beeper_conversation_id"] == "!second"

    assert cli.main(
        [
            "link-identity",
            "--identity-id",
            str(source_identity.id),
            "--person-id",
            str(target_person.id),
        ]
    ) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["changed"] is True
    db_session.expire_all()
    assert db_session.get(Identity, source_identity.id).person_id == target_person.id
