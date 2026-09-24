from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from ten_texter.control import PreparedOwnerCommand
from ten_texter.enums import (
    ConversationKind,
    OutboxStatus,
    ProcessingStatus,
    TaskStatus,
    Transport,
)
from ten_texter.live_tennis import (
    LiveTennisTarget,
    LiveTennisTestError,
    _AllowlistedOwnerCommandHandler,
    _ingest_new_target_messages,
    _reconcile_initial_send,
    poll_live_tennis_test,
    resolve_allowlisted_target,
    run_live_tennis_test,
)
from ten_texter.models import (
    Conversation,
    ConversationParticipant,
    Identity,
    MessageRevision,
    OutboxMessage,
    Person,
    TaskInstance,
)
from ten_texter.workflows import CoordinationWorkflow, ParticipantSendPlan
from tests.test_schema import NOW


def add_allowlisted_target(session: Session, *, second_direct: bool = False) -> dict[str, object]:
    person = Person(display_name="Shehraan Canada", metadata_json={})
    identity = Identity(
        person_id=1,
        beeper_user_id="@whatsapp_lid-test:beeper.local",
        network="WhatsApp",
        display_name="Shehraan Canada",
        metadata_json={},
    )
    session.add(person)
    session.flush()
    identity.person_id = person.id
    session.add(identity)
    session.flush()
    direct = Conversation(
        beeper_conversation_id="!test-direct:beeper.local",
        network="WhatsApp",
        kind=ConversationKind.DIRECT,
        title="Shehraan Canada",
        counterparty_person_id=person.id,
        metadata_json={},
    )
    session.add(direct)
    session.flush()
    session.add(
        ConversationParticipant(conversation_id=direct.id, identity_id=identity.id)
    )
    if second_direct:
        duplicate = Conversation(
            beeper_conversation_id="!test-direct-duplicate:beeper.local",
            network="WhatsApp",
            kind=ConversationKind.DIRECT,
            title="Shehraan Canada",
            counterparty_person_id=person.id,
            metadata_json={},
        )
        session.add(duplicate)
        session.flush()
        session.add(
            ConversationParticipant(conversation_id=duplicate.id, identity_id=identity.id)
        )
    session.flush()
    return {
        "person": person,
        "identity": identity,
        "conversation": direct,
    }


def test_allowlisted_target_resolves_only_the_exact_direct_whatsapp_chat(
    db_session: Session,
) -> None:
    seeded = add_allowlisted_target(db_session)

    target = resolve_allowlisted_target(db_session)

    assert target.person_id == seeded["person"].id  # type: ignore[union-attr]
    assert target.identity_id == seeded["identity"].id  # type: ignore[union-attr]
    assert target.conversation_id == seeded["conversation"].id  # type: ignore[union-attr]
    assert target.beeper_conversation_id == "!test-direct:beeper.local"


def test_allowlisted_target_refuses_duplicate_direct_chats(db_session: Session) -> None:
    add_allowlisted_target(db_session, second_direct=True)

    with pytest.raises(LiveTennisTestError, match="exactly one"):
        resolve_allowlisted_target(db_session)


def test_unsupported_target_message_is_recorded_but_not_sent_for_processing(
    db_session: Session,
) -> None:
    add_allowlisted_target(db_session)
    db_session.commit()
    target = resolve_allowlisted_target(db_session)

    provider_ids, revision_ids = _ingest_new_target_messages(
        sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False),
        target,
        [
            {
                "id": "unsupported-1",
                "chatID": target.beeper_conversation_id,
                "senderID": target.beeper_user_id,
                "isSender": False,
                "type": "IMAGE",
                "timestamp": "2026-09-17T16:00:00Z",
            }
        ],
        set(),
        owner_chat_id=99,
    )

    assert provider_ids == ["unsupported-1"]
    assert revision_ids == []


def test_command_guard_rejects_a_different_person_or_conversation() -> None:
    target = LiveTennisTarget(
        person_id=10,
        identity_id=20,
        conversation_id=30,
        beeper_user_id="target",
        beeper_conversation_id="target-chat",
    )
    plan = SimpleNamespace(
        recurrence_rule=None,
        topic_key="tennis",
        duration_minutes=60,
        scheduled_at=NOW + timedelta(days=1),
    )
    wrong_send = ParticipantSendPlan(
        person_id=999,
        conversation_id=998,
        final_text="Can you play tennis tomorrow?",
    )
    prepared = PreparedOwnerCommand(plan=plan, sends=(wrong_send,))

    class Delegate:
        def __init__(self) -> None:
            self.applied = False

        def prepare_command(self, _parsed: object, _update: object) -> PreparedOwnerCommand:
            return prepared

        def apply_command(self, _session: Session, _prepared: object, _update: object) -> None:
            self.applied = True

        def prepare_decision(self, *_args: object) -> object:
            return None

        def apply_decision(self, *_args: object) -> None:
            return None

    delegate = Delegate()
    guarded = _AllowlistedOwnerCommandHandler(delegate, target)  # type: ignore[arg-type]

    with pytest.raises(LiveTennisTestError, match="hard allowlist"):
        guarded.prepare_command(object(), object())
    assert delegate.applied is False


def test_live_runner_processes_only_the_allowlisted_beeper_outbox(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seeded = add_allowlisted_target(db_session)
    db_session.commit()
    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    target_person = seeded["person"]
    target_conversation = seeded["conversation"]

    class Parser:
        def parse(self, _text: str) -> object:
            return object()

    class Handler:
        def prepare_command(self, _parsed: object, _update: object) -> PreparedOwnerCommand:
            plan = SimpleNamespace(
                recurrence_rule=None,
                topic_key="tennis",
                duration_minutes=60,
                scheduled_at=NOW + timedelta(days=1),
            )
            return PreparedOwnerCommand(
                plan=plan,
                sends=(
                    ParticipantSendPlan(
                        person_id=target_person.id,  # type: ignore[union-attr]
                        conversation_id=target_conversation.id,  # type: ignore[union-attr]
                        final_text="Can you play tennis tomorrow at 5 PM for 60 minutes?",
                    ),
                ),
            )

        def apply_command(
            self,
            session: Session,
            prepared: PreparedOwnerCommand,
            _update: object,
        ) -> None:
            CoordinationWorkflow(session, owner_chat_id=99).start(
                scheduled_at=prepared.plan.scheduled_at,  # type: ignore[union-attr]
                duration_minutes=prepared.plan.duration_minutes,  # type: ignore[union-attr]
                location=None,
                topic_key=prepared.plan.topic_key,  # type: ignore[union-attr]
                participant_sends=list(prepared.sends),
            )

        def prepare_decision(self, *_args: object) -> object:
            return None

        def apply_decision(self, *_args: object) -> None:
            return None

    class Beeper:
        def list_messages(self, _chat_id: str, **_kwargs: object) -> list[dict[str, object]]:
            return []

    class Outbox:
        def __init__(self) -> None:
            self.processed: list[int] = []
            self.reconciled: list[int] = []

        def process(self, outbox_id: int) -> OutboxStatus:
            self.processed.append(outbox_id)
            with factory.begin() as session:
                message = session.get(OutboxMessage, outbox_id)
                assert message is not None
                message.status = OutboxStatus.SENT
            return OutboxStatus.SENT

        def reconcile(self, outbox_id: int) -> OutboxStatus:
            self.reconciled.append(outbox_id)
            with factory.begin() as session:
                message = session.get(OutboxMessage, outbox_id)
                assert message is not None
                message.status = OutboxStatus.SENT
            return OutboxStatus.SENT

    outbox = Outbox()
    runtime = SimpleNamespace(
        control=SimpleNamespace(parser=Parser(), handler=Handler()),
        beeper=Beeper(),
        outbox=outbox,
    )
    monkeypatch.setattr("ten_texter.live_tennis.build_runtime", lambda _app: runtime)
    app = SimpleNamespace(
        settings=SimpleNamespace(
            real_transports_enabled=True,
            owner_id=7,
            owner_chat_id=99,
        ),
        sessions=factory,
    )

    report = run_live_tennis_test(app, confirm_real_send=True)

    assert report.ok is True
    assert report.target_beeper_conversation_id == "!test-direct:beeper.local"
    assert outbox.processed == [report.initial_outbox_id]
    with factory() as session:
        sent = session.get(OutboxMessage, report.initial_outbox_id)
        assert sent is not None
        assert sent.transport is Transport.BEEPER
        assert sent.status is OutboxStatus.SENT
        assert session.scalar(
            select(TaskInstance.status).where(TaskInstance.id == report.task_instance_id)
        ) is TaskStatus.ACTIVE

    with factory.begin() as session:
        initial = session.get(OutboxMessage, report.initial_outbox_id)
        assert initial is not None
        initial.status = OutboxStatus.RECONCILING
    resumed = poll_live_tennis_test(
        app,
        task_instance_id=report.task_instance_id,
        confirm_real_send=True,
    )
    assert resumed.ok is True
    assert resumed.reply_provider_message_ids == ()
    assert outbox.processed == [report.initial_outbox_id]
    assert outbox.reconciled == [report.initial_outbox_id]


def test_live_poll_recovers_a_pending_allowlisted_reply(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seeded = add_allowlisted_target(db_session)
    db_session.commit()
    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    target_person = seeded["person"]
    target_conversation = seeded["conversation"]

    class Parser:
        def parse(self, _text: str) -> object:
            return object()

    class Handler:
        def prepare_command(self, _parsed: object, _update: object) -> PreparedOwnerCommand:
            plan = SimpleNamespace(
                recurrence_rule=None,
                topic_key="tennis",
                duration_minutes=60,
                scheduled_at=NOW + timedelta(days=1),
            )
            return PreparedOwnerCommand(
                plan=plan,
                sends=(
                    ParticipantSendPlan(
                        person_id=target_person.id,  # type: ignore[union-attr]
                        conversation_id=target_conversation.id,  # type: ignore[union-attr]
                        final_text="Can you play tennis tomorrow at 5 PM for 60 minutes?",
                    ),
                ),
            )

        def apply_command(
            self,
            session: Session,
            prepared: PreparedOwnerCommand,
            _update: object,
        ) -> None:
            CoordinationWorkflow(session, owner_chat_id=99).start(
                scheduled_at=prepared.plan.scheduled_at,  # type: ignore[union-attr]
                duration_minutes=prepared.plan.duration_minutes,  # type: ignore[union-attr]
                location=None,
                topic_key=prepared.plan.topic_key,  # type: ignore[union-attr]
                participant_sends=list(prepared.sends),
            )

        def prepare_decision(self, *_args: object) -> object:
            return None

        def apply_decision(self, *_args: object) -> None:
            return None

    class Beeper:
        def list_messages(self, _chat_id: str, **_kwargs: object) -> list[dict[str, object]]:
            return []

    class Outbox:
        def __init__(self) -> None:
            self.processed: list[int] = []

        def process(self, outbox_id: int) -> OutboxStatus:
            self.processed.append(outbox_id)
            with factory.begin() as session:
                message = session.get(OutboxMessage, outbox_id)
                assert message is not None
                message.status = OutboxStatus.SENT
            return OutboxStatus.SENT

    outbox = Outbox()
    runtime = SimpleNamespace(
        control=SimpleNamespace(parser=Parser(), handler=Handler()),
        beeper=Beeper(),
        outbox=outbox,
        sessions=factory,
    )

    def process_revisions() -> None:
        with factory.begin() as session:
            revisions = list(
                session.scalars(
                    select(MessageRevision).where(
                        MessageRevision.processing_status == ProcessingStatus.PENDING
                    )
                )
            )
            assert len(revisions) == 1
            revisions[0].processing_status = ProcessingStatus.PROCESSED

    runtime._process_revisions = process_revisions
    monkeypatch.setattr("ten_texter.live_tennis.build_runtime", lambda _app: runtime)
    app = SimpleNamespace(
        settings=SimpleNamespace(
            real_transports_enabled=True,
            owner_id=7,
            owner_chat_id=99,
        ),
        sessions=factory,
    )

    report = run_live_tennis_test(app, confirm_real_send=True)
    with factory() as session:
        target = resolve_allowlisted_target(session)
    provider_ids, revision_ids = _ingest_new_target_messages(
        factory,
        target,
        [
            {
                "id": "reply-1",
                "chatID": target.beeper_conversation_id,
                "senderID": target.beeper_user_id,
                "isSender": False,
                "type": "TEXT",
                "text": "Yes",
                "timestamp": "2026-09-17T16:00:00Z",
            }
        ],
        set(),
        owner_chat_id=99,
    )

    assert provider_ids == ["reply-1"]
    assert len(revision_ids) == 1
    resumed = poll_live_tennis_test(
        app,
        task_instance_id=report.task_instance_id,
        confirm_real_send=True,
    )

    assert resumed.ok is True
    assert resumed.reply_provider_message_ids == ("reply-1",)
    assert resumed.reply_revision_ids == tuple(revision_ids)
    assert outbox.processed == [report.initial_outbox_id]
    with factory() as session:
        revision = session.get(MessageRevision, revision_ids[0])
        assert revision is not None
        assert revision.processing_status is ProcessingStatus.PROCESSED


def test_live_runner_requires_explicit_real_send_confirmation() -> None:
    app = SimpleNamespace(
        settings=SimpleNamespace(real_transports_enabled=True),
        sessions=None,
    )

    with pytest.raises(LiveTennisTestError, match="confirm_real_send"):
        run_live_tennis_test(app)


def test_reconciling_send_is_reconciled_without_retrying() -> None:
    calls: list[int] = []

    class Outbox:
        def reconcile(self, outbox_id: int) -> OutboxStatus:
            calls.append(outbox_id)
            return OutboxStatus.SENT

    status = _reconcile_initial_send(
        SimpleNamespace(outbox=Outbox()),
        41,
        poll_rounds=2,
        poll_delay_seconds=0,
    )

    assert status is OutboxStatus.SENT
    assert calls == [41]


def test_live_error_payload_includes_task_id_for_resume() -> None:
    from ten_texter.cli import _live_tennis_error_payload

    payload = _live_tennis_error_payload(
        LiveTennisTestError("send is still reconciling", task_instance_id=73)
    )

    assert payload == {
        "ok": False,
        "error": "send is still reconciling",
        "task_instance_id": 73,
    }
