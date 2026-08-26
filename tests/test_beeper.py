from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from ten_texter.beeper import BeeperDesktopAdapter, BeeperSyncService
from ten_texter.enums import MessageKind, OutboxStatus, Transport
from ten_texter.models import Conversation, Identity, Message, MessageRevision, Person
from ten_texter.outbox import AllowingRevalidator, OutboxService, OutboxWorker
from tests.test_schema import NOW, seed_core


def chat_payload(*, full_name: str = "Sam", chat_id: str = "!direct:beeper") -> dict[str, object]:
    return {
        "id": chat_id,
        "accountID": "discordgo",
        "network": "Discord",
        "type": "single",
        "title": full_name,
        "participants": {
            "hasMore": False,
            "items": [
                {"id": "@self:beeper", "fullName": "Owner", "isSelf": True},
                {
                    "id": "@discord_123:beeper",
                    "fullName": full_name,
                    "username": full_name.lower(),
                    "isSelf": False,
                },
            ],
        },
    }


def test_chat_identity_sync_uses_stable_ids_and_updates_metadata(db_session: Session) -> None:
    service = BeeperSyncService(db_session)
    first = service.sync_chat(chat_payload())
    identity = db_session.scalar(
        select(Identity).where(Identity.beeper_user_id == "@discord_123:beeper")
    )
    person_id = identity.person_id
    second = service.sync_chat(chat_payload(full_name="Samuel"))
    assert first.id == second.id
    assert identity.id == db_session.scalar(
        select(Identity.id).where(Identity.beeper_user_id == "@discord_123:beeper")
    )
    assert db_session.get(Identity, identity.id).display_name == "Samuel"
    assert db_session.scalar(select(func.count(Person.id)).where(Person.id == person_id)) == 1


def test_direct_and_group_chats_remain_distinct_pinned_destinations(db_session: Session) -> None:
    service = BeeperSyncService(db_session)
    direct = service.sync_chat(chat_payload(chat_id="!direct:beeper"))
    group_payload = chat_payload(chat_id="!group:beeper")
    group_payload["type"] = "group"
    group = service.sync_chat(group_payload)
    assert direct.id != group.id
    assert direct.beeper_conversation_id == "!direct:beeper"
    assert group.beeper_conversation_id == "!group:beeper"
    assert group.counterparty_person_id is None


def test_beeper_message_maps_sort_key_edits_and_reply_link(db_session: Session) -> None:
    service = BeeperSyncService(db_session)
    conversation = service.sync_chat(chat_payload())
    message_payload = {
        "id": "beeper-message-1",
        "chatID": conversation.beeper_conversation_id,
        "senderID": "@discord_123:beeper",
        "sortKey": "00000100",
        "timestamp": "2026-08-26T15:00:00Z",
        "type": "TEXT",
        "text": "yes",
        "linkedMessageID": "prior-message",
    }
    first = service.ingest_message(message_payload, received_at=NOW)
    message_payload.update(
        {
            "sortKey": "00000101",
            "editedTimestamp": "2026-08-26T15:01:00Z",
            "text": "no",
        }
    )
    second = service.ingest_message(message_payload, received_at=NOW)
    message = db_session.get(Message, first.message_id)
    assert message.provider_reply_to_message_id == "prior-message"
    assert message.current_revision_id == second.revision_id
    assert db_session.get(MessageRevision, second.revision_id).provider_sort_key == "00000101"


class Response:
    def __init__(self, payload: dict[str, object], *, success: bool = True):
        self.payload = payload
        self.is_success = success
        self.status_code = 200 if success else 500

    def json(self) -> dict[str, object]:
        return self.payload

    def raise_for_status(self) -> None:
        if not self.is_success:
            raise RuntimeError("HTTP error")


class Client:
    def __init__(self):
        self.posts: list[str] = []
        self.gets: list[str] = []

    def post(self, url: str, **_: object) -> Response:
        self.posts.append(url)
        return Response({"chatID": "conv:alex", "pendingMessageID": "pending:1"})

    def get(self, url: str, **_: object) -> Response:
        self.gets.append(url)
        if url.endswith("pending%3A1"):
            return Response(
                {
                    "id": "final:1",
                    "chatID": "conv:alex",
                    "text": "Are you available?",
                    "sendStatus": {"status": "SUCCESS"},
                }
            )
        return Response({"items": []})


class Validator:
    def validate(self, **_: object) -> bool:
        return True


def test_pending_send_maps_to_reconciling_then_final_id(db_session: Session) -> None:
    core = seed_core(db_session)
    message = OutboxService(db_session).create_beeper(
        task_instance_id=core["task"].id,
        conversation_id=core["conversation"].id,
        participant_ids=[core["participant"].id],
        final_text="Are you available?",
        message_kind=MessageKind.INITIAL,
        idempotency_key="beeper-send",
    )
    db_session.commit()
    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    client = Client()
    adapter = BeeperDesktopAdapter(
        factory,
        access_token="fake-token",
        enabled=True,
        client=client,  # type: ignore[arg-type]
    )
    worker = OutboxWorker(
        factory,
        revalidator=AllowingRevalidator(),
        validator=Validator(),
        adapters={Transport.BEEPER: adapter},
    )
    assert worker.process(message.id) is OutboxStatus.RECONCILING
    assert len(client.posts) == 1
    assert "/v1/chats/conv%3Aalex/messages" in client.posts[0]
    assert worker.reconcile(message.id) is OutboxStatus.SENT
    assert any(url.endswith("pending%3A1") for url in client.gets)


def test_beeper_transport_disabled_by_default(db_session: Session) -> None:
    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    adapter = BeeperDesktopAdapter(factory, access_token=None)
    from ten_texter.outbox import DeliveryRequest

    result = adapter.send(DeliveryRequest(1, Transport.BEEPER, "1", "text", "key"))
    assert not result.success and result.definitely_not_sent


def test_beeper_poll_syncs_chats_and_ingests_inbound_messages(db_session: Session) -> None:
    class PollClient:
        def get(self, url: str, **_: object) -> Response:
            if url.endswith("/v1/chats"):
                return Response({"items": [chat_payload()]})
            if url.endswith("/v1/chats/%21direct%3Abeeper/messages"):
                return Response(
                    {
                        "items": [
                            {
                                "id": "incoming:1",
                                "chatID": "!direct:beeper",
                                "senderID": "@discord_123:beeper",
                                "sortKey": "00000100",
                                "timestamp": "2026-08-26T15:00:00Z",
                                "type": "TEXT",
                                "text": "yes",
                                "isSender": False,
                            }
                        ]
                    }
                )
            raise AssertionError(f"unexpected URL: {url}")

    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    adapter = BeeperDesktopAdapter(
        factory,
        access_token="fake-token",
        enabled=True,
        client=PollClient(),  # type: ignore[arg-type]
    )

    revision_ids = adapter.poll_inbound()

    assert len(revision_ids) == 1
    with factory() as session:
        revision = session.get(MessageRevision, revision_ids[0])
        assert revision.text == "yes"
        assert revision.processing_status.value == "PENDING"
