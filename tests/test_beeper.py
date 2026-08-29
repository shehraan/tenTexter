from __future__ import annotations

from datetime import UTC, timedelta

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from ten_texter.beeper import BeeperDesktopAdapter, BeeperSyncService
from ten_texter.enums import DecisionCloseReason, DecisionStatus, MessageKind, OutboxStatus, Transport
from ten_texter.models import BeeperOutboxDestination, Conversation, ConversationParticipant, DecisionRequest, DecisionRequestPrompt, Identity, Message, MessageRevision, OutboxDeliveryAttempt, OutboxMessage, Person, TaskInstance, TaskParticipant
from ten_texter.enums import AvailabilityStatus, TaskStatus
from ten_texter.domain import utc_now
from ten_texter.control import ProductionOwnerCommandHandler
from ten_texter.telegram import TelegramControlGateway
from ten_texter.outbox import AllowingRevalidator, DeliveryRequest, OutboxService, OutboxWorker
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


def test_complete_participant_sync_deactivates_departed_but_incomplete_does_not(db_session: Session) -> None:
    service = BeeperSyncService(db_session)
    conversation = service.sync_chat(chat_payload())
    identity = db_session.scalar(select(Identity).where(Identity.beeper_user_id == "@discord_123:beeper"))
    membership = db_session.get(
        ConversationParticipant,
        {"conversation_id": conversation.id, "identity_id": identity.id},
    )
    incomplete = chat_payload()
    incomplete["participants"] = {"hasMore": True, "items": []}
    service.sync_chat(incomplete)
    assert membership.is_current
    complete = chat_payload()
    complete["participants"] = {"hasMore": False, "items": []}
    service.sync_chat(complete)
    assert not membership.is_current
    assert membership.left_at is not None


def test_departed_membership_preserves_history_but_cannot_pin_or_route(db_session: Session) -> None:
    service = BeeperSyncService(db_session)
    conversation = service.sync_chat(chat_payload())
    service.ingest_message({
        "id": "historical", "chatID": conversation.beeper_conversation_id,
        "senderID": "@discord_123:beeper", "timestamp": "2026-08-26T15:00:00Z",
        "type": "TEXT", "text": "old message",
    })
    complete = chat_payload()
    complete["participants"] = {"hasMore": False, "items": []}
    service.sync_chat(complete)
    assert db_session.scalar(select(Message).where(Message.provider_message_id == "historical")) is not None
    assert conversation.counterparty_person_id is None
    identity = db_session.scalar(select(Identity).where(Identity.beeper_user_id == "@discord_123:beeper"))
    assert ProductionOwnerCommandHandler._route_candidates(db_session) == []
    task = TaskInstance(
        scheduled_at=utc_now(), duration_minutes=30, topic_key="inactive", status=TaskStatus.ACTIVE
    )
    db_session.add(task)
    db_session.flush()
    db_session.add(TaskParticipant(
        task_instance_id=task.id, person_id=identity.person_id,
        conversation_id=conversation.id, availability_status=AvailabilityStatus.UNKNOWN,
    ))
    with pytest.raises(IntegrityError):
        db_session.flush()


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
    def __init__(
        self,
        payload: dict[str, object],
        *,
        success: bool = True,
        status_code: int | None = None,
    ):
        self.payload = payload
        self.is_success = success
        self.status_code = status_code if status_code is not None else (200 if success else 500)

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


class SlowPendingClient(Client):
    def get(self, url: str, **_: object) -> Response:
        self.gets.append(url)
        if url.endswith("pending%3A1"):
            return Response(
                {
                    "id": "pending:1",
                    "chatID": "conv:alex",
                    "text": "Are you available?",
                    "sendStatus": {"status": "PENDING"},
                }
            )
        return Response({"items": []})


class EventuallySuccessfulClient(SlowPendingClient):
    def __init__(self) -> None:
        super().__init__()
        self.finalized = False

    def get(self, url: str, **_: object) -> Response:
        if url.endswith("pending%3A1") and self.finalized:
            self.gets.append(url)
            return Response(
                {
                    "id": "final:1",
                    "chatID": "conv:alex",
                    "text": "Are you available?",
                    "sendStatus": {"status": "SUCCESS"},
                }
            )
        return super().get(url)


class AuthorizationFailureClient(Client):
    def post(self, url: str, **_: object) -> Response:
        self.posts.append(url)
        return Response(
            {"error": "Insufficient permissions. Required scopes: write"},
            success=False,
            status_code=403,
        )


class ServerFailureClient(Client):
    def post(self, url: str, **_: object) -> Response:
        self.posts.append(url)
        return Response({"error": "unknown server failure"}, success=False)


class NetworkFailureClient(Client):
    def post(self, url: str, **_: object) -> Response:
        self.posts.append(url)
        raise TimeoutError("timed out after request began")


class MalformedResponse(Response):
    def json(self) -> dict[str, object]:
        raise ValueError("not JSON")


class MalformedClient(Client):
    def post(self, url: str, **_: object) -> Response:
        self.posts.append(url)
        return MalformedResponse({})


class Validator:
    def validate(self, **_: object) -> bool:
        return True


PENDING_GRACE = timedelta(minutes=5)


def pending_worker(
    factory: sessionmaker[Session],
    client: Client,
    *,
    owner_chat_id: int | None = None,
) -> OutboxWorker:
    return OutboxWorker(
        factory,
        revalidator=AllowingRevalidator(),
        validator=Validator(),
        adapters={
            Transport.BEEPER: BeeperDesktopAdapter(
                factory,
                access_token="fake-token",
                enabled=True,
                client=client,  # type: ignore[arg-type]
            )
        },
        owner_chat_id=owner_chat_id,
        pending_reconciliation_grace=PENDING_GRACE,
    )


def attempt_started_at(factory: sessionmaker[Session], outbox_id: int):
    with factory() as session:
        attempt = session.scalar(
            select(OutboxDeliveryAttempt)
            .where(OutboxDeliveryAttempt.outbox_message_id == outbox_id)
            .order_by(OutboxDeliveryAttempt.id.desc())
        )
        assert attempt is not None
        value = attempt.started_at
        return value.astimezone(UTC)


def create_beeper_send(db_session: Session, *, key: str) -> OutboxMessage:
    core = seed_core(db_session)
    message = OutboxService(db_session).create_beeper(
        task_instance_id=core["task"].id,
        conversation_id=core["conversation"].id,
        participant_ids=[core["participant"].id],
        final_text="Are you available?",
        message_kind=MessageKind.INITIAL,
        idempotency_key=key,
    )
    db_session.commit()
    return message


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


def test_pending_send_unresolved_immediately_does_not_request_owner_decision(
    db_session: Session,
) -> None:
    core = seed_core(db_session)
    message = OutboxService(db_session).create_beeper(
        task_instance_id=core["task"].id,
        conversation_id=core["conversation"].id,
        participant_ids=[core["participant"].id],
        final_text="Are you available?",
        message_kind=MessageKind.INITIAL,
        idempotency_key="beeper-slow-pending",
    )
    db_session.commit()
    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    worker = OutboxWorker(
        factory,
        revalidator=AllowingRevalidator(),
        validator=Validator(),
        adapters={
            Transport.BEEPER: BeeperDesktopAdapter(
                factory,
                access_token="fake-token",
                enabled=True,
                client=SlowPendingClient(),  # type: ignore[arg-type]
            )
        },
    )

    assert worker.process(message.id) is OutboxStatus.RECONCILING
    assert worker.reconcile(message.id) is OutboxStatus.RECONCILING
    with factory() as session:
        assert session.scalar(
            select(func.count(DecisionRequest.id)).where(
                DecisionRequest.outbox_message_id == message.id
            )
        ) == 0


def test_pending_send_stays_reconciling_on_repeated_ticks_within_grace(
    db_session: Session,
) -> None:
    message = create_beeper_send(db_session, key="beeper-pending-repeat")
    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    client = SlowPendingClient()
    worker = pending_worker(factory, client)

    assert worker.process(message.id) is OutboxStatus.RECONCILING
    started_at = attempt_started_at(factory, message.id)
    assert worker.reconcile(message.id, at=started_at + timedelta(minutes=1)) is OutboxStatus.RECONCILING
    assert worker.reconcile(message.id, at=started_at + timedelta(minutes=4)) is OutboxStatus.RECONCILING

    with factory() as session:
        assert session.scalar(
            select(func.count(DecisionRequest.id)).where(
                DecisionRequest.outbox_message_id == message.id
            )
        ) == 0
        assert session.scalar(
            select(func.count(OutboxDeliveryAttempt.id)).where(
                OutboxDeliveryAttempt.outbox_message_id == message.id
            )
        ) == 1
    assert len(client.posts) == 1


def test_pending_send_reconciles_to_final_id_during_grace(db_session: Session) -> None:
    message = create_beeper_send(db_session, key="beeper-pending-success")
    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    client = EventuallySuccessfulClient()
    worker = pending_worker(factory, client)

    assert worker.process(message.id) is OutboxStatus.RECONCILING
    started_at = attempt_started_at(factory, message.id)
    assert worker.reconcile(message.id, at=started_at + timedelta(minutes=1)) is OutboxStatus.RECONCILING
    client.finalized = True
    assert worker.reconcile(message.id, at=started_at + timedelta(minutes=2)) is OutboxStatus.SENT

    with factory() as session:
        attempt = session.scalar(
            select(OutboxDeliveryAttempt)
            .where(OutboxDeliveryAttempt.outbox_message_id == message.id)
            .order_by(OutboxDeliveryAttempt.id.desc())
        )
        assert attempt is not None
        assert attempt.provider_message_id == "final:1"
        assert session.scalar(
            select(func.count(DecisionRequest.id)).where(
                DecisionRequest.outbox_message_id == message.id
            )
        ) == 0
    assert len(client.posts) == 1


def test_pending_send_unresolved_after_grace_creates_exactly_one_decision(
    db_session: Session,
) -> None:
    message = create_beeper_send(db_session, key="beeper-pending-expired")
    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    client = SlowPendingClient()
    worker = pending_worker(factory, client)

    assert worker.process(message.id) is OutboxStatus.RECONCILING
    started_at = attempt_started_at(factory, message.id)
    after_grace = started_at + PENDING_GRACE + timedelta(seconds=1)
    assert worker.reconcile(message.id, at=after_grace) is OutboxStatus.RECONCILING
    assert worker.reconcile(message.id, at=after_grace + timedelta(minutes=1)) is OutboxStatus.RECONCILING

    with factory() as session:
        decisions = list(
            session.scalars(
                select(DecisionRequest).where(
                    DecisionRequest.outbox_message_id == message.id,
                    DecisionRequest.type == "UNCERTAIN_DELIVERY",
                )
            )
        )
        assert len(decisions) == 1
    assert len(client.posts) == 1


def test_pending_reconciliation_grace_survives_worker_restart(db_session: Session) -> None:
    message = create_beeper_send(db_session, key="beeper-pending-restart")
    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    client = SlowPendingClient()
    assert pending_worker(factory, client).process(message.id) is OutboxStatus.RECONCILING
    started_at = attempt_started_at(factory, message.id)

    restarted = pending_worker(factory, client)
    assert restarted.reconcile(
        message.id,
        at=started_at + timedelta(minutes=2),
    ) is OutboxStatus.RECONCILING
    with factory() as session:
        assert session.scalar(
            select(func.count(DecisionRequest.id)).where(
                DecisionRequest.outbox_message_id == message.id
            )
        ) == 0
    assert len(client.posts) == 1


def test_pending_send_late_success_closes_uncertain_delivery_decision(
    db_session: Session,
) -> None:
    message = create_beeper_send(db_session, key="beeper-pending-late-success")
    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    client = EventuallySuccessfulClient()
    worker = pending_worker(factory, client)

    assert worker.process(message.id) is OutboxStatus.RECONCILING
    started_at = attempt_started_at(factory, message.id)
    assert worker.reconcile(
        message.id,
        at=started_at + PENDING_GRACE + timedelta(seconds=1),
    ) is OutboxStatus.RECONCILING
    with factory() as session:
        decision = session.scalar(
            select(DecisionRequest).where(
                DecisionRequest.outbox_message_id == message.id,
                DecisionRequest.type == "UNCERTAIN_DELIVERY",
            )
        )
        assert decision is not None
        assert decision.status is DecisionStatus.PENDING

    client.finalized = True
    assert worker.reconcile(
        message.id,
        at=started_at + PENDING_GRACE + timedelta(minutes=1),
    ) is OutboxStatus.SENT
    with factory() as session:
        decision = session.scalar(
            select(DecisionRequest).where(
                DecisionRequest.outbox_message_id == message.id,
                DecisionRequest.type == "UNCERTAIN_DELIVERY",
            )
        )
        attempt = session.scalar(
            select(OutboxDeliveryAttempt)
            .where(OutboxDeliveryAttempt.outbox_message_id == message.id)
            .order_by(OutboxDeliveryAttempt.id.desc())
        )
        assert decision is not None
        assert decision.status is DecisionStatus.CLOSED
        assert decision.close_reason is DecisionCloseReason.SUBJECT_RESOLVED
        assert attempt is not None
        assert attempt.provider_message_id == "final:1"


def test_beeper_transport_disabled_by_default(db_session: Session) -> None:
    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    adapter = BeeperDesktopAdapter(factory, access_token=None)
    from ten_texter.outbox import DeliveryRequest

    result = adapter.send(DeliveryRequest(1, Transport.BEEPER, "1", "text", "key"))
    assert not result.success and result.definitely_not_sent


def test_authorization_rejection_is_definitely_not_sent(db_session: Session) -> None:
    message = create_beeper_send(db_session, key="beeper-authorization-rejected")
    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    client = AuthorizationFailureClient()
    adapter = BeeperDesktopAdapter(
        factory,
        access_token="fake-token",
        enabled=True,
        client=client,  # type: ignore[arg-type]
    )
    destination = db_session.get(BeeperOutboxDestination, message.id)
    assert destination is not None
    request = DeliveryRequest(
        message.id,
        Transport.BEEPER,
        str(destination.conversation_id),
        message.final_text,
        message.idempotency_key,
    )
    result = adapter.send(request)
    assert result.definitely_not_sent
    assert not result.boundary_crossed

    worker = pending_worker(factory, client)
    assert worker.process(message.id) is OutboxStatus.PENDING
    with factory() as session:
        stored = session.get(OutboxMessage, message.id)
        assert stored is not None
        assert stored.status is OutboxStatus.PENDING
        assert session.scalar(
            select(func.count(DecisionRequest.id)).where(
                DecisionRequest.outbox_message_id == message.id
            )
        ) == 0


def test_server_failure_is_uncertain_and_cannot_send_twice(db_session: Session) -> None:
    core = seed_core(db_session)
    message = OutboxService(db_session).create_beeper(
        task_instance_id=core["task"].id,
        conversation_id=core["conversation"].id,
        participant_ids=[core["participant"].id],
        final_text="Are you available?",
        message_kind=MessageKind.INITIAL,
        idempotency_key="beeper-5xx",
    )
    db_session.commit()
    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    client = ServerFailureClient()
    worker = OutboxWorker(
        factory,
        revalidator=AllowingRevalidator(),
        validator=Validator(),
        adapters={Transport.BEEPER: BeeperDesktopAdapter(factory, access_token="fake", enabled=True, client=client)},
        owner_chat_id=99,
    )
    assert worker.process(message.id) is OutboxStatus.RECONCILING
    assert worker.process(message.id) is OutboxStatus.RECONCILING
    assert len(client.posts) == 1
    assert worker.reconcile(message.id) is OutboxStatus.RECONCILING
    with factory() as session:
        decision = session.scalar(select(DecisionRequest).where(DecisionRequest.outbox_message_id == message.id))
        assert decision is not None
        assert session.scalar(select(DecisionRequestPrompt).where(DecisionRequestPrompt.decision_request_id == decision.id)) is not None
        decision_id = decision.id
    handler = ProductionOwnerCommandHandler(
        factory, owner_chat_id=99, resolver=object(), generation=object()  # type: ignore[arg-type]
    )
    control = TelegramControlGateway(factory, owner_id=7, parser=object(), handler=handler)  # type: ignore[arg-type]
    def answer(update_id: int, action: str) -> dict[str, object]:
        return {"update_id": update_id, "callback_query": {"id": str(update_id), "from": {"id": 7}, "message": {"message_id": 1, "chat": {"id": 99, "type": "private"}}, "data": f"decision:{decision_id}:{action}"}}
    unsafe = control.receive(answer(800, "yes"))
    with pytest.raises(Exception, match="only permits"):
        control.process(unsafe.telegram_update_row_id)
    safe = control.receive(answer(801, "keep_reconciling"))
    control.process(safe.telegram_update_row_id)
    assert worker.process(message.id) is OutboxStatus.RECONCILING
    assert len(client.posts) == 1


def test_network_and_malformed_send_outcomes_are_uncertain(db_session: Session) -> None:
    core = seed_core(db_session)
    db_session.commit()
    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    from ten_texter.outbox import DeliveryRequest

    request = DeliveryRequest(1, Transport.BEEPER, str(core["conversation"].id), "text", "key")
    for client in (NetworkFailureClient(), MalformedClient()):
        result = BeeperDesktopAdapter(
            factory, access_token="fake", enabled=True, client=client  # type: ignore[arg-type]
        ).send(request)
        assert result.boundary_crossed
        assert not result.definitely_not_sent


def test_invalid_local_destination_is_known_unsent(db_session: Session) -> None:
    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    from ten_texter.outbox import DeliveryRequest

    result = BeeperDesktopAdapter(
        factory, access_token="fake", enabled=True, client=Client()  # type: ignore[arg-type]
    ).send(DeliveryRequest(1, Transport.BEEPER, "not-an-internal-id", "text", "key"))
    assert not result.boundary_crossed
    assert result.definitely_not_sent


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


def test_beeper_poll_truncates_recent_pages_and_remains_idempotent(db_session: Session) -> None:
    class BoundedPollClient:
        def __init__(self) -> None:
            self.chat_gets = 0
            self.message_gets = 0

        def get(self, url: str, **kwargs: object) -> Response:
            params = kwargs.get("params")
            assert params == {}
            if url.endswith("/v1/chats"):
                self.chat_gets += 1
                return Response(
                    {
                        "items": [chat_payload()],
                        "hasMore": True,
                    }
                )
            if url.endswith("/v1/chats/%21direct%3Abeeper/messages"):
                self.message_gets += 1
                return Response(
                    {
                        "items": [
                            {
                                "id": "recent-incoming",
                                "chatID": "!direct:beeper",
                                "senderID": "@discord_123:beeper",
                                "sortKey": "00000200",
                                "timestamp": "2026-08-28T15:00:00Z",
                                "type": "TEXT",
                                "text": "recent reply",
                                "isSender": False,
                            }
                        ],
                        "hasMore": True,
                    }
                )
            raise AssertionError(f"unexpected URL: {url}")

    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    client = BoundedPollClient()
    adapter = BeeperDesktopAdapter(
        factory,
        access_token="fake-token",
        enabled=True,
        client=client,  # type: ignore[arg-type]
    )

    first = adapter.poll_inbound()
    second = adapter.poll_inbound()

    assert first == second
    assert client.chat_gets == 2
    assert client.message_gets == 2
    with factory() as session:
        conversation = session.scalar(
            select(Conversation).where(
                Conversation.beeper_conversation_id == "!direct:beeper"
            )
        )
        assert conversation is not None
        assert session.scalar(select(func.count(Message.id))) == 1
        assert session.scalar(select(func.count(MessageRevision.id))) == 1


def test_poll_ingests_stable_sender_missing_from_incomplete_participants(
    db_session: Session,
) -> None:
    omitted_sender_id = "@discord_omitted:beeper"
    incomplete_chat = chat_payload(chat_id="!incomplete:beeper")
    incomplete_chat["participants"] = {
        "hasMore": True,
        "items": [{"id": "@self:beeper", "fullName": "Owner", "isSelf": True}],
    }

    class MissingParticipantClient:
        def get(self, url: str, **_: object) -> Response:
            if url.endswith("/v1/chats"):
                return Response({"items": [incomplete_chat], "hasMore": False})
            if url.endswith("/v1/chats/%21incomplete%3Abeeper/messages"):
                return Response(
                    {
                        "items": [
                            {
                                "id": "message-from-omitted-sender",
                                "chatID": "!incomplete:beeper",
                                "senderID": omitted_sender_id,
                                "senderName": "Omitted Sender",
                                "sortKey": "00000300",
                                "timestamp": "2026-08-28T16:00:00Z",
                                "type": "TEXT",
                                "text": "I can make it.",
                                "isSender": False,
                            }
                        ],
                        "hasMore": False,
                    }
                )
            raise AssertionError(f"unexpected URL: {url}")

    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    adapter = BeeperDesktopAdapter(
        factory,
        access_token="fake-token",
        enabled=True,
        client=MissingParticipantClient(),  # type: ignore[arg-type]
    )

    first = adapter.poll_inbound()
    second = adapter.poll_inbound()

    assert first == second
    with factory() as session:
        identity = session.scalar(
            select(Identity).where(Identity.beeper_user_id == omitted_sender_id)
        )
        conversation = session.scalar(
            select(Conversation).where(
                Conversation.beeper_conversation_id == "!incomplete:beeper"
            )
        )
        assert identity is not None
        assert identity.display_name == "Omitted Sender"
        assert conversation is not None
        membership = session.get(
            ConversationParticipant,
            {"conversation_id": conversation.id, "identity_id": identity.id},
        )
        assert membership is not None
        assert not membership.is_current
        assert membership.left_at is not None
        message = session.scalar(
            select(Message).where(Message.provider_message_id == "message-from-omitted-sender")
        )
        assert message is not None
        assert message.sender_identity_id == identity.id
        assert session.scalar(select(func.count(Message.id))) == 1
        assert session.scalar(select(func.count(MessageRevision.id))) == 1
