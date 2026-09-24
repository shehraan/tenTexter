from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from ten_texter.beeper import BeeperDesktopAdapter, BeeperSyncService
from ten_texter.enums import ContentSupport, DecisionCloseReason, DecisionStatus, MessageKind, OutboxStatus, ParentTerminalPolicy, ProcessingStatus, Transport
from ten_texter.models import BeeperOutboxDestination, BeeperSyncCheckpoint, Conversation, ConversationParticipant, DecisionRequest, DecisionRequestPrompt, Identity, Message, MessageRevision, OutboxDeliveryAttempt, OutboxMessage, Person, TaskInstance, TaskParticipant
from ten_texter.enums import AvailabilityStatus, TaskStatus
from ten_texter.domain import DecisionService, DomainError, utc_now
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


def test_beeper_same_sort_key_edit_uses_event_time_ordering(db_session: Session) -> None:
    service = BeeperSyncService(db_session)
    conversation = service.sync_chat(chat_payload())
    original = {
        "id": "same-position-edit",
        "chatID": conversation.beeper_conversation_id,
        "senderID": "@discord_123:beeper",
        "sortKey": "00000100",
        "timestamp": "2026-08-26T15:00:00Z",
        "type": "TEXT",
        "text": "before",
    }
    first = service.ingest_message(original, received_at=NOW)
    edited = dict(original)
    edited.update(
        {
            "editedTimestamp": "2026-08-26T15:01:00Z",
            "text": "after",
        }
    )
    second = service.ingest_message(edited, received_at=NOW + timedelta(seconds=1))

    message = db_session.get(Message, first.message_id)
    assert message is not None
    assert not second.ordering_conflict
    assert message.current_revision_id == second.revision_id
    assert db_session.scalar(
        select(func.count(DecisionRequest.id)).where(
            DecisionRequest.type == "MESSAGE_ORDERING_CONFLICT"
        )
    ) == 0


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


class ScriptedPollClient:
    def __init__(
        self,
        calls: list[tuple[str, dict[str, str], dict[str, object] | Exception]],
    ) -> None:
        self.calls = calls

    def get(self, url: str, **kwargs: object) -> Response:
        assert self.calls, f"unexpected GET {url}"
        suffix, expected_params, result = self.calls.pop(0)
        assert url.endswith(suffix)
        assert kwargs.get("params") == expected_params
        if isinstance(result, Exception):
            raise result
        return Response(result)


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


class TimestampSearchClient(SlowPendingClient):
    def __init__(self) -> None:
        super().__init__()
        self.search_timestamp: str | None = None

    def get(self, url: str, **_: object) -> Response:
        if url.endswith("pending%3A1"):
            self.gets.append(url)
            return Response({"error": "pending message is no longer available"}, success=False, status_code=404)
        assert self.search_timestamp is not None
        self.gets.append(url)
        return Response(
            {
                "items": [
                    {
                        "id": "search:1",
                        "chatID": "conv:alex",
                        "text": "Are you available?",
                        "isSender": True,
                        "timestamp": self.search_timestamp,
                    }
                ]
            }
        )


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
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


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


def test_pending_send_reconciles_timestamp_match_from_utc_sqlite_value(
    db_session: Session,
) -> None:
    message = create_beeper_send(db_session, key="beeper-pending-timestamp-match")
    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    client = TimestampSearchClient()
    worker = pending_worker(factory, client)

    assert worker.process(message.id) is OutboxStatus.RECONCILING
    with factory() as session:
        attempt = session.scalar(
            select(OutboxDeliveryAttempt)
            .where(OutboxDeliveryAttempt.outbox_message_id == message.id)
            .order_by(OutboxDeliveryAttempt.id.desc())
        )
        assert attempt is not None
        client.search_timestamp = f"{attempt.started_at.isoformat()}Z"

    assert worker.reconcile(message.id) is OutboxStatus.SENT
    with factory() as session:
        attempt = session.scalar(
            select(OutboxDeliveryAttempt)
            .where(OutboxDeliveryAttempt.outbox_message_id == message.id)
            .order_by(OutboxDeliveryAttempt.id.desc())
        )
        assert attempt is not None
        assert attempt.provider_message_id == "search:1"


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
                return Response(
                    {
                        "items": [chat_payload()],
                        "hasMore": False,
                        "newestCursor": "chat-head",
                        "oldestCursor": "chat-tail",
                    }
                )
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
                        ],
                        "hasMore": False,
                        "newestCursor": "message-head",
                        "oldestCursor": "message-tail",
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


def test_beeper_poll_uses_durable_cursors_and_remains_idempotent(db_session: Session) -> None:
    class BoundedPollClient:
        def __init__(self) -> None:
            self.chat_gets = 0
            self.message_gets = 0

        def get(self, url: str, **kwargs: object) -> Response:
            params = kwargs.get("params")
            if url.endswith("/v1/chats"):
                self.chat_gets += 1
                if params == {"cursor": "chat-new", "direction": "after"}:
                    return Response(
                        {
                            "items": [],
                            "hasMore": False,
                            "newestCursor": "chat-new",
                            "oldestCursor": "chat-new",
                        }
                    )
                assert params == {}
                return Response(
                    {
                        "items": [chat_payload()],
                        "hasMore": False,
                        "newestCursor": "chat-new",
                        "oldestCursor": "chat-old",
                    }
                )
            if url.endswith("/v1/chats/%21direct%3Abeeper/messages"):
                self.message_gets += 1
                assert params == {}
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
                        "hasMore": False,
                        "newestCursor": "message-new",
                        "oldestCursor": "message-old",
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
                return Response(
                    {
                        "items": [incomplete_chat],
                        "hasMore": False,
                        "newestCursor": "chat-head",
                        "oldestCursor": "chat-tail",
                    }
                )
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
                        "newestCursor": "message-head",
                        "oldestCursor": "message-tail",
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


SYNC_NOW = datetime(2026, 9, 14, 16, 0, tzinfo=UTC)


def _sync_chat(chat_id: str, *, activity: str = "2026-09-14T15:00:00Z") -> dict[str, object]:
    payload = chat_payload(chat_id=chat_id)
    payload["lastActivity"] = activity
    return payload


def _sync_message(
    message_id: str,
    chat_id: str,
    *,
    text: str | None = "yes",
    timestamp: str = "2026-09-14T15:00:00Z",
    sort_key: str = "00000100",
    edited_timestamp: str | None = None,
    deleted: bool = False,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": message_id,
        "chatID": chat_id,
        "senderID": "@discord_123:beeper",
        "sortKey": sort_key,
        "timestamp": timestamp,
        "type": "TEXT",
        "text": text,
        "isSender": False,
    }
    if edited_timestamp is not None:
        payload["editedTimestamp"] = edited_timestamp
    if deleted:
        payload["isDeleted"] = True
    return payload


def _page(
    items: list[dict[str, object]],
    *,
    newest: str,
    oldest: str,
    more: bool = False,
) -> dict[str, object]:
    return {
        "items": items,
        "hasMore": more,
        "newestCursor": newest,
        "oldestCursor": oldest,
    }


def test_chat_feed_page_two_resumes_from_durable_checkpoint_after_restart(
    db_session: Session,
) -> None:
    chat_one = "!page-one:beeper"
    chat_two = "!page-two:beeper"
    client = ScriptedPollClient(
        [
            (
                "/v1/chats",
                {},
                _page([_sync_chat(chat_one)], newest="chat-head", oldest="chat-page-2", more=True),
            ),
            (
                "/v1/chats/%21page-one%3Abeeper/messages",
                {},
                _page([_sync_message("page-one-message", chat_one)], newest="one-head", oldest="one-tail"),
            ),
            (
                "/v1/chats",
                {"cursor": "chat-page-2", "direction": "before"},
                _page([_sync_chat(chat_two)], newest="chat-page-2", oldest="chat-tail"),
            ),
            (
                "/v1/chats/%21page-two%3Abeeper/messages",
                {},
                _page([_sync_message("page-two-message", chat_two)], newest="two-head", oldest="two-tail"),
            ),
            (
                "/v1/chats/%21page-one%3Abeeper/messages",
                {},
                _page([_sync_message("page-one-message", chat_one)], newest="one-head", oldest="one-tail"),
            ),
        ]
    )
    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    first = BeeperDesktopAdapter(
        factory, access_token="fake", enabled=True, client=client  # type: ignore[arg-type]
    )
    first.poll_inbound(now=SYNC_NOW)
    restarted = BeeperDesktopAdapter(
        factory, access_token="fake", enabled=True, client=client  # type: ignore[arg-type]
    )
    restarted.poll_inbound(now=SYNC_NOW + timedelta(seconds=2))

    with factory() as session:
        assert session.scalar(select(func.count(Message.id))) == 2
        checkpoint = session.get(BeeperSyncCheckpoint, "chat-feed")
        assert checkpoint is not None
        assert checkpoint.bootstrap_complete
        assert checkpoint.newest_cursor == "chat-head"
    assert client.calls == []


def test_message_page_two_resumes_from_durable_checkpoint_after_restart(
    db_session: Session,
) -> None:
    chat_id = "!message-pages:beeper"
    client = ScriptedPollClient(
        [
            ("/v1/chats", {}, _page([_sync_chat(chat_id)], newest="chat-head", oldest="chat-tail")),
            (
                "/v1/chats/%21message-pages%3Abeeper/messages",
                {},
                _page(
                    [_sync_message("newer", chat_id, sort_key="00000200")],
                    newest="message-head",
                    oldest="message-page-2",
                    more=True,
                ),
            ),
            (
                "/v1/chats",
                {"cursor": "chat-head", "direction": "after"},
                _page([], newest="chat-head", oldest="chat-head"),
            ),
            (
                "/v1/chats/%21message-pages%3Abeeper/messages",
                {"cursor": "message-page-2", "direction": "before"},
                _page(
                    [_sync_message("older", chat_id, sort_key="00000100")],
                    newest="message-page-2",
                    oldest="message-tail",
                ),
            ),
        ]
    )
    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    BeeperDesktopAdapter(
        factory, access_token="fake", enabled=True, client=client  # type: ignore[arg-type]
    ).poll_inbound(now=SYNC_NOW)
    BeeperDesktopAdapter(
        factory, access_token="fake", enabled=True, client=client  # type: ignore[arg-type]
    ).poll_inbound(now=SYNC_NOW + timedelta(seconds=2))

    with factory() as session:
        assert set(session.scalars(select(Message.provider_message_id))) == {"newer", "older"}
        checkpoint = session.scalar(
            select(BeeperSyncCheckpoint).where(
                BeeperSyncCheckpoint.scope == "CONVERSATION"
            )
        )
        assert checkpoint is not None and checkpoint.bootstrap_complete
        assert checkpoint.newest_cursor == "message-head"


def test_incomplete_conversation_bootstrap_uses_backfill_cursor_when_chat_feed_touches_it(
    db_session: Session,
) -> None:
    chat_id = "!incomplete-bootstrap:beeper"
    client = ScriptedPollClient(
        [
            (
                "/v1/chats",
                {},
                _page(
                    [_sync_chat(chat_id)],
                    newest="chat-head",
                    oldest="chat-page-2",
                    more=True,
                ),
            ),
            (
                "/v1/chats/%21incomplete-bootstrap%3Abeeper/messages",
                {},
                _page(
                    [_sync_message("bootstrap-recent", chat_id)],
                    newest="message-head-1",
                    oldest="message-tail-1",
                    more=True,
                ),
            ),
            (
                "/v1/chats",
                {"cursor": "chat-page-2", "direction": "before"},
                _page(
                    [_sync_chat(chat_id)],
                    newest="chat-page-2",
                    oldest="chat-tail",
                ),
            ),
            (
                "/v1/chats/%21incomplete-bootstrap%3Abeeper/messages",
                {"cursor": "message-tail-1", "direction": "before"},
                _page(
                    [
                        _sync_message(
                            "bootstrap-older",
                            chat_id,
                            timestamp="2026-09-13T15:00:00Z",
                            sort_key="00000099",
                        )
                    ],
                    newest="message-tail-1",
                    oldest="message-tail-2",
                ),
            ),
        ]
    )
    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    adapter = BeeperDesktopAdapter(
        factory,
        access_token="fake-token",
        enabled=True,
        client=client,  # type: ignore[arg-type]
    )

    first = adapter.poll_inbound(now=SYNC_NOW)
    second = adapter.poll_inbound(now=SYNC_NOW + timedelta(seconds=2))

    assert len(first) == 1
    assert len(second) == 1
    with factory() as session:
        checkpoint = session.scalar(
            select(BeeperSyncCheckpoint).where(
                BeeperSyncCheckpoint.scope == "CONVERSATION"
            )
        )
        assert checkpoint is not None
        assert checkpoint.bootstrap_complete is True
        assert checkpoint.backfill_cursor is None
        assert session.scalar(select(func.count(Message.id))) == 2


def test_failed_message_page_does_not_advance_chat_checkpoint(
    db_session: Session,
) -> None:
    chat_id = "!failed-page:beeper"
    client = ScriptedPollClient(
        [
            ("/v1/chats", {}, _page([_sync_chat(chat_id)], newest="chat-head", oldest="chat-tail")),
            (
                "/v1/chats/%21failed-page%3Abeeper/messages",
                {},
                RuntimeError("provider unavailable"),
            ),
        ]
    )
    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    adapter = BeeperDesktopAdapter(
        factory, access_token="fake", enabled=True, client=client  # type: ignore[arg-type]
    )
    with pytest.raises(RuntimeError, match="provider unavailable"):
        adapter.poll_inbound(now=SYNC_NOW)
    with factory() as session:
        assert session.get(BeeperSyncCheckpoint, "chat-feed") is None
        assert session.scalar(select(func.count(Conversation.id))) == 0


def test_malformed_conversation_scan_retains_checkpoint_without_blocking_poll(
    db_session: Session,
) -> None:
    chat_id = "!blocked-conversation:beeper"
    conversation = BeeperSyncService(db_session).sync_chat(chat_payload(chat_id=chat_id))
    checkpoint = BeeperSyncCheckpoint(
        checkpoint_key="conversation:1",
        scope="CONVERSATION",
        conversation_id=conversation.id,
        backfill_cursor="message-tail",
        bootstrap_cutoff_at=SYNC_NOW,
        bootstrap_complete=False,
        created_at=SYNC_NOW,
        updated_at=SYNC_NOW,
    )
    db_session.add(checkpoint)
    db_session.flush()
    db_session.commit()
    client = ScriptedPollClient(
        [
            (
                "/v1/chats/%21blocked-conversation%3Abeeper/messages",
                {"cursor": "message-tail", "direction": "before"},
                {"items": [], "hasMore": True},
            )
        ]
    )
    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    adapter = BeeperDesktopAdapter(
        factory, access_token="fake", enabled=True, client=client  # type: ignore[arg-type]
    )

    assert adapter._advance_conversation_scan(SYNC_NOW, excluded_keys=frozenset()) == []
    assert client.calls == []
    assert checkpoint.bootstrap_complete is False
    assert checkpoint.backfill_cursor == "message-tail"
    assert checkpoint.updated_at == SYNC_NOW


def test_ingestion_failure_rolls_back_messages_and_checkpoint_progress(
    db_session: Session,
) -> None:
    chat_id = "!bad-message:beeper"
    malformed = _sync_message("bad", chat_id)
    malformed.pop("senderID")
    client = ScriptedPollClient(
        [
            ("/v1/chats", {}, _page([_sync_chat(chat_id)], newest="chat-head", oldest="chat-tail")),
            (
                "/v1/chats/%21bad-message%3Abeeper/messages",
                {},
                _page([malformed], newest="message-head", oldest="message-tail"),
            ),
        ]
    )
    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    adapter = BeeperDesktopAdapter(
        factory, access_token="fake", enabled=True, client=client  # type: ignore[arg-type]
    )
    with pytest.raises(DomainError, match="stable chat/message/sender identifiers"):
        adapter.poll_inbound(now=SYNC_NOW)
    with factory() as session:
        assert session.scalar(select(func.count(BeeperSyncCheckpoint.checkpoint_key))) == 0
        assert session.scalar(select(func.count(Conversation.id))) == 0
        assert session.scalar(select(func.count(Message.id))) == 0


def test_pagination_without_required_cursor_fails_closed(db_session: Session) -> None:
    client = ScriptedPollClient(
        [
            (
                "/v1/chats",
                {},
                {"items": [_sync_chat("!missing-cursor:beeper")], "hasMore": True},
            )
        ]
    )
    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    adapter = BeeperDesktopAdapter(
        factory, access_token="fake", enabled=True, client=client  # type: ignore[arg-type]
    )
    with pytest.raises(RuntimeError, match="missing required cursors"):
        adapter.poll_inbound(now=SYNC_NOW)
    with factory() as session:
        assert session.scalar(select(func.count(BeeperSyncCheckpoint.checkpoint_key))) == 0


@pytest.mark.parametrize(
    "page",
    [
        {"hasMore": False, "newestCursor": "head", "oldestCursor": "tail"},
        {"items": [], "newestCursor": "head", "oldestCursor": "tail"},
    ],
)
def test_incomplete_beeper_page_envelope_does_not_advance_checkpoint(
    db_session: Session,
    page: dict[str, object],
) -> None:
    client = ScriptedPollClient([("/v1/chats", {}, page)])
    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    adapter = BeeperDesktopAdapter(
        factory,
        access_token="fake",
        enabled=True,
        client=client,  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError):
        adapter.poll_inbound(now=SYNC_NOW)

    with factory() as session:
        assert session.scalar(select(func.count(BeeperSyncCheckpoint.checkpoint_key))) == 0


def test_missing_beeper_provider_timestamp_never_becomes_ordering_metadata(
    db_session: Session,
) -> None:
    service = BeeperSyncService(db_session)
    conversation = service.sync_chat(chat_payload())
    first = service.ingest_message(
        {
            "id": "edited-without-time",
            "chatID": conversation.beeper_conversation_id,
            "senderID": "@discord_123:beeper",
            "timestamp": "2026-08-26T15:00:00Z",
            "type": "TEXT",
            "text": "yes",
        },
        received_at=NOW,
    )
    second = service.ingest_message(
        {
            "id": "edited-without-time",
            "chatID": conversation.beeper_conversation_id,
            "senderID": "@discord_123:beeper",
            "type": "TEXT",
            "text": "no",
        },
        received_at=NOW + timedelta(minutes=1),
    )

    message = db_session.get(Message, first.message_id)
    revision = db_session.get(MessageRevision, second.revision_id)
    assert message is not None and revision is not None
    assert second.ordering_conflict
    assert message.current_revision_id == first.revision_id
    assert revision.provider_event_at is None
    assert revision.received_at.replace(tzinfo=UTC) == NOW + timedelta(minutes=1)
    assert message.created_at.replace(tzinfo=UTC) == NOW


def test_empty_terminal_incremental_page_retains_last_committed_checkpoint(
    db_session: Session,
) -> None:
    chat_id = "!retain-checkpoint:beeper"
    client = ScriptedPollClient(
        [
            ("/v1/chats", {}, _page([_sync_chat(chat_id)], newest="chat-head", oldest="chat-tail")),
            (
                "/v1/chats/%21retain-checkpoint%3Abeeper/messages",
                {},
                _page([_sync_message("known", chat_id)], newest="message-head", oldest="message-tail"),
            ),
            (
                "/v1/chats",
                {"cursor": "chat-head", "direction": "after"},
                {"items": [], "hasMore": False},
            ),
            (
                "/v1/chats/%21retain-checkpoint%3Abeeper/messages",
                {},
                {"items": [], "hasMore": False},
            ),
        ]
    )
    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    adapter = BeeperDesktopAdapter(
        factory, access_token="fake", enabled=True, client=client  # type: ignore[arg-type]
    )
    adapter.poll_inbound(now=SYNC_NOW)
    adapter.poll_inbound(now=SYNC_NOW + timedelta(seconds=2))
    with factory() as session:
        checkpoint = session.get(BeeperSyncCheckpoint, "chat-feed")
        assert checkpoint is not None
        assert checkpoint.newest_cursor == "chat-head"
        assert checkpoint.updated_at == (SYNC_NOW + timedelta(seconds=2)).replace(tzinfo=None)


def test_initial_sync_stops_at_thirty_day_cutoff(db_session: Session) -> None:
    chat_id = "!cutoff:beeper"
    client = ScriptedPollClient(
        [
            ("/v1/chats", {}, _page([_sync_chat(chat_id)], newest="chat-head", oldest="chat-tail")),
            (
                "/v1/chats/%21cutoff%3Abeeper/messages",
                {},
                _page(
                    [
                        _sync_message("recent", chat_id),
                        _sync_message(
                            "too-old",
                            chat_id,
                            timestamp="2026-07-01T12:00:00Z",
                            sort_key="00000001",
                        ),
                    ],
                    newest="message-head",
                    oldest="message-old",
                    more=True,
                ),
            ),
        ]
    )
    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    BeeperDesktopAdapter(
        factory, access_token="fake", enabled=True, client=client  # type: ignore[arg-type]
    ).poll_inbound(now=SYNC_NOW)
    with factory() as session:
        assert list(session.scalars(select(Message.provider_message_id))) == ["recent"]
        checkpoint = session.scalar(
            select(BeeperSyncCheckpoint).where(
                BeeperSyncCheckpoint.scope == "CONVERSATION"
            )
        )
        assert checkpoint is not None and checkpoint.bootstrap_complete
        assert checkpoint.backfill_cursor is None


def test_reactivated_chat_uses_forward_message_cursor(db_session: Session) -> None:
    chat_id = "!reactivated:beeper"
    client = ScriptedPollClient(
        [
            ("/v1/chats", {}, _page([_sync_chat(chat_id)], newest="chat-head", oldest="chat-tail")),
            (
                "/v1/chats/%21reactivated%3Abeeper/messages",
                {},
                _page([_sync_message("first", chat_id)], newest="message-head", oldest="message-tail"),
            ),
            (
                "/v1/chats",
                {"cursor": "chat-head", "direction": "after"},
                _page([_sync_chat(chat_id)], newest="chat-next", oldest="chat-next"),
            ),
            (
                "/v1/chats/%21reactivated%3Abeeper/messages",
                {"cursor": "message-head", "direction": "after"},
                _page(
                    [_sync_message("second", chat_id, sort_key="00000200")],
                    newest="message-next",
                    oldest="message-next",
                ),
            ),
        ]
    )
    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    adapter = BeeperDesktopAdapter(
        factory, access_token="fake", enabled=True, client=client  # type: ignore[arg-type]
    )
    adapter.poll_inbound(now=SYNC_NOW)
    adapter.poll_inbound(now=SYNC_NOW + timedelta(seconds=2))
    with factory() as session:
        assert set(session.scalars(select(Message.provider_message_id))) == {"first", "second"}


def test_reconciliation_ingests_explicit_edit_and_deletion_but_not_absence(
    db_session: Session,
) -> None:
    chat_id = "!reconcile:beeper"
    original = [
        _sync_message("edited", chat_id, text="before", sort_key="00000100"),
        _sync_message("deleted", chat_id, text="present", sort_key="00000200"),
        _sync_message("absent", chat_id, text="unchanged", sort_key="00000300"),
    ]
    changed = [
        _sync_message(
            "edited",
            chat_id,
            text="after",
            sort_key="00000400",
            edited_timestamp="2026-09-14T15:30:00Z",
        ),
        _sync_message(
            "deleted",
            chat_id,
            text="ignored",
            sort_key="00000500",
            edited_timestamp="2026-09-14T15:31:00Z",
            deleted=True,
        ),
    ]
    client = ScriptedPollClient(
        [
            ("/v1/chats", {}, _page([_sync_chat(chat_id)], newest="chat-head", oldest="chat-tail")),
            (
                "/v1/chats/%21reconcile%3Abeeper/messages",
                {},
                _page(original, newest="message-head", oldest="message-tail"),
            ),
            (
                "/v1/chats",
                {"cursor": "chat-head", "direction": "after"},
                _page([], newest="chat-head", oldest="chat-head"),
            ),
            (
                "/v1/chats/%21reconcile%3Abeeper/messages",
                {},
                _page(changed, newest="message-reconciled", oldest="message-tail"),
            ),
        ]
    )
    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    adapter = BeeperDesktopAdapter(
        factory, access_token="fake", enabled=True, client=client  # type: ignore[arg-type]
    )
    adapter.poll_inbound(now=SYNC_NOW)
    adapter.poll_inbound(now=SYNC_NOW + timedelta(seconds=2))

    with factory() as session:
        rows = {
            message.provider_message_id: session.get(MessageRevision, message.current_revision_id)
            for message in session.scalars(select(Message))
        }
        assert rows["edited"].text == "after"
        assert rows["deleted"].is_deleted and rows["deleted"].text is None
        assert rows["absent"].text == "unchanged" and not rows["absent"].is_deleted
        assert session.scalar(select(func.count(MessageRevision.id))) == 5


def test_reconciliation_preserves_tombstone_when_legacy_unsupported_revision_has_same_timestamp(
    db_session: Session,
) -> None:
    chat_id = "!unsupported-then-deleted:beeper"
    timestamp = "2026-09-14T15:00:00Z"
    unsupported = _sync_message(
        "unsupported-then-deleted",
        chat_id,
        text=None,
        timestamp=timestamp,
        sort_key="00000100",
    )
    tombstone = _sync_message(
        "unsupported-then-deleted",
        chat_id,
        text=None,
        timestamp=timestamp,
        sort_key="00000100",
        deleted=True,
    )
    client = ScriptedPollClient(
        [
            (
                "/v1/chats",
                {},
                _page(
                    [_sync_chat(chat_id)],
                    newest="chat-head",
                    oldest="chat-tail",
                ),
            ),
            (
                "/v1/chats/%21unsupported-then-deleted%3Abeeper/messages",
                {},
                _page(
                    [unsupported],
                    newest="message-head",
                    oldest="message-tail",
                ),
            ),
            (
                "/v1/chats",
                {"cursor": "chat-head", "direction": "after"},
                _page([], newest="chat-head", oldest="chat-head"),
            ),
            (
                "/v1/chats/%21unsupported-then-deleted%3Abeeper/messages",
                {},
                _page(
                    [tombstone],
                    newest="message-tombstone",
                    oldest="message-tail",
                ),
            ),
        ]
    )
    factory = sessionmaker(
        bind=db_session.bind, expire_on_commit=False, autoflush=False
    )
    adapter = BeeperDesktopAdapter(
        factory,
        access_token="fake",
        enabled=True,
        client=client,  # type: ignore[arg-type]
    )

    adapter.poll_inbound(now=SYNC_NOW)
    adapter.poll_inbound(now=SYNC_NOW + timedelta(seconds=2))

    with factory.begin() as session:
        message = session.scalar(
            select(Message).where(
                Message.provider_message_id == "unsupported-then-deleted"
            )
        )
        revisions = list(
            session.scalars(
                select(MessageRevision)
                .where(MessageRevision.message_id == message.id)
                .order_by(MessageRevision.id)
            )
        )
        assert len(revisions) == 2
        assert not revisions[0].is_deleted
        assert revisions[0].content_support is ContentSupport.UNSUPPORTED
        assert revisions[1].is_deleted
        assert message.current_revision_id == revisions[1].id
        assert revisions[1].processing_status is ProcessingStatus.PROCESSED
        assert session.scalar(
            select(func.count(DecisionRequest.id)).where(
                DecisionRequest.message_revision_id == revisions[1].id,
                DecisionRequest.type == "MESSAGE_ORDERING_CONFLICT",
                DecisionRequest.status == DecisionStatus.PENDING,
            )
        ) == 0
        repeated = BeeperSyncService(session).ingest_message(
            tombstone,
            received_at=SYNC_NOW + timedelta(seconds=3),
        )
        assert repeated.duplicate
        assert (
            session.scalar(
                select(func.count(MessageRevision.id)).where(
                    MessageRevision.message_id == message.id
                )
            )
            == 2
        )


def test_replayed_tombstone_repairs_legacy_pending_ordering_conflict(
    db_session: Session,
) -> None:
    chat_id = "!legacy-tombstone-conflict:beeper"
    timestamp = "2026-09-14T15:00:00Z"
    service = BeeperSyncService(db_session)
    conversation = service.sync_chat(chat_payload(chat_id=chat_id))
    original = service.ingest_message(
        _sync_message(
            "legacy-tombstone",
            chat_id,
            text=None,
            timestamp=timestamp,
            sort_key="00000100",
        )
    )
    message = db_session.get(Message, original.message_id)
    assert message is not None
    deleted_hash = hashlib.sha256(b"deleted\0").hexdigest()
    legacy_key = f"{timestamp}:{deleted_hash}"
    legacy_tombstone = MessageRevision(
        message_id=message.id,
        provider_revision_key=legacy_key,
        provider_sort_key="00000100",
        provider_event_at=datetime.fromisoformat(timestamp.replace("Z", "+00:00")),
        content_hash=deleted_hash,
        is_deleted=True,
        text=None,
        received_at=NOW,
        content_support=ContentSupport.SUPPORTED,
        processing_status=ProcessingStatus.PENDING,
    )
    db_session.add(legacy_tombstone)
    db_session.flush()
    decision = DecisionService(db_session).create(
        decision_type="MESSAGE_ORDERING_CONFLICT",
        subject_kind="message_revision",
        subject_id=legacy_tombstone.id,
        context={"message_id": message.id},
        parent_terminal_policy=ParentTerminalPolicy.SURVIVE,
    )
    db_session.flush()

    result = service.ingest_message(
        _sync_message(
            "legacy-tombstone",
            chat_id,
            text=None,
            timestamp=timestamp,
            sort_key="00000100",
            deleted=True,
        ),
        received_at=NOW + timedelta(seconds=1),
    )

    assert result.duplicate
    assert message.current_revision_id == legacy_tombstone.id
    assert legacy_tombstone.processing_status is ProcessingStatus.PROCESSED
    assert legacy_tombstone.lease_expires_at is None
    assert db_session.get(DecisionRequest, decision.id).status is DecisionStatus.CLOSED
    assert db_session.get(DecisionRequest, decision.id).close_reason is DecisionCloseReason.SUBJECT_RESOLVED
    assert conversation.id == message.conversation_id


def test_explicit_tombstone_closes_conflict_attached_to_prior_revision(
    db_session: Session,
) -> None:
    chat_id = "!stale-ordering-decision:beeper"
    timestamp = "2026-09-14T15:00:00Z"
    service = BeeperSyncService(db_session)
    service.sync_chat(chat_payload(chat_id=chat_id))
    original = service.ingest_message(
        _sync_message(
            "stale-ordering-message",
            chat_id,
            text=None,
            timestamp=timestamp,
            sort_key="00000100",
        )
    )
    decision = DecisionService(db_session).create(
        decision_type="MESSAGE_ORDERING_CONFLICT",
        subject_kind="message_revision",
        subject_id=original.revision_id,
        context={"message_id": original.message_id},
        parent_terminal_policy=ParentTerminalPolicy.SURVIVE,
    )
    db_session.flush()

    deleted = service.ingest_message(
        _sync_message(
            "stale-ordering-message",
            chat_id,
            text=None,
            timestamp=timestamp,
            sort_key="00000100",
            deleted=True,
        ),
        received_at=NOW + timedelta(seconds=1),
    )

    message = db_session.get(Message, original.message_id)
    assert message is not None
    assert message.current_revision_id == deleted.revision_id
    stored_decision = db_session.get(DecisionRequest, decision.id)
    assert stored_decision is not None
    assert stored_decision.status is DecisionStatus.CLOSED
    assert stored_decision.close_reason is DecisionCloseReason.SUBJECT_RESOLVED


def test_replayed_same_sort_edit_repairs_legacy_pending_conflict(
    db_session: Session,
) -> None:
    chat_id = "!legacy-same-sort-edit:beeper"
    timestamp = "2026-09-14T15:00:00Z"
    edited_timestamp = "2026-09-14T15:01:00Z"
    service = BeeperSyncService(db_session)
    conversation = service.sync_chat(chat_payload(chat_id=chat_id))
    original = service.ingest_message(
        _sync_message(
            "legacy-same-sort-edit",
            chat_id,
            text="before",
            timestamp=timestamp,
            sort_key="00000100",
        )
    )
    message = db_session.get(Message, original.message_id)
    assert message is not None
    edited_hash = hashlib.sha256(b"content\0after").hexdigest()
    legacy_key = f"{edited_timestamp}:{hashlib.sha256(b'after').hexdigest()}"
    legacy_edit = MessageRevision(
        message_id=message.id,
        provider_revision_key=legacy_key,
        provider_sort_key="00000100",
        provider_event_at=datetime.fromisoformat(edited_timestamp.replace("Z", "+00:00")),
        content_hash=edited_hash,
        is_deleted=False,
        text="after",
        received_at=NOW,
        content_support=ContentSupport.SUPPORTED,
        processing_status=ProcessingStatus.PENDING,
    )
    db_session.add(legacy_edit)
    db_session.flush()
    decision = DecisionService(db_session).create(
        decision_type="MESSAGE_ORDERING_CONFLICT",
        subject_kind="message_revision",
        subject_id=legacy_edit.id,
        context={"message_id": message.id},
        parent_terminal_policy=ParentTerminalPolicy.SURVIVE,
    )
    db_session.flush()

    result = service.ingest_message(
        _sync_message(
            "legacy-same-sort-edit",
            chat_id,
            text="after",
            timestamp=timestamp,
            sort_key="00000100",
            edited_timestamp=edited_timestamp,
        ),
        received_at=NOW + timedelta(seconds=1),
    )

    assert result.duplicate
    assert message.current_revision_id == legacy_edit.id
    assert legacy_edit.processing_status is ProcessingStatus.PROCESSED
    stored_decision = db_session.get(DecisionRequest, decision.id)
    assert stored_decision is not None
    assert stored_decision.status is DecisionStatus.CLOSED
    assert stored_decision.close_reason is DecisionCloseReason.SUBJECT_RESOLVED
    assert conversation.id == message.conversation_id


def test_reconciliation_page_progress_resumes_after_restart(db_session: Session) -> None:
    chat_id = "!reconcile-pages:beeper"
    client = ScriptedPollClient(
        [
            ("/v1/chats", {}, _page([_sync_chat(chat_id)], newest="chat-head", oldest="chat-tail")),
            (
                "/v1/chats/%21reconcile-pages%3Abeeper/messages",
                {},
                _page([_sync_message("first", chat_id)], newest="message-head", oldest="message-tail"),
            ),
            (
                "/v1/chats",
                {"cursor": "chat-head", "direction": "after"},
                _page([], newest="chat-head", oldest="chat-head"),
            ),
            (
                "/v1/chats/%21reconcile-pages%3Abeeper/messages",
                {},
                _page(
                    [_sync_message("first", chat_id)],
                    newest="audit-head",
                    oldest="audit-page-2",
                    more=True,
                ),
            ),
            (
                "/v1/chats",
                {"cursor": "chat-head", "direction": "after"},
                _page([], newest="chat-head", oldest="chat-head"),
            ),
            (
                "/v1/chats/%21reconcile-pages%3Abeeper/messages",
                {"cursor": "audit-page-2", "direction": "before"},
                _page([], newest="audit-page-2", oldest="audit-tail"),
            ),
        ]
    )
    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    for offset in (0, 2, 4):
        BeeperDesktopAdapter(
            factory, access_token="fake", enabled=True, client=client  # type: ignore[arg-type]
        ).poll_inbound(now=SYNC_NOW + timedelta(seconds=offset))
    with factory() as session:
        checkpoint = session.scalar(
            select(BeeperSyncCheckpoint).where(
                BeeperSyncCheckpoint.scope == "CONVERSATION"
            )
        )
        assert checkpoint is not None
        assert checkpoint.reconciliation_cursor is None
        assert checkpoint.reconciliation_cutoff_at is None
        assert checkpoint.last_reconciled_at == (SYNC_NOW + timedelta(seconds=4)).replace(
            tzinfo=None
        )
