from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import quote

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from ten_texter.domain import DomainError, utc_now
from ten_texter.enums import ContentSupport, ConversationKind, Transport
from ten_texter.inbound import InboundEvent, IngestResult, MessageIngestor
from ten_texter.models import (
    Conversation,
    ConversationParticipant,
    Identity,
    OutboxDeliveryAttempt,
    Person,
)
from ten_texter.outbox import DeliveryRequest, DeliveryResult


def _parse_time(value: str | None) -> datetime:
    if not value:
        return utc_now()
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class BeeperSyncService:
    def __init__(self, session: Session):
        self.session = session

    def sync_chat(self, payload: dict[str, Any]) -> Conversation:
        chat_id = payload.get("id")
        if not isinstance(chat_id, str) or not chat_id:
            raise DomainError("Beeper chat is missing stable id")
        chat_type = payload.get("type")
        if chat_type not in {"single", "group"}:
            raise DomainError("Beeper chat type is unsupported")
        conversation = self.session.scalar(
            select(Conversation).where(Conversation.beeper_conversation_id == chat_id)
        )
        if conversation is None:
            conversation = Conversation(
                beeper_conversation_id=chat_id,
                network=str(payload.get("network") or "unknown"),
                kind=ConversationKind.DIRECT if chat_type == "single" else ConversationKind.GROUP,
                title=payload.get("title"),
                metadata_json={},
            )
            self.session.add(conversation)
            self.session.flush()
        else:
            conversation.network = str(payload.get("network") or conversation.network)
            conversation.kind = ConversationKind.DIRECT if chat_type == "single" else ConversationKind.GROUP
            conversation.title = payload.get("title")
        participant_payload = payload.get("participants") or {}
        items = participant_payload.get("items") if isinstance(participant_payload, dict) else []
        counterparty_person_id: int | None = None
        for user in items or []:
            if not isinstance(user, dict) or user.get("isSelf") is True:
                continue
            identity = self._sync_identity(user, network=conversation.network)
            membership = self.session.get(
                ConversationParticipant,
                {"conversation_id": conversation.id, "identity_id": identity.id},
            )
            if membership is None:
                self.session.add(
                    ConversationParticipant(conversation_id=conversation.id, identity_id=identity.id)
                )
            if conversation.kind is ConversationKind.DIRECT:
                counterparty_person_id = identity.person_id
        conversation.counterparty_person_id = (
            counterparty_person_id if conversation.kind is ConversationKind.DIRECT else None
        )
        conversation.metadata_json = {
            "account_id": payload.get("accountID"),
            "participants_complete": not bool(participant_payload.get("hasMore"))
            if isinstance(participant_payload, dict)
            else False,
        }
        self.session.flush()
        return conversation

    def _sync_identity(self, user: dict[str, Any], *, network: str) -> Identity:
        user_id = user.get("id")
        if not isinstance(user_id, str) or not user_id:
            raise DomainError("Beeper participant missing stable user id")
        identity = self.session.scalar(select(Identity).where(Identity.beeper_user_id == user_id))
        if identity is None:
            display = str(user.get("fullName") or user.get("username") or user_id)
            person = Person(display_name=display, metadata_json={})
            self.session.add(person)
            self.session.flush()
            identity = Identity(
                person_id=person.id,
                beeper_user_id=user_id,
                network=network,
                metadata_json={},
            )
            self.session.add(identity)
            self.session.flush()
        identity.network = network
        identity.username = user.get("username")
        identity.display_name = user.get("fullName")
        identity.metadata_json = {
            "email": user.get("email"),
            "phone_number": user.get("phoneNumber"),
            "cannot_message": user.get("cannotMessage"),
        }
        return identity

    def ingest_message(self, payload: dict[str, Any], *, received_at: datetime | None = None) -> IngestResult:
        chat_id = payload.get("chatID")
        sender_id = payload.get("senderID")
        message_id = payload.get("id")
        if not all(isinstance(value, str) and value for value in (chat_id, sender_id, message_id)):
            raise DomainError("Beeper message lacks stable chat/message/sender identifiers")
        conversation = self.session.scalar(
            select(Conversation).where(Conversation.beeper_conversation_id == chat_id)
        )
        identity = self.session.scalar(select(Identity).where(Identity.beeper_user_id == sender_id))
        if conversation is None or identity is None:
            raise DomainError("Beeper chat and sender must be synced before message ingestion")
        message_type = payload.get("type")
        is_deleted = bool(payload.get("isDeleted"))
        text_value = payload.get("text")
        supported = message_type in {"TEXT", "NOTICE"} and isinstance(text_value, str)
        if is_deleted:
            text_value = None
            supported = True
        body_hash = hashlib.sha256((text_value or "<deleted>").encode()).hexdigest()
        revision_marker = payload.get("editedTimestamp") or payload.get("timestamp") or "unknown"
        event = InboundEvent(
            conversation_id=conversation.id,
            provider_message_id=message_id,
            sender_identity_id=identity.id,
            provider_revision_key=f"{revision_marker}:{body_hash}",
            provider_sort_key=str(payload.get("sortKey")) if payload.get("sortKey") is not None else None,
            provider_event_at=_parse_time(payload.get("editedTimestamp") or payload.get("timestamp")),
            created_at=_parse_time(payload.get("timestamp")),
            received_at=received_at or utc_now(),
            text=text_value if isinstance(text_value, str) else None,
            provider_reply_to_message_id=payload.get("linkedMessageID"),
            is_deleted=is_deleted,
            content_support=ContentSupport.SUPPORTED if supported else ContentSupport.UNSUPPORTED,
        )
        return MessageIngestor(self.session).ingest(event)


class BeeperDesktopAdapter:
    """Beeper Desktop REST v1 adapter; disabled unless explicitly configured."""

    def __init__(
        self,
        sessions: sessionmaker[Session],
        *,
        base_url: str = "http://127.0.0.1:23373",
        access_token: str | None,
        enabled: bool = False,
        client: httpx.Client | None = None,
    ):
        if enabled and not access_token:
            raise ValueError("enabled Beeper adapter requires an access token")
        self.sessions = sessions
        self.base_url = base_url.rstrip("/")
        self.access_token = access_token
        self.enabled = enabled
        self.client = client or httpx.Client(timeout=30)

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.access_token}"} if self.access_token else {}

    def info(self) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        response = self.client.get(f"{self.base_url}/v1/info", headers=self.headers)
        response.raise_for_status()
        return response.json()

    def list_chats(self) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        response = self.client.get(f"{self.base_url}/v1/chats", headers=self.headers)
        response.raise_for_status()
        payload = response.json()
        return list(payload.get("items") or [])

    def send(self, request: DeliveryRequest) -> DeliveryResult:
        if request.transport is not Transport.BEEPER:
            return DeliveryResult(False, False, definitely_not_sent=True, error="wrong transport")
        if not self.enabled:
            return DeliveryResult(False, False, definitely_not_sent=True, error="Beeper transport disabled")
        chat_id = self._provider_chat_id(request.destination)
        try:
            response = self.client.post(
                f"{self.base_url}/v1/chats/{quote(chat_id, safe='')}/messages",
                headers=self.headers,
                json={"text": request.text},
            )
        except Exception as exc:
            return DeliveryResult(False, True, error=str(exc))
        try:
            payload = response.json()
        except Exception as exc:
            return DeliveryResult(False, True, error=f"invalid Beeper response: {exc}")
        if response.is_success and isinstance(payload.get("pendingMessageID"), str):
            # A pending ID proves the boundary was crossed, not that the network accepted the send.
            return DeliveryResult(
                False,
                True,
                pending_provider_id=payload["pendingMessageID"],
            )
        return DeliveryResult(
            False,
            True,
            definitely_not_sent=not response.is_success,
            error=str(payload.get("error") or response.status_code),
        )

    def reconcile(
        self,
        request: DeliveryRequest,
        *,
        provider_message_id: str | None,
        pending_provider_id: str | None,
    ) -> str | None:
        if provider_message_id is not None:
            return provider_message_id
        if not self.enabled:
            return None
        chat_id = self._provider_chat_id(request.destination)
        if pending_provider_id is not None:
            response = self.client.get(
                f"{self.base_url}/v1/chats/{quote(chat_id, safe='')}/messages/{quote(pending_provider_id, safe='')}",
                headers=self.headers,
            )
            if response.is_success:
                payload = response.json()
                send_status = (payload.get("sendStatus") or {}).get("status")
                final_id = payload.get("id")
                if send_status == "SUCCESS" and isinstance(final_id, str):
                    return final_id
                if send_status == "PENDING":
                    return None
        # Exact destination/sender/text/timestamp search is the final automatic fallback.
        response = self.client.get(
            f"{self.base_url}/v1/chats/{quote(chat_id, safe='')}/messages",
            headers=self.headers,
        )
        if not response.is_success:
            return None
        payload = response.json()
        started_at = self._attempt_started_at(request.outbox_id)
        matches: list[str] = []
        for item in payload.get("items") or []:
            if item.get("isSender") is not True or item.get("text") != request.text:
                continue
            timestamp = _parse_time(item.get("timestamp"))
            if abs(timestamp - started_at) <= timedelta(minutes=2) and isinstance(item.get("id"), str):
                matches.append(item["id"])
        return matches[0] if len(matches) == 1 else None

    def _provider_chat_id(self, internal_destination: str) -> str:
        try:
            conversation_id = int(internal_destination)
        except ValueError as exc:
            raise DomainError("invalid internal Beeper destination") from exc
        with self.sessions() as session:
            conversation = session.get(Conversation, conversation_id)
            if conversation is None:
                raise DomainError("Beeper destination conversation not found")
            return conversation.beeper_conversation_id

    def _attempt_started_at(self, outbox_id: int) -> datetime:
        with self.sessions() as session:
            attempt = session.scalar(
                select(OutboxDeliveryAttempt)
                .where(OutboxDeliveryAttempt.outbox_message_id == outbox_id)
                .order_by(OutboxDeliveryAttempt.id.desc())
            )
            if attempt is None:
                return utc_now()
            value = attempt.started_at
            return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
