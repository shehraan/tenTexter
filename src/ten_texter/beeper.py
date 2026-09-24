from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import quote

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from ten_texter.domain import DomainError, utc_now
from ten_texter.enums import ContentSupport, ConversationKind, Transport
from ten_texter.inbound import (
    InboundEvent,
    IngestResult,
    MessageIngestor,
    immutable_content_hash,
)
from ten_texter.models import (
    BeeperSyncCheckpoint,
    Conversation,
    ConversationParticipant,
    Identity,
    Message,
    MessageRevision,
    OutboxDeliveryAttempt,
    Person,
)
from ten_texter.outbox import DeliveryRequest, DeliveryResult


LOGGER = logging.getLogger(__name__)


def _parse_time(value: object) -> datetime:
    if not isinstance(value, str) or not value:
        raise RuntimeError("Beeper timestamp is missing or malformed")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeError("Beeper timestamp is malformed") from exc


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _optional_provider_time(value: object) -> datetime | None:
    if value is None:
        return None
    return _aware(_parse_time(value))


@dataclass(frozen=True, slots=True)
class BeeperPage:
    items: tuple[dict[str, Any], ...]
    has_more: bool
    newest_cursor: str | None
    oldest_cursor: str | None


@dataclass(frozen=True, slots=True)
class _CheckpointSnapshot:
    checkpoint_key: str
    newest_cursor: str | None
    backfill_cursor: str | None
    bootstrap_cutoff_at: datetime
    bootstrap_complete: bool
    reconciliation_cursor: str | None
    reconciliation_cutoff_at: datetime | None
    last_reconciled_at: datetime | None
    updated_at: datetime


class BeeperSyncService:
    def __init__(self, session: Session, *, owner_chat_id: int | None = None):
        self.session = session
        self.owner_chat_id = owner_chat_id

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
        participant_payload = payload.get("participants")
        if not isinstance(participant_payload, dict):
            raise DomainError("Beeper chat participants payload is malformed")
        items = participant_payload.get("items")
        has_more = participant_payload.get("hasMore")
        if not isinstance(items, list) or not isinstance(has_more, bool):
            raise DomainError("Beeper chat participants page is malformed")
        if any(not isinstance(user, dict) for user in items):
            raise DomainError("Beeper chat participants contain a malformed item")
        participants_complete = not has_more
        counterparty_person_id: int | None = None
        current_identity_ids: set[int] = set()
        for user in items:
            if user.get("isSelf") is True:
                continue
            identity = self._sync_identity(user, network=conversation.network)
            membership = self.session.get(
                ConversationParticipant,
                {"conversation_id": conversation.id, "identity_id": identity.id},
            )
            if membership is None:
                membership = ConversationParticipant(conversation_id=conversation.id, identity_id=identity.id)
                self.session.add(membership)
            else:
                membership.is_current = True
                membership.left_at = None
            current_identity_ids.add(identity.id)
            if conversation.kind is ConversationKind.DIRECT:
                counterparty_person_id = identity.person_id
        if participants_complete:
            departed = list(
                self.session.scalars(
                    select(ConversationParticipant).where(
                        ConversationParticipant.conversation_id == conversation.id,
                        ConversationParticipant.is_current.is_(True),
                        ConversationParticipant.identity_id.not_in(current_identity_ids),
                    )
                )
            )
            timestamp = utc_now()
            for membership in departed:
                membership.is_current = False
                membership.left_at = timestamp
            conversation.counterparty_person_id = (
                counterparty_person_id if conversation.kind is ConversationKind.DIRECT else None
            )
        elif conversation.kind is not ConversationKind.DIRECT:
            conversation.counterparty_person_id = None
        conversation.metadata_json = {
            "account_id": payload.get("accountID"),
            "participants_complete": participants_complete,
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
        if conversation is None:
            raise DomainError("Beeper chat must be synced before message ingestion")
        identity = self.session.scalar(select(Identity).where(Identity.beeper_user_id == sender_id))
        if identity is None:
            sender_name = payload.get("senderName")
            identity = self._sync_identity(
                {
                    "id": sender_id,
                    "fullName": sender_name if isinstance(sender_name, str) else None,
                },
                network=conversation.network,
            )
        membership = self.session.get(
            ConversationParticipant,
            {"conversation_id": conversation.id, "identity_id": identity.id},
        )
        if membership is None:
            # A message proves historical participation, not current membership.
            self.session.add(
                ConversationParticipant(
                    conversation_id=conversation.id,
                    identity_id=identity.id,
                    is_current=False,
                    left_at=received_at or utc_now(),
                )
            )
        message_type = payload.get("type")
        is_deleted = bool(payload.get("isDeleted"))
        text_value = payload.get("text")
        supported = message_type in {"TEXT", "NOTICE"} and isinstance(text_value, str)
        if is_deleted:
            text_value = None
            supported = True
        revision_marker = payload.get("editedTimestamp") or payload.get("timestamp") or "unknown"
        legacy_body_hash = hashlib.sha256(
            (text_value or "<deleted>").encode()
        ).hexdigest()
        provider_revision_key = f"{revision_marker}:{legacy_body_hash}"
        existing_message = self.session.scalar(
            select(Message).where(
                Message.conversation_id == conversation.id,
                Message.provider_message_id == message_id,
            )
        )
        legacy_revision = (
            self.session.scalar(
                select(MessageRevision).where(
                    MessageRevision.message_id == existing_message.id,
                    MessageRevision.provider_revision_key == provider_revision_key,
                )
            )
            if existing_message is not None
            else None
        )
        immutable_hash = immutable_content_hash(
            text=text_value if isinstance(text_value, str) else None,
            is_deleted=is_deleted,
        )
        if legacy_revision is not None and (
            legacy_revision.content_hash != immutable_hash
            or legacy_revision.is_deleted != is_deleted
        ):
            # Older releases synthesized keys from timestamp plus display text,
            # which collapses missing/empty content and deletion tombstones.
            # Keep legacy keys stable unless that lossy key is already occupied.
            provider_revision_key = f"{revision_marker}:{immutable_hash}"
        arrival = _aware(received_at or utc_now())
        provider_time = payload.get("editedTimestamp") or payload.get("timestamp")
        event = InboundEvent(
            conversation_id=conversation.id,
            provider_message_id=message_id,
            sender_identity_id=identity.id,
            provider_revision_key=provider_revision_key,
            provider_sort_key=str(payload.get("sortKey")) if payload.get("sortKey") is not None else None,
            provider_event_at=(
                _optional_provider_time(provider_time)
                if provider_time is not None
                else None
            ),
            created_at=arrival,
            received_at=arrival,
            text=text_value if isinstance(text_value, str) else None,
            provider_reply_to_message_id=payload.get("linkedMessageID"),
            is_deleted=is_deleted,
            content_support=ContentSupport.SUPPORTED if supported else ContentSupport.UNSUPPORTED,
        )
        return MessageIngestor(self.session, owner_chat_id=self.owner_chat_id).ingest(event)


class BeeperDesktopAdapter:
    """Beeper Desktop REST v1 adapter; disabled unless explicitly configured."""

    SYNC_WINDOW = timedelta(days=30)
    CHAT_FEED_KEY = "chat-feed"

    def __init__(
        self,
        sessions: sessionmaker[Session],
        *,
        base_url: str = "http://127.0.0.1:23373",
        access_token: str | None,
        enabled: bool = False,
        client: httpx.Client | None = None,
        owner_chat_id: int | None = None,
    ):
        if enabled and not access_token:
            raise ValueError("enabled Beeper adapter requires an access token")
        self.sessions = sessions
        self.base_url = base_url.rstrip("/")
        self.access_token = access_token
        self.enabled = enabled
        self.client = client or httpx.Client(timeout=30)
        self.owner_chat_id = owner_chat_id
        self._blocked_scan_keys: set[str] = set()

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.access_token}"} if self.access_token else {}

    def info(self) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        response = self.client.get(f"{self.base_url}/v1/info", headers=self.headers)
        response.raise_for_status()
        return response.json()

    def list_chats(
        self,
        *,
        max_pages: int = 20,
        allow_truncated: bool = False,
    ) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        return self._list_pages(
            f"{self.base_url}/v1/chats",
            max_pages=max_pages,
            allow_truncated=allow_truncated,
        )

    def list_messages(
        self,
        chat_id: str,
        *,
        max_pages: int = 20,
        allow_truncated: bool = False,
    ) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        return self._list_pages(
            f"{self.base_url}/v1/chats/{quote(chat_id, safe='')}/messages",
            max_pages=max_pages,
            allow_truncated=allow_truncated,
        )

    def _list_pages(
        self,
        url: str,
        *,
        max_pages: int = 20,
        allow_truncated: bool = False,
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        cursor: str | None = None
        for page in range(max_pages):
            result = self._get_page(url, cursor=cursor, direction="before" if cursor else None)
            items.extend(result.items)
            if not result.has_more:
                break
            if allow_truncated and page + 1 == max_pages:
                break
            cursor = result.oldest_cursor
        else:
            raise RuntimeError("Beeper pagination exceeded the bounded poll limit")
        return items

    def _get_page(
        self,
        url: str,
        *,
        cursor: str | None,
        direction: str | None,
        require_cursors: bool = False,
    ) -> BeeperPage:
        params: dict[str, str] = {}
        if cursor is not None:
            params["cursor"] = cursor
            params["direction"] = direction or "before"
        response = self.client.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("Beeper page response must be an object")
        if "items" not in payload or not isinstance(payload["items"], list):
            raise RuntimeError("Beeper page items must be an array")
        raw_items = payload["items"]
        if any(not isinstance(item, dict) for item in raw_items):
            raise RuntimeError("Beeper page contains a malformed item")
        newest = payload.get("newestCursor")
        oldest = payload.get("oldestCursor")
        if newest is not None and (not isinstance(newest, str) or not newest):
            raise RuntimeError("Beeper newest cursor is malformed")
        if oldest is not None and (not isinstance(oldest, str) or not oldest):
            raise RuntimeError("Beeper oldest cursor is malformed")
        if "hasMore" not in payload or not isinstance(payload["hasMore"], bool):
            raise RuntimeError("Beeper page hasMore flag is missing or malformed")
        has_more = payload["hasMore"]
        # Beeper omits cursors for an empty terminal page. There is no
        # continuation to persist in that case, so retain the last committed
        # checkpoint. Any page with data, or any page that claims to have a
        # continuation, still requires both cursors so synchronization fails
        # closed rather than guessing progress.
        if require_cursors and (raw_items or has_more) and (newest is None or oldest is None):
            raise RuntimeError("Beeper synchronized page is missing required cursors")
        continuation = newest if direction == "after" else oldest
        if has_more and continuation is None:
            raise RuntimeError("Beeper pagination indicated more data without a cursor")
        return BeeperPage(
            tuple(raw_items),
            has_more,
            newest,
            oldest,
        )

    def poll_inbound(self, *, now: datetime | None = None) -> list[int]:
        """Advance durable REST synchronization by bounded provider pages."""
        if not self.enabled:
            return []
        timestamp = _aware(now or utc_now())
        revision_ids, touched = self._poll_chat_feed(timestamp)
        revision_ids.extend(
            self._advance_conversation_scan(timestamp, excluded_keys=touched)
        )
        return revision_ids

    def _poll_chat_feed(
        self, timestamp: datetime
    ) -> tuple[list[int], frozenset[str]]:
        cutoff = timestamp - self.SYNC_WINDOW
        with self.sessions() as session:
            checkpoint = session.get(BeeperSyncCheckpoint, self.CHAT_FEED_KEY)
            snapshot = self._snapshot(checkpoint)
        if checkpoint is None:
            cursor = None
            direction = None
            bootstrap = True
        elif not checkpoint.bootstrap_complete:
            cursor = checkpoint.backfill_cursor
            direction = "before"
            bootstrap = True
        else:
            cursor = checkpoint.newest_cursor
            direction = "after" if cursor is not None else None
            bootstrap = False
        chat_page = self._get_page(
            f"{self.base_url}/v1/chats",
            cursor=cursor,
            direction=direction,
            require_cursors=True,
        )
        eligible_chats = tuple(
            chat
            for chat in chat_page.items
            if not bootstrap or self._at_or_after(chat.get("lastActivity"), cutoff)
        )
        message_requests: list[
            tuple[
                dict[str, Any],
                str,
                str | None,
                str | None,
                _CheckpointSnapshot | None,
                bool,
            ]
        ] = []
        seen_provider_ids: set[str] = set()
        with self.sessions() as session:
            for chat in eligible_chats:
                provider_id = chat.get("id")
                if not isinstance(provider_id, str) or not provider_id:
                    raise RuntimeError("Beeper chat page contains a chat without a stable id")
                if provider_id in seen_provider_ids:
                    continue
                seen_provider_ids.add(provider_id)
                conversation = session.scalar(
                    select(Conversation).where(
                        Conversation.beeper_conversation_id == provider_id
                    )
                )
                message_checkpoint = (
                    session.get(
                        BeeperSyncCheckpoint,
                        self._conversation_checkpoint_key(conversation.id),
                    )
                    if conversation is not None
                    else None
                )
                message_snapshot = self._snapshot(message_checkpoint)
                if message_checkpoint is None:
                    message_cursor = None
                    message_direction = None
                    message_bootstrap = True
                elif not message_checkpoint.bootstrap_complete:
                    message_cursor = message_checkpoint.backfill_cursor
                    message_direction = "before" if message_cursor is not None else None
                    message_bootstrap = True
                else:
                    message_cursor = message_checkpoint.newest_cursor
                    message_direction = "after" if message_cursor is not None else None
                    message_bootstrap = False
                message_requests.append(
                    (
                        chat,
                        provider_id,
                        message_cursor,
                        message_direction,
                        message_snapshot,
                        message_bootstrap,
                    )
                )
        fetched_messages: list[
            tuple[dict[str, Any], BeeperPage, _CheckpointSnapshot | None, bool]
        ] = []
        for (
            chat,
            provider_id,
            message_cursor,
            message_direction,
            message_snapshot,
            message_bootstrap,
        ) in message_requests:
            message_page = self._get_page(
                f"{self.base_url}/v1/chats/{quote(provider_id, safe='')}/messages",
                cursor=message_cursor,
                direction=message_direction,
                require_cursors=True,
            )
            fetched_messages.append(
                (chat, message_page, message_snapshot, message_bootstrap)
            )

        revision_ids: list[int] = []
        touched_keys: set[str] = set()
        with self.sessions.begin() as session:
            current_feed = session.get(BeeperSyncCheckpoint, self.CHAT_FEED_KEY)
            self._require_snapshot(current_feed, snapshot)
            sync = BeeperSyncService(session, owner_chat_id=self.owner_chat_id)
            for chat, message_page, message_snapshot, message_bootstrap in fetched_messages:
                conversation = sync.sync_chat(chat)
                key = self._conversation_checkpoint_key(conversation.id)
                touched_keys.add(key)
                current_message_checkpoint = session.get(BeeperSyncCheckpoint, key)
                self._require_snapshot(current_message_checkpoint, message_snapshot)
                message_cutoff = cutoff
                messages = (
                    item
                    for item in message_page.items
                    if not message_bootstrap
                    or self._at_or_after(
                        item.get("editedTimestamp") or item.get("timestamp"),
                        message_cutoff,
                    )
                )
                revision_ids.extend(
                    self._ingest_messages(sync, conversation, messages)
                )
                self._advance_message_checkpoint(
                    session,
                    current_message_checkpoint,
                    conversation=conversation,
                    page=message_page,
                    cutoff=message_cutoff,
                    timestamp=timestamp,
                    bootstrap=message_bootstrap,
                    provider_activity_at=self._optional_time(chat.get("lastActivity")),
                )
            self._advance_feed_checkpoint(
                session,
                current_feed,
                page=chat_page,
                cutoff=cutoff,
                timestamp=timestamp,
                bootstrap=bootstrap,
            )
        return revision_ids, frozenset(touched_keys)

    def _advance_conversation_scan(
        self,
        timestamp: datetime,
        *,
        excluded_keys: frozenset[str],
    ) -> list[int]:
        excluded_scan_keys = excluded_keys | self._blocked_scan_keys
        not_touched = (
            BeeperSyncCheckpoint.checkpoint_key.not_in(excluded_scan_keys)
            if excluded_scan_keys
            else True
        )
        with self.sessions() as session:
            checkpoint = session.scalar(
                select(BeeperSyncCheckpoint)
                .where(
                    BeeperSyncCheckpoint.scope == "CONVERSATION",
                    BeeperSyncCheckpoint.bootstrap_complete.is_(False),
                    not_touched,
                )
                .order_by(BeeperSyncCheckpoint.created_at, BeeperSyncCheckpoint.checkpoint_key)
                .limit(1)
            )
            scan_kind = "bootstrap"
            if checkpoint is None:
                checkpoint = session.scalar(
                    select(BeeperSyncCheckpoint)
                    .where(
                        BeeperSyncCheckpoint.scope == "CONVERSATION",
                        BeeperSyncCheckpoint.reconciliation_cursor.is_not(None),
                        not_touched,
                    )
                    .order_by(BeeperSyncCheckpoint.updated_at, BeeperSyncCheckpoint.checkpoint_key)
                    .limit(1)
                )
                scan_kind = "reconcile"
            if checkpoint is None:
                checkpoint = session.scalar(
                    select(BeeperSyncCheckpoint)
                    .where(
                        BeeperSyncCheckpoint.scope == "CONVERSATION",
                        not_touched,
                    )
                    .order_by(
                        BeeperSyncCheckpoint.last_reconciled_at.asc().nulls_first(),
                        BeeperSyncCheckpoint.checkpoint_key,
                    )
                    .limit(1)
                )
                scan_kind = "reconcile"
            if checkpoint is None:
                return []
            snapshot = self._snapshot(checkpoint)
            conversation = session.get(Conversation, checkpoint.conversation_id)
            if conversation is None:
                raise RuntimeError("Beeper checkpoint conversation is unavailable")
            provider_id = conversation.beeper_conversation_id
            if scan_kind == "bootstrap":
                cursor = checkpoint.backfill_cursor
                cutoff = _aware(checkpoint.bootstrap_cutoff_at)
            else:
                cursor = checkpoint.reconciliation_cursor
                cutoff = _aware(
                    checkpoint.reconciliation_cutoff_at or timestamp - self.SYNC_WINDOW
                )
        try:
            page = self._get_page(
                f"{self.base_url}/v1/chats/{quote(provider_id, safe='')}/messages",
                cursor=cursor,
                direction="before" if cursor is not None else None,
                require_cursors=True,
            )
        except RuntimeError as exc:
            if str(exc) != "Beeper synchronized page is missing required cursors":
                raise
            if snapshot.checkpoint_key not in self._blocked_scan_keys:
                LOGGER.warning(
                    "Beeper conversation scan is blocked by missing cursors; "
                    "retaining checkpoint %s",
                    snapshot.checkpoint_key,
                )
                self._blocked_scan_keys.add(snapshot.checkpoint_key)
            return []
        revision_ids: list[int] = []
        with self.sessions.begin() as session:
            current = session.get(BeeperSyncCheckpoint, snapshot.checkpoint_key)
            self._require_snapshot(current, snapshot)
            assert current is not None
            conversation = session.get(Conversation, current.conversation_id)
            if conversation is None:
                raise RuntimeError("Beeper checkpoint conversation is unavailable")
            sync = BeeperSyncService(session, owner_chat_id=self.owner_chat_id)
            revision_ids.extend(
                self._ingest_messages(
                    sync,
                    conversation,
                    (
                        item
                        for item in page.items
                        if self._at_or_after(
                            item.get("editedTimestamp") or item.get("timestamp"),
                            cutoff,
                        )
                    ),
                )
            )
            reached_cutoff = self._page_reaches_cutoff(
                page.items,
                cutoff,
                fields=("editedTimestamp", "timestamp"),
            )
            if cursor is None:
                current.newest_cursor = page.newest_cursor or current.newest_cursor
            if scan_kind == "bootstrap":
                complete = not page.has_more or reached_cutoff
                current.bootstrap_complete = complete
                current.backfill_cursor = None if complete else page.oldest_cursor
            else:
                complete = not page.has_more or reached_cutoff
                if complete:
                    current.reconciliation_cursor = None
                    current.reconciliation_cutoff_at = None
                    current.last_reconciled_at = timestamp
                else:
                    current.reconciliation_cursor = page.oldest_cursor
                    current.reconciliation_cutoff_at = cutoff
            current.updated_at = timestamp
        return revision_ids

    @staticmethod
    def _conversation_checkpoint_key(conversation_id: int) -> str:
        return f"conversation:{conversation_id}"

    @staticmethod
    def _snapshot(
        checkpoint: BeeperSyncCheckpoint | None,
    ) -> _CheckpointSnapshot | None:
        if checkpoint is None:
            return None
        return _CheckpointSnapshot(
            checkpoint.checkpoint_key,
            checkpoint.newest_cursor,
            checkpoint.backfill_cursor,
            checkpoint.bootstrap_cutoff_at,
            checkpoint.bootstrap_complete,
            checkpoint.reconciliation_cursor,
            checkpoint.reconciliation_cutoff_at,
            checkpoint.last_reconciled_at,
            checkpoint.updated_at,
        )

    @classmethod
    def _require_snapshot(
        cls,
        checkpoint: BeeperSyncCheckpoint | None,
        expected: _CheckpointSnapshot | None,
    ) -> None:
        if cls._snapshot(checkpoint) != expected:
            raise RuntimeError("Beeper synchronization checkpoint changed concurrently")

    @staticmethod
    def _optional_time(value: object) -> datetime | None:
        if value is None:
            return None
        return _aware(_parse_time(value))

    @classmethod
    def _at_or_after(cls, value: object, cutoff: datetime) -> bool:
        parsed = cls._optional_time(value)
        return parsed is None or parsed >= cutoff

    @classmethod
    def _page_reaches_cutoff(
        cls,
        items: tuple[dict[str, Any], ...],
        cutoff: datetime,
        *,
        fields: tuple[str, ...],
    ) -> bool:
        for item in items:
            value = next((item.get(field) for field in fields if item.get(field)), None)
            parsed = cls._optional_time(value)
            if parsed is not None and parsed < cutoff:
                return True
        return False

    @staticmethod
    def _ingest_messages(
        sync: BeeperSyncService,
        conversation: Conversation,
        messages: Any,
    ) -> list[int]:
        revision_ids: list[int] = []
        ordered = sorted(
            messages,
            key=lambda item: str(item.get("sortKey") or item.get("timestamp") or ""),
        )
        for item in ordered:
            if item.get("isSender") is True:
                continue
            payload = dict(item)
            payload.setdefault("chatID", conversation.beeper_conversation_id)
            result = sync.ingest_message(payload)
            revision_ids.append(result.revision_id)
        return revision_ids

    @classmethod
    def _advance_message_checkpoint(
        cls,
        session: Session,
        checkpoint: BeeperSyncCheckpoint | None,
        *,
        conversation: Conversation,
        page: BeeperPage,
        cutoff: datetime,
        timestamp: datetime,
        bootstrap: bool,
        provider_activity_at: datetime | None,
    ) -> BeeperSyncCheckpoint:
        reached_cutoff = cls._page_reaches_cutoff(
            page.items,
            cutoff,
            fields=("editedTimestamp", "timestamp"),
        )
        if checkpoint is None:
            complete = not page.has_more or reached_cutoff
            checkpoint = BeeperSyncCheckpoint(
                checkpoint_key=cls._conversation_checkpoint_key(conversation.id),
                scope="CONVERSATION",
                conversation_id=conversation.id,
                newest_cursor=page.newest_cursor,
                backfill_cursor=None if complete else page.oldest_cursor,
                bootstrap_cutoff_at=cutoff,
                bootstrap_complete=complete,
                provider_activity_at=provider_activity_at,
                created_at=timestamp,
                updated_at=timestamp,
            )
            session.add(checkpoint)
            return checkpoint
        if not bootstrap:
            checkpoint.newest_cursor = page.newest_cursor or checkpoint.newest_cursor
        if bootstrap:
            complete = not page.has_more or reached_cutoff
            checkpoint.bootstrap_complete = complete
            checkpoint.backfill_cursor = None if complete else page.oldest_cursor
        checkpoint.provider_activity_at = provider_activity_at or checkpoint.provider_activity_at
        checkpoint.updated_at = timestamp
        return checkpoint

    @classmethod
    def _advance_feed_checkpoint(
        cls,
        session: Session,
        checkpoint: BeeperSyncCheckpoint | None,
        *,
        page: BeeperPage,
        cutoff: datetime,
        timestamp: datetime,
        bootstrap: bool,
    ) -> BeeperSyncCheckpoint:
        reached_cutoff = cls._page_reaches_cutoff(
            page.items,
            cutoff,
            fields=("lastActivity",),
        )
        if checkpoint is None:
            complete = not page.has_more or reached_cutoff
            checkpoint = BeeperSyncCheckpoint(
                checkpoint_key=cls.CHAT_FEED_KEY,
                scope="CHAT_FEED",
                newest_cursor=page.newest_cursor,
                backfill_cursor=None if complete else page.oldest_cursor,
                bootstrap_cutoff_at=cutoff,
                bootstrap_complete=complete,
                created_at=timestamp,
                updated_at=timestamp,
            )
            session.add(checkpoint)
            return checkpoint
        if not bootstrap:
            checkpoint.newest_cursor = page.newest_cursor or checkpoint.newest_cursor
        if bootstrap:
            complete = not page.has_more or reached_cutoff
            checkpoint.bootstrap_complete = complete
            checkpoint.backfill_cursor = None if complete else page.oldest_cursor
        checkpoint.updated_at = timestamp
        return checkpoint

    def send(self, request: DeliveryRequest) -> DeliveryResult:
        if request.transport is not Transport.BEEPER:
            return DeliveryResult(False, False, definitely_not_sent=True, error="wrong transport")
        if not self.enabled:
            return DeliveryResult(False, False, definitely_not_sent=True, error="Beeper transport disabled")
        try:
            chat_id = self._provider_chat_id(request.destination)
        except DomainError as exc:
            return DeliveryResult(False, False, definitely_not_sent=True, error=str(exc))
        try:
            response = self.client.post(
                f"{self.base_url}/v1/chats/{quote(chat_id, safe='')}/messages",
                headers=self.headers,
                json={"text": request.text},
            )
        except Exception as exc:
            return DeliveryResult(False, True, error=str(exc))
        if response.status_code in {401, 403}:
            try:
                rejection = response.json()
                error = str(rejection.get("error") or response.status_code)
            except Exception:
                error = str(response.status_code)
            return DeliveryResult(
                False,
                False,
                definitely_not_sent=True,
                error=error,
            )
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
        if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
            return None
        started_at = self._attempt_started_at(request.outbox_id)
        matches: list[str] = []
        for item in payload["items"]:
            if not isinstance(item, dict):
                return None
            if item.get("isSender") is not True or item.get("text") != request.text:
                continue
            timestamp = _optional_provider_time(item.get("timestamp"))
            if timestamp is None:
                continue
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
            return _aware(attempt.started_at)
