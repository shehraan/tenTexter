from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import httpx
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from ten_texter.domain import DomainError
from ten_texter.enums import TelegramUpdateStatus, Transport
from ten_texter.models import (
    DecisionRequestPrompt,
    OutboxDeliveryAttempt,
    TelegramUpdate,
)
from ten_texter.outbox import DeliveryRequest, DeliveryResult


class OwnerTaskParser(Protocol):
    def parse(self, text: str) -> object: ...


class OwnerCommandHandler(Protocol):
    def prepare_command(self, parsed: object, update: TelegramUpdate) -> object: ...

    def apply_command(self, session: Session, parsed: object, update: TelegramUpdate) -> None: ...

    def prepare_decision(self, decision_id: int, payload: dict[str, Any], update: TelegramUpdate) -> object: ...

    def apply_decision(self, session: Session, decision_id: int, payload: dict[str, Any], update: TelegramUpdate) -> None: ...


@dataclass(frozen=True, slots=True)
class ReceivedUpdate:
    outcome: str
    telegram_update_row_id: int | None = None


def _extract(raw: dict[str, Any]) -> tuple[int | None, int | None, str | None, str | None, int | None]:
    callback = raw.get("callback_query")
    if isinstance(callback, dict):
        sender = callback.get("from", {}).get("id")
        message = callback.get("message") or {}
        chat = message.get("chat") or {}
        return sender, chat.get("id"), chat.get("type"), callback.get("data"), None
    message = raw.get("message")
    if isinstance(message, dict):
        sender = message.get("from", {}).get("id")
        chat = message.get("chat") or {}
        reply = message.get("reply_to_message") or {}
        return sender, chat.get("id"), chat.get("type"), message.get("text"), reply.get("message_id")
    return None, None, None, None, None


class TelegramControlGateway:
    def __init__(
        self,
        sessions: sessionmaker[Session],
        *,
        owner_id: int,
        parser: OwnerTaskParser,
        handler: OwnerCommandHandler,
        require_private_chat: bool = True,
    ):
        self.sessions = sessions
        self.owner_id = owner_id
        self.parser = parser
        self.handler = handler
        self.require_private_chat = require_private_chat

    def receive(self, raw: dict[str, Any]) -> ReceivedUpdate:
        update_id = raw.get("update_id")
        sender_id, chat_id, chat_type, _text, _reply = _extract(raw)
        if not isinstance(update_id, int) or sender_id != self.owner_id:
            return ReceivedUpdate("UNAUTHORIZED")
        if self.require_private_chat and chat_type != "private":
            return ReceivedUpdate("UNAUTHORIZED")
        if not isinstance(chat_id, int):
            return ReceivedUpdate("UNSUPPORTED")
        with self.sessions.begin() as session:
            existing = session.scalar(
                select(TelegramUpdate).where(TelegramUpdate.telegram_update_id == update_id)
            )
            if existing is not None:
                return ReceivedUpdate("DUPLICATE", existing.id)
            row = TelegramUpdate(
                telegram_update_id=update_id,
                sender_user_id=sender_id,
                chat_id=chat_id,
                payload_json=raw,
                status=TelegramUpdateStatus.PENDING,
            )
            session.add(row)
            session.flush()
            return ReceivedUpdate("PERSISTED", row.id)

    def process(self, row_id: int) -> TelegramUpdateStatus:
        with self.sessions() as read_session:
            update = read_session.get(TelegramUpdate, row_id)
            if update is None:
                raise DomainError("Telegram update not found")
            if update.status is not TelegramUpdateStatus.PENDING:
                return update.status
            raw = dict(update.payload_json)
            _sender, _chat, _type, text, reply_to = _extract(raw)
            decision_id = self._resolve_decision(read_session, raw, reply_to)
        parsed: object | None = None
        prepared: object | None = None
        if decision_id is None:
            if not isinstance(text, str) or not text.strip():
                raise DomainError("owner update has no supported instruction")
            parsed = self.parser.parse(text)
            prepare = getattr(self.handler, "prepare_command", None)
            prepared = prepare(parsed, update) if prepare is not None else parsed
        else:
            prepare_decision = getattr(self.handler, "prepare_decision", None)
            prepared = (
                prepare_decision(decision_id, raw, update)
                if prepare_decision is not None
                else raw
            )
        with self.sessions.begin() as session:
            update = session.get(TelegramUpdate, row_id)
            if update is None:
                raise DomainError("Telegram update not found")
            if update.status is not TelegramUpdateStatus.PENDING:
                return update.status
            if decision_id is not None:
                self.handler.apply_decision(session, decision_id, prepared, update)
            else:
                self.handler.apply_command(session, prepared, update)
            update.status = TelegramUpdateStatus.PROCESSED
            update.error_details = None
            session.flush()
            return update.status

    @staticmethod
    def _resolve_decision(
        session: Session,
        raw: dict[str, Any],
        reply_to_message_id: int | None,
    ) -> int | None:
        callback = raw.get("callback_query")
        if isinstance(callback, dict):
            data = callback.get("data")
            if isinstance(data, str) and data.startswith("decision:"):
                parts = data.split(":", 2)
                if len(parts) >= 2 and parts[1].isdigit():
                    return int(parts[1])
        if reply_to_message_id is None:
            return None
        return session.scalar(
            select(DecisionRequestPrompt.decision_request_id)
            .join(
                OutboxDeliveryAttempt,
                OutboxDeliveryAttempt.outbox_message_id == DecisionRequestPrompt.outbox_message_id,
            )
            .where(OutboxDeliveryAttempt.provider_message_id == str(reply_to_message_id))
        )


class TelegramBotAdapter:
    """HTTP Bot API boundary. It is inert unless explicitly enabled."""

    def __init__(
        self,
        *,
        token: str | None,
        enabled: bool = False,
        client: httpx.Client | None = None,
    ):
        if enabled and not token:
            raise ValueError("enabled Telegram adapter requires a bot token")
        self.token = token
        self.enabled = enabled
        self.client = client or httpx.Client(timeout=35)

    @property
    def base_url(self) -> str:
        if not self.token:
            return "https://api.telegram.org/bot-disabled"
        return f"https://api.telegram.org/bot{self.token}"

    def poll(self, *, offset: int | None = None, timeout: int = 30) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        params: dict[str, object] = {"timeout": timeout, "allowed_updates": ["message", "callback_query"]}
        if offset is not None:
            params["offset"] = offset
        response = self.client.get(f"{self.base_url}/getUpdates", params=params)
        response.raise_for_status()
        payload = response.json()
        if payload.get("ok") is not True or not isinstance(payload.get("result"), list):
            raise RuntimeError("invalid Telegram getUpdates response")
        return payload["result"]

    def send(self, request: DeliveryRequest) -> DeliveryResult:
        if request.transport is not Transport.TELEGRAM:
            return DeliveryResult(False, False, definitely_not_sent=True, error="wrong transport")
        if not self.enabled:
            return DeliveryResult(False, False, definitely_not_sent=True, error="Telegram transport disabled")
        try:
            response = self.client.post(
                f"{self.base_url}/sendMessage",
                json={"chat_id": int(request.destination), "text": request.text},
            )
        except Exception as exc:
            return DeliveryResult(False, True, error=str(exc))
        try:
            payload = response.json()
        except Exception as exc:
            return DeliveryResult(False, True, error=f"invalid Telegram response: {exc}")
        if response.is_success and payload.get("ok") is True:
            result = payload.get("result") or {}
            message_id = result.get("message_id")
            return DeliveryResult(True, True, provider_message_id=str(message_id))
        return DeliveryResult(
            False,
            True,
            definitely_not_sent=payload.get("ok") is False,
            error=str(payload.get("description") or response.status_code),
        )

    def reconcile(
        self,
        _request: DeliveryRequest,
        *,
        provider_message_id: str | None,
        pending_provider_id: str | None,
    ) -> str | None:
        # Bot API has no general exact-history search endpoint. A known final ID is strongest evidence.
        return provider_message_id
