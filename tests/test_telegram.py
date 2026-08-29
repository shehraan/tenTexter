from __future__ import annotations

from sqlalchemy import func, select
import pytest
from sqlalchemy.orm import Session, sessionmaker

from ten_texter.domain import DecisionService
from ten_texter.enums import (
    AttemptResult,
    MessageKind,
    OutboxStatus,
    TelegramUpdateStatus,
    Transport,
)
from ten_texter.models import (
    DecisionRequestPrompt,
    OutboxDeliveryAttempt,
    OutboxMessage,
    Person,
    TelegramUpdate,
)
from ten_texter.outbox import DeliveryRequest, OutboxService
from ten_texter.policy import DatabaseContextProvider
from ten_texter.telegram import TelegramBotAdapter, TelegramControlGateway
from ten_texter.runtime import AgentRuntime
from ten_texter.validator import DatabaseValidatorContextProvider
from tests.test_schema import NOW, seed_core


class Parser:
    def __init__(self):
        self.calls: list[str] = []

    def parse(self, text: str) -> object:
        self.calls.append(text)
        return {"instruction": text}


class Handler:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.commands = 0
        self.decisions: list[int] = []

    def apply_command(self, session: Session, parsed: object, update: TelegramUpdate) -> None:
        session.add(Person(display_name=f"command:{update.telegram_update_id}", metadata_json={}))
        self.commands += 1
        if self.fail:
            raise RuntimeError("crash")

    def apply_decision(self, _session: Session, decision_id: int, _payload: object, _update: TelegramUpdate) -> None:
        self.decisions.append(decision_id)


def gateway(session: Session, parser: Parser, handler: Handler) -> TelegramControlGateway:
    factory = sessionmaker(bind=session.bind, expire_on_commit=False, autoflush=False)
    return TelegramControlGateway(factory, owner_id=7, parser=parser, handler=handler)


def raw(update_id: int, *, sender: int = 7, text: str = "schedule tennis") -> dict[str, object]:
    return {
        "update_id": update_id,
        "message": {
            "message_id": update_id,
            "from": {"id": sender},
            "chat": {"id": 99, "type": "private"},
            "text": text,
        },
    }


def test_unauthorized_sender_never_reaches_parser_or_domain(db_session: Session) -> None:
    parser = Parser()
    handler = Handler()
    result = gateway(db_session, parser, handler).receive(raw(1, sender=8))
    assert result.outcome == "UNAUTHORIZED"
    assert parser.calls == []
    assert handler.commands == 0
    assert db_session.scalar(select(func.count(TelegramUpdate.id))) == 0


def test_receipt_is_durable_before_effects_and_resumes_from_pending(db_session: Session) -> None:
    parser = Parser()
    handler = Handler()
    control = gateway(db_session, parser, handler)
    received = control.receive(raw(2))
    assert received.outcome == "PERSISTED"
    assert parser.calls == []
    db_session.expire_all()
    assert db_session.get(TelegramUpdate, received.telegram_update_row_id).status is TelegramUpdateStatus.PENDING
    assert control.process(received.telegram_update_row_id) is TelegramUpdateStatus.PROCESSED
    assert parser.calls == ["schedule tennis"]
    assert handler.commands == 1


def test_duplicate_update_has_no_duplicate_effect(db_session: Session) -> None:
    parser = Parser()
    handler = Handler()
    control = gateway(db_session, parser, handler)
    first = control.receive(raw(3))
    control.process(first.telegram_update_row_id)
    duplicate = control.receive(raw(3))
    assert duplicate.outcome == "DUPLICATE"
    assert control.process(duplicate.telegram_update_row_id) is TelegramUpdateStatus.PROCESSED
    assert handler.commands == 1


def test_effect_and_processed_status_rollback_together(db_session: Session) -> None:
    parser = Parser()
    handler = Handler(fail=True)
    control = gateway(db_session, parser, handler)
    received = control.receive(raw(4))
    try:
        control.process(received.telegram_update_row_id)
    except RuntimeError:
        pass
    db_session.expire_all()
    assert db_session.get(TelegramUpdate, received.telegram_update_row_id).status is TelegramUpdateStatus.PENDING
    assert db_session.scalar(select(func.count(Person.id)).where(Person.display_name == "command:4")) == 0


def test_runtime_failure_notifies_owner_once_and_keeps_update_pending(
    db_session: Session,
) -> None:
    parser = Parser()
    handler = Handler(fail=True)
    control = gateway(db_session, parser, handler)
    received = control.receive(raw(40))
    runtime = AgentRuntime.__new__(AgentRuntime)
    runtime.sessions = sessionmaker(
        bind=db_session.bind, expire_on_commit=False, autoflush=False
    )
    runtime.control = control
    runtime.owner_chat_id = 99

    class NoUpdates:
        def poll(self, **_: object) -> list[dict[str, object]]:
            return []

    runtime.telegram = NoUpdates()

    with pytest.raises(RuntimeError, match="crash"):
        runtime._poll_telegram()
    with pytest.raises(RuntimeError, match="crash"):
        runtime._poll_telegram()

    db_session.expire_all()
    update = db_session.get(TelegramUpdate, received.telegram_update_row_id)
    assert update.status is TelegramUpdateStatus.PENDING
    notices = list(
        db_session.scalars(
            select(OutboxMessage).where(
                OutboxMessage.idempotency_key
                == f"telegram-update:{update.id}:processing-failed"
            )
        )
    )
    assert len(notices) == 1
    assert notices[0].final_text == (
        "Telegram command 40 could not be processed and remains pending for retry."
    )
    context = DatabaseValidatorContextProvider(
        runtime.sessions,
        facts=DatabaseContextProvider(),
        owner_chat_id=99,
    ).context_for(notices[0].id, notices[0].message_kind)
    assert context.allowed_claims == (notices[0].final_text,)


def test_explicit_callback_decision_id_has_priority(db_session: Session) -> None:
    core = seed_core(db_session)
    decision = DecisionService(db_session).create(
        decision_type="TEST",
        subject_kind="message_revision",
        subject_id=core["revision"].id,
        context={},
    )
    db_session.commit()
    parser = Parser()
    handler = Handler()
    control = gateway(db_session, parser, handler)
    callback = {
        "update_id": 5,
        "callback_query": {
            "id": "callback",
            "from": {"id": 7},
            "message": {"message_id": 10, "chat": {"id": 99, "type": "private"}},
            "data": f"decision:{decision.id}:approve",
        },
    }
    received = control.receive(callback)
    control.process(received.telegram_update_row_id)
    assert handler.decisions == [decision.id]
    assert parser.calls == []


def test_reply_to_owner_outbox_resolves_prompt_without_guess(db_session: Session) -> None:
    core = seed_core(db_session)
    decision = DecisionService(db_session).create(
        decision_type="TEST",
        subject_kind="message_revision",
        subject_id=core["revision"].id,
        context={},
    )
    prompt = OutboxService(db_session).create_owner(
        telegram_chat_id=99,
        final_text="Choose",
        message_kind=MessageKind.NOTIFICATION,
        idempotency_key="decision-prompt",
    )
    prompt.status = OutboxStatus.SENT
    db_session.add_all(
        [
            DecisionRequestPrompt(decision_request_id=decision.id, outbox_message_id=prompt.id),
            OutboxDeliveryAttempt(
                outbox_message_id=prompt.id,
                started_at=NOW,
                finished_at=NOW,
                result=AttemptResult.SUCCESS,
                provider_message_id="42",
            ),
        ]
    )
    db_session.commit()
    parser = Parser()
    handler = Handler()
    control = gateway(db_session, parser, handler)
    reply = raw(6, text="approve")
    reply["message"]["reply_to_message"] = {"message_id": 42}  # type: ignore[index]
    received = control.receive(reply)
    control.process(received.telegram_update_row_id)
    assert handler.decisions == [decision.id]


class Response:
    is_success = True
    status_code = 200

    def json(self) -> dict[str, object]:
        return {"ok": True, "result": {"message_id": 77}}


class Client:
    def __init__(self):
        self.posts = 0

    def post(self, *_: object, **__: object) -> Response:
        self.posts += 1
        return Response()


def test_telegram_transport_is_disabled_by_default_and_uses_outbox_request(db_session: Session) -> None:
    request = DeliveryRequest(1, Transport.TELEGRAM, "99", "hello", "owner:1")
    disabled = TelegramBotAdapter(token=None)
    result = disabled.send(request)
    assert not result.success and result.definitely_not_sent
    client = Client()
    enabled = TelegramBotAdapter(token="fake-token", enabled=True, client=client)  # type: ignore[arg-type]
    result = enabled.send(request)
    assert result.success and result.provider_message_id == "77"
    assert client.posts == 1
