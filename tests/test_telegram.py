from __future__ import annotations

import logging

import httpx
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
from ten_texter.logging import configure_logging
from ten_texter.outbox import (
    AllowingRevalidator,
    DeliveryRequest,
    OutboxService,
    OutboxWorker,
)
from ten_texter.policy import DatabaseContextProvider
from ten_texter.telegram import (
    TelegramBotAdapter,
    TelegramControlGateway,
    TelegramTransportError,
)
from ten_texter.runtime import AgentRuntime
from ten_texter.validator import DatabaseValidatorContextProvider
from tests.test_schema import NOW, seed_core


SECRET_TOKEN = "telegram-secret-sentinel"


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
    assert result.outcome == "FAILED"
    assert parser.calls == []
    assert handler.commands == 0
    assert db_session.scalar(select(func.count(TelegramUpdate.id))) == 1
    update = db_session.scalar(select(TelegramUpdate))
    assert update is not None
    assert update.status is TelegramUpdateStatus.FAILED
    assert update.error_details == "unauthorized sender"


def test_structurally_unsupported_update_is_durable_and_does_not_reach_parser(
    db_session: Session,
) -> None:
    parser = Parser()
    handler = Handler()
    malformed = raw(2)
    malformed["message"]["chat"] = {"type": "private"}  # type: ignore[index]

    result = gateway(db_session, parser, handler).receive(malformed)

    assert result.outcome == "FAILED"
    update = db_session.get(TelegramUpdate, result.telegram_update_row_id)
    assert update is not None
    assert update.status is TelegramUpdateStatus.FAILED
    assert update.chat_id is None
    assert update.error_details == "update has no supported chat"
    assert parser.calls == []
    assert handler.commands == 0


def test_boolean_sender_id_cannot_match_numeric_owner_id(db_session: Session) -> None:
    parser = Parser()
    handler = Handler()
    malformed = raw(4, sender=True)

    result = gateway(db_session, parser, handler).receive(malformed)

    assert result.outcome == "FAILED"
    update = db_session.get(TelegramUpdate, result.telegram_update_row_id)
    assert update is not None
    assert update.status is TelegramUpdateStatus.FAILED
    assert update.error_details == "unauthorized sender"
    assert parser.calls == []
    assert handler.commands == 0


def test_failed_telegram_update_advances_runtime_offset(db_session: Session) -> None:
    parser = Parser()
    handler = Handler()
    control = gateway(db_session, parser, handler)
    rejected = control.receive(raw(3, sender=8))
    assert rejected.outcome == "FAILED"

    class RecordingTelegram:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def poll(self, **kwargs: object) -> list[dict[str, object]]:
            self.calls.append(kwargs)
            return []

    telegram = RecordingTelegram()
    runtime = AgentRuntime.__new__(AgentRuntime)
    runtime.sessions = sessionmaker(
        bind=db_session.bind, expire_on_commit=False, autoflush=False
    )
    runtime.telegram = telegram
    runtime.control = control
    runtime.owner_chat_id = 99

    runtime._poll_telegram()

    assert telegram.calls == [{"offset": 4, "timeout": 0}]


def test_telegram_poll_rejects_update_without_stable_id() -> None:
    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True, "result": [{"message": {}}]})

    adapter = TelegramBotAdapter(
        token="fake-token",
        enabled=True,
        client=httpx.Client(transport=httpx.MockTransport(respond)),
    )

    with pytest.raises(TelegramTransportError, match="invalid update item"):
        adapter.poll(timeout=0)


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


def test_telegram_http_request_logs_never_include_bot_token(caplog: pytest.LogCaptureFixture) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/getUpdates"):
            return httpx.Response(200, json={"ok": True, "result": []})
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 77}})

    httpx_logger = logging.getLogger("httpx")
    httpcore_logger = logging.getLogger("httpcore")
    original_httpx_level = httpx_logger.level
    original_httpcore_level = httpcore_logger.level
    try:
        httpx_logger.setLevel(logging.NOTSET)
        httpcore_logger.setLevel(logging.NOTSET)
        caplog.set_level(logging.INFO)
        configure_logging("INFO")
        adapter = TelegramBotAdapter(
            token=SECRET_TOKEN,
            enabled=True,
            client=httpx.Client(transport=httpx.MockTransport(respond)),
        )

        assert adapter.poll(timeout=0) == []
        result = adapter.send(
            DeliveryRequest(1, Transport.TELEGRAM, "99", "hello", "owner:logs")
        )

        assert result.success
        assert SECRET_TOKEN not in caplog.text
    finally:
        httpx_logger.setLevel(original_httpx_level)
        httpcore_logger.setLevel(original_httpcore_level)


def test_telegram_poll_http_error_is_safe_for_runtime_and_logs(
    db_session: Session,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def reject(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"ok": False, "error_code": 401, "description": "Unauthorized"},
        )

    adapter = TelegramBotAdapter(
        token=SECRET_TOKEN,
        enabled=True,
        client=httpx.Client(transport=httpx.MockTransport(reject)),
    )
    runtime = AgentRuntime.__new__(AgentRuntime)
    runtime.sessions = sessionmaker(
        bind=db_session.bind, expire_on_commit=False, autoflush=False
    )
    runtime.telegram = adapter
    runtime.control = object()
    runtime.owner_chat_id = 99
    runtime.outbox = object()
    health_events: list[tuple[str, bool, str]] = []
    runtime.health = type(
        "HealthRecorder",
        (),
        {
            "record": lambda self, session, dependency, *, healthy, details="": (
                health_events.append((dependency, healthy, details))
            )
        },
    )()
    runtime._poll_beeper = lambda: None
    runtime._process_revisions = lambda: None
    runtime._sweep_tasks = lambda _timestamp: None
    runtime._poll_recurrence = lambda _timestamp: None
    runtime._initialize_recurring_tasks = lambda: None
    runtime._poll_triggers = lambda _timestamp: None
    runtime._recovery_pass = False

    caplog.set_level(logging.ERROR)
    tick = runtime.run_once(now=NOW)

    assert set(tick.errors) == {"telegram"}
    surfaced = tick.errors["telegram"]
    assert "Telegram getUpdates failed" in surfaced
    assert "HTTP 401" in surfaced
    assert "code 401" in surfaced
    assert "Unauthorized" in surfaced
    assert SECRET_TOKEN not in surfaced
    assert SECRET_TOKEN not in caplog.text
    assert ("telegram", False, "TelegramTransportError") in health_events
    assert all(SECRET_TOKEN not in details for _, _, details in health_events)


class ValidatingEverything:
    def validate(self, **_: object) -> bool:
        return True


class RaisingTelegramClient:
    def post(self, url: str, **_: object) -> object:
        request = httpx.Request("POST", url)
        response = httpx.Response(
            403,
            request=request,
            json={"ok": False, "error_code": 403, "description": "Forbidden"},
        )
        raise httpx.HTTPStatusError(
            f"Telegram rejected request at {url}",
            request=request,
            response=response,
        )


def test_telegram_send_exception_is_safe_in_delivery_result_and_persisted_attempt(
    db_session: Session,
) -> None:
    message = OutboxService(db_session).create_owner(
        telegram_chat_id=99,
        final_text="owner notice",
        message_kind=MessageKind.NOTIFICATION,
        idempotency_key="owner:telegram-redaction",
    )
    db_session.commit()
    sessions = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    adapter = TelegramBotAdapter(
        token=SECRET_TOKEN,
        enabled=True,
        client=RaisingTelegramClient(),  # type: ignore[arg-type]
    )

    direct = adapter.send(
        DeliveryRequest(message.id, Transport.TELEGRAM, "99", "owner notice", message.idempotency_key)
    )
    assert not direct.success
    assert direct.error is not None
    assert "Telegram sendMessage failed" in direct.error
    assert "HTTP 403" in direct.error
    assert "code 403" in direct.error
    assert "Forbidden" in direct.error
    assert SECRET_TOKEN not in direct.error

    status = OutboxWorker(
        sessions,
        revalidator=AllowingRevalidator(),
        validator=ValidatingEverything(),
        adapters={Transport.TELEGRAM: adapter},
    ).process(message.id)

    assert status is OutboxStatus.RECONCILING
    with sessions() as session:
        attempt = session.scalar(
            select(OutboxDeliveryAttempt).where(
                OutboxDeliveryAttempt.outbox_message_id == message.id
            )
        )
        assert attempt is not None
        assert attempt.error_details is not None
        assert "HTTP 403" in attempt.error_details
        assert "Forbidden" in attempt.error_details
        assert SECRET_TOKEN not in attempt.error_details
