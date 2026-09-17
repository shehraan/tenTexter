from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ten_texter.beeper import BeeperSyncService
from ten_texter.control import PreparedOwnerCommand
from ten_texter.domain import utc_now
from ten_texter.enums import (
    ConversationKind,
    DestinationKind,
    OutboxStatus,
    ProcessingStatus,
    TaskStatus,
    TelegramUpdateStatus,
    Transport,
)
from ten_texter.models import (
    BeeperOutboxDestination,
    Conversation,
    ConversationParticipant,
    Identity,
    Message,
    MessageRevision,
    OutboxMessage,
    OutboxMessageParticipant,
    Person,
    TaskInstance,
    TaskParticipant,
    TelegramOutboxDestination,
    TelegramUpdate,
)
from ten_texter.runtime import build_runtime
from ten_texter.telegram import OwnerCommandHandler, TelegramControlGateway
from ten_texter.workflows import ParticipantSendPlan


ALLOWLISTED_TARGET_NAME = "Shehraan Canada"
ALLOWLISTED_TARGET_NETWORK = "WhatsApp"
MAX_POLL_ROUNDS = 10
MAX_POLL_DELAY_SECONDS = 60.0


class LiveTennisTestError(RuntimeError):
    """The guarded live tennis test could not safely complete."""

    def __init__(self, message: str, *, task_instance_id: int | None = None):
        super().__init__(message)
        self.task_instance_id = task_instance_id


@dataclass(frozen=True, slots=True)
class LiveTennisTarget:
    person_id: int
    identity_id: int
    conversation_id: int
    beeper_user_id: str
    beeper_conversation_id: str


@dataclass(frozen=True, slots=True)
class LiveTennisReport:
    ok: bool
    command: str
    target_name: str
    target_beeper_conversation_id: str
    telegram_update_id: int
    telegram_update_row_id: int
    task_instance_id: int
    initial_outbox_id: int
    initial_outbox_status: str
    generated_text: str
    rounds_checked: int
    reply_provider_message_ids: tuple[str, ...]
    reply_revision_ids: tuple[int, ...]
    owner_outbox_ids: tuple[int, ...]
    owner_outbox_statuses: tuple[str, ...]
    task_status: str
    availability_status: str


@dataclass(frozen=True, slots=True)
class LiveTennisPollReport:
    ok: bool
    task_instance_id: int
    target_name: str
    target_beeper_conversation_id: str
    rounds_checked: int
    reply_provider_message_ids: tuple[str, ...]
    reply_revision_ids: tuple[int, ...]
    owner_outbox_ids: tuple[int, ...]
    owner_outbox_statuses: tuple[str, ...]
    task_status: str
    availability_status: str


def resolve_allowlisted_target(session: Session) -> LiveTennisTarget:
    """Resolve the one exact direct WhatsApp destination permitted by this test."""
    rows = list(
        session.execute(
            select(Person, Identity, Conversation)
            .join(Identity, Identity.person_id == Person.id)
            .join(
                ConversationParticipant,
                ConversationParticipant.identity_id == Identity.id,
            )
            .join(
                Conversation,
                Conversation.id == ConversationParticipant.conversation_id,
            )
            .where(
                Person.display_name == ALLOWLISTED_TARGET_NAME,
                Person.archived_at.is_(None),
                Identity.archived_at.is_(None),
                Identity.network.ilike(ALLOWLISTED_TARGET_NETWORK),
                Conversation.archived_at.is_(None),
                Conversation.network.ilike(ALLOWLISTED_TARGET_NETWORK),
                Conversation.kind == ConversationKind.DIRECT,
                Conversation.counterparty_person_id == Person.id,
                ConversationParticipant.is_current.is_(True),
            )
            .order_by(Person.id, Identity.id, Conversation.id)
        )
    )
    if len(rows) != 1:
        raise LiveTennisTestError(
            "Refusing live test: expected exactly one current direct WhatsApp "
            f"conversation for {ALLOWLISTED_TARGET_NAME!r}, found {len(rows)}."
        )

    person, identity, conversation = rows[0]
    if not identity.beeper_user_id or not conversation.beeper_conversation_id:
        raise LiveTennisTestError(
            "Refusing live test: allowlisted target is missing a stable Beeper ID."
        )

    current_members = list(
        session.scalars(
            select(ConversationParticipant.identity_id).where(
                ConversationParticipant.conversation_id == conversation.id,
                ConversationParticipant.is_current.is_(True),
            )
        )
    )
    if current_members != [identity.id]:
        raise LiveTennisTestError(
            "Refusing live test: the selected direct conversation does not have "
            "exactly one current counterparty membership."
        )

    return LiveTennisTarget(
        person_id=person.id,
        identity_id=identity.id,
        conversation_id=conversation.id,
        beeper_user_id=identity.beeper_user_id,
        beeper_conversation_id=conversation.beeper_conversation_id,
    )


class _AllowlistedOwnerCommandHandler:
    """Guard the production owner handler before it can create a task."""

    def __init__(self, delegate: OwnerCommandHandler, target: LiveTennisTarget):
        self.delegate = delegate
        self.target = target

    def prepare_command(self, parsed: object, update: TelegramUpdate) -> PreparedOwnerCommand:
        prepared = self.delegate.prepare_command(parsed, update)
        self._assert_prepared(prepared)
        return prepared

    def apply_command(
        self,
        session: Session,
        prepared: object,
        update: TelegramUpdate,
    ) -> None:
        if not isinstance(prepared, PreparedOwnerCommand):
            raise LiveTennisTestError("Refusing live test: command was not prepared.")
        self._assert_prepared(prepared)
        self.delegate.apply_command(session, prepared, update)

    def prepare_decision(
        self,
        decision_id: int,
        payload: dict[str, Any],
        update: TelegramUpdate,
    ) -> object:
        return self.delegate.prepare_decision(decision_id, payload, update)

    def apply_decision(
        self,
        session: Session,
        decision_id: int,
        payload: object,
        update: TelegramUpdate,
    ) -> None:
        self.delegate.apply_decision(session, decision_id, payload, update)

    def _assert_prepared(self, prepared: object) -> None:
        if not isinstance(prepared, PreparedOwnerCommand):
            raise LiveTennisTestError(
                "Refusing live test: owner handler returned an unsupported plan."
            )
        if prepared.review_reason is not None:
            raise LiveTennisTestError(
                "Refusing live test: owner command requires review: "
                f"{prepared.review_reason}"
            )
        if prepared.plan is None or len(prepared.sends) != 1:
            raise LiveTennisTestError(
                "Refusing live test: expected one executable one-time tennis send."
            )
        plan = prepared.plan
        if getattr(plan, "recurrence_rule", None) is not None:
            raise LiveTennisTestError("Refusing live test: recurring commands are not allowed.")
        if str(getattr(plan, "topic_key", "")).casefold() != "tennis":
            raise LiveTennisTestError(
                "Refusing live test: parsed topic was not exactly tennis."
            )
        if getattr(plan, "duration_minutes", None) != 60:
            raise LiveTennisTestError(
                "Refusing live test: parsed duration was not exactly 60 minutes."
            )
        scheduled_at = getattr(plan, "scheduled_at", None)
        if not isinstance(scheduled_at, datetime) or scheduled_at.tzinfo is None:
            raise LiveTennisTestError(
                "Refusing live test: parsed start time was not timezone-aware."
            )
        send = prepared.sends[0]
        if not isinstance(send, ParticipantSendPlan):
            raise LiveTennisTestError(
                "Refusing live test: participant send plan had an unsupported shape."
            )
        if send.person_id != self.target.person_id:
            raise LiveTennisTestError(
                "Refusing live test: resolved person is outside the hard allowlist."
            )
        if send.conversation_id != self.target.conversation_id:
            raise LiveTennisTestError(
                "Refusing live test: resolved conversation is outside the hard allowlist."
            )
        if not send.final_text.strip():
            raise LiveTennisTestError(
                "Refusing live test: generated participant text was empty."
            )


def run_live_tennis_test(
    app: object,
    *,
    confirm_real_send: bool = False,
    poll_rounds: int = 1,
    poll_delay_seconds: float = 0.0,
) -> LiveTennisReport:
    """Run one real tennis coordination flow against the hard-coded test account.

    The command is injected after Telegram authentication and persistence, because a
    Telegram bot cannot create an inbound message from the owner's personal account.
    Participant delivery still uses the production validator, policy, durable Outbox,
    and Beeper adapter. Only the exact allowlisted WhatsApp conversation is eligible.
    """
    if not confirm_real_send:
        raise LiveTennisTestError(
            "Refusing live send: pass confirm_real_send=True (or the CLI confirmation flag)."
        )
    _validate_poll_config(poll_rounds, poll_delay_seconds)

    settings = app.settings
    if not settings.real_transports_enabled:
        raise LiveTennisTestError(
            "Refusing live send: TEN_TEXTER_REAL_TRANSPORTS_ENABLED is not true."
        )
    if settings.owner_id is None or settings.owner_chat_id is None:
        raise LiveTennisTestError(
            "Refusing live send: owner ID and owner chat ID are required."
        )

    with app.sessions() as session:
        target = resolve_allowlisted_target(session)
        _assert_clean_live_test_start(session, target)
        known_provider_message_ids = set(
            session.scalars(
                select(Message.provider_message_id).where(
                    Message.conversation_id == target.conversation_id
                )
            )
        )
        existing_task_ids = set(session.scalars(select(TaskInstance.id)))

    runtime = build_runtime(app)
    command = (
        f"Ask {target.beeper_conversation_id} about tennis tomorrow at 5 PM "
        "for 60 minutes"
    )
    update_id = _next_synthetic_update_id(app.sessions)
    raw_update = {
        "update_id": update_id,
        "message": {
            "message_id": abs(update_id),
            "from": {"id": settings.owner_id},
            "chat": {"id": settings.owner_chat_id, "type": "private"},
            "text": command,
        },
    }
    guarded_handler = _AllowlistedOwnerCommandHandler(runtime.control.handler, target)
    gateway = TelegramControlGateway(
        app.sessions,
        owner_id=settings.owner_id,
        parser=runtime.control.parser,
        handler=guarded_handler,
    )
    received = gateway.receive(raw_update)
    if received.outcome != "PERSISTED" or received.telegram_update_row_id is None:
        raise LiveTennisTestError(
            f"Synthetic owner update was not persisted: {received.outcome}."
        )

    try:
        gateway.process(received.telegram_update_row_id)
    except Exception as exc:
        _fail_synthetic_update(
            app.sessions,
            received.telegram_update_row_id,
            f"{type(exc).__name__}: {exc}",
        )
        if isinstance(exc, LiveTennisTestError):
            raise
        raise LiveTennisTestError(
            f"Owner command could not be processed: {type(exc).__name__}: {exc}"
        ) from exc

    with app.sessions() as session:
        task_ids = set(session.scalars(select(TaskInstance.id))) - existing_task_ids
    if len(task_ids) != 1:
        raise LiveTennisTestError(
            "Refusing live send: expected exactly one task created by the synthetic command, "
            f"found {len(task_ids)}."
        )
    task_id = next(iter(task_ids))

    task_outboxes = _validate_task_outboxes(app.sessions, task_id, target, settings.owner_chat_id)
    participant_outboxes = [
        message for message in task_outboxes if message.transport is Transport.BEEPER
    ]
    if len(participant_outboxes) != 1:
        raise LiveTennisTestError(
            "Refusing live send: expected exactly one participant outbox message, "
            f"found {len(participant_outboxes)}."
        )
    initial = participant_outboxes[0]
    if initial.status is not OutboxStatus.PENDING:
        raise LiveTennisTestError(
            "Refusing live send: initial participant outbox was not pending."
        )
    try:
        try:
            initial_status = runtime.outbox.process(initial.id)
        except Exception as exc:
            raise LiveTennisTestError(
                f"Initial participant send could not be processed: {type(exc).__name__}: {exc}",
                task_instance_id=task_id,
            ) from exc
        if initial_status is OutboxStatus.RECONCILING:
            initial_status = _reconcile_initial_send(
                runtime,
                initial.id,
                poll_rounds=poll_rounds,
                poll_delay_seconds=poll_delay_seconds,
            )
    except LiveTennisTestError as exc:
        if exc.task_instance_id is None:
            exc.task_instance_id = task_id
        raise
    if initial_status is not OutboxStatus.SENT:
        raise LiveTennisTestError(
            f"Initial participant send did not reach SENT; status is {initial_status.value}.",
            task_instance_id=task_id,
        )

    poll = _poll_live_tennis_task(
        runtime,
        app.sessions,
        target,
        task_id,
        owner_chat_id=settings.owner_chat_id,
        known_provider_message_ids=known_provider_message_ids,
        known_task_outbox_ids={message.id for message in task_outboxes},
        poll_rounds=poll_rounds,
        poll_delay_seconds=poll_delay_seconds,
    )

    return LiveTennisReport(
        ok=initial_status is OutboxStatus.SENT and poll.ok,
        command=command,
        target_name=ALLOWLISTED_TARGET_NAME,
        target_beeper_conversation_id=target.beeper_conversation_id,
        telegram_update_id=update_id,
        telegram_update_row_id=received.telegram_update_row_id,
        task_instance_id=task_id,
        initial_outbox_id=initial.id,
        initial_outbox_status=initial_status.value,
        generated_text=initial.final_text,
        rounds_checked=poll.rounds_checked,
        reply_provider_message_ids=poll.reply_provider_message_ids,
        reply_revision_ids=poll.reply_revision_ids,
        owner_outbox_ids=poll.owner_outbox_ids,
        owner_outbox_statuses=poll.owner_outbox_statuses,
        task_status=poll.task_status,
        availability_status=poll.availability_status,
    )


def poll_live_tennis_test(
    app: object,
    *,
    task_instance_id: int,
    confirm_real_send: bool = False,
    poll_rounds: int = 1,
    poll_delay_seconds: float = 0.0,
) -> LiveTennisPollReport:
    """Poll and process one existing allowlisted tennis task without a new participant send."""
    if not confirm_real_send:
        raise LiveTennisTestError(
            "Refusing live poll: pass confirm_real_send=True (or the CLI confirmation flag)."
        )
    _validate_poll_config(poll_rounds, poll_delay_seconds)
    settings = app.settings
    if not settings.real_transports_enabled:
        raise LiveTennisTestError(
            "Refusing live poll: TEN_TEXTER_REAL_TRANSPORTS_ENABLED is not true."
        )
    if settings.owner_id is None or settings.owner_chat_id is None:
        raise LiveTennisTestError(
            "Refusing live poll: owner ID and owner chat ID are required."
        )

    with app.sessions() as session:
        target = resolve_allowlisted_target(session)
        task = session.get(TaskInstance, task_instance_id)
        if task is None or task.status is not TaskStatus.ACTIVE:
            raise LiveTennisTestError(
                "Refusing live poll: task must exist and still be ACTIVE."
            )
        participants = list(
            session.scalars(
                select(TaskParticipant).where(
                    TaskParticipant.task_instance_id == task_instance_id
                )
            )
        )
        if len(participants) != 1 or (
            participants[0].person_id != target.person_id
            or participants[0].conversation_id != target.conversation_id
        ):
            raise LiveTennisTestError(
                "Refusing live poll: task is not pinned exclusively to the allowlisted target."
            )
        task_outboxes = _validate_task_outboxes(
            app.sessions, task_instance_id, target, settings.owner_chat_id
        )
        participant_outboxes = [
            message for message in task_outboxes if message.transport is Transport.BEEPER
        ]
        if (
            len(participant_outboxes) != 1
            or participant_outboxes[0].status
            not in {OutboxStatus.SENT, OutboxStatus.RECONCILING}
        ):
            raise LiveTennisTestError(
                "Refusing live poll: the task does not have exactly one sent or "
                "reconciling participant message."
            )
        initial_outbox_needs_reconcile = (
            participant_outboxes[0].status is OutboxStatus.RECONCILING
        )
        recovery_replies = _find_recoverable_target_revisions(session, target)
        known_provider_message_ids = set(
            session.scalars(
                select(Message.provider_message_id).where(
                    Message.conversation_id == target.conversation_id
                )
            )
        )

    runtime = build_runtime(app)
    if initial_outbox_needs_reconcile:
        try:
            initial_status = _reconcile_initial_send(
                runtime,
                participant_outboxes[0].id,
                poll_rounds=poll_rounds,
                poll_delay_seconds=poll_delay_seconds,
            )
        except LiveTennisTestError as exc:
            if exc.task_instance_id is None:
                exc.task_instance_id = task_instance_id
            raise
        if initial_status is not OutboxStatus.SENT:
            raise LiveTennisTestError(
                "Initial participant send is still RECONCILING; resume the same "
                "task after the Beeper delivery can be resolved.",
                task_instance_id=task_instance_id,
            )
    poll = _poll_live_tennis_task(
        runtime,
        app.sessions,
        target,
        task_instance_id,
        owner_chat_id=settings.owner_chat_id,
        known_provider_message_ids=known_provider_message_ids,
        known_task_outbox_ids={message.id for message in task_outboxes},
        recovery_replies=recovery_replies,
        poll_rounds=poll_rounds,
        poll_delay_seconds=poll_delay_seconds,
    )
    return LiveTennisPollReport(
        ok=all(status == OutboxStatus.SENT.value for status in poll.owner_outbox_statuses),
        task_instance_id=task_instance_id,
        target_name=ALLOWLISTED_TARGET_NAME,
        target_beeper_conversation_id=target.beeper_conversation_id,
        rounds_checked=poll.rounds_checked,
        reply_provider_message_ids=poll.reply_provider_message_ids,
        reply_revision_ids=poll.reply_revision_ids,
        owner_outbox_ids=poll.owner_outbox_ids,
        owner_outbox_statuses=poll.owner_outbox_statuses,
        task_status=poll.task_status,
        availability_status=poll.availability_status,
    )


def _validate_poll_config(poll_rounds: int, poll_delay_seconds: float) -> None:
    if not 1 <= poll_rounds <= MAX_POLL_ROUNDS:
        raise LiveTennisTestError(
            f"poll_rounds must be between 1 and {MAX_POLL_ROUNDS}."
        )
    if not 0 <= poll_delay_seconds <= MAX_POLL_DELAY_SECONDS:
        raise LiveTennisTestError(
            f"poll_delay_seconds must be between 0 and {MAX_POLL_DELAY_SECONDS}."
        )


def _assert_clean_live_test_start(session: Session, target: LiveTennisTarget) -> None:
    active_task = session.scalar(
        select(TaskInstance.id)
        .join(TaskParticipant, TaskParticipant.task_instance_id == TaskInstance.id)
        .where(
            TaskParticipant.person_id == target.person_id,
            TaskInstance.status == TaskStatus.ACTIVE,
        )
        .limit(1)
    )
    if active_task is not None:
        raise LiveTennisTestError(
            "Refusing live test: the allowlisted participant already has an active task."
        )

    pending_target_send = session.scalar(
        select(OutboxMessage.id)
        .join(
            BeeperOutboxDestination,
            BeeperOutboxDestination.outbox_message_id == OutboxMessage.id,
        )
        .where(
            OutboxMessage.transport == Transport.BEEPER,
            OutboxMessage.status.in_(
                [OutboxStatus.PENDING, OutboxStatus.SENDING, OutboxStatus.RECONCILING]
            ),
            BeeperOutboxDestination.conversation_id == target.conversation_id,
        )
        .limit(1)
    )
    if pending_target_send is not None:
        raise LiveTennisTestError(
            "Refusing live test: the allowlisted conversation already has a pending send."
        )

    _assert_no_active_revisions(session)


def _assert_no_active_revisions(session: Session) -> None:
    active_revisions = list(
        session.scalars(
            select(MessageRevision.id).where(
                MessageRevision.processing_status.in_(
                    [ProcessingStatus.PENDING, ProcessingStatus.PROCESSING]
                )
            )
        )
    )
    if active_revisions:
        raise LiveTennisTestError(
            "Refusing live test: existing inbound revisions must be drained before "
            "the targeted loop can run."
        )


def _find_recoverable_target_revisions(
    session: Session,
    target: LiveTennisTarget,
) -> tuple[tuple[str, int], ...]:
    """Find the one persisted target reply that a resumed live poll may process."""
    rows = list(
        session.execute(
            select(
                MessageRevision.id.label("revision_id"),
                Message.provider_message_id,
                Message.conversation_id,
                Message.sender_identity_id,
                MessageRevision.processing_status,
                MessageRevision.lease_expires_at,
            ).join(Message, Message.id == MessageRevision.message_id)
            .where(
                MessageRevision.processing_status.in_(
                    [ProcessingStatus.PENDING, ProcessingStatus.PROCESSING]
                )
            )
        )
    )
    if not rows:
        return ()

    now = utc_now()
    recoverable = []
    blocked = False
    for row in rows:
        lease_expires_at = row.lease_expires_at
        if lease_expires_at is not None and lease_expires_at.tzinfo is None:
            lease_expires_at = lease_expires_at.replace(tzinfo=UTC)
        is_recoverable = row.processing_status is ProcessingStatus.PENDING or (
            row.processing_status is ProcessingStatus.PROCESSING
            and lease_expires_at is not None
            and lease_expires_at <= now
        )
        if is_recoverable:
            recoverable.append(row)
        else:
            blocked = True

    if blocked or len(recoverable) != 1:
        raise LiveTennisTestError(
            "Refusing live poll: existing inbound revisions are not exactly one "
            "recoverable reply for the allowlisted target."
        )

    row = recoverable[0]
    if (
        row.conversation_id != target.conversation_id
        or row.sender_identity_id != target.identity_id
    ):
        raise LiveTennisTestError(
            "Refusing live poll: the recoverable inbound revision is outside the "
            "allowlisted WhatsApp conversation."
        )
    return ((row.provider_message_id, row.revision_id),)


def _next_synthetic_update_id(sessions: Any) -> int:
    candidate = -time.time_ns()
    with sessions() as session:
        while session.scalar(
            select(TelegramUpdate.id).where(
                TelegramUpdate.telegram_update_id == candidate
            )
        ) is not None:
            candidate -= 1
    return candidate


def _fail_synthetic_update(sessions: Any, row_id: int, details: str) -> None:
    with sessions.begin() as session:
        update = session.get(TelegramUpdate, row_id)
        if update is not None and update.status is TelegramUpdateStatus.PENDING:
            update.status = TelegramUpdateStatus.FAILED
            update.error_details = f"live tennis test: {details}"[:2000]


def _validate_task_outboxes(
    sessions: Any,
    task_id: int,
    target: LiveTennisTarget,
    owner_chat_id: int,
) -> list[OutboxMessage]:
    with sessions() as session:
        messages = list(
            session.scalars(
                select(OutboxMessage)
                .where(OutboxMessage.task_instance_id == task_id)
                .order_by(OutboxMessage.id)
            )
        )
        for message in messages:
            participant_links = list(
                session.scalars(
                    select(OutboxMessageParticipant).where(
                        OutboxMessageParticipant.outbox_message_id == message.id
                    )
                )
            )
            if message.transport is Transport.BEEPER:
                destination = session.get(BeeperOutboxDestination, message.id)
                if (
                    message.destination_kind is not DestinationKind.PARTICIPANT
                    or destination is None
                    or destination.conversation_id != target.conversation_id
                    or len(participant_links) != 1
                ):
                    raise LiveTennisTestError(
                        "Refusing live send: a task participant outbox escaped the "
                        "allowlisted WhatsApp conversation."
                    )
                participant = session.get(
                    TaskParticipant, participant_links[0].task_participant_id
                )
                if (
                    participant is None
                    or participant.task_instance_id != task_id
                    or participant.person_id != target.person_id
                    or participant.conversation_id != target.conversation_id
                ):
                    raise LiveTennisTestError(
                        "Refusing live send: a participant outbox escaped the hard "
                        "person allowlist."
                    )
            elif message.transport is Transport.TELEGRAM:
                destination = session.get(TelegramOutboxDestination, message.id)
                if (
                    message.destination_kind is not DestinationKind.OWNER
                    or destination is None
                    or destination.telegram_chat_id != owner_chat_id
                    or participant_links
                ):
                    raise LiveTennisTestError(
                        "Refusing live send: a task notification has an unexpected destination."
                    )
            else:
                raise LiveTennisTestError(
                    "Refusing live send: task created an unsupported transport."
                )
        return messages


def _poll_live_tennis_task(
    runtime: Any,
    sessions: Any,
    target: LiveTennisTarget,
    task_id: int,
    *,
    owner_chat_id: int,
    known_provider_message_ids: set[str],
    known_task_outbox_ids: set[int],
    recovery_replies: tuple[tuple[str, int], ...] = (),
    poll_rounds: int,
    poll_delay_seconds: float,
) -> LiveTennisPollReport:
    reply_provider_ids: list[str] = []
    reply_revision_ids: list[int] = []
    owner_outbox_ids: list[int] = []
    owner_outbox_statuses: list[str] = []
    rounds_checked = 0

    for round_index in range(poll_rounds):
        if round_index:
            time.sleep(poll_delay_seconds)
        rounds_checked += 1
        if recovery_replies:
            new_provider_ids = [provider_id for provider_id, _ in recovery_replies]
            new_revisions = [revision_id for _, revision_id in recovery_replies]
            recovery_replies = ()
        else:
            try:
                messages = runtime.beeper.list_messages(
                    target.beeper_conversation_id,
                    max_pages=1,
                    allow_truncated=True,
                )
            except Exception as exc:
                raise LiveTennisTestError(
                    f"Target Beeper conversation could not be polled: {type(exc).__name__}: {exc}"
                ) from exc
            new_provider_ids, new_revisions = _ingest_new_target_messages(
                sessions,
                target,
                messages,
                known_provider_message_ids,
                owner_chat_id=owner_chat_id,
            )
        reply_provider_ids.extend(new_provider_ids)
        reply_revision_ids.extend(new_revisions)
        if new_revisions:
            _process_only_new_target_revisions(runtime, target, new_revisions)
            task_outboxes = _validate_task_outboxes(
                sessions, task_id, target, owner_chat_id
            )
            new_outboxes = [
                message
                for message in task_outboxes
                if message.id not in known_task_outbox_ids
            ]
            _process_new_task_outboxes(
                runtime,
                new_outboxes,
                owner_outbox_ids,
                owner_outbox_statuses,
            )
            known_task_outbox_ids.update(message.id for message in new_outboxes)
            break

    with sessions() as session:
        task = session.get(TaskInstance, task_id)
        participant = session.scalar(
            select(TaskParticipant).where(TaskParticipant.task_instance_id == task_id)
        )
        if task is None or participant is None:
            raise LiveTennisTestError("Live test task disappeared while polling.")
        task_outboxes = list(
            session.scalars(
                select(OutboxMessage)
                .where(OutboxMessage.task_instance_id == task_id)
                .order_by(OutboxMessage.id)
            )
        )
        owner_messages = [
            message for message in task_outboxes if message.transport is Transport.TELEGRAM
        ]
        final_owner_ids = tuple(message.id for message in owner_messages)
        final_owner_statuses = tuple(message.status.value for message in owner_messages)

    return LiveTennisPollReport(
        ok=all(status == OutboxStatus.SENT.value for status in final_owner_statuses),
        task_instance_id=task_id,
        target_name=ALLOWLISTED_TARGET_NAME,
        target_beeper_conversation_id=target.beeper_conversation_id,
        rounds_checked=rounds_checked,
        reply_provider_message_ids=tuple(reply_provider_ids),
        reply_revision_ids=tuple(reply_revision_ids),
        owner_outbox_ids=final_owner_ids,
        owner_outbox_statuses=final_owner_statuses,
        task_status=task.status.value,
        availability_status=participant.availability_status.value,
    )


def _process_new_task_outboxes(
    runtime: Any,
    messages: list[OutboxMessage],
    owner_outbox_ids: list[int],
    owner_outbox_statuses: list[str],
) -> None:
    for message in messages:
        if message.status is not OutboxStatus.PENDING:
            continue
        try:
            status = runtime.outbox.process(message.id)
        except Exception as exc:
            raise LiveTennisTestError(
                f"Task outbox {message.id} could not be processed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if message.transport is Transport.TELEGRAM:
            owner_outbox_ids.append(message.id)
            owner_outbox_statuses.append(status.value)
        if status is not OutboxStatus.SENT:
            raise LiveTennisTestError(
                f"Task outbox {message.id} did not reach SENT; status is {status.value}."
            )


def _reconcile_initial_send(
    runtime: Any,
    outbox_id: int,
    *,
    poll_rounds: int,
    poll_delay_seconds: float,
) -> OutboxStatus:
    """Resolve Beeper's asynchronous pending-message response without retrying it."""
    status = OutboxStatus.RECONCILING
    for round_index in range(poll_rounds):
        if round_index:
            time.sleep(poll_delay_seconds)
        reconcile = getattr(runtime.outbox, "reconcile", None)
        if reconcile is None:
            raise LiveTennisTestError(
                "Initial participant send is RECONCILING, but the outbox worker "
                "has no reconciliation operation."
            )
        try:
            status = reconcile(outbox_id)
        except Exception as exc:
            raise LiveTennisTestError(
                f"Initial participant send could not be reconciled: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if status is not OutboxStatus.RECONCILING:
            return status
    return status


def _ingest_new_target_messages(
    sessions: Any,
    target: LiveTennisTarget,
    messages: list[dict[str, Any]],
    known_provider_message_ids: set[str],
    *,
    owner_chat_id: int,
) -> tuple[list[str], list[int]]:
    provider_ids: list[str] = []
    revision_ids: list[int] = []
    with sessions.begin() as session:
        sync = BeeperSyncService(session, owner_chat_id=owner_chat_id)
        for item in messages:
            provider_id = item.get("id")
            if not isinstance(provider_id, str) or not provider_id:
                continue
            if provider_id in known_provider_message_ids:
                continue
            if item.get("chatID", target.beeper_conversation_id) != target.beeper_conversation_id:
                continue
            if item.get("isSender") is not False:
                continue
            if item.get("senderID") != target.beeper_user_id:
                continue
            payload = dict(item)
            payload.setdefault("chatID", target.beeper_conversation_id)
            result = sync.ingest_message(payload)
            known_provider_message_ids.add(provider_id)
            provider_ids.append(provider_id)
            if not result.duplicate and result.revision_id:
                revision = session.get(MessageRevision, result.revision_id)
                if revision is not None and revision.processing_status is ProcessingStatus.PENDING:
                    revision_ids.append(result.revision_id)
    return provider_ids, revision_ids


def _process_only_new_target_revisions(
    runtime: Any,
    target: LiveTennisTarget,
    expected_revision_ids: list[int],
) -> None:
    expected = set(expected_revision_ids)
    with runtime.sessions() as session:
        eligible = set(
            session.scalars(
                select(MessageRevision.id)
                .where(
                    MessageRevision.processing_status.in_(
                        [ProcessingStatus.PENDING, ProcessingStatus.PROCESSING]
                    )
                )
            )
        )
        if eligible != expected:
            raise LiveTennisTestError(
                "Refusing live test: inbound processing would include revisions outside "
                "the newly observed allowlisted reply."
            )
        non_target = list(
            session.scalars(
                select(MessageRevision.id)
                .join(Message, Message.id == MessageRevision.message_id)
                .where(
                    MessageRevision.id.in_(expected),
                    Message.conversation_id != target.conversation_id,
                )
            )
        )
        if non_target:
            raise LiveTennisTestError(
                "Refusing live test: an observed reply revision is outside the "
                "allowlisted conversation."
            )
    try:
        runtime._process_revisions()
    except Exception as exc:
        raise LiveTennisTestError(
            f"Allowlisted reply could not be processed: {type(exc).__name__}: {exc}"
        ) from exc
    with runtime.sessions() as session:
        remaining = set(
            session.scalars(
                select(MessageRevision.id).where(
                    MessageRevision.id.in_(expected),
                    MessageRevision.processing_status.in_(
                        [ProcessingStatus.PENDING, ProcessingStatus.PROCESSING]
                    ),
                )
            )
        )
    if remaining:
        raise LiveTennisTestError(
            "Allowlisted reply was left pending by the inbound worker; refusing to "
            "claim the live loop completed."
        )
