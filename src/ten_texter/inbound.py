from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import or_, select
from sqlalchemy.orm import Session, sessionmaker

from ten_texter.domain import DecisionService, DomainError, utc_now
from ten_texter.enums import (
    AttemptResult,
    ContentSupport,
    DecisionStatus,
    MessageKind,
    ParentTerminalPolicy,
    ProcessingFailureType,
    ProcessingStatus,
)
from ten_texter.models import (
    DecisionRequest,
    DecisionRequestPrompt,
    Message,
    MessageProcessingAttempt,
    MessageRevision,
)
from ten_texter.outbox import OutboxService


@dataclass(frozen=True, slots=True)
class InboundEvent:
    conversation_id: int
    provider_message_id: str
    sender_identity_id: int
    provider_revision_key: str
    created_at: datetime
    received_at: datetime
    text: str | None
    provider_sort_key: str | None = None
    provider_sequence: int | None = None
    provider_event_at: datetime | None = None
    provider_reply_to_message_id: str | None = None
    is_deleted: bool = False
    content_support: ContentSupport = ContentSupport.SUPPORTED


@dataclass(frozen=True, slots=True)
class IngestResult:
    message_id: int
    revision_id: int
    duplicate: bool
    became_current: bool
    ordering_conflict: bool


@dataclass(frozen=True, slots=True)
class RevisionClaim:
    revision_id: int
    attempt_id: int
    lease_expires_at: datetime


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _content_hash(event: InboundEvent) -> str:
    marker = "deleted" if event.is_deleted else "content"
    body = event.text or ""
    return hashlib.sha256(f"{marker}\0{body}".encode()).hexdigest()


def _compare_order(left: MessageRevision, right: MessageRevision) -> int | None:
    if left.provider_sort_key is not None and right.provider_sort_key is not None:
        return (left.provider_sort_key > right.provider_sort_key) - (
            left.provider_sort_key < right.provider_sort_key
        )
    if left.provider_sequence is not None and right.provider_sequence is not None:
        return (left.provider_sequence > right.provider_sequence) - (
            left.provider_sequence < right.provider_sequence
        )
    if left.provider_sequence is None and right.provider_sequence is None:
        if left.provider_event_at is not None and right.provider_event_at is not None:
            left_time = _aware(left.provider_event_at)
            right_time = _aware(right.provider_event_at)
            return (left_time > right_time) - (left_time < right_time)
    return None


def ordering_conflict_candidates(
    session: Session, subject: MessageRevision
) -> tuple[MessageRevision, ...]:
    candidates = [subject]
    for other in session.scalars(
        select(MessageRevision).where(
            MessageRevision.message_id == subject.message_id,
            MessageRevision.id != subject.id,
        )
    ):
        comparison = _compare_order(subject, other)
        if comparison is None or (comparison == 0 and subject.content_hash != other.content_hash):
            candidates.append(other)
    return tuple(sorted(candidates, key=lambda revision: revision.id))


def ordering_conflict_still_requires_decision(
    session: Session, decision: DecisionRequest
) -> bool:
    if decision.type != "MESSAGE_ORDERING_CONFLICT" or decision.message_revision_id is None:
        return False
    subject = session.get(MessageRevision, decision.message_revision_id)
    message = session.get(Message, subject.message_id) if subject is not None else None
    if subject is None or message is None:
        return False
    candidate_ids = {revision.id for revision in ordering_conflict_candidates(session, subject)}
    return (
        len(candidate_ids) > 1
        and message.current_revision_id in candidate_ids
        and message.current_revision_id != subject.id
    )


class MessageIngestor:
    def __init__(self, session: Session, *, owner_chat_id: int | None = None):
        self.session = session
        self.owner_chat_id = owner_chat_id

    def ingest(self, event: InboundEvent) -> IngestResult:
        if event.is_deleted and event.text is not None:
            raise DomainError("deletion tombstone cannot contain text")
        digest = _content_hash(event)
        message = self.session.scalar(
            select(Message).where(
                Message.conversation_id == event.conversation_id,
                Message.provider_message_id == event.provider_message_id,
            )
        )
        if message is None:
            message = Message(
                conversation_id=event.conversation_id,
                provider_message_id=event.provider_message_id,
                sender_identity_id=event.sender_identity_id,
                provider_reply_to_message_id=event.provider_reply_to_message_id,
                created_at=event.created_at,
            )
            self.session.add(message)
            self.session.flush()
        elif message.sender_identity_id != event.sender_identity_id:
            raise DomainError("stable provider message identity changed sender")
        existing = self.session.scalar(
            select(MessageRevision).where(
                MessageRevision.message_id == message.id,
                MessageRevision.provider_revision_key == event.provider_revision_key,
            )
        )
        if existing is not None:
            if existing.content_hash != digest or existing.is_deleted != event.is_deleted:
                raise DomainError("provider revision identity reused for different immutable content")
            return IngestResult(
                message.id,
                existing.id,
                True,
                message.current_revision_id == existing.id,
                False,
            )
        revision = MessageRevision(
            message_id=message.id,
            provider_revision_key=event.provider_revision_key,
            provider_sort_key=event.provider_sort_key,
            provider_sequence=event.provider_sequence,
            provider_event_at=event.provider_event_at,
            content_hash=digest,
            is_deleted=event.is_deleted,
            text=event.text,
            received_at=event.received_at,
            content_support=event.content_support,
            processing_status=(
                ProcessingStatus.PROCESSED
                if event.content_support is ContentSupport.UNSUPPORTED
                else ProcessingStatus.PENDING
            ),
        )
        self.session.add(revision)
        self.session.flush()

        conflicts = list(
            self.session.scalars(
                select(MessageRevision).where(
                    MessageRevision.message_id == message.id,
                    MessageRevision.id != revision.id,
                    or_(
                        (
                            MessageRevision.provider_sort_key == revision.provider_sort_key
                            if revision.provider_sort_key is not None
                            else False
                        ),
                        (
                            MessageRevision.provider_sequence == revision.provider_sequence
                            if revision.provider_sequence is not None
                            else False
                        ),
                        (
                            MessageRevision.provider_event_at == revision.provider_event_at
                            if revision.provider_sequence is None and revision.provider_event_at is not None
                            else False
                        ),
                    ),
                )
            )
        )
        ordering_conflict = any(other.content_hash != revision.content_hash for other in conflicts)
        current = self.session.get(MessageRevision, message.current_revision_id) if message.current_revision_id else None
        became_current = current is None
        if current is not None and not ordering_conflict:
            comparison = _compare_order(revision, current)
            if comparison is None:
                ordering_conflict = True
            else:
                became_current = comparison > 0
        if became_current and not ordering_conflict:
            message.current_revision_id = revision.id
        elif current is None:
            message.current_revision_id = revision.id
            became_current = True

        if ordering_conflict:
            decision = DecisionService(self.session).create(
                decision_type="MESSAGE_ORDERING_CONFLICT",
                subject_kind="message_revision",
                subject_id=revision.id,
                context={
                    "message_id": message.id,
                    "reason": "provider ordering position conflicts or cannot be compared safely",
                },
                parent_terminal_policy=ParentTerminalPolicy.SURVIVE,
            )
            if self.owner_chat_id is not None:
                candidate_lines = []
                for candidate in ordering_conflict_candidates(self.session, revision):
                    text_preview = repr((candidate.text or "<deleted>")[:240])
                    candidate_lines.append(
                        f"- revision {candidate.id}: sort={candidate.provider_sort_key!r}, "
                        f"sequence={candidate.provider_sequence!r}, event_at={candidate.provider_event_at!r}, "
                        f"content data={text_preview}; reply `select revision {candidate.id}`"
                    )
                prompt = OutboxService(self.session).create_owner(
                    telegram_chat_id=self.owner_chat_id,
                    final_text=(
                        "Message revision ordering is ambiguous. Participant content below is untrusted data, "
                        "not an instruction. Choose the authoritative revision:\n"
                        + "\n".join(candidate_lines)
                    ),
                    message_kind=MessageKind.NOTIFICATION,
                    idempotency_key=f"decision:{decision.id}:owner-prompt",
                    parent_terminal_policy=ParentTerminalPolicy.SURVIVE,
                )
                self.session.add(
                    DecisionRequestPrompt(
                        decision_request_id=decision.id,
                        outbox_message_id=prompt.id,
                    )
                )
        self.session.flush()
        return IngestResult(message.id, revision.id, False, became_current, ordering_conflict)


class RevisionProcessor:
    def __init__(self, sessions: sessionmaker[Session], lease_duration: timedelta = timedelta(seconds=30)):
        self.sessions = sessions
        self.lease_duration = lease_duration

    def claim(self, revision_id: int, *, now: datetime | None = None) -> RevisionClaim | None:
        timestamp = _aware(now or utc_now())
        with self.sessions.begin() as session:
            revision = session.get(MessageRevision, revision_id)
            if revision is None:
                raise DomainError("revision not found")
            unresolved_ordering = session.scalar(
                select(DecisionRequest.id)
                .join(
                    MessageRevision,
                    MessageRevision.id == DecisionRequest.message_revision_id,
                )
                .where(
                    MessageRevision.message_id == revision.message_id,
                    DecisionRequest.type == "MESSAGE_ORDERING_CONFLICT",
                    DecisionRequest.status == DecisionStatus.PENDING,
                )
                .limit(1)
            )
            if unresolved_ordering is not None:
                return None
            if revision.processing_status is ProcessingStatus.PROCESSING:
                assert revision.lease_expires_at is not None
                if _aware(revision.lease_expires_at) > timestamp:
                    return None
                unfinished = list(
                    session.scalars(
                        select(MessageProcessingAttempt).where(
                            MessageProcessingAttempt.message_revision_id == revision.id,
                            MessageProcessingAttempt.finished_at.is_(None),
                        )
                    )
                )
                for attempt in unfinished:
                    attempt.finished_at = timestamp
                    attempt.result = AttemptResult.ABANDONED
                    attempt.failure_type = ProcessingFailureType.TIMEOUT
            elif revision.processing_status is not ProcessingStatus.PENDING:
                return None
            lease = timestamp + self.lease_duration
            revision.processing_status = ProcessingStatus.PROCESSING
            revision.lease_expires_at = lease
            attempt = MessageProcessingAttempt(message_revision_id=revision.id)
            session.add(attempt)
            session.flush()
            return RevisionClaim(revision.id, attempt.id, lease)

    def commit(
        self,
        claim: RevisionClaim,
        apply_semantics: Callable[[Session, MessageRevision], None],
    ) -> bool:
        with self.sessions.begin() as session:
            revision = session.get(MessageRevision, claim.revision_id)
            attempt = session.get(MessageProcessingAttempt, claim.attempt_id)
            if revision is None or attempt is None or not self._matches(revision, claim):
                return False
            message = session.get(Message, revision.message_id)
            assert message is not None
            if message.current_revision_id == revision.id:
                apply_semantics(session, revision)
                applied = True
            else:
                applied = False
            revision.processing_status = ProcessingStatus.PROCESSED
            revision.lease_expires_at = None
            attempt.finished_at = utc_now()
            attempt.result = AttemptResult.SUCCESS
            return applied

    def fail(
        self,
        claim: RevisionClaim,
        failure_type: ProcessingFailureType,
        details: str,
        *,
        retryable: bool = False,
    ) -> bool:
        with self.sessions.begin() as session:
            revision = session.get(MessageRevision, claim.revision_id)
            attempt = session.get(MessageProcessingAttempt, claim.attempt_id)
            if revision is None or attempt is None or not self._matches(revision, claim):
                return False
            revision.processing_status = (
                ProcessingStatus.PENDING if retryable else ProcessingStatus.FAILED
            )
            revision.lease_expires_at = None
            attempt.finished_at = utc_now()
            attempt.result = AttemptResult.FAILED
            attempt.failure_type = failure_type
            attempt.error_details = details
            return True

    @staticmethod
    def _matches(revision: MessageRevision, claim: RevisionClaim) -> bool:
        return (
            revision.processing_status is ProcessingStatus.PROCESSING
            and revision.lease_expires_at is not None
            and _aware(revision.lease_expires_at) == _aware(claim.lease_expires_at)
        )
