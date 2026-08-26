from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from ten_texter.domain import (
    AvailabilityService,
    DecisionService,
    DomainError,
    StaleWork,
    utc_now,
)
from ten_texter.enums import (
    AvailabilityEvidence,
    AvailabilityStatus,
    AwaitedResponseStatus,
    DecisionCloseReason,
    DecisionStatus,
    MessageKind,
    ParentTerminalPolicy,
    ProposalStatus,
    TaskStatus,
)
from ten_texter.models import (
    AwaitedResponse,
    AwaitedResponsePrompt,
    DecisionRequest,
    DecisionRequestAwaitedResponseCandidate,
    Identity,
    Message,
    MessageRevision,
    OutboxDeliveryAttempt,
    OutboxMessage,
    Proposal,
    TaskEvent,
    TaskInstance,
    TaskParticipant,
    DecisionRequestPrompt,
)
from ten_texter.outbox import OutboxService


class SemanticCorrelator(Protocol):
    def choose(self, revision: MessageRevision, candidates: list[AwaitedResponse]) -> int | None: ...


@dataclass(frozen=True, slots=True)
class AtomicProposal:
    field: str
    operation: str
    old_value: object
    proposed_value: object


@dataclass(frozen=True, slots=True)
class Classification:
    kind: str
    availability: AvailabilityStatus | None = None
    evidence: AvailabilityEvidence = AvailabilityEvidence.FIRST_PARTY
    proposals: tuple[AtomicProposal, ...] = field(default_factory=tuple)


class Classifier(Protocol):
    def classify(self, revision: MessageRevision, awaited_response: AwaitedResponse) -> Classification: ...


@dataclass(frozen=True, slots=True)
class CorrelationResult:
    outcome: str
    awaited_response_id: int | None = None
    decision_request_id: int | None = None


@dataclass(frozen=True, slots=True)
class PreparedCorrelation:
    outcome: str
    awaited_response_id: int | None = None
    source: str | None = None
    classification: Classification | None = None
    candidate_ids: tuple[int, ...] = ()
    decision_type: str | None = None


class CorrelationOrchestrator:
    def __init__(
        self,
        session: Session,
        *,
        semantic: SemanticCorrelator,
        classifier: Classifier,
        owner_chat_id: int | None = None,
    ):
        self.session = session
        self.semantic = semantic
        self.classifier = classifier
        self.owner_chat_id = owner_chat_id

    def prepare(self, revision_id: int) -> PreparedCorrelation:
        """Run correlation/classification reads and model calls without a write transaction."""
        revision = self.session.get(MessageRevision, revision_id)
        if revision is None:
            raise DomainError("revision not found")
        message = self.session.get(Message, revision.message_id)
        if message is None or message.current_revision_id != revision.id:
            raise StaleWork("revision is not current")
        if revision.awaited_response_id is not None:
            return PreparedCorrelation("ALREADY_CORRELATED", revision.awaited_response_id)

        lineage = list(
            self.session.scalars(
                select(AwaitedResponse)
                .join(MessageRevision, MessageRevision.awaited_response_id == AwaitedResponse.id)
                .where(
                    MessageRevision.message_id == message.id,
                    MessageRevision.id != revision.id,
                    MessageRevision.awaited_response_id.is_not(None),
                )
                .distinct()
            )
        )
        if len(lineage) == 1:
            return self._prepare_known(revision, lineage[0], "REVISION_LINEAGE")
        if len(lineage) > 1:
            return self._prepare_ambiguous(lineage, "REVISION_LINEAGE_CONFLICT")

        reply = self._reply_candidates(message)
        if reply:
            narrowed = self._narrow_to_sender(reply, message.sender_identity_id) or reply
            if len(narrowed) == 1:
                return self._prepare_known(revision, narrowed[0], "PROVIDER_REPLY")
            return self._prepare_ambiguous(narrowed, "CORRELATION_AMBIGUITY")

        exact = self._exact_conversation_candidates(message)
        if len(exact) == 1:
            return self._prepare_known(revision, exact[0], "EXACT_CONVERSATION")
        if len(exact) > 1:
            relevant = self._temporally_relevant(exact, message)
            if len(relevant) == 1:
                return self._prepare_known(revision, relevant[0], "RECENCY")
            candidates = relevant or exact
            chosen = self.semantic.choose(revision, candidates)
            if chosen is not None and sum(item.id == chosen for item in candidates) == 1:
                awaited = next(item for item in candidates if item.id == chosen)
                return self._prepare_known(revision, awaited, "SEMANTIC")
            return self._prepare_ambiguous(candidates, "CORRELATION_AMBIGUITY")

        cross = self._same_person_other_conversation(message)
        if cross:
            return self._prepare_ambiguous(cross, "CROSS_CONVERSATION_RESPONSE")
        return PreparedCorrelation("UNMATCHED")

    def apply_prepared(self, revision_id: int, plan: PreparedCorrelation) -> CorrelationResult:
        revision = self.session.get(MessageRevision, revision_id)
        if revision is None:
            raise DomainError("revision not found")
        message = self.session.get(Message, revision.message_id)
        if message is None or message.current_revision_id != revision.id:
            raise StaleWork("revision is not current")
        if plan.outcome == "ALREADY_CORRELATED":
            return CorrelationResult(plan.outcome, plan.awaited_response_id)
        if plan.outcome == "UNMATCHED":
            return CorrelationResult("UNMATCHED")
        if plan.awaited_response_id is not None:
            awaited = self.session.get(AwaitedResponse, plan.awaited_response_id)
            if awaited is None:
                raise StaleWork("prepared awaited response disappeared")
            return self._apply_known(
                revision,
                awaited,
                source=plan.source or "PREPARED",
                classification=plan.classification,
            )
        candidates = [
            candidate
            for candidate_id in plan.candidate_ids
            if (candidate := self.session.get(AwaitedResponse, candidate_id)) is not None
        ]
        if len(candidates) != len(plan.candidate_ids):
            raise StaleWork("prepared correlation candidates changed")
        return self._ambiguous(
            revision,
            candidates,
            plan.decision_type or "CORRELATION_AMBIGUITY",
        )

    def _prepare_known(
        self,
        revision: MessageRevision,
        awaited: AwaitedResponse,
        source: str,
    ) -> PreparedCorrelation:
        return PreparedCorrelation(
            outcome="KNOWN",
            awaited_response_id=awaited.id,
            source=source,
            classification=self.classifier.classify(revision, awaited),
        )

    @staticmethod
    def _prepare_ambiguous(
        candidates: list[AwaitedResponse],
        decision_type: str,
    ) -> PreparedCorrelation:
        return PreparedCorrelation(
            outcome=decision_type,
            candidate_ids=tuple(candidate.id for candidate in candidates),
            decision_type=decision_type,
        )

    def process(self, revision_id: int) -> CorrelationResult:
        revision = self.session.get(MessageRevision, revision_id)
        if revision is None:
            raise DomainError("revision not found")
        message = self.session.get(Message, revision.message_id)
        if message is None or message.current_revision_id != revision.id:
            raise StaleWork("revision is not current")
        if revision.awaited_response_id is not None:
            awaited = self.session.get(AwaitedResponse, revision.awaited_response_id)
            assert awaited is not None
            return CorrelationResult("ALREADY_CORRELATED", awaited.id)

        lineage_candidates = list(
            self.session.scalars(
                select(AwaitedResponse)
                .join(MessageRevision, MessageRevision.awaited_response_id == AwaitedResponse.id)
                .where(
                    MessageRevision.message_id == message.id,
                    MessageRevision.id != revision.id,
                    MessageRevision.awaited_response_id.is_not(None),
                )
                .distinct()
            )
        )
        if len(lineage_candidates) == 1:
            return self._apply_known(revision, lineage_candidates[0], source="REVISION_LINEAGE")
        if len(lineage_candidates) > 1:
            return self._ambiguous(revision, lineage_candidates, "REVISION_LINEAGE_CONFLICT")

        reply_candidates = self._reply_candidates(message)
        if reply_candidates:
            candidates = self._narrow_to_sender(reply_candidates, message.sender_identity_id)
            if not candidates:
                candidates = reply_candidates
            return self._resolve_candidates(revision, candidates, source="PROVIDER_REPLY")

        exact = self._exact_conversation_candidates(message)
        if len(exact) == 1:
            return self._apply_known(revision, exact[0], source="EXACT_CONVERSATION")
        if len(exact) > 1:
            temporally_relevant = self._temporally_relevant(exact, message)
            if len(temporally_relevant) == 1:
                return self._apply_known(revision, temporally_relevant[0], source="RECENCY")
            candidates = temporally_relevant or exact
            chosen = self.semantic.choose(revision, candidates)
            if chosen is not None and sum(candidate.id == chosen for candidate in candidates) == 1:
                return self._apply_known(
                    revision,
                    next(candidate for candidate in candidates if candidate.id == chosen),
                    source="SEMANTIC",
                )
            return self._ambiguous(revision, candidates, "CORRELATION_AMBIGUITY")

        cross_conversation = self._same_person_other_conversation(message)
        if cross_conversation:
            return self._ambiguous(revision, cross_conversation, "CROSS_CONVERSATION_RESPONSE")
        return CorrelationResult("UNMATCHED")

    def _reply_candidates(self, message: Message) -> list[AwaitedResponse]:
        if not message.provider_reply_to_message_id:
            return []
        return list(
            self.session.scalars(
                select(AwaitedResponse)
                .join(AwaitedResponsePrompt, AwaitedResponsePrompt.awaited_response_id == AwaitedResponse.id)
                .join(OutboxMessage, OutboxMessage.id == AwaitedResponsePrompt.outbox_message_id)
                .join(OutboxDeliveryAttempt, OutboxDeliveryAttempt.outbox_message_id == OutboxMessage.id)
                .where(OutboxDeliveryAttempt.provider_message_id == message.provider_reply_to_message_id)
            )
        )

    def _exact_conversation_candidates(self, message: Message) -> list[AwaitedResponse]:
        return list(
            self.session.scalars(
                select(AwaitedResponse)
                .join(TaskParticipant, TaskParticipant.id == AwaitedResponse.task_participant_id)
                .where(
                    TaskParticipant.conversation_id == message.conversation_id,
                    AwaitedResponse.status == AwaitedResponseStatus.OPEN,
                )
            )
        )

    def _same_person_other_conversation(self, message: Message) -> list[AwaitedResponse]:
        sender = self.session.get(Identity, message.sender_identity_id)
        if sender is None:
            return []
        return list(
            self.session.scalars(
                select(AwaitedResponse)
                .join(TaskParticipant, TaskParticipant.id == AwaitedResponse.task_participant_id)
                .where(
                    TaskParticipant.person_id == sender.person_id,
                    TaskParticipant.conversation_id != message.conversation_id,
                    AwaitedResponse.status == AwaitedResponseStatus.OPEN,
                )
            )
        )

    def _narrow_to_sender(self, candidates: list[AwaitedResponse], sender_identity_id: int) -> list[AwaitedResponse]:
        sender = self.session.get(Identity, sender_identity_id)
        if sender is None:
            return []
        return [
            candidate
            for candidate in candidates
            if self.session.get(TaskParticipant, candidate.task_participant_id).person_id == sender.person_id  # type: ignore[union-attr]
        ]

    @staticmethod
    def _temporally_relevant(candidates: list[AwaitedResponse], message: Message) -> list[AwaitedResponse]:
        created = message.created_at
        return [
            candidate
            for candidate in candidates
            if candidate.created_at <= created
            and (candidate.expires_at is None or candidate.expires_at >= created)
        ]

    def _resolve_candidates(
        self,
        revision: MessageRevision,
        candidates: list[AwaitedResponse],
        *,
        source: str,
    ) -> CorrelationResult:
        if len(candidates) == 1:
            return self._apply_known(revision, candidates[0], source=source)
        return self._ambiguous(revision, candidates, "CORRELATION_AMBIGUITY")

    def _apply_known(
        self,
        revision: MessageRevision,
        awaited: AwaitedResponse,
        *,
        source: str,
        classification: Classification | None = None,
    ) -> CorrelationResult:
        participant = self.session.get(TaskParticipant, awaited.task_participant_id)
        assert participant is not None
        task = self.session.get(TaskInstance, participant.task_instance_id)
        assert task is not None
        revision.awaited_response_id = awaited.id
        if task.status is not TaskStatus.ACTIVE:
            decision = self._decision(
                revision,
                "LATE_TERMINAL_MESSAGE",
                {"awaited_response_id": awaited.id, "correlation_source": source},
                [],
            )
            self.session.flush()
            return CorrelationResult("LATE_TERMINAL", awaited.id, decision.id)

        classification = classification or self.classifier.classify(revision, awaited)
        proposals_to_create = self._reconcile_previous_effects(
            revision,
            participant,
            classification,
        )
        if classification.kind == "AMBIGUOUS":
            awaited.status = AwaitedResponseStatus.AMBIGUOUS
            self.session.flush()
            return CorrelationResult("INTERPRETATION_AMBIGUOUS", awaited.id)
        if classification.kind == "AVAILABILITY" and classification.availability is not None:
            AvailabilityService(self.session).apply(
                participant.id,
                revision.id,
                classification.availability,
                classification.evidence,
            )
            awaited.status = AwaitedResponseStatus.SATISFIED
        elif classification.kind == "COUNTERPROPOSAL" and classification.proposals:
            for atomic in proposals_to_create:
                proposal = Proposal(
                        task_instance_id=task.id,
                        proposed_by_participant_id=participant.id,
                        source_message_revision_id=revision.id,
                        field=atomic.field,
                        operation=atomic.operation,
                        old_value=atomic.old_value,
                        proposed_value=atomic.proposed_value,
                        status=ProposalStatus.PENDING,
                    )
                self.session.add(proposal)
                self.session.flush()
                decision = DecisionService(self.session).create(
                    decision_type="COUNTERPROPOSAL",
                    subject_kind="proposal",
                    subject_id=proposal.id,
                    context={
                        "field": proposal.field,
                        "operation": proposal.operation,
                        "proposed_value": proposal.proposed_value,
                    },
                    task_instance_id=task.id,
                    parent_terminal_policy=ParentTerminalPolicy.TERMINATE,
                )
                if self.owner_chat_id is not None:
                    prompt = OutboxService(self.session).create_owner(
                        telegram_chat_id=self.owner_chat_id,
                        task_instance_id=task.id,
                        final_text=(
                            f"Participant proposed {proposal.field} {proposal.operation}: "
                            f"{proposal.proposed_value}. Reply `approve` or `reject`."
                        ),
                        message_kind=MessageKind.NOTIFICATION,
                        idempotency_key=f"proposal:{proposal.id}:owner-prompt",
                        parent_terminal_policy=ParentTerminalPolicy.TERMINATE,
                    )
                    self.session.add(
                        DecisionRequestPrompt(
                            decision_request_id=decision.id,
                            outbox_message_id=prompt.id,
                        )
                    )
            awaited.status = AwaitedResponseStatus.SATISFIED
        else:
            awaited.status = AwaitedResponseStatus.AMBIGUOUS
            self.session.flush()
            return CorrelationResult("INTERPRETATION_AMBIGUOUS", awaited.id)
        self.session.flush()
        return CorrelationResult("CORRELATED", awaited.id)

    def _reconcile_previous_effects(
        self,
        revision: MessageRevision,
        participant: TaskParticipant,
        classification: Classification,
    ) -> tuple[AtomicProposal, ...]:
        """Diff semantic effects owned by earlier revisions of the same message."""
        prior_revision_ids = list(
            self.session.scalars(
                select(MessageRevision.id).where(
                    MessageRevision.message_id == revision.message_id,
                    MessageRevision.id != revision.id,
                )
            )
        )
        if not prior_revision_ids:
            return classification.proposals

        if (
            participant.availability_source_revision_id in prior_revision_ids
            and classification.kind != "AVAILABILITY"
        ):
            participant.availability_status = AvailabilityStatus.UNKNOWN
            participant.availability_evidence = None
            participant.availability_source_revision_id = None
            participant.updated_at = utc_now()
            self.session.add(
                TaskEvent(
                    task_instance_id=participant.task_instance_id,
                    task_participant_id=participant.id,
                    source_message_revision_id=revision.id,
                    event_type="AVAILABILITY_RETRACTED_BY_EDIT",
                    payload_json={},
                )
            )

        pending = list(
            self.session.scalars(
                select(Proposal).where(
                    Proposal.source_message_revision_id.in_(prior_revision_ids),
                    Proposal.status == ProposalStatus.PENDING,
                )
            )
        )
        unmatched = list(classification.proposals if classification.kind == "COUNTERPROPOSAL" else ())
        timestamp = utc_now()
        for proposal in pending:
            matching_index = next(
                (
                    index
                    for index, atomic in enumerate(unmatched)
                    if (
                        proposal.field,
                        proposal.operation,
                        proposal.old_value,
                        proposal.proposed_value,
                    )
                    == (
                        atomic.field,
                        atomic.operation,
                        atomic.old_value,
                        atomic.proposed_value,
                    )
                ),
                None,
            )
            if matching_index is not None:
                proposal.source_message_revision_id = revision.id
                unmatched.pop(matching_index)
                continue
            proposal.status = ProposalStatus.SUPERSEDED
            proposal.resolved_at = timestamp
            decisions = list(
                self.session.scalars(
                    select(DecisionRequest).where(
                        DecisionRequest.proposal_id == proposal.id,
                        DecisionRequest.status == DecisionStatus.PENDING,
                    )
                )
            )
            for decision in decisions:
                decision.status = DecisionStatus.CLOSED
                decision.close_reason = DecisionCloseReason.SUBJECT_RESOLVED
                decision.resolved_at = timestamp
        return tuple(unmatched)

    def _ambiguous(
        self,
        revision: MessageRevision,
        candidates: list[AwaitedResponse],
        decision_type: str,
    ) -> CorrelationResult:
        decision = self._decision(
            revision,
            decision_type,
            {"candidate_awaited_response_ids": [candidate.id for candidate in candidates]},
            candidates,
        )
        self.session.flush()
        return CorrelationResult(decision_type, decision_request_id=decision.id)

    def _decision(
        self,
        revision: MessageRevision,
        decision_type: str,
        context: dict[str, object],
        candidates: list[AwaitedResponse],
    ) -> DecisionRequest:
        existing = self.session.scalar(
            select(DecisionRequest).where(
                DecisionRequest.message_revision_id == revision.id,
                DecisionRequest.type == decision_type,
            )
        )
        if existing is not None:
            return existing
        decision = DecisionService(self.session).create(
            decision_type=decision_type,
            subject_kind="message_revision",
            subject_id=revision.id,
            context=context,
            parent_terminal_policy=ParentTerminalPolicy.SURVIVE,
        )
        self.session.add_all(
            [
                DecisionRequestAwaitedResponseCandidate(
                    decision_request_id=decision.id,
                    awaited_response_id=candidate.id,
                )
                for candidate in candidates
            ]
        )
        if self.owner_chat_id is not None and candidates:
            lines = ["Choose the response this message belongs to:"]
            for candidate in candidates:
                participant = self.session.get(TaskParticipant, candidate.task_participant_id)
                assert participant is not None
                task = self.session.get(TaskInstance, participant.task_instance_id)
                assert task is not None
                lines.append(
                    f"- {candidate.id}: {task.topic_key} at {task.scheduled_at.isoformat()}, "
                    f"participant {participant.person_id}, conversation {participant.conversation_id}, "
                    f"expected {candidate.expected_response_type}; reply `select {candidate.id}`"
                )
            prompt = OutboxService(self.session).create_owner(
                telegram_chat_id=self.owner_chat_id,
                final_text="\n".join(lines),
                message_kind=MessageKind.NOTIFICATION,
                idempotency_key=f"decision:{decision.id}:owner-prompt",
                task_instance_id=decision.task_instance_id,
                parent_terminal_policy=ParentTerminalPolicy.SURVIVE,
            )
            self.session.add(
                DecisionRequestPrompt(
                    decision_request_id=decision.id,
                    outbox_message_id=prompt.id,
                )
            )
        return decision
