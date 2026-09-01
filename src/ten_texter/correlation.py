from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from ten_texter.decision_prompts import (
    correlation_ambiguity_prompt,
    counterproposal_prompt,
)
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
    Person,
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
class AvailabilitySubjectCandidate:
    task_participant_id: int
    display_name: str


# Matches TaskPlan's v1 maximum participant-reference contract. Exact-task state
# created outside that path can exceed it, so completeness is reported explicitly.
MAX_AVAILABILITY_SUBJECT_CANDIDATES = 100


@dataclass(frozen=True, slots=True)
class AvailabilitySubjectContext:
    candidates: tuple[AvailabilitySubjectCandidate, ...]
    complete: bool = True
    ambiguous_display_names: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        counts: dict[str, int] = {}
        for candidate in self.candidates:
            normalized = " ".join(candidate.display_name.casefold().split())
            counts[normalized] = counts.get(normalized, 0) + 1
        object.__setattr__(
            self,
            "ambiguous_display_names",
            tuple(
                sorted(
                    name
                    for name, count in counts.items()
                    if not name or count > 1
                )
            ),
        )

    @property
    def selectable_ids(self) -> frozenset[int]:
        if not self.complete:
            return frozenset()
        ambiguous = set(self.ambiguous_display_names)
        return frozenset(
            candidate.task_participant_id
            for candidate in self.candidates
            if " ".join(candidate.display_name.casefold().split()) not in ambiguous
        )


@dataclass(frozen=True, slots=True)
class Classification:
    kind: str
    availability: AvailabilityStatus | None = None
    evidence: AvailabilityEvidence = AvailabilityEvidence.FIRST_PARTY
    subject_task_participant_id: int | None = None
    proposals: tuple[AtomicProposal, ...] = field(default_factory=tuple)


class Classifier(Protocol):
    def classify(
        self,
        revision: MessageRevision,
        awaited_response: AwaitedResponse,
        third_party_subject_context: AvailabilitySubjectContext,
    ) -> Classification: ...


def third_party_subject_context(
    session: Session,
    revision: MessageRevision,
    awaited_response: AwaitedResponse,
) -> AvailabilitySubjectContext:
    """Return other participants from the exact correlated task as a bounded model set."""
    awaited_participant = session.get(
        TaskParticipant,
        awaited_response.task_participant_id,
    )
    message = session.get(Message, revision.message_id)
    sender = session.get(Identity, message.sender_identity_id) if message is not None else None
    if awaited_participant is None or sender is None:
        raise DomainError("availability subject context is unavailable")
    rows = list(
        session.execute(
            select(TaskParticipant.id, Person.display_name)
            .join(Person, Person.id == TaskParticipant.person_id)
            .where(
                TaskParticipant.task_instance_id == awaited_participant.task_instance_id,
                TaskParticipant.person_id != sender.person_id,
            )
            .order_by(TaskParticipant.id)
            .limit(MAX_AVAILABILITY_SUBJECT_CANDIDATES + 1)
        )
    )
    complete = len(rows) <= MAX_AVAILABILITY_SUBJECT_CANDIDATES
    return AvailabilitySubjectContext(
        candidates=tuple(
            AvailabilitySubjectCandidate(
                task_participant_id=participant_id,
                display_name=display_name,
            )
            for participant_id, display_name in rows[:MAX_AVAILABILITY_SUBJECT_CANDIDATES]
        ),
        complete=complete,
    )


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
            classification=self.classifier.classify(
                revision,
                awaited,
                third_party_subject_context(self.session, revision, awaited),
            ),
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

        classification = classification or self.classifier.classify(
            revision,
            awaited,
            third_party_subject_context(self.session, revision, awaited),
        )
        availability_target: TaskParticipant | None = None
        if classification.kind == "AVAILABILITY" and classification.availability is not None:
            availability_target = self._availability_target(
                revision,
                task,
                classification,
            )
        prior_revision_ids = list(
            self.session.scalars(
                select(MessageRevision.id).where(
                    MessageRevision.message_id == revision.message_id,
                    MessageRevision.id != revision.id,
                )
            )
        )
        lineage_owns_prior_effect, independent_effect_exists = (
            self._awaited_response_effect_ownership(
                revision,
                awaited,
                participant,
                prior_revision_ids,
            )
        )
        may_replace_awaited_status = (
            awaited.status is not AwaitedResponseStatus.SATISFIED
            or (lineage_owns_prior_effect and not independent_effect_exists)
        )
        proposals_to_create = self._reconcile_previous_effects(
            revision,
            participant,
            classification,
            availability_target,
            prior_revision_ids,
        )
        if classification.kind == "AMBIGUOUS":
            if may_replace_awaited_status:
                awaited.status = AwaitedResponseStatus.AMBIGUOUS
            self.session.flush()
            return CorrelationResult("INTERPRETATION_AMBIGUOUS", awaited.id)
        if classification.kind == "AVAILABILITY" and classification.availability is not None:
            assert availability_target is not None
            applied = AvailabilityService(self.session).apply(
                availability_target.id,
                revision.id,
                classification.availability,
                classification.evidence,
            )
            if availability_target.id == participant.id:
                awaited.status = AwaitedResponseStatus.SATISFIED
            elif may_replace_awaited_status:
                awaited.status = AwaitedResponseStatus.OPEN
            if self.owner_chat_id is not None and applied:
                person = self.session.get(Person, availability_target.person_id)
                assert person is not None
                status = availability_target.availability_status.value.lower()
                OutboxService(self.session).create_owner(
                    telegram_chat_id=self.owner_chat_id,
                    task_instance_id=task.id,
                    final_text=(
                        f"{person.display_name} is {status} for {task.topic_key}."
                    ),
                    message_kind=MessageKind.NOTIFICATION,
                    idempotency_key=(
                        f"task:{task.id}:availability:{availability_target.id}:revision:{revision.id}"
                    ),
                    parent_terminal_policy=ParentTerminalPolicy.SURVIVE,
                )
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
                        final_text=counterproposal_prompt(
                            proposal.field,
                            proposal.operation,
                            proposal.proposed_value,
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
            if may_replace_awaited_status:
                awaited.status = AwaitedResponseStatus.AMBIGUOUS
            self.session.flush()
            return CorrelationResult("INTERPRETATION_AMBIGUOUS", awaited.id)
        self.session.flush()
        return CorrelationResult("CORRELATED", awaited.id)

    def _awaited_response_effect_ownership(
        self,
        revision: MessageRevision,
        awaited: AwaitedResponse,
        participant: TaskParticipant,
        prior_revision_ids: list[int],
    ) -> tuple[bool, bool]:
        """Reconstruct semantic ownership from exact revision-backed state."""
        source_revision = (
            self.session.get(
                MessageRevision,
                participant.availability_source_revision_id,
            )
            if participant.availability_source_revision_id is not None
            else None
        )
        lineage_availability = (
            participant.availability_source_revision_id in prior_revision_ids
        )
        independent_availability = (
            source_revision is not None
            and source_revision.message_id != revision.message_id
            and source_revision.awaited_response_id == awaited.id
        )
        lineage_proposal = self.session.scalar(
            select(Proposal.id).where(
                Proposal.source_message_revision_id.in_(prior_revision_ids),
                Proposal.proposed_by_participant_id == participant.id,
                Proposal.status != ProposalStatus.SUPERSEDED,
            )
        )
        independent_proposal = self.session.scalar(
            select(Proposal.id)
            .join(
                MessageRevision,
                MessageRevision.id == Proposal.source_message_revision_id,
            )
            .where(
                MessageRevision.message_id != revision.message_id,
                MessageRevision.awaited_response_id == awaited.id,
                Proposal.proposed_by_participant_id == participant.id,
                Proposal.status != ProposalStatus.SUPERSEDED,
            )
        )
        return (
            lineage_availability or lineage_proposal is not None,
            independent_availability or independent_proposal is not None,
        )

    def _availability_target(
        self,
        revision: MessageRevision,
        task: TaskInstance,
        classification: Classification,
    ) -> TaskParticipant:
        message = self.session.get(Message, revision.message_id)
        sender = (
            self.session.get(Identity, message.sender_identity_id)
            if message is not None
            else None
        )
        if sender is None:
            raise DomainError("availability sender identity is unavailable")
        if classification.evidence is AvailabilityEvidence.FIRST_PARTY:
            if classification.subject_task_participant_id is not None:
                raise DomainError("first-party availability cannot select another subject")
            target = self.session.scalar(
                select(TaskParticipant).where(
                    TaskParticipant.task_instance_id == task.id,
                    TaskParticipant.person_id == sender.person_id,
                )
            )
            if target is None:
                raise DomainError("first-party availability sender is not a task participant")
            return target
        subject_id = classification.subject_task_participant_id
        if subject_id is None:
            raise DomainError("third-party availability requires an exact subject")
        target = self.session.get(TaskParticipant, subject_id)
        if target is None or target.task_instance_id != task.id:
            raise DomainError("availability subject must belong to the same task")
        if target.person_id == sender.person_id:
            raise DomainError("third-party availability subject must be another person")
        return target

    def _reconcile_previous_effects(
        self,
        revision: MessageRevision,
        participant: TaskParticipant,
        classification: Classification,
        availability_target: TaskParticipant | None,
        prior_revision_ids: list[int],
    ) -> tuple[AtomicProposal, ...]:
        """Diff semantic effects owned by earlier revisions of the same message."""
        if not prior_revision_ids:
            return classification.proposals

        prior_effect_participants = list(
            self.session.scalars(
                select(TaskParticipant).where(
                    TaskParticipant.task_instance_id == participant.task_instance_id,
                    TaskParticipant.availability_source_revision_id.in_(prior_revision_ids),
                )
            )
        )
        for prior_participant in prior_effect_participants:
            if (
                classification.kind == "AVAILABILITY"
                and availability_target is not None
                and prior_participant.id == availability_target.id
            ):
                continue
            prior_participant.availability_status = AvailabilityStatus.UNKNOWN
            prior_participant.availability_evidence = None
            prior_participant.availability_source_revision_id = None
            prior_participant.updated_at = utc_now()
            self.session.add(
                TaskEvent(
                    task_instance_id=prior_participant.task_instance_id,
                    task_participant_id=prior_participant.id,
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
            prompt_candidates: list[tuple[int, str, str, int, int, str]] = []
            for candidate in sorted(candidates, key=lambda value: value.id):
                participant = self.session.get(TaskParticipant, candidate.task_participant_id)
                assert participant is not None
                task = self.session.get(TaskInstance, participant.task_instance_id)
                assert task is not None
                scheduled_at = task.scheduled_at.replace(
                    tzinfo=task.scheduled_at.tzinfo or UTC
                ).astimezone(UTC).isoformat()
                prompt_candidates.append(
                    (
                        candidate.id,
                        task.topic_key,
                        scheduled_at,
                        participant.person_id,
                        participant.conversation_id,
                        candidate.expected_response_type,
                    )
                )
            prompt = OutboxService(self.session).create_owner(
                telegram_chat_id=self.owner_chat_id,
                final_text=correlation_ambiguity_prompt(prompt_candidates),
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
