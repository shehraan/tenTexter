from __future__ import annotations

from datetime import timedelta

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from ten_texter.correlation import AtomicProposal, Classification, CorrelationOrchestrator
from ten_texter.domain import AvailabilityService, DomainError, TaskService, distinct_response_count
from ten_texter.enums import (
    AttemptResult,
    AvailabilityEvidence,
    AvailabilityStatus,
    AwaitedResponseStatus,
    ContentSupport,
    ConversationKind,
    DecisionStatus,
    MessageKind,
    OutboxStatus,
    ParentTerminalPolicy,
    ProcessingStatus,
    ProposalStatus,
    TaskStatus,
)
from ten_texter.models import (
    AwaitedResponse,
    AwaitedResponsePrompt,
    BeeperOutboxDestination,
    Conversation,
    ConversationParticipant,
    DecisionRequestAwaitedResponseCandidate,
    Identity,
    Message,
    MessageRevision,
    OutboxDeliveryAttempt,
    OutboxMessage,
    Person,
    Proposal,
    TaskInstance,
    TaskParticipant,
    TaskEvent,
    DecisionRequest,
    DecisionRequestPrompt,
)
from ten_texter.control import ProductionOwnerCommandHandler
from ten_texter.model_clients import MessageClassifier
from ten_texter.correlation import MAX_AVAILABILITY_SUBJECT_CANDIDATES
from ten_texter.telegram import TelegramControlGateway
from ten_texter.outbox import OutboxService
from ten_texter.policy import DatabaseContextProvider
from ten_texter.validator import DatabaseValidatorContextProvider, IndependentMessageValidator
from tests.test_schema import NOW, seed_core


class Semantic:
    def __init__(self, choice: int | None = None):
        self.choice = choice

    def choose(self, *_: object) -> int | None:
        return self.choice


class Classifier:
    def __init__(self, classification: Classification):
        self.classification = classification

    def classify(self, *_: object) -> Classification:
        return self.classification


def orchestrator(session: Session, classification: Classification, choice: int | None = None) -> CorrelationOrchestrator:
    return CorrelationOrchestrator(
        session,
        semantic=Semantic(choice),
        classifier=Classifier(classification),
    )


def awaited(session: Session, core: dict[str, object], *, created_at=NOW - timedelta(minutes=5)) -> AwaitedResponse:
    row = AwaitedResponse(
        task_participant_id=core["participant"].id,
        expected_response_type="availability",
        status=AwaitedResponseStatus.OPEN,
        created_at=created_at,
        expires_at=NOW + timedelta(hours=1),
    )
    session.add(row)
    session.flush()
    return row


def add_participant(
    session: Session,
    core: dict[str, object],
    *,
    name: str,
    task_instance_id: int | None = None,
) -> TaskParticipant:
    person = Person(display_name=name, metadata_json={})
    session.add(person)
    session.flush()
    identity = Identity(
        person_id=person.id,
        beeper_user_id=f"beeper:{name.casefold()}:{person.id}",
        network="discord",
        metadata_json={},
    )
    conversation = Conversation(
        beeper_conversation_id=f"conv:{name.casefold()}:{person.id}",
        network="discord",
        kind=ConversationKind.DIRECT,
        counterparty_person_id=person.id,
        metadata_json={},
    )
    session.add_all([identity, conversation])
    session.flush()
    session.add(
        ConversationParticipant(
            conversation_id=conversation.id,
            identity_id=identity.id,
        )
    )
    participant = TaskParticipant(
        task_instance_id=task_instance_id or core["task"].id,
        person_id=person.id,
        conversation_id=conversation.id,
        availability_status=AvailabilityStatus.UNKNOWN,
    )
    session.add(participant)
    session.flush()
    return participant


def test_exact_conversation_single_open_response_correlates(db_session: Session) -> None:
    core = seed_core(db_session)
    response = awaited(db_session, core)
    result = orchestrator(
        db_session,
        Classification(kind="AVAILABILITY", availability=AvailabilityStatus.AVAILABLE),
    ).process(core["revision"].id)
    assert result.awaited_response_id == response.id
    assert response.status is AwaitedResponseStatus.SATISFIED
    assert core["revision"].awaited_response_id == response.id
    assert core["participant"].availability_status is AvailabilityStatus.AVAILABLE


def test_first_party_availability_mutates_and_satisfies_sender(
    db_session: Session,
) -> None:
    core = seed_core(db_session)
    response = awaited(db_session, core)

    orchestrator(
        db_session,
        Classification(
            kind="AVAILABILITY",
            availability=AvailabilityStatus.AVAILABLE,
            evidence=AvailabilityEvidence.FIRST_PARTY,
        ),
    ).process(core["revision"].id)

    assert core["participant"].availability_status is AvailabilityStatus.AVAILABLE
    assert core["participant"].availability_evidence is AvailabilityEvidence.FIRST_PARTY
    assert response.status is AwaitedResponseStatus.SATISFIED


def test_first_party_unavailability_mutates_sender(db_session: Session) -> None:
    core = seed_core(db_session)
    response = awaited(db_session, core)

    orchestrator(
        db_session,
        Classification(
            kind="AVAILABILITY",
            availability=AvailabilityStatus.UNAVAILABLE,
            evidence=AvailabilityEvidence.FIRST_PARTY,
        ),
    ).process(core["revision"].id)

    assert core["participant"].availability_status is AvailabilityStatus.UNAVAILABLE
    assert core["participant"].availability_evidence is AvailabilityEvidence.FIRST_PARTY
    assert response.status is AwaitedResponseStatus.SATISFIED


def test_third_party_availability_without_subject_fails_closed(
    db_session: Session,
) -> None:
    core = seed_core(db_session)
    response = awaited(db_session, core)

    with pytest.raises(DomainError, match="requires an exact subject"):
        orchestrator(
            db_session,
            Classification(
                kind="AVAILABILITY",
                availability=AvailabilityStatus.AVAILABLE,
                evidence=AvailabilityEvidence.THIRD_PARTY,
            ),
        ).process(core["revision"].id)

    assert core["participant"].availability_status is AvailabilityStatus.UNKNOWN
    assert response.status is AwaitedResponseStatus.OPEN


@pytest.mark.parametrize(
    "status",
    [AvailabilityStatus.AVAILABLE, AvailabilityStatus.UNAVAILABLE],
)
def test_third_party_availability_mutates_named_participant_only(
    db_session: Session,
    status: AvailabilityStatus,
) -> None:
    core = seed_core(db_session)
    response = awaited(db_session, core)
    kyran = add_participant(db_session, core, name="Kyran")

    orchestrator(
        db_session,
        Classification(
            kind="AVAILABILITY",
            availability=status,
            evidence=AvailabilityEvidence.THIRD_PARTY,
            subject_task_participant_id=kyran.id,
        ),
    ).process(core["revision"].id)

    assert kyran.availability_status is status
    assert kyran.availability_evidence is AvailabilityEvidence.THIRD_PARTY
    assert core["participant"].availability_status is AvailabilityStatus.UNKNOWN
    assert response.status is AwaitedResponseStatus.OPEN


def test_real_classifier_receives_only_other_participants_from_exact_task(
    db_session: Session,
) -> None:
    core = seed_core(db_session)
    response = awaited(db_session, core)
    kyran = add_participant(db_session, core, name="Kyran")
    other_task = TaskInstance(
        scheduled_at=NOW + timedelta(days=2),
        duration_minutes=60,
        topic_key="dinner",
    )
    db_session.add(other_task)
    db_session.flush()
    outsider = add_participant(
        db_session,
        core,
        name="Outside",
        task_instance_id=other_task.id,
    )

    class Backend:
        def __init__(self) -> None:
            self.payload: dict[str, object] | None = None

        def infer(self, *, operation: str, payload: dict[str, object]) -> dict[str, object]:
            assert operation == "message_classifier"
            self.payload = payload
            return {
                "kind": "AVAILABILITY",
                "availability": "AVAILABLE",
                "evidence": "THIRD_PARTY",
                "subject_task_participant_id": kyran.id,
                "proposals": [],
            }

    backend = Backend()
    CorrelationOrchestrator(
        db_session,
        semantic=Semantic(),
        classifier=MessageClassifier(backend),
    ).process(core["revision"].id)

    assert backend.payload is not None
    candidates = backend.payload["third_party_subject_candidates"]
    assert candidates == [
        {"task_participant_id": kyran.id, "display_name": "Kyran"}
    ]
    candidate_ids = {candidate["task_participant_id"] for candidate in candidates}
    assert core["participant"].id not in candidate_ids
    assert outsider.id not in candidate_ids
    assert kyran.availability_status is AvailabilityStatus.AVAILABLE
    assert response.status is AwaitedResponseStatus.OPEN


def test_over_bound_subject_context_preserves_first_party_and_disallows_third_party(
    db_session: Session,
) -> None:
    core = seed_core(db_session)
    response = awaited(db_session, core)
    for index in range(MAX_AVAILABILITY_SUBJECT_CANDIDATES + 1):
        add_participant(db_session, core, name=f"Person {index}")

    class Backend:
        def __init__(self) -> None:
            self.payload: dict[str, object] | None = None

        def infer(self, *, operation: str, payload: dict[str, object]) -> dict[str, object]:
            self.payload = payload
            return {
                "kind": "AVAILABILITY",
                "availability": "AVAILABLE",
                "evidence": "FIRST_PARTY",
                "subject_task_participant_id": None,
                "proposals": [],
            }

    backend = Backend()
    CorrelationOrchestrator(
        db_session,
        semantic=Semantic(),
        classifier=MessageClassifier(backend),
    ).process(core["revision"].id)

    assert backend.payload is not None
    assert len(backend.payload["third_party_subject_candidates"]) <= (
        MAX_AVAILABILITY_SUBJECT_CANDIDATES
    )
    assert backend.payload["third_party_subject_candidates_complete"] is False
    assert core["participant"].availability_status is AvailabilityStatus.AVAILABLE
    assert response.status is AwaitedResponseStatus.SATISFIED


def test_duplicate_candidate_names_force_real_classifier_ambiguity(
    db_session: Session,
) -> None:
    core = seed_core(db_session)
    response = awaited(db_session, core)
    first = add_participant(db_session, core, name="Sam Lee")
    second = add_participant(db_session, core, name="  sam   lee ")

    class Backend:
        def __init__(self) -> None:
            self.payload: dict[str, object] | None = None

        def infer(self, *, operation: str, payload: dict[str, object]) -> dict[str, object]:
            self.payload = payload
            return {
                "kind": "AMBIGUOUS",
                "availability": None,
                "evidence": "FIRST_PARTY",
                "subject_task_participant_id": None,
                "proposals": [],
            }

    backend = Backend()
    result = CorrelationOrchestrator(
        db_session,
        semantic=Semantic(),
        classifier=MessageClassifier(backend),
    ).process(core["revision"].id)

    assert backend.payload is not None
    assert backend.payload["ambiguous_third_party_display_names"] == ["sam lee"]
    assert len(backend.payload["third_party_subject_candidates"]) == 2
    assert "duplicate display names" in backend.payload["trusted_instructions"]
    assert result.outcome == "INTERPRETATION_AMBIGUOUS"
    assert first.availability_status is AvailabilityStatus.UNKNOWN
    assert second.availability_status is AvailabilityStatus.UNKNOWN
    assert core["participant"].availability_status is AvailabilityStatus.UNKNOWN
    assert response.status is AwaitedResponseStatus.AMBIGUOUS


def test_multiple_named_candidates_force_real_classifier_ambiguity(
    db_session: Session,
) -> None:
    core = seed_core(db_session)
    response = awaited(db_session, core)
    kyran = add_participant(db_session, core, name="Kyran")
    amith = add_participant(db_session, core, name="Amith")
    classified_revision = MessageRevision(
        message_id=core["message"].id,
        provider_revision_key="r2",
        provider_sequence=2,
        content_hash="f" * 64,
        text="Kyran and Amith are free",
        content_support=ContentSupport.SUPPORTED,
        processing_status=ProcessingStatus.PENDING,
    )
    db_session.add(classified_revision)
    db_session.flush()
    core["message"].current_revision_id = classified_revision.id
    db_session.flush()

    class Backend:
        def __init__(self) -> None:
            self.payload: dict[str, object] | None = None

        def infer(self, *, operation: str, payload: dict[str, object]) -> dict[str, object]:
            self.payload = payload
            return {
                "kind": "AMBIGUOUS",
                "availability": None,
                "evidence": "FIRST_PARTY",
                "subject_task_participant_id": None,
                "proposals": [],
            }

    backend = Backend()
    result = CorrelationOrchestrator(
        db_session,
        semantic=Semantic(),
        classifier=MessageClassifier(backend),
    ).process(classified_revision.id)

    assert backend.payload is not None
    assert backend.payload["untrusted_participant_text"] == "Kyran and Amith are free"
    assert backend.payload["third_party_subject_candidates"] == [
        {"task_participant_id": kyran.id, "display_name": "Kyran"},
        {"task_participant_id": amith.id, "display_name": "Amith"},
    ]
    assert "multiple people are reported" in backend.payload["trusted_instructions"]
    assert result.outcome == "INTERPRETATION_AMBIGUOUS"
    assert core["participant"].availability_status is AvailabilityStatus.UNKNOWN
    assert kyran.availability_status is AvailabilityStatus.UNKNOWN
    assert amith.availability_status is AvailabilityStatus.UNKNOWN
    assert response.status is AwaitedResponseStatus.AMBIGUOUS


def test_unresolved_third_party_subject_does_not_mutate_availability(
    db_session: Session,
) -> None:
    core = seed_core(db_session)
    response = awaited(db_session, core)
    kyran = add_participant(db_session, core, name="Kyran")

    result = orchestrator(
        db_session,
        Classification(kind="AMBIGUOUS"),
    ).process(core["revision"].id)

    assert result.outcome == "INTERPRETATION_AMBIGUOUS"
    assert core["participant"].availability_status is AvailabilityStatus.UNKNOWN
    assert kyran.availability_status is AvailabilityStatus.UNKNOWN
    assert response.status is AwaitedResponseStatus.AMBIGUOUS


def test_non_task_third_party_subject_is_rejected_without_mutation(
    db_session: Session,
) -> None:
    core = seed_core(db_session)
    response = awaited(db_session, core)
    other_task = TaskInstance(
        scheduled_at=NOW + timedelta(days=2),
        duration_minutes=60,
        topic_key="dinner",
    )
    db_session.add(other_task)
    db_session.flush()
    outsider = add_participant(
        db_session,
        core,
        name="Outsider",
        task_instance_id=other_task.id,
    )

    with pytest.raises(DomainError, match="same task"):
        orchestrator(
            db_session,
            Classification(
                kind="AVAILABILITY",
                availability=AvailabilityStatus.AVAILABLE,
                evidence=AvailabilityEvidence.THIRD_PARTY,
                subject_task_participant_id=outsider.id,
            ),
        ).process(core["revision"].id)

    assert core["participant"].availability_status is AvailabilityStatus.UNKNOWN
    assert response.status is AwaitedResponseStatus.OPEN


def test_third_party_availability_replay_is_idempotent(db_session: Session) -> None:
    core = seed_core(db_session)
    awaited(db_session, core)
    kyran = add_participant(db_session, core, name="Kyran")
    service = orchestrator(
        db_session,
        Classification(
            kind="AVAILABILITY",
            availability=AvailabilityStatus.AVAILABLE,
            evidence=AvailabilityEvidence.THIRD_PARTY,
            subject_task_participant_id=kyran.id,
        ),
    )

    service.process(core["revision"].id)
    service.process(core["revision"].id)

    events = list(
        db_session.scalars(
            select(TaskEvent).where(
                TaskEvent.task_participant_id == kyran.id,
                TaskEvent.event_type == "AVAILABILITY_UPDATED",
            )
        )
    )
    assert len(events) == 1


def test_edit_retracts_third_party_effect_and_applies_first_party_effect(
    db_session: Session,
) -> None:
    core = seed_core(db_session)
    response = awaited(db_session, core)
    kyran = add_participant(db_session, core, name="Kyran")
    orchestrator(
        db_session,
        Classification(
            kind="AVAILABILITY",
            availability=AvailabilityStatus.AVAILABLE,
            evidence=AvailabilityEvidence.THIRD_PARTY,
            subject_task_participant_id=kyran.id,
        ),
    ).process(core["revision"].id)
    edit = MessageRevision(
        message_id=core["message"].id,
        provider_revision_key="r2",
        provider_sequence=2,
        content_hash="b" * 64,
        text="Actually, I can't make it",
        content_support=ContentSupport.SUPPORTED,
        processing_status=ProcessingStatus.PENDING,
    )
    db_session.add(edit)
    db_session.flush()
    core["message"].current_revision_id = edit.id
    db_session.flush()

    orchestrator(
        db_session,
        Classification(
            kind="AVAILABILITY",
            availability=AvailabilityStatus.UNAVAILABLE,
            evidence=AvailabilityEvidence.FIRST_PARTY,
        ),
    ).process(edit.id)

    assert kyran.availability_status is AvailabilityStatus.UNKNOWN
    assert kyran.availability_evidence is None
    assert core["participant"].availability_status is AvailabilityStatus.UNAVAILABLE
    assert core["participant"].availability_evidence is AvailabilityEvidence.FIRST_PARTY
    assert response.status is AwaitedResponseStatus.SATISFIED


def test_edit_retracts_first_party_effect_and_opens_sender_for_third_party(
    db_session: Session,
) -> None:
    core = seed_core(db_session)
    response = awaited(db_session, core)
    kyran = add_participant(db_session, core, name="Kyran")
    orchestrator(
        db_session,
        Classification(
            kind="AVAILABILITY",
            availability=AvailabilityStatus.AVAILABLE,
            evidence=AvailabilityEvidence.FIRST_PARTY,
        ),
    ).process(core["revision"].id)
    edit = MessageRevision(
        message_id=core["message"].id,
        provider_revision_key="r2",
        provider_sequence=2,
        content_hash="c" * 64,
        text="Actually, Kyran is free",
        content_support=ContentSupport.SUPPORTED,
        processing_status=ProcessingStatus.PENDING,
    )
    db_session.add(edit)
    db_session.flush()
    core["message"].current_revision_id = edit.id
    db_session.flush()

    orchestrator(
        db_session,
        Classification(
            kind="AVAILABILITY",
            availability=AvailabilityStatus.AVAILABLE,
            evidence=AvailabilityEvidence.THIRD_PARTY,
            subject_task_participant_id=kyran.id,
        ),
    ).process(edit.id)

    assert core["participant"].availability_status is AvailabilityStatus.UNKNOWN
    assert core["participant"].availability_evidence is None
    assert kyran.availability_status is AvailabilityStatus.AVAILABLE
    assert kyran.availability_evidence is AvailabilityEvidence.THIRD_PARTY
    assert response.status is AwaitedResponseStatus.OPEN


def test_third_party_edit_does_not_reopen_response_satisfied_by_separate_message(
    db_session: Session,
) -> None:
    core = seed_core(db_session)
    response = awaited(db_session, core)
    kyran = add_participant(db_session, core, name="Kyran")
    orchestrator(
        db_session,
        Classification(
            kind="AVAILABILITY",
            availability=AvailabilityStatus.AVAILABLE,
            evidence=AvailabilityEvidence.THIRD_PARTY,
            subject_task_participant_id=kyran.id,
        ),
    ).process(core["revision"].id)
    assert response.status is AwaitedResponseStatus.OPEN

    own_message = Message(
        conversation_id=core["conversation"].id,
        provider_message_id="own-message",
        sender_identity_id=core["identity"].id,
        created_at=NOW + timedelta(minutes=1),
    )
    db_session.add(own_message)
    db_session.flush()
    own_revision = MessageRevision(
        message_id=own_message.id,
        provider_revision_key="r1",
        provider_sequence=2,
        content_hash="d" * 64,
        text="I'm free",
        content_support=ContentSupport.SUPPORTED,
        processing_status=ProcessingStatus.PENDING,
    )
    db_session.add(own_revision)
    db_session.flush()
    own_message.current_revision_id = own_revision.id
    db_session.flush()
    orchestrator(
        db_session,
        Classification(
            kind="AVAILABILITY",
            availability=AvailabilityStatus.AVAILABLE,
            evidence=AvailabilityEvidence.FIRST_PARTY,
        ),
    ).process(own_revision.id)
    assert response.status is AwaitedResponseStatus.SATISFIED

    third_party_edit = MessageRevision(
        message_id=core["message"].id,
        provider_revision_key="r2",
        provider_sequence=3,
        content_hash="e" * 64,
        text="Actually, Kyran can't make it",
        content_support=ContentSupport.SUPPORTED,
        processing_status=ProcessingStatus.PENDING,
    )
    db_session.add(third_party_edit)
    db_session.flush()
    core["message"].current_revision_id = third_party_edit.id
    db_session.flush()
    orchestrator(
        db_session,
        Classification(
            kind="AVAILABILITY",
            availability=AvailabilityStatus.UNAVAILABLE,
            evidence=AvailabilityEvidence.THIRD_PARTY,
            subject_task_participant_id=kyran.id,
        ),
    ).process(third_party_edit.id)

    assert response.status is AwaitedResponseStatus.SATISFIED
    assert core["participant"].availability_status is AvailabilityStatus.AVAILABLE
    assert core["participant"].availability_evidence is AvailabilityEvidence.FIRST_PARTY
    assert core["participant"].availability_source_revision_id == own_revision.id
    assert kyran.availability_status is AvailabilityStatus.UNAVAILABLE
    assert kyran.availability_evidence is AvailabilityEvidence.THIRD_PARTY
    assert kyran.availability_source_revision_id == third_party_edit.id


def test_counterproposal_edit_to_third_party_reopens_sender_response(
    db_session: Session,
) -> None:
    core = seed_core(db_session)
    response = awaited(db_session, core)
    kyran = add_participant(db_session, core, name="Kyran")
    orchestrator(
        db_session,
        Classification(
            kind="COUNTERPROPOSAL",
            proposals=(
                AtomicProposal(
                    field="scheduled_at",
                    operation="SET",
                    old_value="2026-08-27T15:00:00Z",
                    proposed_value="2026-08-27T16:00:00Z",
                ),
            ),
        ),
    ).process(core["revision"].id)
    proposal = db_session.scalar(select(Proposal))
    assert proposal is not None
    assert response.status is AwaitedResponseStatus.SATISFIED

    edit = MessageRevision(
        message_id=core["message"].id,
        provider_revision_key="r2",
        provider_sequence=2,
        content_hash="1" * 64,
        text="Actually, Kyran is free",
        content_support=ContentSupport.SUPPORTED,
        processing_status=ProcessingStatus.PENDING,
    )
    db_session.add(edit)
    db_session.flush()
    core["message"].current_revision_id = edit.id
    db_session.flush()
    orchestrator(
        db_session,
        Classification(
            kind="AVAILABILITY",
            availability=AvailabilityStatus.AVAILABLE,
            evidence=AvailabilityEvidence.THIRD_PARTY,
            subject_task_participant_id=kyran.id,
        ),
    ).process(edit.id)

    assert proposal.status is ProposalStatus.SUPERSEDED
    assert response.status is AwaitedResponseStatus.OPEN
    assert core["participant"].availability_status is AvailabilityStatus.UNKNOWN
    assert kyran.availability_status is AvailabilityStatus.AVAILABLE


def test_counterproposal_drops_a_noop_change_before_creating_owner_decision(
    db_session: Session,
) -> None:
    core = seed_core(db_session)
    task = core["task"]
    task.duration_minutes = 30
    db_session.flush()
    awaited(db_session, core)
    current_start = NOW + timedelta(days=1)
    proposed_start = current_start + timedelta(hours=1)

    result = orchestrator(
        db_session,
        Classification(
            kind="COUNTERPROPOSAL",
            proposals=(
                AtomicProposal(
                    field="duration",
                    operation="REPLACE",
                    old_value=None,
                    proposed_value="30 minutes",
                ),
                AtomicProposal(
                    field="scheduled_at",
                    operation="SET",
                    old_value=current_start.isoformat(),
                    proposed_value=proposed_start.isoformat(),
                ),
            ),
        ),
    ).process(core["revision"].id)

    assert result.outcome == "CORRELATED"
    proposals = list(db_session.scalars(select(Proposal).order_by(Proposal.id)))
    assert [(proposal.field, proposal.proposed_value) for proposal in proposals] == [
        ("scheduled_at", proposed_start.isoformat())
    ]


@pytest.mark.parametrize("edited_kind", ["AMBIGUOUS", "OTHER"])
def test_older_message_edit_does_not_override_independent_satisfied_response(
    db_session: Session,
    edited_kind: str,
) -> None:
    core = seed_core(db_session)
    response = awaited(db_session, core)
    kyran = add_participant(db_session, core, name="Kyran")
    orchestrator(
        db_session,
        Classification(
            kind="AVAILABILITY",
            availability=AvailabilityStatus.AVAILABLE,
            evidence=AvailabilityEvidence.THIRD_PARTY,
            subject_task_participant_id=kyran.id,
        ),
    ).process(core["revision"].id)

    own_message = Message(
        conversation_id=core["conversation"].id,
        provider_message_id=f"own-{edited_kind.casefold()}",
        sender_identity_id=core["identity"].id,
        created_at=NOW + timedelta(minutes=1),
    )
    db_session.add(own_message)
    db_session.flush()
    own_revision = MessageRevision(
        message_id=own_message.id,
        provider_revision_key="r1",
        provider_sequence=2,
        content_hash=("2" if edited_kind == "AMBIGUOUS" else "3") * 64,
        text="I'm free",
        content_support=ContentSupport.SUPPORTED,
        processing_status=ProcessingStatus.PENDING,
    )
    db_session.add(own_revision)
    db_session.flush()
    own_message.current_revision_id = own_revision.id
    db_session.flush()
    orchestrator(
        db_session,
        Classification(
            kind="AVAILABILITY",
            availability=AvailabilityStatus.AVAILABLE,
            evidence=AvailabilityEvidence.FIRST_PARTY,
        ),
    ).process(own_revision.id)

    old_edit = MessageRevision(
        message_id=core["message"].id,
        provider_revision_key="r2",
        provider_sequence=3,
        content_hash=("4" if edited_kind == "AMBIGUOUS" else "5") * 64,
        text="unclear edit",
        content_support=ContentSupport.SUPPORTED,
        processing_status=ProcessingStatus.PENDING,
    )
    db_session.add(old_edit)
    db_session.flush()
    core["message"].current_revision_id = old_edit.id
    db_session.flush()
    orchestrator(
        db_session,
        Classification(kind=edited_kind),
    ).process(old_edit.id)

    assert response.status is AwaitedResponseStatus.SATISFIED
    assert core["participant"].availability_status is AvailabilityStatus.AVAILABLE
    assert core["participant"].availability_source_revision_id == own_revision.id
    assert kyran.availability_status is AvailabilityStatus.UNKNOWN


def add_evidence_revision(
    db_session: Session,
    core: dict[str, object],
    *,
    provider_message_id: str,
    provider_sequence: int,
) -> MessageRevision:
    message = Message(
        conversation_id=core["conversation"].id,
        provider_message_id=provider_message_id,
        sender_identity_id=core["identity"].id,
        created_at=NOW - timedelta(minutes=1),
    )
    db_session.add(message)
    db_session.flush()
    revision = MessageRevision(
        message_id=message.id,
        provider_revision_key="r1",
        provider_sequence=provider_sequence,
        content_hash=provider_message_id[0] * 64,
        text="prior evidence",
        content_support=ContentSupport.SUPPORTED,
        processing_status=ProcessingStatus.PROCESSED,
    )
    db_session.add(revision)
    db_session.flush()
    message.current_revision_id = revision.id
    db_session.flush()
    return revision


def test_older_availability_does_not_enqueue_stale_owner_notification(
    db_session: Session,
) -> None:
    core = seed_core(db_session)
    awaited(db_session, core)
    newer = add_evidence_revision(
        db_session,
        core,
        provider_message_id="newer",
        provider_sequence=2,
    )
    AvailabilityService(db_session).apply(
        core["participant"].id,
        newer.id,
        AvailabilityStatus.AVAILABLE,
        AvailabilityEvidence.FIRST_PARTY,
    )
    service = CorrelationOrchestrator(
        db_session,
        semantic=Semantic(),
        classifier=Classifier(
            Classification(
                kind="AVAILABILITY",
                availability=AvailabilityStatus.UNAVAILABLE,
                evidence=AvailabilityEvidence.FIRST_PARTY,
            )
        ),
        owner_chat_id=99,
    )

    service.process(core["revision"].id)

    assert core["participant"].availability_status is AvailabilityStatus.AVAILABLE
    assert db_session.scalar(select(func.count(OutboxMessage.id))) == 0


def test_owner_notification_uses_committed_uncertain_availability(
    db_session: Session,
) -> None:
    core = seed_core(db_session)
    awaited(db_session, core)
    equal_order = add_evidence_revision(
        db_session,
        core,
        provider_message_id="equal",
        provider_sequence=1,
    )
    AvailabilityService(db_session).apply(
        core["participant"].id,
        equal_order.id,
        AvailabilityStatus.AVAILABLE,
        AvailabilityEvidence.FIRST_PARTY,
    )
    service = CorrelationOrchestrator(
        db_session,
        semantic=Semantic(),
        classifier=Classifier(
            Classification(
                kind="AVAILABILITY",
                availability=AvailabilityStatus.UNAVAILABLE,
                evidence=AvailabilityEvidence.FIRST_PARTY,
            )
        ),
        owner_chat_id=99,
    )

    service.process(core["revision"].id)

    notice = db_session.scalar(select(OutboxMessage))
    assert core["participant"].availability_status is AvailabilityStatus.UNCERTAIN
    assert notice is not None
    assert notice.final_text == "Alex is uncertain for tennis."


def test_availability_reply_creates_one_durable_owner_notification(
    db_session: Session,
) -> None:
    core = seed_core(db_session)
    response = awaited(db_session, core)
    service = CorrelationOrchestrator(
        db_session,
        semantic=Semantic(),
        classifier=Classifier(
            Classification(kind="AVAILABILITY", availability=AvailabilityStatus.AVAILABLE)
        ),
        owner_chat_id=99,
    )

    service.process(core["revision"].id)
    service.process(core["revision"].id)

    notices = list(
        db_session.scalars(
            select(OutboxMessage).where(
                OutboxMessage.idempotency_key
                == (
                    f"task:{core['task'].id}:availability:{core['participant'].id}:"
                    f"revision:{core['revision'].id}"
                )
            )
        )
    )
    assert response.status is AwaitedResponseStatus.SATISFIED
    assert len(notices) == 1
    assert notices[0].final_text == "Alex is available for tennis."
    db_session.commit()
    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    context = DatabaseValidatorContextProvider(
        factory,
        facts=DatabaseContextProvider(),
        owner_chat_id=99,
    ).context_for(notices[0].id, notices[0].message_kind)
    assert "Alex availability is available" in context.allowed_claims


def test_correlation_ambiguity_is_not_guessed(db_session: Session) -> None:
    core = seed_core(db_session)
    first = awaited(db_session, core, created_at=NOW - timedelta(minutes=10))
    second = awaited(db_session, core, created_at=NOW - timedelta(minutes=5))
    result = orchestrator(db_session, Classification(kind="AMBIGUOUS")).process(core["revision"].id)
    assert result.outcome == "CORRELATION_AMBIGUITY"
    assert core["revision"].awaited_response_id is None
    assert first.status is AwaitedResponseStatus.OPEN
    assert second.status is AwaitedResponseStatus.OPEN
    candidate_count = db_session.scalar(
        select(func.count(DecisionRequestAwaitedResponseCandidate.awaited_response_id)).where(
            DecisionRequestAwaitedResponseCandidate.decision_request_id == result.decision_request_id
        )
    )
    assert candidate_count == 2


def test_ambiguous_message_owner_selection_continues_semantic_processing(db_session: Session) -> None:
    core = seed_core(db_session)
    first = awaited(db_session, core, created_at=NOW - timedelta(minutes=10))
    second = awaited(db_session, core, created_at=NOW - timedelta(minutes=5))
    result = CorrelationOrchestrator(
        db_session,
        semantic=Semantic(),
        classifier=Classifier(Classification(kind="AMBIGUOUS")),
        owner_chat_id=99,
    ).process(core["revision"].id)
    prompt = db_session.scalar(
        select(DecisionRequestPrompt).where(
            DecisionRequestPrompt.decision_request_id == result.decision_request_id
        )
    )
    assert prompt is not None
    db_session.commit()
    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    handler = ProductionOwnerCommandHandler(
        factory,
        owner_chat_id=99,
        resolver=object(),  # type: ignore[arg-type]
        generation=object(),  # type: ignore[arg-type]
        classifier=Classifier(
            Classification(kind="AVAILABILITY", availability=AvailabilityStatus.AVAILABLE)
        ),
    )
    control = TelegramControlGateway(factory, owner_id=7, parser=object(), handler=handler)  # type: ignore[arg-type]
    raw = {
        "update_id": 700,
        "callback_query": {
            "id": "selection",
            "from": {"id": 7},
            "message": {"message_id": 1, "chat": {"id": 99, "type": "private"}},
            "data": f"decision:{result.decision_request_id}:awaited_response:{second.id}",
        },
    }
    received = control.receive(raw)
    control.process(received.telegram_update_row_id)
    with factory() as session:
        revision = session.get(MessageRevision, core["revision"].id)
        assert revision.awaited_response_id == second.id
        assert session.get(AwaitedResponse, second.id).status is AwaitedResponseStatus.SATISFIED
        assert session.get(AwaitedResponse, first.id).status is AwaitedResponseStatus.OPEN
        assert session.get(type(core["participant"]), core["participant"].id).availability_status is AvailabilityStatus.AVAILABLE
        decision = session.get(DecisionRequest, result.decision_request_id)
        assert decision.status.value == "CLOSED"


def test_owner_correlation_selection_classifies_with_task_proposal_context(
    db_session: Session,
) -> None:
    core = seed_core(db_session)
    first = awaited(db_session, core, created_at=NOW - timedelta(minutes=10))
    second = awaited(db_session, core, created_at=NOW - timedelta(minutes=5))
    result = CorrelationOrchestrator(
        db_session,
        semantic=Semantic(),
        classifier=Classifier(Classification(kind="AMBIGUOUS")),
        owner_chat_id=99,
    ).process(core["revision"].id)
    db_session.commit()

    class RecordingClassifier:
        def __init__(self) -> None:
            self.context = None

        def classify(self, *args: object, **kwargs: object) -> Classification:
            self.context = args[3] if len(args) > 3 else kwargs.get("task_context")
            return Classification(kind="OTHER")

    classifier = RecordingClassifier()
    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    handler = ProductionOwnerCommandHandler(
        factory,
        owner_chat_id=99,
        owner_timezone="America/Toronto",
        resolver=object(),  # type: ignore[arg-type]
        generation=object(),  # type: ignore[arg-type]
        classifier=classifier,
    )

    prepared = handler.prepare_decision(
        result.decision_request_id,
        {
            "callback_query": {
                "data": f"decision:{result.decision_request_id}:awaited_response:{second.id}"
            }
        },
        object(),  # type: ignore[arg-type]
    )

    assert prepared.action == "select"
    assert classifier.context is not None
    assert classifier.context.duration_minutes == core["task"].duration_minutes
    assert classifier.context.topic_key == core["task"].topic_key
    assert classifier.context.owner_timezone == "America/Toronto"
    assert first.status is AwaitedResponseStatus.OPEN


def test_stale_and_repeated_owner_correlation_answers_have_no_second_effect(db_session: Session) -> None:
    core = seed_core(db_session)
    first = awaited(db_session, core)
    second = awaited(db_session, core, created_at=NOW - timedelta(minutes=4))
    result = CorrelationOrchestrator(
        db_session, semantic=Semantic(), classifier=Classifier(Classification(kind="AMBIGUOUS")), owner_chat_id=99
    ).process(core["revision"].id)
    core["revision"].awaited_response_id = first.id
    db_session.commit()
    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    handler = ProductionOwnerCommandHandler(
        factory, owner_chat_id=99, resolver=object(), generation=object(),
        classifier=Classifier(Classification(kind="AVAILABILITY", availability=AvailabilityStatus.UNAVAILABLE)),
    )  # type: ignore[arg-type]
    control = TelegramControlGateway(factory, owner_id=7, parser=object(), handler=handler)  # type: ignore[arg-type]
    def callback(update_id: int) -> dict[str, object]:
        return {"update_id": update_id, "callback_query": {"id": str(update_id), "from": {"id": 7}, "message": {"message_id": 1, "chat": {"id": 99, "type": "private"}}, "data": f"decision:{result.decision_request_id}:awaited_response:{second.id}"}}
    first_update = control.receive(callback(701))
    control.process(first_update.telegram_update_row_id)
    repeated = control.receive(callback(702))
    control.process(repeated.telegram_update_row_id)
    with factory() as session:
        decision = session.get(DecisionRequest, result.decision_request_id)
        assert decision.close_reason.value == "SUBJECT_RESOLVED"
        assert session.get(MessageRevision, core["revision"].id).awaited_response_id == first.id


def test_known_correlation_interpretation_ambiguity_marks_awaited(db_session: Session) -> None:
    core = seed_core(db_session)
    response = awaited(db_session, core)
    result = orchestrator(db_session, Classification(kind="AMBIGUOUS")).process(core["revision"].id)
    assert result.outcome == "INTERPRETATION_AMBIGUOUS"
    assert core["revision"].awaited_response_id == response.id
    assert response.status is AwaitedResponseStatus.AMBIGUOUS


def test_response_count_is_distinct_participants_not_awaited_rows(db_session: Session) -> None:
    core = seed_core(db_session)
    first = awaited(db_session, core)
    second = awaited(db_session, core)
    first.status = AwaitedResponseStatus.SATISFIED
    second.status = AwaitedResponseStatus.AMBIGUOUS
    db_session.flush()
    assert distinct_response_count(db_session, core["task"].id) == 1


def test_cross_conversation_same_person_asks_owner(db_session: Session) -> None:
    core = seed_core(db_session)
    response = awaited(db_session, core)
    second_identity = Identity(
        person_id=core["person"].id,
        beeper_user_id="beeper:alex:instagram",
        network="instagram",
        metadata_json={},
    )
    other_conversation = Conversation(
        beeper_conversation_id="conv:alex:instagram",
        network="instagram",
        kind=ConversationKind.DIRECT,
        counterparty_person_id=core["person"].id,
        metadata_json={},
    )
    db_session.add_all([second_identity, other_conversation])
    db_session.flush()
    db_session.add(
        ConversationParticipant(conversation_id=other_conversation.id, identity_id=second_identity.id)
    )
    db_session.flush()
    message = Message(
        conversation_id=other_conversation.id,
        provider_message_id="cross-message",
        sender_identity_id=second_identity.id,
        created_at=NOW,
    )
    db_session.add(message)
    db_session.flush()
    revision = MessageRevision(
        message_id=message.id,
        provider_revision_key="cross-r1",
        provider_sequence=1,
        content_hash="c" * 64,
        text="yes",
        content_support=ContentSupport.SUPPORTED,
        processing_status=ProcessingStatus.PENDING,
    )
    db_session.add(revision)
    db_session.flush()
    message.current_revision_id = revision.id
    db_session.flush()
    result = orchestrator(
        db_session,
        Classification(kind="AVAILABILITY", availability=AvailabilityStatus.AVAILABLE),
    ).process(revision.id)
    assert result.outcome == "CROSS_CONVERSATION_RESPONSE"
    assert revision.awaited_response_id is None
    assert response.status is AwaitedResponseStatus.OPEN


def test_provider_reply_linkage_precedes_ambiguous_conversation(db_session: Session) -> None:
    core = seed_core(db_session)
    chosen = awaited(db_session, core)
    awaited(db_session, core, created_at=NOW - timedelta(minutes=2))
    sent = OutboxService(db_session).create_beeper(
        task_instance_id=core["task"].id,
        conversation_id=core["conversation"].id,
        participant_ids=[core["participant"].id],
        final_text="question",
        message_kind=MessageKind.INITIAL,
        idempotency_key="prompt",
    )
    sent.status = OutboxStatus.SENT
    attempt = OutboxDeliveryAttempt(
        outbox_message_id=sent.id,
        started_at=NOW,
        finished_at=NOW,
        result=AttemptResult.SUCCESS,
        provider_message_id="provider:prompt",
    )
    db_session.add_all([attempt, AwaitedResponsePrompt(awaited_response_id=chosen.id, outbox_message_id=sent.id)])
    core["message"].provider_reply_to_message_id = "provider:prompt"
    db_session.flush()
    result = orchestrator(
        db_session,
        Classification(kind="AVAILABILITY", availability=AvailabilityStatus.AVAILABLE),
    ).process(core["revision"].id)
    assert result.awaited_response_id == chosen.id
    assert chosen.status is AwaitedResponseStatus.SATISFIED


def test_late_terminal_reply_creates_one_owner_prompt(db_session: Session) -> None:
    core = seed_core(db_session)
    chosen = awaited(db_session, core)
    sent = OutboxService(db_session).create_beeper(
        task_instance_id=core["task"].id,
        conversation_id=core["conversation"].id,
        participant_ids=[core["participant"].id],
        final_text="question",
        message_kind=MessageKind.INITIAL,
        idempotency_key="late-terminal-prompt",
    )
    sent.status = OutboxStatus.SENT
    db_session.add_all(
        [
            OutboxDeliveryAttempt(
                outbox_message_id=sent.id,
                started_at=NOW,
                finished_at=NOW,
                result=AttemptResult.SUCCESS,
                provider_message_id="provider:late-terminal-prompt",
            ),
            AwaitedResponsePrompt(
                awaited_response_id=chosen.id,
                outbox_message_id=sent.id,
            ),
        ]
    )
    core["message"].provider_reply_to_message_id = "provider:late-terminal-prompt"
    db_session.flush()
    TaskService(db_session).terminalize(core["task"].id, TaskStatus.COMPLETED)

    service = CorrelationOrchestrator(
        db_session,
        semantic=Semantic(),
        classifier=Classifier(Classification(kind="OTHER")),
        owner_chat_id=99,
    )
    first = service.process(core["revision"].id)
    second = service.process(core["revision"].id)

    assert first.outcome == "LATE_TERMINAL"
    assert second.outcome == "ALREADY_CORRELATED"
    assert core["task"].status is TaskStatus.COMPLETED
    assert chosen.status is AwaitedResponseStatus.CANCELLED
    assert core["participant"].availability_status is AvailabilityStatus.UNKNOWN
    prompts = list(
        db_session.scalars(
            select(OutboxMessage)
            .join(DecisionRequestPrompt)
            .join(DecisionRequest)
            .where(DecisionRequest.type == "LATE_TERMINAL_MESSAGE")
        )
    )
    assert len(prompts) == 1
    assert core["revision"].text in prompts[0].final_text
    assert "`dismiss`" in prompts[0].final_text
    db_session.commit()
    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    context = DatabaseValidatorContextProvider(
        factory,
        facts=DatabaseContextProvider(),
        owner_chat_id=99,
    ).context_for(prompts[0].id, prompts[0].message_kind)
    assert any("terminal" in claim for claim in context.allowed_claims)
    assert not any(core["revision"].text in claim for claim in context.allowed_claims)
    assert any(core["revision"].text in item for item in context.untrusted_data)

    class ExactLatePromptBackend:
        def infer(self, *, operation: str, payload: dict[str, object]) -> dict[str, object]:
            assert operation == "message_validator"
            assert payload["exact_text"] == prompts[0].final_text
            assert tuple(payload["allowed_claims"]) == context.allowed_claims
            assert tuple(payload["untrusted_data"]) == context.untrusted_data
            return {"category": "VALID", "critique": None}

    result = IndependentMessageValidator(ExactLatePromptBackend()).review(
        text=prompts[0].final_text,
        context=context,
    )
    assert result.category.value == "VALID"


def test_late_terminal_prompt_dismisses_only_by_reply_to_current_revision(
    db_session: Session,
) -> None:
    core = seed_core(db_session)
    chosen = awaited(db_session, core)
    sent = OutboxService(db_session).create_beeper(
        task_instance_id=core["task"].id,
        conversation_id=core["conversation"].id,
        participant_ids=[core["participant"].id],
        final_text="question",
        message_kind=MessageKind.INITIAL,
        idempotency_key="late-terminal-dismiss-source",
    )
    sent.status = OutboxStatus.SENT
    db_session.add_all(
        [
            OutboxDeliveryAttempt(
                outbox_message_id=sent.id,
                started_at=NOW,
                finished_at=NOW,
                result=AttemptResult.SUCCESS,
                provider_message_id="provider:late-terminal-dismiss-source",
            ),
            AwaitedResponsePrompt(
                awaited_response_id=chosen.id,
                outbox_message_id=sent.id,
            ),
        ]
    )
    core["message"].provider_reply_to_message_id = (
        "provider:late-terminal-dismiss-source"
    )
    db_session.flush()
    TaskService(db_session).terminalize(core["task"].id, TaskStatus.CANCELLED)
    result = CorrelationOrchestrator(
        db_session,
        semantic=Semantic(),
        classifier=Classifier(Classification(kind="OTHER")),
        owner_chat_id=99,
    ).process(core["revision"].id)
    prompt = db_session.scalar(
        select(OutboxMessage)
        .join(DecisionRequestPrompt)
        .where(DecisionRequestPrompt.decision_request_id == result.decision_request_id)
    )
    assert prompt is not None
    prompt.status = OutboxStatus.SENT
    db_session.add(
        OutboxDeliveryAttempt(
            outbox_message_id=prompt.id,
            started_at=NOW,
            finished_at=NOW,
            result=AttemptResult.SUCCESS,
            provider_message_id="901",
        )
    )
    db_session.commit()

    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    handler = ProductionOwnerCommandHandler(
        factory,
        owner_chat_id=99,
        resolver=object(),  # type: ignore[arg-type]
        generation=object(),  # type: ignore[arg-type]
    )
    control = TelegramControlGateway(
        factory,
        owner_id=7,
        parser=object(),  # type: ignore[arg-type]
        handler=handler,
    )
    standalone = {
        "update_id": 900,
        "message": {
            "message_id": 900,
            "from": {"id": 7},
            "chat": {"id": 99, "type": "private"},
            "text": "dismiss",
        },
    }
    with factory() as session:
        assert control._resolve_decision(session, standalone, None) is None

    reply = {
        "update_id": 901,
        "message": {
            "message_id": 902,
            "from": {"id": 7},
            "chat": {"id": 99, "type": "private"},
            "text": "dismiss",
            "reply_to_message": {"message_id": 901},
        },
    }
    received = control.receive(reply)
    control.process(received.telegram_update_row_id)
    with factory() as session:
        decision = session.get(DecisionRequest, result.decision_request_id)
        assert decision.status is DecisionStatus.CLOSED
        assert decision.close_reason.value == "DISMISSED"
        assert decision.resolution_json is None
        assert session.get(TaskInstance, core["task"].id).status is TaskStatus.CANCELLED
        assert (
            session.get(TaskParticipant, core["participant"].id).availability_status
            is AvailabilityStatus.UNKNOWN
        )


def test_edit_makes_old_late_terminal_decision_subject_resolved(
    db_session: Session,
) -> None:
    core = seed_core(db_session)
    chosen = awaited(db_session, core)
    sent = OutboxService(db_session).create_beeper(
        task_instance_id=core["task"].id,
        conversation_id=core["conversation"].id,
        participant_ids=[core["participant"].id],
        final_text="question",
        message_kind=MessageKind.INITIAL,
        idempotency_key="late-terminal-edit-source",
    )
    sent.status = OutboxStatus.SENT
    db_session.add_all(
        [
            OutboxDeliveryAttempt(
                outbox_message_id=sent.id,
                started_at=NOW,
                finished_at=NOW,
                result=AttemptResult.SUCCESS,
                provider_message_id="provider:late-terminal-edit-source",
            ),
            AwaitedResponsePrompt(
                awaited_response_id=chosen.id,
                outbox_message_id=sent.id,
            ),
        ]
    )
    core["message"].provider_reply_to_message_id = "provider:late-terminal-edit-source"
    db_session.flush()
    TaskService(db_session).terminalize(core["task"].id, TaskStatus.COMPLETED)
    service = CorrelationOrchestrator(
        db_session,
        semantic=Semantic(),
        classifier=Classifier(Classification(kind="OTHER")),
        owner_chat_id=99,
    )
    old_result = service.process(core["revision"].id)
    old_prompt = db_session.scalar(
        select(OutboxMessage)
        .join(DecisionRequestPrompt)
        .where(DecisionRequestPrompt.decision_request_id == old_result.decision_request_id)
    )
    assert old_prompt is not None
    old_prompt.status = OutboxStatus.SENT
    db_session.add(
        OutboxDeliveryAttempt(
            outbox_message_id=old_prompt.id,
            started_at=NOW,
            finished_at=NOW,
            result=AttemptResult.SUCCESS,
            provider_message_id="902",
        )
    )
    edited = MessageRevision(
        message_id=core["message"].id,
        provider_revision_key="r2",
        provider_sequence=2,
        content_hash="b" * 64,
        text="Actually, this is the edited late reply.",
        content_support=ContentSupport.SUPPORTED,
        processing_status=ProcessingStatus.PENDING,
    )
    db_session.add(edited)
    db_session.flush()
    core["message"].current_revision_id = edited.id
    new_result = service.process(edited.id)
    db_session.commit()

    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    handler = ProductionOwnerCommandHandler(
        factory,
        owner_chat_id=99,
        resolver=object(),  # type: ignore[arg-type]
        generation=object(),  # type: ignore[arg-type]
    )
    control = TelegramControlGateway(
        factory,
        owner_id=7,
        parser=object(),  # type: ignore[arg-type]
        handler=handler,
    )
    reply = {
        "update_id": 902,
        "message": {
            "message_id": 903,
            "from": {"id": 7},
            "chat": {"id": 99, "type": "private"},
            "text": "dismiss",
            "reply_to_message": {"message_id": 902},
        },
    }
    received = control.receive(reply)
    control.process(received.telegram_update_row_id)

    with factory() as session:
        old_decision = session.get(DecisionRequest, old_result.decision_request_id)
        current_decision = session.get(DecisionRequest, new_result.decision_request_id)
        assert old_decision.close_reason.value == "SUBJECT_RESOLVED"
        assert current_decision.status is DecisionStatus.PENDING
        assert current_decision.message_revision_id == edited.id
        assert session.get(TaskInstance, core["task"].id).status is TaskStatus.COMPLETED
        assert (
            session.get(TaskParticipant, core["participant"].id).availability_status
            is AvailabilityStatus.UNKNOWN
        )
        assert session.scalar(
            select(func.count(DecisionRequestPrompt.outbox_message_id)).where(
                DecisionRequestPrompt.decision_request_id.in_(
                    [old_decision.id, current_decision.id]
                )
            )
        ) == 2
