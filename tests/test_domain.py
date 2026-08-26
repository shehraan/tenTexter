from __future__ import annotations

from datetime import timedelta

import pytest
from sqlalchemy.orm import Session

from ten_texter.domain import (
    AvailabilityService,
    ContactRuleService,
    DecisionService,
    DomainError,
    ProposalService,
    StaleWork,
    TaskService,
)
from ten_texter.enums import (
    AvailabilityEvidence,
    AvailabilityStatus,
    AwaitedResponseStatus,
    ContactRuleScope,
    ContactRuleSource,
    DecisionCloseReason,
    DecisionStatus,
    DisclosureGrantStatus,
    DisclosureInactiveReason,
    DestinationKind,
    MessageKind,
    OutboxCancelReason,
    OutboxStatus,
    ParentTerminalPolicy,
    ProposalStatus,
    RuleStrength,
    TargetSelector,
    TaskStatus,
    Transport,
    TriggerActionType,
    TriggerExecutionStatus,
    TriggerInactiveReason,
    TriggerStatus,
)
from ten_texter.models import (
    AwaitedResponse,
    ContactRule,
    DecisionRequest,
    DisclosureGrant,
    OutboxMessage,
    Proposal,
    TaskTrigger,
    TriggerExecution,
)
from tests.test_schema import NOW, seed_core


def test_task_creation_rejects_unpinned_person(db_session: Session) -> None:
    core = seed_core(db_session)
    with pytest.raises(DomainError):
        TaskService(db_session).create(
            scheduled_at=NOW,
            duration_minutes=30,
            topic_key="Bad Route",
            participants=[(9999, core["conversation"].id)],  # type: ignore[union-attr]
        )


def test_terminalization_is_atomic_and_respects_survive(db_session: Session) -> None:
    core = seed_core(db_session)
    task = core["task"]
    participant = core["participant"]
    revision = core["revision"]
    awaited = AwaitedResponse(
        task_participant_id=participant.id,
        expected_response_type="availability",
        status=AwaitedResponseStatus.OPEN,
    )
    trigger = TaskTrigger(
        task_instance_id=task.id,
        condition_json={"kind": "AT_TIME"},
        target_selector=TargetSelector.ALL_PARTICIPANTS,
        action_type=TriggerActionType.SEND_MESSAGE,
        action_payload_json={"goal": "remind"},
        status=TriggerStatus.ACTIVE,
    )
    grant = DisclosureGrant(
        source_person_id=core["person"].id,
        source_conversation_id=core["conversation"].id,
        destination_conversation_id=_second_conversation(db_session, core),
        task_instance_id=task.id,
        status=DisclosureGrantStatus.ACTIVE,
        expires_at=NOW + timedelta(days=1),
    )
    proposal = Proposal(
        task_instance_id=task.id,
        proposed_by_participant_id=participant.id,
        source_message_revision_id=revision.id,
        field="location",
        operation="SET",
        old_value=None,
        proposed_value="park",
        status=ProposalStatus.PENDING,
    )
    cancelled_send = OutboxMessage(
        task_instance_id=task.id,
        transport=Transport.BEEPER,
        destination_kind=DestinationKind.PARTICIPANT,
        final_text="pending",
        message_kind=MessageKind.REMINDER,
        status=OutboxStatus.PENDING,
        idempotency_key="terminate-send",
        parent_terminal_policy=ParentTerminalPolicy.TERMINATE,
    )
    surviving_send = OutboxMessage(
        task_instance_id=task.id,
        transport=Transport.TELEGRAM,
        destination_kind=DestinationKind.OWNER,
        final_text="survive",
        message_kind=MessageKind.NOTIFICATION,
        status=OutboxStatus.PENDING,
        idempotency_key="survive-send",
        parent_terminal_policy=ParentTerminalPolicy.SURVIVE,
    )
    db_session.add_all([awaited, trigger, grant, proposal, cancelled_send, surviving_send])
    db_session.flush()
    execution = TriggerExecution(
        task_trigger_id=trigger.id,
        fire_key="one",
        status=TriggerExecutionStatus.PENDING,
    )
    db_session.add(execution)
    db_session.flush()
    decision = DecisionService(db_session).create(
        decision_type="PROPOSAL",
        subject_kind="proposal",
        subject_id=proposal.id,
        context={},
        task_instance_id=task.id,
    )

    TaskService(db_session).terminalize(task.id, TaskStatus.CANCELLED)
    db_session.expire_all()

    assert db_session.get(AwaitedResponse, awaited.id).status is AwaitedResponseStatus.CANCELLED
    stored_trigger = db_session.get(TaskTrigger, trigger.id)
    assert (stored_trigger.status, stored_trigger.inactive_reason) == (
        TriggerStatus.INACTIVE,
        TriggerInactiveReason.TASK_TERMINAL,
    )
    assert db_session.get(DisclosureGrant, grant.id).inactive_reason is DisclosureInactiveReason.TASK_TERMINAL
    assert db_session.get(Proposal, proposal.id).status is ProposalStatus.PARENT_TERMINAL
    assert db_session.get(DecisionRequest, decision.id).close_reason is DecisionCloseReason.PARENT_TERMINAL
    assert db_session.get(OutboxMessage, cancelled_send.id).cancel_reason is OutboxCancelReason.PARENT_TERMINAL
    assert db_session.get(OutboxMessage, surviving_send.id).status is OutboxStatus.PENDING
    assert db_session.get(TriggerExecution, execution.id).status is TriggerExecutionStatus.CANCELLED


def _second_conversation(session: Session, core: dict[str, object]) -> int:
    from ten_texter.enums import ConversationKind
    from ten_texter.models import Conversation

    conversation = Conversation(
        beeper_conversation_id="conv:group",
        network="discord",
        kind=ConversationKind.GROUP,
        metadata_json={},
    )
    session.add(conversation)
    session.flush()
    return conversation.id


def test_stale_decision_answer_closes_without_effect(db_session: Session) -> None:
    core = seed_core(db_session)
    proposal = Proposal(
        task_instance_id=core["task"].id,
        proposed_by_participant_id=core["participant"].id,
        source_message_revision_id=core["revision"].id,
        field="location",
        operation="SET",
        old_value=None,
        proposed_value="park",
        status=ProposalStatus.PENDING,
    )
    db_session.add(proposal)
    db_session.flush()
    decision = DecisionService(db_session).create(
        decision_type="PROPOSAL",
        subject_kind="proposal",
        subject_id=proposal.id,
        context={},
        task_instance_id=core["task"].id,
    )
    ProposalService(db_session).resolve(proposal.id, accept=False)
    applied = False

    def apply(_decision: DecisionRequest, _resolution: dict[str, object]) -> None:
        nonlocal applied
        applied = True

    DecisionService(db_session).answer(
        decision.id,
        {"accept": True},
        subject_still_requires_decision=lambda _: proposal.status is ProposalStatus.PENDING,
        apply=apply,
    )
    assert not applied
    assert decision.close_reason is DecisionCloseReason.SUBJECT_RESOLVED
    assert decision.resolution_json is None


def test_older_availability_cannot_overwrite_newer(db_session: Session) -> None:
    core = seed_core(db_session)
    first = core["revision"]
    AvailabilityService(db_session).apply(
        core["participant"].id,
        first.id,
        AvailabilityStatus.AVAILABLE,
        AvailabilityEvidence.FIRST_PARTY,
    )
    # A distinct provider message can be current while still carrying older provider ordering evidence.
    from ten_texter.enums import ContentSupport, ProcessingStatus
    from ten_texter.models import Message, MessageRevision

    older_message = Message(
        conversation_id=core["conversation"].id,
        provider_message_id="older-event",
        sender_identity_id=core["identity"].id,
        created_at=NOW,
    )
    db_session.add(older_message)
    db_session.flush()
    older = MessageRevision(
        message_id=older_message.id,
        provider_revision_key="older-r1",
        provider_sequence=0,
        content_hash="b" * 64,
        text="no",
        content_support=ContentSupport.SUPPORTED,
        processing_status=ProcessingStatus.PENDING,
    )
    db_session.add(older)
    db_session.flush()
    older_message.current_revision_id = older.id
    db_session.flush()
    assert not AvailabilityService(db_session).apply(
        core["participant"].id,
        older.id,
        AvailabilityStatus.UNAVAILABLE,
        AvailabilityEvidence.FIRST_PARTY,
    )
    assert core["participant"].availability_status is AvailabilityStatus.AVAILABLE


def test_stale_revision_cannot_mutate_availability(db_session: Session) -> None:
    core = seed_core(db_session)
    core["message"].current_revision_id = None
    db_session.flush()
    with pytest.raises(StaleWork):
        AvailabilityService(db_session).apply(
            core["participant"].id,
            core["revision"].id,
            AvailabilityStatus.AVAILABLE,
            AvailabilityEvidence.FIRST_PARTY,
        )


def test_contact_rule_override_cannot_cross_person_or_skip_narrowing(db_session: Session) -> None:
    core = seed_core(db_session)
    service = ContactRuleService(db_session)
    original = service.create(
        person_id=core["person"].id,
        scope=ContactRuleScope.TOPIC,
        type="DO_NOT_CONTACT",
        value="tennis",
        source=ContactRuleSource.PARTICIPANT_REQUESTED,
        strength=RuleStrength.STRONG,
        topic_key="tennis",
    )
    with pytest.raises(DomainError):
        service.create(
            person_id=999,
            scope=ContactRuleScope.TASK_INSTANCE,
            type="ALLOW",
            value="once",
            source=ContactRuleSource.USER_CONFIGURED,
            strength=RuleStrength.STRONG,
            task_instance_id=core["task"].id,
            overrides_contact_rule_id=original.id,
        )
    with pytest.raises(DomainError):
        service.create(
            person_id=core["person"].id,
            scope=ContactRuleScope.TOPIC,
            type="ALLOW",
            value="once",
            source=ContactRuleSource.USER_CONFIGURED,
            strength=RuleStrength.STRONG,
            topic_key="tennis",
            overrides_contact_rule_id=original.id,
        )


def test_proposal_revalidates_precondition(db_session: Session) -> None:
    core = seed_core(db_session)
    proposal = Proposal(
        task_instance_id=core["task"].id,
        proposed_by_participant_id=core["participant"].id,
        source_message_revision_id=core["revision"].id,
        field="location",
        operation="SET",
        old_value="old location",
        proposed_value="new location",
        status=ProposalStatus.PENDING,
    )
    db_session.add(proposal)
    db_session.flush()
    ProposalService(db_session).resolve(proposal.id, accept=True)
    assert proposal.status is ProposalStatus.SUPERSEDED
    assert core["task"].location is None
