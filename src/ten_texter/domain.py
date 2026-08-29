from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.orm import Session

from ten_texter.enums import (
    AttemptResult,
    AvailabilityEvidence,
    AvailabilityStatus,
    AwaitedResponseStatus,
    ContactRuleScope,
    ContactRuleSource,
    DecisionCloseReason,
    DecisionStatus,
    DisclosureGrantStatus,
    DisclosureInactiveReason,
    OutboxCancelReason,
    OutboxStatus,
    ParentTerminalPolicy,
    ProposalStatus,
    RuleStrength,
    TaskStatus,
    TriggerExecutionStatus,
    TriggerInactiveReason,
    TriggerStatus,
)
from ten_texter.models import (
    AwaitedResponse,
    ContactRule,
    ConversationParticipant,
    DecisionRequest,
    DisclosureGrant,
    DisclosureGrantScope,
    Identity,
    Message,
    MessageRevision,
    OutboxMessage,
    Proposal,
    TaskEvent,
    TaskInstance,
    TaskParticipant,
    TaskTrigger,
    TriggerExecution,
)


class DomainError(RuntimeError):
    pass


class InvalidTransition(DomainError):
    pass


class StaleWork(DomainError):
    pass


def utc_now() -> datetime:
    return datetime.now(UTC)


def normalize_topic_key(value: str) -> str:
    normalized = "-".join(value.strip().lower().split())
    if not normalized:
        raise DomainError("topic_key must not be empty")
    return normalized


def _normalize_absolute_instant(value: datetime, *, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise DomainError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


class TaskService:
    def __init__(self, session: Session):
        self.session = session

    def create(
        self,
        *,
        scheduled_at: datetime,
        duration_minutes: int,
        topic_key: str,
        participants: Iterable[tuple[int, int]] = (),
        task_definition_id: int | None = None,
        occurrence_key: str | None = None,
        location: str | None = None,
        coordination_close_offset_minutes: int = 60,
    ) -> TaskInstance:
        scheduled_at = _normalize_absolute_instant(
            scheduled_at,
            field="scheduled_at",
        )
        participant_pairs = list(participants)
        person_ids = [person_id for person_id, _conversation_id in participant_pairs]
        if len(person_ids) != len(set(person_ids)):
            raise DomainError("duplicate canonical person in task participants")
        task = TaskInstance(
            task_definition_id=task_definition_id,
            occurrence_key=occurrence_key,
            scheduled_at=scheduled_at,
            duration_minutes=duration_minutes,
            location=location,
            topic_key=normalize_topic_key(topic_key),
            coordination_close_offset_minutes=coordination_close_offset_minutes,
            status=TaskStatus.ACTIVE,
        )
        self.session.add(task)
        self.session.flush()
        for person_id, conversation_id in participant_pairs:
            membership = self.session.scalar(
                select(ConversationParticipant)
                .join(Identity, Identity.id == ConversationParticipant.identity_id)
                .where(
                    ConversationParticipant.conversation_id == conversation_id,
                    ConversationParticipant.is_current.is_(True),
                    Identity.person_id == person_id,
                )
            )
            if membership is None:
                raise DomainError("pinned conversation does not contain an identity for the person")
            self.session.add(
                TaskParticipant(
                    task_instance_id=task.id,
                    person_id=person_id,
                    conversation_id=conversation_id,
                    availability_status=AvailabilityStatus.UNKNOWN,
                )
            )
        self.session.flush()
        return task

    def reschedule(self, task_id: int, scheduled_at: datetime) -> TaskInstance:
        scheduled_at = _normalize_absolute_instant(
            scheduled_at,
            field="scheduled_at",
        )
        task = self._active(task_id)
        task.scheduled_at = scheduled_at
        task.updated_at = utc_now()
        # v1 grants are task-relative and never outlive the coordination close window.
        self.session.execute(
            update(DisclosureGrant)
            .where(
                DisclosureGrant.task_instance_id == task.id,
                DisclosureGrant.status == DisclosureGrantStatus.ACTIVE,
            )
            .values(
                expires_at=scheduled_at
                + timedelta(minutes=task.coordination_close_offset_minutes)
            )
        )
        self.session.add(
            TaskEvent(task_instance_id=task.id, event_type="RESCHEDULED", payload_json={"scheduled_at": scheduled_at.isoformat()})
        )
        self.session.flush()
        return task

    def terminalize(self, task_id: int, status: TaskStatus) -> TaskInstance:
        if status is TaskStatus.ACTIVE:
            raise InvalidTransition("terminalization requires a terminal status")
        task = self.session.get(TaskInstance, task_id)
        if task is None:
            raise DomainError("task not found")
        if task.status is not TaskStatus.ACTIVE:
            if task.status is status:
                return task
            raise InvalidTransition("terminal task state is immutable")
        timestamp = utc_now()
        task.status = status
        task.updated_at = timestamp
        participant_ids = select(TaskParticipant.id).where(TaskParticipant.task_instance_id == task.id)
        self.session.execute(
            update(AwaitedResponse)
            .where(
                AwaitedResponse.task_participant_id.in_(participant_ids),
                AwaitedResponse.status.in_([AwaitedResponseStatus.OPEN, AwaitedResponseStatus.AMBIGUOUS]),
            )
            .values(status=AwaitedResponseStatus.CANCELLED)
        )
        self.session.execute(
            update(TaskTrigger)
            .where(TaskTrigger.task_instance_id == task.id, TaskTrigger.status == TriggerStatus.ACTIVE)
            .values(status=TriggerStatus.INACTIVE, inactive_reason=TriggerInactiveReason.TASK_TERMINAL)
        )
        self.session.execute(
            update(DisclosureGrant)
            .where(
                DisclosureGrant.task_instance_id == task.id,
                DisclosureGrant.status == DisclosureGrantStatus.ACTIVE,
            )
            .values(status=DisclosureGrantStatus.INACTIVE, inactive_reason=DisclosureInactiveReason.TASK_TERMINAL)
        )
        self.session.execute(
            update(DecisionRequest)
            .where(
                DecisionRequest.task_instance_id == task.id,
                DecisionRequest.status == DecisionStatus.PENDING,
                DecisionRequest.parent_terminal_policy == ParentTerminalPolicy.TERMINATE,
            )
            .values(
                status=DecisionStatus.CLOSED,
                close_reason=DecisionCloseReason.PARENT_TERMINAL,
                resolved_at=timestamp,
            )
        )
        self.session.execute(
            update(Proposal)
            .where(Proposal.task_instance_id == task.id, Proposal.status == ProposalStatus.PENDING)
            .values(status=ProposalStatus.PARENT_TERMINAL, resolved_at=timestamp)
        )
        self.session.execute(
            update(OutboxMessage)
            .where(
                OutboxMessage.task_instance_id == task.id,
                OutboxMessage.status == OutboxStatus.PENDING,
                OutboxMessage.parent_terminal_policy == ParentTerminalPolicy.TERMINATE,
            )
            .values(status=OutboxStatus.CANCELLED, cancel_reason=OutboxCancelReason.PARENT_TERMINAL)
        )
        trigger_ids = select(TaskTrigger.id).where(TaskTrigger.task_instance_id == task.id)
        self.session.execute(
            update(TriggerExecution)
            .where(
                TriggerExecution.task_trigger_id.in_(trigger_ids),
                TriggerExecution.status == TriggerExecutionStatus.PENDING,
            )
            .values(status=TriggerExecutionStatus.CANCELLED)
        )
        self.session.add(TaskEvent(task_instance_id=task.id, event_type="TERMINALIZED", payload_json={"status": status.value}))
        self.session.flush()
        return task

    def _active(self, task_id: int) -> TaskInstance:
        task = self.session.get(TaskInstance, task_id)
        if task is None:
            raise DomainError("task not found")
        if task.status is not TaskStatus.ACTIVE:
            raise InvalidTransition("task is terminal")
        return task


def _revision_order(revision: MessageRevision) -> tuple[str, Any] | None:
    if revision.provider_sequence is not None:
        return ("sequence", revision.provider_sequence)
    if revision.provider_event_at is not None:
        value = revision.provider_event_at
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return ("timestamp", value)
    return None


class AvailabilityService:
    def __init__(self, session: Session):
        self.session = session

    def apply(
        self,
        participant_id: int,
        revision_id: int,
        status: AvailabilityStatus,
        evidence: AvailabilityEvidence,
    ) -> bool:
        if status is AvailabilityStatus.UNKNOWN:
            raise DomainError("semantic evidence cannot set UNKNOWN")
        participant = self.session.get(TaskParticipant, participant_id)
        revision = self.session.get(MessageRevision, revision_id)
        if participant is None or revision is None:
            raise DomainError("participant or revision not found")
        task = self.session.get(TaskInstance, participant.task_instance_id)
        message = self.session.get(Message, revision.message_id)
        if task is None or task.status is not TaskStatus.ACTIVE:
            raise StaleWork("terminal task cannot accept availability mutation")
        if message is None or message.current_revision_id != revision.id:
            raise StaleWork("revision is not current")

        chosen_status = status
        if participant.availability_source_revision_id is not None:
            previous = self.session.get(MessageRevision, participant.availability_source_revision_id)
            assert previous is not None
            incoming_order = _revision_order(revision)
            prior_order = _revision_order(previous)
            if incoming_order is None or prior_order is None or incoming_order[0] != prior_order[0]:
                chosen_status = AvailabilityStatus.UNCERTAIN
            elif incoming_order[1] < prior_order[1]:
                return False
            elif incoming_order[1] == prior_order[1]:
                if revision.id == previous.id:
                    return False
                chosen_status = AvailabilityStatus.UNCERTAIN

        participant.availability_status = chosen_status
        participant.availability_evidence = evidence
        participant.availability_source_revision_id = revision.id
        participant.updated_at = utc_now()
        self.session.add(
            TaskEvent(
                task_instance_id=participant.task_instance_id,
                task_participant_id=participant.id,
                source_message_revision_id=revision.id,
                event_type="AVAILABILITY_UPDATED",
                payload_json={"status": chosen_status.value, "evidence": evidence.value},
            )
        )
        self.session.flush()
        return True


class ProposalService:
    MUTABLE_FIELDS = {"scheduled_at", "duration_minutes", "location"}

    def __init__(self, session: Session):
        self.session = session

    def resolve(self, proposal_id: int, *, accept: bool) -> Proposal:
        proposal = self.session.get(Proposal, proposal_id)
        if proposal is None:
            raise DomainError("proposal not found")
        if proposal.status is not ProposalStatus.PENDING:
            raise InvalidTransition("proposal is not pending")
        task = self.session.get(TaskInstance, proposal.task_instance_id)
        assert task is not None
        timestamp = utc_now()
        if not accept:
            proposal.status = ProposalStatus.REJECTED
        elif task.status is not TaskStatus.ACTIVE:
            proposal.status = ProposalStatus.PARENT_TERMINAL
        elif proposal.field not in self.MUTABLE_FIELDS or proposal.operation != "SET":
            proposal.status = ProposalStatus.SUPERSEDED
        else:
            current = getattr(task, proposal.field)
            expected = proposal.old_value
            comparable = current.isoformat() if isinstance(current, datetime) else current
            if comparable != expected:
                proposal.status = ProposalStatus.SUPERSEDED
            else:
                value = proposal.proposed_value
                if proposal.field == "scheduled_at":
                    value = datetime.fromisoformat(value)
                setattr(task, proposal.field, value)
                task.updated_at = timestamp
                proposal.status = ProposalStatus.ACCEPTED
        proposal.resolved_at = timestamp
        self.session.flush()
        return proposal


class DecisionService:
    SUBJECT_COLUMNS = {
        "proposal": "proposal_id",
        "contact_rule": "contact_rule_id",
        "outbox_message": "outbox_message_id",
        "message_revision": "message_revision_id",
        "telegram_update": "telegram_update_id",
        "trigger_execution": "trigger_execution_id",
    }

    def __init__(self, session: Session):
        self.session = session

    def create(
        self,
        *,
        decision_type: str,
        subject_kind: str,
        subject_id: int,
        context: dict[str, Any],
        task_instance_id: int | None = None,
        parent_terminal_policy: ParentTerminalPolicy = ParentTerminalPolicy.TERMINATE,
        expires_at: datetime | None = None,
    ) -> DecisionRequest:
        column = self.SUBJECT_COLUMNS.get(subject_kind)
        if column is None:
            raise DomainError("unknown decision subject kind")
        decision = DecisionRequest(
            task_instance_id=task_instance_id,
            type=decision_type,
            status=DecisionStatus.PENDING,
            context_json=context,
            parent_terminal_policy=parent_terminal_policy,
            expires_at=expires_at,
            **{column: subject_id},
        )
        self.session.add(decision)
        self.session.flush()
        return decision

    def answer(
        self,
        decision_id: int,
        resolution: dict[str, Any],
        *,
        subject_still_requires_decision: Callable[[DecisionRequest], bool],
        apply: Callable[[DecisionRequest, dict[str, Any]], bool | None],
    ) -> DecisionRequest:
        decision = self._pending(decision_id)
        timestamp = utc_now()
        if not subject_still_requires_decision(decision):
            decision.status = DecisionStatus.CLOSED
            decision.close_reason = DecisionCloseReason.SUBJECT_RESOLVED
            decision.resolved_at = timestamp
            self.session.flush()
            return decision
        applied = apply(decision, resolution)
        if applied is False:
            decision.status = DecisionStatus.CLOSED
            decision.close_reason = DecisionCloseReason.SUBJECT_RESOLVED
            decision.resolved_at = timestamp
            self.session.flush()
            return decision
        decision.status = DecisionStatus.CLOSED
        decision.close_reason = DecisionCloseReason.ANSWERED
        decision.resolved_at = timestamp
        decision.resolution_json = resolution
        self.session.flush()
        return decision

    def close(self, decision_id: int, reason: DecisionCloseReason) -> DecisionRequest:
        if reason is DecisionCloseReason.ANSWERED:
            raise DomainError("answered decisions require a resolution")
        decision = self._pending(decision_id)
        decision.status = DecisionStatus.CLOSED
        decision.close_reason = reason
        decision.resolved_at = utc_now()
        self.session.flush()
        return decision

    def _pending(self, decision_id: int) -> DecisionRequest:
        decision = self.session.get(DecisionRequest, decision_id)
        if decision is None:
            raise DomainError("decision not found")
        if decision.status is not DecisionStatus.PENDING:
            raise InvalidTransition("decision is not pending")
        return decision


_SCOPE_SPECIFICITY = {
    ContactRuleScope.GLOBAL: 0,
    ContactRuleScope.TOPIC: 1,
    ContactRuleScope.TASK_DEFINITION: 2,
    ContactRuleScope.TASK_INSTANCE: 3,
}


class ContactRuleService:
    def __init__(self, session: Session):
        self.session = session

    def create(self, **values: Any) -> ContactRule:
        if values.get("scope") is ContactRuleScope.TOPIC:
            values["topic_key"] = normalize_topic_key(values.get("topic_key", ""))
        override_id = values.get("overrides_contact_rule_id")
        if override_id is not None:
            original = self.session.get(ContactRule, override_id)
            if original is None:
                raise DomainError("overridden rule not found")
            if original.person_id != values.get("person_id"):
                raise DomainError("override cannot cross Person")
            scope = values.get("scope")
            if not isinstance(scope, ContactRuleScope) or _SCOPE_SPECIFICITY[scope] <= _SCOPE_SPECIFICITY[original.scope]:
                raise DomainError("override must be narrower than the original")
            if values.get("type") != "ALLOW" or values.get("source") is not ContactRuleSource.USER_CONFIGURED:
                raise DomainError("approved exceptions must be explicit owner-configured ALLOW rules")
            if original.scope in {ContactRuleScope.TOPIC, ContactRuleScope.TASK_DEFINITION} and scope is not ContactRuleScope.TASK_INSTANCE:
                raise DomainError("scoped boundary exceptions must target one applicable task instance")
            if scope is ContactRuleScope.TASK_INSTANCE:
                task = self.session.get(TaskInstance, values.get("task_instance_id"))
                if task is None:
                    raise DomainError("override task instance not found")
                if original.scope is ContactRuleScope.TOPIC and task.topic_key != original.topic_key:
                    raise DomainError("override task does not match original topic scope")
                if (
                    original.scope is ContactRuleScope.TASK_DEFINITION
                    and task.task_definition_id != original.task_definition_id
                ):
                    raise DomainError("override task does not match original definition scope")
            cursor = original
            seen: set[int] = set()
            while cursor.overrides_contact_rule_id is not None:
                if cursor.id in seen:
                    raise DomainError("contact rule override cycle")
                seen.add(cursor.id)
                cursor = self.session.get(ContactRule, cursor.overrides_contact_rule_id)
                if cursor is None:
                    raise DomainError("broken override chain")
        rule = ContactRule(**values)
        self.session.add(rule)
        self.session.flush()
        return rule

    def revoke(self, rule_id: int, reason: str = "REVOKED") -> ContactRule:
        rule = self.session.get(ContactRule, rule_id)
        if rule is None:
            raise DomainError("contact rule not found")
        if rule.revoked_at is None:
            rule.revoked_at = utc_now()
            rule.revoked_reason = reason
            self.session.flush()
        return rule


class DisclosureGrantService:
    def __init__(self, session: Session):
        self.session = session

    def create(self, *, scopes: Iterable[Any], **values: Any) -> DisclosureGrant:
        grant = DisclosureGrant(status=DisclosureGrantStatus.ACTIVE, **values)
        self.session.add(grant)
        self.session.flush()
        scope_values = list(scopes)
        if not scope_values:
            raise DomainError("disclosure grant requires at least one atomic scope")
        self.session.add_all([DisclosureGrantScope(disclosure_grant_id=grant.id, scope=scope) for scope in scope_values])
        self.session.flush()
        return grant

    def deactivate(self, grant_id: int, reason: DisclosureInactiveReason) -> DisclosureGrant:
        grant = self.session.get(DisclosureGrant, grant_id)
        if grant is None:
            raise DomainError("grant not found")
        if grant.status is DisclosureGrantStatus.ACTIVE:
            grant.status = DisclosureGrantStatus.INACTIVE
            grant.inactive_reason = reason
            self.session.flush()
        return grant


class AwaitedResponseService:
    TERMINAL = {
        AwaitedResponseStatus.SATISFIED,
        AwaitedResponseStatus.EXPIRED,
        AwaitedResponseStatus.CANCELLED,
    }

    def __init__(self, session: Session):
        self.session = session

    def create(self, task_participant_id: int, expected_response_type: str, expires_at: datetime | None = None) -> AwaitedResponse:
        response = AwaitedResponse(
            task_participant_id=task_participant_id,
            expected_response_type=expected_response_type,
            status=AwaitedResponseStatus.OPEN,
            expires_at=expires_at,
        )
        self.session.add(response)
        self.session.flush()
        return response

    def transition(self, response_id: int, status: AwaitedResponseStatus) -> AwaitedResponse:
        response = self.session.get(AwaitedResponse, response_id)
        if response is None:
            raise DomainError("awaited response not found")
        if response.status in self.TERMINAL:
            if response.status is status:
                return response
            raise InvalidTransition("awaited response is terminal")
        if status is AwaitedResponseStatus.OPEN:
            raise InvalidTransition("cannot reopen awaited response")
        response.status = status
        self.session.flush()
        return response


def distinct_response_count(session: Session, task_instance_id: int) -> int:
    return int(
        session.scalar(
            select(func.count(func.distinct(TaskParticipant.id)))
            .join(AwaitedResponse, AwaitedResponse.task_participant_id == TaskParticipant.id)
            .where(
                TaskParticipant.task_instance_id == task_instance_id,
                AwaitedResponse.status.in_([AwaitedResponseStatus.SATISFIED, AwaitedResponseStatus.AMBIGUOUS]),
            )
        )
        or 0
    )
