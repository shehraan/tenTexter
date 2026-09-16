from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC

from sqlalchemy import select
from sqlalchemy.orm import Session

from ten_texter.decision_prompts import (
    OrderingPromptCandidate,
    bounded_revision_preview,
    correlation_ambiguity_prompt,
    counterproposal_prompt,
    late_terminal_message_prompt,
    late_terminal_message_trusted_claims,
    late_terminal_message_untrusted_data,
    ordering_conflict_prompt,
    ordering_conflict_trusted_claims,
    ordering_conflict_untrusted_data,
    owner_command_review_prompt,
    uncertain_delivery_prompt,
    validator_block_prompt,
    contact_boundary_ambiguity_prompt,
    contact_rule_exception_prompt,
)
from ten_texter.enums import (
    AwaitedResponseStatus,
    DecisionStatus,
    DestinationKind,
    MessageKind,
    OutboxStatus,
    ProposalStatus,
    TaskStatus,
    Transport,
)
from ten_texter.models import (
    AwaitedResponse,
    DecisionRequest,
    DecisionRequestAwaitedResponseCandidate,
    DecisionRequestPrompt,
    Identity,
    Message,
    MessageRevision,
    OutboxMessage,
    Proposal,
    Person,
    TaskInstance,
    TaskParticipant,
    TelegramOutboxDestination,
    TelegramUpdate,
)


_CORRELATION_DECISION_TYPES = {
    "CORRELATION_AMBIGUITY",
    "CROSS_CONVERSATION_RESPONSE",
    "REVISION_LINEAGE_CONFLICT",
}


@dataclass(frozen=True, slots=True)
class DecisionPromptAuthorization:
    expected_text: str
    allowed_claims: tuple[str, ...]
    untrusted_data: tuple[str, ...] = ()

    @property
    def applicability_token(self) -> tuple[object, ...]:
        return (
            self.expected_text,
            self.allowed_claims,
            self.untrusted_data,
        )


@dataclass(frozen=True, slots=True)
class LateTerminalSubject:
    task_id: int
    topic_key: str
    task_status: str
    sender_name: str
    revision_id: int
    participant_text: str | None


def late_terminal_subject(
    session: Session, decision: DecisionRequest
) -> LateTerminalSubject | None:
    if decision.type != "LATE_TERMINAL_MESSAGE" or decision.message_revision_id is None:
        return None
    revision = session.get(MessageRevision, decision.message_revision_id)
    inbound = session.get(Message, revision.message_id) if revision is not None else None
    awaited = (
        session.get(AwaitedResponse, revision.awaited_response_id)
        if revision is not None and revision.awaited_response_id is not None
        else None
    )
    participant = (
        session.get(TaskParticipant, awaited.task_participant_id)
        if awaited is not None
        else None
    )
    task = (
        session.get(TaskInstance, participant.task_instance_id)
        if participant is not None
        else None
    )
    sender = (
        session.get(Identity, inbound.sender_identity_id)
        if inbound is not None
        else None
    )
    person = session.get(Person, sender.person_id) if sender is not None else None
    if (
        revision is None
        or inbound is None
        or inbound.current_revision_id != revision.id
        or awaited is None
        or participant is None
        or task is None
        or task.status is TaskStatus.ACTIVE
        or decision.task_instance_id != task.id
        or person is None
    ):
        return None
    return LateTerminalSubject(
        task_id=task.id,
        topic_key=task.topic_key,
        task_status=task.status.value,
        sender_name=person.display_name,
        revision_id=revision.id,
        participant_text=revision.text,
    )


def decision_for_prompt(
    session: Session, message: OutboxMessage
) -> DecisionRequest | None:
    return session.scalar(
        select(DecisionRequest)
        .join(
            DecisionRequestPrompt,
            DecisionRequestPrompt.decision_request_id == DecisionRequest.id,
        )
        .where(DecisionRequestPrompt.outbox_message_id == message.id)
    )


def authorize_decision_prompt(
    session: Session,
    message: OutboxMessage,
    *,
    owner_chat_id: int | None,
) -> DecisionPromptAuthorization | None:
    if (
        owner_chat_id is None
        or message.transport is not Transport.TELEGRAM
        or message.destination_kind is not DestinationKind.OWNER
        or message.message_kind is not MessageKind.NOTIFICATION
        or message.trigger_execution_id is not None
        or message.corrects_outbox_message_id is not None
    ):
        return None
    destination = session.get(TelegramOutboxDestination, message.id)
    if destination is None or destination.telegram_chat_id != owner_chat_id:
        return None
    decision = decision_for_prompt(session, message)
    if (
        decision is None
        or decision.status is not DecisionStatus.PENDING
        or message.task_instance_id != decision.task_instance_id
        or message.parent_terminal_policy is not decision.parent_terminal_policy
    ):
        return None
    authorization = _authorization_for_subject(session, decision, message)
    if authorization is None or message.final_text != authorization.expected_text:
        return None
    return authorization


def _authorization_for_subject(
    session: Session,
    decision: DecisionRequest,
    prompt: OutboxMessage,
) -> DecisionPromptAuthorization | None:
    if decision.type == "CONTACT_BOUNDARY_AMBIGUITY" and decision.message_revision_id is not None:
        from ten_texter.contact_boundaries import boundary_context
        revision = session.get(MessageRevision, decision.message_revision_id)
        inbound = session.get(Message, revision.message_id) if revision is not None else None
        if (revision is None or inbound is None or inbound.current_revision_id != revision.id
                or prompt.idempotency_key != f"decision:{decision.id}:owner-prompt"):
            return None
        awaited = (
            session.get(AwaitedResponse, revision.awaited_response_id)
            if revision.awaited_response_id is not None
            else None
        )
        participant = (
            session.get(TaskParticipant, awaited.task_participant_id)
            if awaited is not None
            else None
        )
        context = boundary_context(
            session,
            revision,
            attributable_task_id=(
                participant.task_instance_id if participant is not None else None
            ),
        )
        expected_scope = context.attributable_task_id
        if expected_scope is None and context.complete and len(context.candidates) == 1:
            expected_scope = context.candidates[0].task_instance_id
        if decision.task_instance_id != expected_scope:
            return None
        text = contact_boundary_ambiguity_prompt(decision.id, context.candidates)
        claims = (
            f"Participant contact boundary decision {decision.id} has ambiguous scope.",
            *(f"Task {item.task_instance_id} has topic {item.topic_key}." for item in context.candidates),
            "The owner may set a listed scope or dismiss by replying to this Telegram message.",
        )
        return DecisionPromptAuthorization(text, claims, (f"participant text={revision.text!r}",))

    if (
        decision.type == "CONTACT_RULE_EXCEPTION"
        and decision.contact_rule_id is not None
        and decision.task_instance_id is not None
    ):
        from ten_texter.contact_boundaries import contact_rule_exception_subject

        subject = contact_rule_exception_subject(
            session,
            rule_id=decision.contact_rule_id,
            task_id=decision.task_instance_id,
        )
        if (
            subject is None
            or prompt.idempotency_key != f"decision:{decision.id}:owner-prompt"
        ):
            return None
        rule = subject.rule
        task = subject.task
        person = subject.person
        if rule.scope.value == "GLOBAL":
            scope_description = "global"
        elif rule.scope.value == "TOPIC":
            scope_description = f"topic {rule.topic_key}"
        else:
            scope_description = f"task definition {rule.task_definition_id}"
        text = contact_rule_exception_prompt(
            rule.id,
            task.id,
            person.display_name,
            scope_description,
            rule.value,
        )
        return DecisionPromptAuthorization(
            text,
            (
                f"{person.display_name} has contact rule {rule.id} scoped to {scope_description}.",
                f"Contact rule {rule.id} blocks task {task.id}.",
                "The owner may allow only this task or respect the participant boundary.",
            ),
            (f"contact rule {rule.id} participant boundary text={rule.value!r}",),
        )
    if decision.type == "UNCERTAIN_DELIVERY" and decision.outbox_message_id is not None:
        subject = session.get(OutboxMessage, decision.outbox_message_id)
        if (
            subject is None
            or subject.status is not OutboxStatus.RECONCILING
            or prompt.idempotency_key != f"decision:{decision.id}:owner-prompt"
        ):
            return None
        text = uncertain_delivery_prompt(subject.id)
        return DecisionPromptAuthorization(text, (text,))

    if decision.type in {
        "VALIDATOR_AUTHORITY_VIOLATION",
        "VALIDATOR_REPAIR_REQUIRED",
    } and decision.outbox_message_id is not None:
        subject = session.get(OutboxMessage, decision.outbox_message_id)
        if (
            subject is None
            or subject.status is not OutboxStatus.PENDING
            or prompt.idempotency_key != f"decision:{decision.id}:owner-prompt"
        ):
            return None
        text = validator_block_prompt(subject.id)
        return DecisionPromptAuthorization(text, (text,))

    if decision.type == "OWNER_COMMAND_REVIEW" and decision.telegram_update_id is not None:
        update = session.get(TelegramUpdate, decision.telegram_update_id)
        if (
            update is None
            or prompt.idempotency_key != f"telegram-update:{update.id}:review"
        ):
            return None
        text = owner_command_review_prompt(update.telegram_update_id)
        return DecisionPromptAuthorization(text, (text,))

    if decision.type == "COUNTERPROPOSAL" and decision.proposal_id is not None:
        proposal = session.get(Proposal, decision.proposal_id)
        if (
            proposal is None
            or proposal.status is not ProposalStatus.PENDING
            or proposal.task_instance_id != decision.task_instance_id
            or prompt.idempotency_key != f"proposal:{proposal.id}:owner-prompt"
        ):
            return None
        text = counterproposal_prompt(
            proposal.field,
            proposal.operation,
            proposal.proposed_value,
        )
        return DecisionPromptAuthorization(text, (text,))

    if decision.type == "LATE_TERMINAL_MESSAGE":
        subject = late_terminal_subject(session, decision)
        if (
            subject is None
            or prompt.idempotency_key != f"decision:{decision.id}:owner-prompt"
        ):
            return None
        text = late_terminal_message_prompt(
            task_id=subject.task_id,
            topic_key=subject.topic_key,
            task_status=subject.task_status,
            sender_name=subject.sender_name,
            revision_id=subject.revision_id,
            participant_text=subject.participant_text,
        )
        return DecisionPromptAuthorization(
            text,
            late_terminal_message_trusted_claims(
                task_id=subject.task_id,
                topic_key=subject.topic_key,
                task_status=subject.task_status,
                sender_name=subject.sender_name,
                revision_id=subject.revision_id,
            ),
            late_terminal_message_untrusted_data(
                revision_id=subject.revision_id,
                participant_text=subject.participant_text,
            ),
        )

    if (
        decision.type in _CORRELATION_DECISION_TYPES
        and decision.message_revision_id is not None
    ):
        revision = session.get(MessageRevision, decision.message_revision_id)
        inbound = session.get(Message, revision.message_id) if revision is not None else None
        candidates = list(
            session.scalars(
                select(AwaitedResponse)
                .join(
                    DecisionRequestAwaitedResponseCandidate,
                    DecisionRequestAwaitedResponseCandidate.awaited_response_id
                    == AwaitedResponse.id,
                )
                .where(
                    DecisionRequestAwaitedResponseCandidate.decision_request_id
                    == decision.id
                )
                .order_by(AwaitedResponse.id)
            )
        )
        if (
            revision is None
            or inbound is None
            or inbound.current_revision_id != revision.id
            or not candidates
            or any(candidate.status is not AwaitedResponseStatus.OPEN for candidate in candidates)
            or prompt.idempotency_key != f"decision:{decision.id}:owner-prompt"
        ):
            return None
        prompt_candidates: list[tuple[int, str, str, int, int, str]] = []
        for candidate in candidates:
            participant = session.get(TaskParticipant, candidate.task_participant_id)
            task = (
                session.get(TaskInstance, participant.task_instance_id)
                if participant is not None
                else None
            )
            if participant is None or task is None:
                return None
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
        text = correlation_ambiguity_prompt(prompt_candidates)
        return DecisionPromptAuthorization(text, (text,))

    if decision.type == "MESSAGE_ORDERING_CONFLICT" and decision.message_revision_id is not None:
        from ten_texter.inbound import (
            ordering_conflict_candidates,
            ordering_conflict_still_requires_decision,
        )

        revision = session.get(MessageRevision, decision.message_revision_id)
        if (
            revision is None
            or not ordering_conflict_still_requires_decision(session, decision)
            or prompt.idempotency_key != f"decision:{decision.id}:owner-prompt"
        ):
            return None
        prompt_candidates: list[OrderingPromptCandidate] = []
        for candidate in sorted(
            ordering_conflict_candidates(session, revision), key=lambda value: value.id
        ):
            event_at = (
                candidate.provider_event_at.replace(
                    tzinfo=candidate.provider_event_at.tzinfo or UTC
                ).astimezone(UTC).isoformat()
                if candidate.provider_event_at is not None
                else None
            )
            prompt_candidates.append(
                (
                    candidate.id,
                    candidate.provider_sort_key,
                    candidate.provider_sequence,
                    event_at,
                    bounded_revision_preview(candidate.text),
                )
            )
        if not prompt_candidates:
            return None
        text = ordering_conflict_prompt(prompt_candidates)
        return DecisionPromptAuthorization(
            text,
            ordering_conflict_trusted_claims(prompt_candidates),
            ordering_conflict_untrusted_data(prompt_candidates),
        )
    return None
