from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import distinct, func, select
from sqlalchemy.orm import Session

from ten_texter.decision_prompts import mass_contact_confirmation_prompt
from ten_texter.domain import DecisionService
from ten_texter.enums import (
    DecisionCloseReason,
    DecisionStatus,
    DestinationKind,
    MessageKind,
    ParentTerminalPolicy,
    TaskStatus,
    Transport,
)
from ten_texter.models import (
    DecisionRequest,
    DecisionRequestPrompt,
    OutboxMessage,
    OutboxMessageParticipant,
    TaskInstance,
    TaskParticipant,
)
from ten_texter.outbox import OutboxService


MASS_CONTACT_THRESHOLD = 25
MASS_CONTACT_DECISION_TYPE = "MASS_CONTACT_CONFIRMATION"


@dataclass(frozen=True, slots=True)
class MassContactSubject:
    task: TaskInstance
    anchor: OutboxMessage
    participant_count: int


def mass_contact_participant_count(session: Session, task_id: int) -> int:
    return int(
        session.scalar(
            select(func.count(distinct(TaskParticipant.person_id))).where(
                TaskParticipant.task_instance_id == task_id
            )
        )
        or 0
    )


def _anchor(session: Session, task_id: int) -> OutboxMessage | None:
    return session.scalar(
        select(OutboxMessage)
        .join(
            OutboxMessageParticipant,
            OutboxMessageParticipant.outbox_message_id == OutboxMessage.id,
        )
        .join(
            TaskParticipant,
            TaskParticipant.id == OutboxMessageParticipant.task_participant_id,
        )
        .where(
            OutboxMessage.task_instance_id == task_id,
            OutboxMessage.transport == Transport.BEEPER,
            OutboxMessage.destination_kind == DestinationKind.PARTICIPANT,
            TaskParticipant.task_instance_id == task_id,
        )
        .order_by(OutboxMessage.id)
        .limit(1)
    )


def mass_contact_subject(
    session: Session,
    decision: DecisionRequest,
) -> MassContactSubject | None:
    if (
        decision.type != MASS_CONTACT_DECISION_TYPE
        or decision.task_instance_id is None
        or decision.outbox_message_id is None
        or decision.parent_terminal_policy is not ParentTerminalPolicy.TERMINATE
    ):
        return None
    task = session.get(TaskInstance, decision.task_instance_id)
    anchor = _anchor(session, decision.task_instance_id)
    participant_count = mass_contact_participant_count(
        session, decision.task_instance_id
    )
    if (
        task is None
        or task.status is not TaskStatus.ACTIVE
        or participant_count <= MASS_CONTACT_THRESHOLD
        or anchor is None
        or anchor.id != decision.outbox_message_id
        or anchor.task_instance_id != task.id
    ):
        return None
    return MassContactSubject(task, anchor, participant_count)


def mass_contact_is_approved(session: Session, task_id: int) -> bool:
    decisions = session.scalars(
        select(DecisionRequest)
        .where(
            DecisionRequest.task_instance_id == task_id,
            DecisionRequest.type == MASS_CONTACT_DECISION_TYPE,
            DecisionRequest.status == DecisionStatus.CLOSED,
            DecisionRequest.close_reason == DecisionCloseReason.ANSWERED,
        )
        .order_by(DecisionRequest.id)
    )
    for decision in decisions:
        subject = mass_contact_subject(session, decision)
        if subject is not None and decision.resolution_json == {
            "action": "approve_mass_contact",
            "participant_count": subject.participant_count,
            "threshold": MASS_CONTACT_THRESHOLD,
        }:
            return True
    return False


def ensure_mass_contact_decision(
    session: Session,
    *,
    task_id: int,
    owner_chat_id: int,
) -> DecisionRequest | None:
    if (
        mass_contact_participant_count(session, task_id) <= MASS_CONTACT_THRESHOLD
        or mass_contact_is_approved(session, task_id)
    ):
        return None
    anchor = _anchor(session, task_id)
    if anchor is None:
        return None
    pending = list(
        session.scalars(
            select(DecisionRequest)
            .where(
                DecisionRequest.task_instance_id == task_id,
                DecisionRequest.type == MASS_CONTACT_DECISION_TYPE,
                DecisionRequest.status == DecisionStatus.PENDING,
            )
            .order_by(DecisionRequest.id)
        )
    )
    decision = next(
        (item for item in pending if mass_contact_subject(session, item) is not None),
        None,
    )
    if decision is None:
        participant_count = mass_contact_participant_count(session, task_id)
        decision = DecisionService(session).create(
            decision_type=MASS_CONTACT_DECISION_TYPE,
            subject_kind="outbox_message",
            subject_id=anchor.id,
            context={
                "participant_count": participant_count,
                "threshold": MASS_CONTACT_THRESHOLD,
            },
            task_instance_id=task_id,
            parent_terminal_policy=ParentTerminalPolicy.TERMINATE,
        )
    subject = mass_contact_subject(session, decision)
    if subject is None:
        return None
    prompt = OutboxService(session).create_owner(
        telegram_chat_id=owner_chat_id,
        final_text=mass_contact_confirmation_prompt(
            task_id=task_id,
            participant_count=subject.participant_count,
            threshold=MASS_CONTACT_THRESHOLD,
        ),
        message_kind=MessageKind.NOTIFICATION,
        idempotency_key=f"decision:{decision.id}:owner-prompt",
        task_instance_id=task_id,
        parent_terminal_policy=ParentTerminalPolicy.TERMINATE,
    )
    if (
        session.get(
            DecisionRequestPrompt,
            {
                "decision_request_id": decision.id,
                "outbox_message_id": prompt.id,
            },
        )
        is None
    ):
        session.add(
            DecisionRequestPrompt(
                decision_request_id=decision.id,
                outbox_message_id=prompt.id,
            )
        )
        session.flush()
    return decision
