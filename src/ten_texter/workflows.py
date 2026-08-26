from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.orm import Session

from ten_texter.domain import DecisionService, DomainError, ProposalService, TaskService
from ten_texter.enums import (
    DecisionStatus,
    MessageKind,
    ParentTerminalPolicy,
    ProposalStatus,
)
from ten_texter.models import (
    AwaitedResponsePrompt,
    DecisionRequest,
    Proposal,
    TaskInstance,
    TaskParticipant,
)
from ten_texter.outbox import OutboxService
from ten_texter.domain import AwaitedResponseService


@dataclass(frozen=True, slots=True)
class ParticipantSendPlan:
    person_id: int
    conversation_id: int
    final_text: str


@dataclass(frozen=True, slots=True)
class CoordinationStart:
    task_id: int
    outbox_ids: tuple[int, ...]
    awaited_response_ids: tuple[int, ...]


class CoordinationWorkflow:
    """Deterministic composition root for already parsed, resolved, and validated intent."""

    def __init__(self, session: Session, *, owner_chat_id: int):
        self.session = session
        self.owner_chat_id = owner_chat_id

    def start(
        self,
        *,
        scheduled_at: datetime,
        duration_minutes: int,
        location: str | None,
        topic_key: str,
        participant_sends: list[ParticipantSendPlan],
    ) -> CoordinationStart:
        task = TaskService(self.session).create(
            scheduled_at=scheduled_at,
            duration_minutes=duration_minutes,
            location=location,
            topic_key=topic_key,
            participants=[
                (plan.person_id, plan.conversation_id) for plan in participant_sends
            ],
        )
        return self.start_existing(task.id, participant_sends)

    def start_existing(
        self,
        task_id: int,
        participant_sends: list[ParticipantSendPlan],
    ) -> CoordinationStart:
        task = self.session.get(TaskInstance, task_id)
        if task is None:
            raise DomainError("task not found")
        outbox_ids: list[int] = []
        awaited_ids: list[int] = []
        participants_by_person = {
            participant.person_id: participant
            for participant in self.session.query(TaskParticipant).filter(
                TaskParticipant.task_instance_id == task.id
            )
        }
        outbox = OutboxService(self.session)
        awaited_service = AwaitedResponseService(self.session)
        for plan in participant_sends:
            participant = participants_by_person.get(plan.person_id)
            if participant is None or participant.conversation_id != plan.conversation_id:
                raise DomainError("send plan does not match a pinned task participant")
            message = outbox.create_beeper(
                task_instance_id=task.id,
                conversation_id=plan.conversation_id,
                participant_ids=[participant.id],
                final_text=plan.final_text,
                message_kind=MessageKind.INITIAL,
                idempotency_key=f"task:{task.id}:initial:participant:{participant.id}",
            )
            awaited = awaited_service.create(participant.id, "availability")
            self.session.add(
                AwaitedResponsePrompt(
                    awaited_response_id=awaited.id,
                    outbox_message_id=message.id,
                )
            )
            outbox_ids.append(message.id)
            awaited_ids.append(awaited.id)
        self.session.flush()
        return CoordinationStart(task.id, tuple(outbox_ids), tuple(awaited_ids))

    def notify_owner(self, task_id: int, text: str, *, key: str) -> int:
        task = self.session.get(TaskInstance, task_id)
        if task is None:
            raise DomainError("task not found")
        message = OutboxService(self.session).create_owner(
            telegram_chat_id=self.owner_chat_id,
            task_instance_id=task.id,
            final_text=text,
            message_kind=MessageKind.NOTIFICATION,
            idempotency_key=f"task:{task.id}:owner:{key}",
            parent_terminal_policy=ParentTerminalPolicy.SURVIVE,
        )
        return message.id

    def approve_proposal(self, decision_id: int) -> DecisionRequest:
        decision = self.session.get(DecisionRequest, decision_id)
        if decision is None or decision.proposal_id is None:
            raise DomainError("decision is not for a proposal")
        proposal = self.session.get(Proposal, decision.proposal_id)
        assert proposal is not None

        def still_requires(_decision: DecisionRequest) -> bool:
            return proposal.status is ProposalStatus.PENDING

        def apply(_decision: DecisionRequest, _resolution: dict[str, object]) -> bool:
            resolved = ProposalService(self.session).resolve(proposal.id, accept=True)
            return resolved.status is ProposalStatus.ACCEPTED

        return DecisionService(self.session).answer(
            decision.id,
            {"accept": True},
            subject_still_requires_decision=still_requires,
            apply=apply,
        )
