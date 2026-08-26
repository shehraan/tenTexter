from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from ten_texter.domain import DecisionService, DomainError, ProposalService
from ten_texter.enums import (
    DecisionStatus,
    MessageKind,
    ParentTerminalPolicy,
    ProposalStatus,
)
from ten_texter.model_clients import EntityResolverAssistant, TaskPlan
from ten_texter.models import (
    Conversation,
    ConversationParticipant,
    DecisionRequest,
    DecisionRequestPrompt,
    Identity,
    Person,
    Proposal,
    TelegramUpdate,
    TaskDefinition,
    TaskDefinitionParticipant,
)
from ten_texter.outbox import OutboxService
from ten_texter.validator import (
    GenerationOutcome,
    ValidatedGenerationPipeline,
    ValidatorContext,
)
from ten_texter.workflows import CoordinationWorkflow, ParticipantSendPlan


@dataclass(frozen=True, slots=True)
class ResolvedRoute:
    person_id: int
    conversation_id: int
    display_name: str


@dataclass(frozen=True, slots=True)
class PreparedOwnerCommand:
    plan: TaskPlan
    sends: tuple[ParticipantSendPlan, ...] = ()
    review_reason: str | None = None


class ProductionOwnerCommandHandler:
    """Turns validated owner plans into one deterministic coordination transaction."""

    def __init__(
        self,
        sessions: sessionmaker[Session],
        *,
        owner_chat_id: int,
        resolver: EntityResolverAssistant,
        generation: ValidatedGenerationPipeline,
    ):
        self.sessions = sessions
        self.owner_chat_id = owner_chat_id
        self.resolver = resolver
        self.generation = generation

    def prepare_command(self, parsed: object, _update: TelegramUpdate) -> PreparedOwnerCommand:
        if not isinstance(parsed, TaskPlan):
            raise DomainError("owner parser returned an unsupported plan")
        with self.sessions() as session:
            candidates = self._route_candidates(session)
        routes: list[ResolvedRoute] = []
        for reference in parsed.participant_references:
            route = self._resolve_route(reference, candidates)
            if route is None:
                return PreparedOwnerCommand(
                    parsed,
                    review_reason=f"Participant route is ambiguous or unavailable: {reference}",
                )
            if any(existing.person_id == route.person_id for existing in routes):
                return PreparedOwnerCommand(
                    parsed,
                    review_reason=f"Participant was resolved more than once: {reference}",
                )
            routes.append(route)

        if parsed.recurrence_rule is not None:
            return PreparedOwnerCommand(
                parsed,
                tuple(
                    ParticipantSendPlan(
                        person_id=route.person_id,
                        conversation_id=route.conversation_id,
                        final_text="",
                    )
                    for route in routes
                ),
            )

        task_claims = (
            f"topic: {parsed.topic_key}",
            f"scheduled_at: {parsed.scheduled_at.isoformat()}",
            f"duration_minutes: {parsed.duration_minutes}",
            *((f"location: {parsed.location}",) if parsed.location else ()),
        )
        sends: list[ParticipantSendPlan] = []
        for route in routes:
            facts = [
                {"claim": claim} for claim in task_claims
            ] + [{"participant": route.display_name}]
            result = self.generation.run(
                goal=f"Ask {route.display_name} whether they are available for the planned activity.",
                facts=facts,
                constraints=[
                    "Ask only about this coordination task.",
                    "Do not make commitments on the owner's behalf.",
                    "Do not mention information from another conversation.",
                ],
                context=ValidatorContext(
                    message_kind=MessageKind.INITIAL,
                    allowed_claims=task_claims + (f"participant: {route.display_name}",),
                    constraints=(
                        "No cross-conversation participant facts are authorized for this initial message.",
                    ),
                ),
            )
            if result.outcome is not GenerationOutcome.READY or result.text is None:
                return PreparedOwnerCommand(
                    parsed,
                    review_reason=(
                        f"Initial message generation requires owner review: {result.category.value}"
                    ),
                )
            sends.append(
                ParticipantSendPlan(
                    person_id=route.person_id,
                    conversation_id=route.conversation_id,
                    final_text=result.text,
                )
            )
        return PreparedOwnerCommand(parsed, tuple(sends))

    def apply_command(
        self,
        session: Session,
        prepared: object,
        update: TelegramUpdate,
    ) -> None:
        if not isinstance(prepared, PreparedOwnerCommand):
            raise DomainError("owner command was not prepared")
        if prepared.review_reason is not None:
            decision = DecisionService(session).create(
                decision_type="OWNER_COMMAND_REVIEW",
                subject_kind="telegram_update",
                subject_id=update.id,
                context={"reason": prepared.review_reason},
                parent_terminal_policy=ParentTerminalPolicy.SURVIVE,
            )
            prompt = OutboxService(session).create_owner(
                telegram_chat_id=self.owner_chat_id,
                final_text=prepared.review_reason,
                message_kind=MessageKind.NOTIFICATION,
                idempotency_key=f"telegram-update:{update.id}:review",
                parent_terminal_policy=ParentTerminalPolicy.SURVIVE,
            )
            session.add(
                DecisionRequestPrompt(
                    decision_request_id=decision.id,
                    outbox_message_id=prompt.id,
                )
            )
            return
        if prepared.plan.recurrence_rule is not None and prepared.plan.timezone is not None:
            local_time = prepared.plan.scheduled_at.astimezone(
                ZoneInfo(prepared.plan.timezone)
            ).strftime("%H:%M")
            definition = TaskDefinition(
                name=f"Recurring {prepared.plan.topic_key}",
                recurrence_rule=prepared.plan.recurrence_rule,
                default_time=local_time,
                timezone=prepared.plan.timezone,
                default_duration_minutes=prepared.plan.duration_minutes,
                default_location=prepared.plan.location,
                default_topic_key=prepared.plan.topic_key,
                next_occurrence_at=prepared.plan.scheduled_at,
            )
            session.add(definition)
            session.flush()
            session.add_all(
                [
                    TaskDefinitionParticipant(
                        task_definition_id=definition.id,
                        person_id=send.person_id,
                    )
                    for send in prepared.sends
                ]
            )
            return
        CoordinationWorkflow(session, owner_chat_id=self.owner_chat_id).start(
            scheduled_at=prepared.plan.scheduled_at,
            duration_minutes=prepared.plan.duration_minutes,
            location=prepared.plan.location,
            topic_key=prepared.plan.topic_key,
            participant_sends=list(prepared.sends),
        )

    def apply_decision(
        self,
        session: Session,
        decision_id: int,
        payload: dict[str, Any],
        _update: TelegramUpdate,
    ) -> None:
        decision = session.get(DecisionRequest, decision_id)
        if decision is None or decision.status is not DecisionStatus.PENDING:
            raise DomainError("decision is not pending")
        accepted = self._accepted(payload)
        if decision.proposal_id is not None:
            proposal = session.get(Proposal, decision.proposal_id)
            assert proposal is not None

            def still_requires(_decision: DecisionRequest) -> bool:
                return proposal.status is ProposalStatus.PENDING

            def apply(_decision: DecisionRequest, _resolution: dict[str, object]) -> bool:
                ProposalService(session).resolve(proposal.id, accept=accepted)
                return True

            DecisionService(session).answer(
                decision.id,
                {"accept": accepted},
                subject_still_requires_decision=still_requires,
                apply=apply,
            )
            return
        DecisionService(session).answer(
            decision.id,
            {"acknowledged": accepted},
            subject_still_requires_decision=lambda _decision: True,
            apply=lambda _decision, _resolution: True,
        )

    @staticmethod
    def _accepted(payload: dict[str, Any]) -> bool:
        callback = payload.get("callback_query")
        if isinstance(callback, dict):
            value = str(callback.get("data") or "").rsplit(":", 1)[-1].casefold()
        else:
            message = payload.get("message") or {}
            value = str(message.get("text") or "").strip().casefold()
        if value in {"approve", "accept", "yes", "y"}:
            return True
        if value in {"reject", "decline", "no", "n"}:
            return False
        raise DomainError("owner decision answer must be an explicit yes/no choice")

    @staticmethod
    def _route_candidates(session: Session) -> list[dict[str, object]]:
        rows = session.execute(
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
            .where(Person.archived_at.is_(None), Conversation.archived_at.is_(None))
            .order_by(Person.id, Conversation.id)
        )
        candidates: list[dict[str, object]] = []
        for index, (person, identity, conversation) in enumerate(rows, start=1):
            candidates.append(
                {
                    "id": index,
                    "person_id": person.id,
                    "conversation_id": conversation.id,
                    "display_name": person.display_name,
                    "person_name": person.display_name,
                    "identity_username": identity.username,
                    "identity_display_name": identity.display_name,
                    "beeper_user_id": identity.beeper_user_id,
                    "conversation_title": conversation.title,
                    "beeper_conversation_id": conversation.beeper_conversation_id,
                    "network": conversation.network,
                }
            )
        return candidates

    def _resolve_route(
        self,
        reference: str,
        candidates: list[dict[str, object]],
    ) -> ResolvedRoute | None:
        needle = reference.strip().casefold()
        keys = (
            "person_name",
            "identity_username",
            "identity_display_name",
            "beeper_user_id",
            "conversation_title",
            "beeper_conversation_id",
        )
        exact = [
            candidate
            for candidate in candidates
            if any(
                isinstance(candidate.get(key), str)
                and str(candidate[key]).strip().casefold() == needle
                for key in keys
            )
        ]
        pool = exact or candidates
        chosen: dict[str, object] | None = pool[0] if len(pool) == 1 else None
        if chosen is None and pool:
            selected_id = self.resolver.resolve(reference, pool)
            matches = [candidate for candidate in pool if candidate["id"] == selected_id]
            chosen = matches[0] if len(matches) == 1 else None
        if chosen is None:
            return None
        return ResolvedRoute(
            person_id=int(chosen["person_id"]),
            conversation_id=int(chosen["conversation_id"]),
            display_name=str(chosen["display_name"]),
        )
