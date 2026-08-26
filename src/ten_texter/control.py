from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from ten_texter.correlation import Classification, Classifier, CorrelationOrchestrator, PreparedCorrelation
from ten_texter.domain import DecisionService, DomainError, ProposalService
from ten_texter.inbound import (
    ordering_conflict_candidates,
    ordering_conflict_still_requires_decision,
)
from ten_texter.enums import (
    AwaitedResponseStatus,
    DecisionStatus,
    MessageKind,
    OutboxCancelReason,
    ParentTerminalPolicy,
    OutboxStatus,
    ProposalStatus,
)
from ten_texter.model_clients import EntityResolverAssistant, TaskPlan
from ten_texter.models import (
    Conversation,
    ConversationParticipant,
    DecisionRequest,
    DecisionRequestPrompt,
    DecisionRequestAwaitedResponseCandidate,
    AwaitedResponse,
    Identity,
    Person,
    Proposal,
    Message,
    MessageRevision,
    OutboxMessage,
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


@dataclass(frozen=True, slots=True)
class PreparedOwnerDecision:
    action: str
    awaited_response_id: int | None = None
    classification: Classification | None = None
    message_revision_id: int | None = None


class _UnusedSemantic:
    def choose(self, revision: MessageRevision, candidates: list[AwaitedResponse]) -> int | None:
        return None


class ProductionOwnerCommandHandler:
    """Turns validated owner plans into one deterministic coordination transaction."""

    def __init__(
        self,
        sessions: sessionmaker[Session],
        *,
        owner_chat_id: int,
        resolver: EntityResolverAssistant,
        generation: ValidatedGenerationPipeline,
        classifier: Classifier | None = None,
    ):
        self.sessions = sessions
        self.owner_chat_id = owner_chat_id
        self.resolver = resolver
        self.generation = generation
        self.classifier = classifier

    def prepare_decision(
        self, decision_id: int, payload: dict[str, Any], _update: TelegramUpdate
    ) -> PreparedOwnerDecision:
        action, selected_id = self._decision_action(payload)
        with self.sessions() as session:
            decision = session.get(DecisionRequest, decision_id)
            if decision is None:
                raise DomainError("decision not found")
            if decision.status is not DecisionStatus.PENDING:
                return PreparedOwnerDecision("already_closed")
            if decision.type == "MESSAGE_ORDERING_CONFLICT":
                if action != "select_revision" or selected_id is None:
                    raise DomainError(
                        "message ordering decisions require `select revision <revision-id>`"
                    )
                subject = session.get(MessageRevision, decision.message_revision_id)
                if subject is None:
                    return PreparedOwnerDecision("stale")
                candidate_ids = {
                    revision.id
                    for revision in ordering_conflict_candidates(session, subject)
                }
                if selected_id not in candidate_ids:
                    raise DomainError(
                        "selected revision is not a candidate for this ordering conflict"
                    )
                if not ordering_conflict_still_requires_decision(session, decision):
                    return PreparedOwnerDecision("stale", message_revision_id=selected_id)
                return PreparedOwnerDecision(
                    "select_revision", message_revision_id=selected_id
                )
            if decision.message_revision_id is not None:
                if action != "select" or selected_id is None:
                    raise DomainError("correlation decisions require `select <awaited-response-id>`")
                allowed = session.get(
                    DecisionRequestAwaitedResponseCandidate,
                    {"decision_request_id": decision.id, "awaited_response_id": selected_id},
                )
                if allowed is None:
                    raise DomainError("selected response is not a candidate for this decision")
                revision = session.get(MessageRevision, decision.message_revision_id)
                awaited = session.get(AwaitedResponse, selected_id)
                message = session.get(Message, revision.message_id) if revision is not None else None
                if (
                    revision is None
                    or awaited is None
                    or message is None
                    or message.current_revision_id != revision.id
                    or revision.awaited_response_id is not None
                    or awaited.status is not AwaitedResponseStatus.OPEN
                ):
                    return PreparedOwnerDecision("stale", selected_id)
                if self.classifier is None:
                    raise DomainError("correlation decision classifier is unavailable")
                return PreparedOwnerDecision(
                    "select",
                    selected_id,
                    self.classifier.classify(revision, awaited),
                )
        return PreparedOwnerDecision(action, selected_id)

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
                final_text=(
                    f"{prepared.review_reason}\n"
                    "This command cannot be applied safely. Reply `dismiss` and submit a corrected command separately."
                ),
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
        payload: object,
        _update: TelegramUpdate,
    ) -> None:
        decision = session.get(DecisionRequest, decision_id)
        if decision is None:
            raise DomainError("decision not found")
        if decision.status is not DecisionStatus.PENDING:
            return
        if not isinstance(payload, PreparedOwnerDecision):
            raise DomainError("decision answer was not prepared")
        if decision.proposal_id is not None:
            accepted = payload.action in {"approve", "accept", "yes"}
            if payload.action not in {"approve", "accept", "yes", "reject", "decline", "no"}:
                raise DomainError("proposal decision requires approve or reject")
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
        if decision.type == "MESSAGE_ORDERING_CONFLICT":
            subject = session.get(MessageRevision, decision.message_revision_id)
            selected = (
                session.get(MessageRevision, payload.message_revision_id)
                if payload.message_revision_id is not None
                else None
            )

            def still_requires_ordering(_decision: DecisionRequest) -> bool:
                if not ordering_conflict_still_requires_decision(session, decision):
                    return False
                if subject is None or selected is None or selected.message_id != subject.message_id:
                    return False
                return selected.id in {
                    revision.id
                    for revision in ordering_conflict_candidates(session, subject)
                }

            def apply_ordering(
                _decision: DecisionRequest, _resolution: dict[str, object]
            ) -> bool:
                assert subject is not None and selected is not None
                message = session.get(Message, subject.message_id)
                if message is None:
                    return False
                message.current_revision_id = selected.id
                return True

            DecisionService(session).answer(
                decision.id,
                {"message_revision_id": payload.message_revision_id},
                subject_still_requires_decision=still_requires_ordering,
                apply=apply_ordering,
            )
            return
        if decision.message_revision_id is not None:
            revision = session.get(MessageRevision, decision.message_revision_id)
            message = session.get(Message, revision.message_id) if revision is not None else None
            awaited = (
                session.get(AwaitedResponse, payload.awaited_response_id)
                if payload.awaited_response_id
                else None
            )

            def still_requires_correlation(_decision: DecisionRequest) -> bool:
                return bool(
                    payload.action == "select"
                    and revision is not None
                    and message is not None
                    and message.current_revision_id == revision.id
                    and revision.awaited_response_id is None
                    and awaited is not None
                    and awaited.status is AwaitedResponseStatus.OPEN
                )

            def apply_correlation(_decision: DecisionRequest, _resolution: dict[str, object]) -> bool:
                assert revision is not None and awaited is not None and payload.classification is not None
                CorrelationOrchestrator(
                    session,
                    semantic=_UnusedSemantic(),
                    classifier=self.classifier,  # type: ignore[arg-type]
                    owner_chat_id=self.owner_chat_id,
                ).apply_prepared(
                    revision.id,
                    PreparedCorrelation(
                        "KNOWN", awaited.id, "OWNER_SELECTION", payload.classification
                    ),
                )
                return True

            DecisionService(session).answer(
                decision.id,
                {"awaited_response_id": payload.awaited_response_id},
                subject_still_requires_decision=still_requires_correlation,
                apply=apply_correlation,
            )
            return
        if decision.outbox_message_id is not None:
            message = session.get(OutboxMessage, decision.outbox_message_id)
            if decision.type == "UNCERTAIN_DELIVERY":
                expected = "keep_reconciling"
            elif decision.type in {
                "VALIDATOR_AUTHORITY_VIOLATION",
                "VALIDATOR_REPAIR_REQUIRED",
            }:
                expected = "keep_blocked"
            else:
                raise DomainError("unsupported Outbox decision type")
            if payload.action != expected:
                raise DomainError(f"{decision.type} only permits `{expected.replace('_', ' ')}`")

            def apply_outbox_decision(
                _decision: DecisionRequest, _resolution: dict[str, object]
            ) -> bool:
                if message is None:
                    return False
                if expected == "keep_blocked":
                    message.status = OutboxStatus.CANCELLED
                    message.cancel_reason = OutboxCancelReason.POLICY_BLOCKED
                return True

            DecisionService(session).answer(
                decision.id,
                {"action": expected},
                subject_still_requires_decision=lambda _decision: bool(
                    message is not None
                    and message.status
                    is (OutboxStatus.RECONCILING if expected == "keep_reconciling" else OutboxStatus.PENDING)
                ),
                apply=apply_outbox_decision,
            )
            return
        if decision.telegram_update_id is not None:
            if payload.action != "dismiss":
                raise DomainError("owner-command review only permits `dismiss`; submit a corrected command separately")
            DecisionService(session).answer(
                decision.id,
                {"action": "dismiss"},
                subject_still_requires_decision=lambda _decision: True,
                apply=lambda _decision, _resolution: True,
            )
            return
        raise DomainError("unsupported decision subject")

    @staticmethod
    def _decision_action(payload: dict[str, Any]) -> tuple[str, int | None]:
        callback = payload.get("callback_query")
        if isinstance(callback, dict):
            raw = str(callback.get("data") or "")
            parts = raw.split(":")
            value = parts[2].casefold() if len(parts) >= 3 else ""
            selected = (
                int(parts[3])
                if value in {"select", "awaited_response", "message_revision"}
                and len(parts) >= 4
                and parts[3].isdigit()
                else None
            )
            if value == "awaited_response":
                value = "select"
            elif value == "message_revision":
                value = "select_revision"
        else:
            message = payload.get("message") or {}
            raw = str(message.get("text") or "").strip().casefold()
            parts = raw.split()
            if len(parts) == 3 and parts[:2] == ["select", "revision"]:
                value = "select_revision"
                selected = int(parts[2]) if parts[2].isdigit() else None
            else:
                value = (
                    "_".join(parts[:2])
                    if parts[:2] in [["keep", "reconciling"], ["keep", "blocked"]]
                    else (parts[0] if parts else "")
                )
                selected = (
                    int(parts[1])
                    if value == "select" and len(parts) == 2 and parts[1].isdigit()
                    else None
                )
        aliases = {"yes": "yes", "y": "yes", "no": "no", "n": "no"}
        return aliases.get(value, value), selected


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
            .where(
                Person.archived_at.is_(None),
                Conversation.archived_at.is_(None),
                ConversationParticipant.is_current.is_(True),
            )
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
