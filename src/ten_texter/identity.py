from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ten_texter.domain import DomainError
from ten_texter.models import (
    ContactRule,
    Conversation,
    ConversationParticipant,
    DisclosureGrant,
    Identity,
    Person,
    TaskDefinitionParticipant,
    TaskParticipant,
)


@dataclass(frozen=True, slots=True)
class IdentityMembershipView:
    conversation_id: int
    beeper_conversation_id: str
    title: str | None
    is_current: bool


@dataclass(frozen=True, slots=True)
class IdentityView:
    identity_id: int
    beeper_user_id: str
    network: str
    username: str | None
    identity_display_name: str | None
    person_id: int
    person_display_name: str
    identity_archived: bool
    person_archived: bool
    person_identity_ids: tuple[int, ...]
    person_state_counts: tuple[tuple[str, int], ...]
    memberships: tuple[IdentityMembershipView, ...]


@dataclass(frozen=True, slots=True)
class IdentityLinkResult:
    identity_id: int
    previous_person_id: int
    person_id: int
    changed: bool


class IdentityLinkingService:
    """Exact owner-directed Identity reassignment with conservative state checks."""

    def __init__(self, session: Session):
        self.session = session

    def inspect(self) -> tuple[IdentityView, ...]:
        records: list[IdentityView] = []
        for identity in self.session.scalars(select(Identity).order_by(Identity.id)):
            person = self.session.get(Person, identity.person_id)
            if person is None:
                raise DomainError("identity has no Person")
            memberships = tuple(
                IdentityMembershipView(
                    conversation_id=conversation.id,
                    beeper_conversation_id=conversation.beeper_conversation_id,
                    title=conversation.title,
                    is_current=membership.is_current,
                )
                for membership, conversation in self.session.execute(
                    select(ConversationParticipant, Conversation)
                    .join(
                        Conversation,
                        Conversation.id == ConversationParticipant.conversation_id,
                    )
                    .where(ConversationParticipant.identity_id == identity.id)
                    .order_by(Conversation.id)
                )
            )
            records.append(
                IdentityView(
                    identity_id=identity.id,
                    beeper_user_id=identity.beeper_user_id,
                    network=identity.network,
                    username=identity.username,
                    identity_display_name=identity.display_name,
                    person_id=person.id,
                    person_display_name=person.display_name,
                    identity_archived=identity.archived_at is not None,
                    person_archived=person.archived_at is not None,
                    person_identity_ids=tuple(
                        self.session.scalars(
                            select(Identity.id)
                            .where(Identity.person_id == person.id)
                            .order_by(Identity.id)
                        )
                    ),
                    person_state_counts=tuple(
                        sorted(self._person_state_counts(person.id).items())
                    ),
                    memberships=memberships,
                )
            )
        return tuple(records)

    def link(self, *, identity_id: int, target_person_id: int) -> IdentityLinkResult:
        identity = self.session.get(Identity, identity_id)
        target = self.session.get(Person, target_person_id)
        if identity is None:
            raise DomainError("identity not found")
        if target is None:
            raise DomainError("target Person not found")
        previous_person_id = identity.person_id
        if previous_person_id == target.id:
            return IdentityLinkResult(identity.id, previous_person_id, target.id, False)
        if target.archived_at is not None:
            raise DomainError("cannot link an identity to an archived Person")

        blockers = self._person_state_counts(previous_person_id)
        populated = {name: count for name, count in blockers.items() if count}
        if populated:
            details = ", ".join(f"{name}={count}" for name, count in sorted(populated.items()))
            raise DomainError(
                "source Person has Person-scoped state requiring explicit resolution: " + details
            )

        memberships = list(
            self.session.scalars(
                select(ConversationParticipant).where(
                    ConversationParticipant.identity_id == identity.id
                )
            )
        )
        for membership in memberships:
            conflicting_person = self.session.scalar(
                select(Identity.person_id)
                .join(
                    ConversationParticipant,
                    ConversationParticipant.identity_id == Identity.id,
                )
                .where(
                    ConversationParticipant.conversation_id == membership.conversation_id,
                    ConversationParticipant.identity_id != identity.id,
                    ConversationParticipant.is_current.is_(True),
                    Identity.person_id == previous_person_id,
                )
                .limit(1)
            )
            conversation = self.session.get(Conversation, membership.conversation_id)
            if (
                conversation is not None
                and conversation.counterparty_person_id == previous_person_id
                and conflicting_person is not None
            ):
                raise DomainError(
                    "direct conversation still contains another identity for the source Person"
                )

        identity.person_id = target.id
        for membership in memberships:
            conversation = self.session.get(Conversation, membership.conversation_id)
            if conversation is not None and conversation.counterparty_person_id == previous_person_id:
                conversation.counterparty_person_id = target.id
        self.session.flush()
        return IdentityLinkResult(identity.id, previous_person_id, target.id, True)

    def _person_state_counts(self, person_id: int) -> dict[str, int]:
        models_and_columns = {
            "contact_rules": ContactRule.person_id,
            "task_participants": TaskParticipant.person_id,
            "task_definition_participants": TaskDefinitionParticipant.person_id,
            "disclosure_grants": DisclosureGrant.source_person_id,
        }
        return {
            name: int(
                self.session.scalar(
                    select(func.count()).select_from(column.class_).where(column == person_id)
                )
                or 0
            )
            for name, column in models_and_columns.items()
        }
