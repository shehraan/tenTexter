from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from ten_texter.correlation import AtomicProposal, Classification, CorrelationOrchestrator
from ten_texter.decision_prompts import (
    CounterproposalPromptContext,
    counterproposal_prompt,
    legacy_counterproposal_prompt,
)
from ten_texter.enums import (
    AwaitedResponseStatus,
    MessageKind,
    ParentTerminalPolicy,
    ProposalStatus,
)
from ten_texter.domain import DecisionService
from ten_texter.models import (
    AwaitedResponse,
    DecisionRequest,
    DecisionRequestPrompt,
    OutboxMessage,
)
from ten_texter.policy import DatabaseContextProvider, PolicyRevalidator, PreSendDecision
from ten_texter.validator import DatabaseValidatorContextProvider
from ten_texter.outbox import OutboxService
from ten_texter.models import Proposal
from tests.test_schema import NOW, seed_core


def test_scheduled_counterproposal_is_friendly_and_localized() -> None:
    context = CounterproposalPromptContext(
        participant_name="Shehraan Canada",
        topic_key="tennis",
        scheduled_at=datetime(2026, 9, 25, 21, 0, tzinfo=UTC),
        duration_minutes=30,
        location=None,
        owner_timezone="America/Toronto",
    )

    rendered = counterproposal_prompt(
        context=context,
        field="scheduled_at",
        operation="SET",
        proposed_value="2026-09-25T22:00:00+00:00",
    )

    assert rendered == (
        "Shehraan Canada suggested moving tennis from Friday, September 25 at 5:00 PM "
        "to 6:00 PM for 30 minutes. Should I accept this change? Reply to this "
        "Telegram message with `approve` or `reject`."
    )
    assert "scheduled_at" not in rendered
    assert "SET" not in rendered
    assert "+00:00" not in rendered


def test_scheduled_counterproposal_includes_new_local_date_when_needed() -> None:
    rendered = counterproposal_prompt(
        context=CounterproposalPromptContext(
            participant_name="Shehraan Canada",
            topic_key="tennis",
            scheduled_at=datetime(2026, 9, 25, 23, 30, tzinfo=UTC),
            duration_minutes=30,
            location=None,
            owner_timezone="America/Toronto",
        ),
        field="scheduled_at",
        operation="SET",
        proposed_value="2026-09-26T04:00:00+00:00",
    )

    assert "from Friday, September 25 at 7:30 PM to Saturday, September 26 at 12:00 AM" in rendered


@pytest.mark.parametrize(
    ("field", "proposed_value", "expected_fragment"),
    [
        (
            "duration_minutes",
            45,
            "changing the duration for tennis from 30 minutes to 45 minutes",
        ),
        (
            "location",
            "River courts",
            'changing the location for tennis from no location to “River courts”',
        ),
    ],
)
def test_non_time_counterproposals_are_humanized(
    field: str,
    proposed_value: object,
    expected_fragment: str,
) -> None:
    rendered = counterproposal_prompt(
        context=CounterproposalPromptContext(
            participant_name="Shehraan Canada",
            topic_key="tennis",
            scheduled_at=datetime(2026, 9, 25, 21, 0, tzinfo=UTC),
            duration_minutes=30,
            location=None,
            owner_timezone="America/Toronto",
        ),
        field=field,
        operation="SET",
        proposed_value=proposed_value,
    )

    assert expected_fragment in rendered


def test_legacy_counterproposal_prompt_is_stable() -> None:
    assert legacy_counterproposal_prompt(
        "scheduled_at", "SET", "2026-09-25T22:00:00+00:00"
    ) == (
        "Participant proposed scheduled_at SET: 2026-09-25T22:00:00+00:00. "
        "Reply to this Telegram message with `approve` or `reject`."
    )


def test_counterproposal_creation_and_validation_share_localized_prompt(
    db_session: Session,
) -> None:
    prompt = _create_scheduled_counterproposal(db_session)
    db_session.commit()

    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    context = DatabaseValidatorContextProvider(
        factory,
        facts=DatabaseContextProvider(),
        owner_chat_id=99,
        owner_timezone="America/Toronto",
    ).context_for(prompt.id, prompt.message_kind)

    assert context.allowed_claims == (prompt.final_text,)
    assert "Friday, September 25 at 5:00 PM" in prompt.final_text
    assert "to 6:00 PM" in prompt.final_text
    assert PolicyRevalidator(
        owner_chat_id=99,
        owner_timezone="America/Toronto",
    ).check(db_session, prompt) is PreSendDecision.READY


def test_existing_legacy_counterproposal_prompt_remains_sendable(
    db_session: Session,
) -> None:
    prompt = _create_legacy_scheduled_counterproposal(db_session)
    legacy_text = legacy_counterproposal_prompt(
        "scheduled_at", "SET", "2026-09-25T22:00:00+00:00"
    )
    prompt.final_text = legacy_text
    db_session.commit()

    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    context = DatabaseValidatorContextProvider(
        factory,
        facts=DatabaseContextProvider(),
        owner_chat_id=99,
        owner_timezone="America/Toronto",
    ).context_for(prompt.id, prompt.message_kind)

    assert context.allowed_claims == (legacy_text,)
    assert PolicyRevalidator(
        owner_chat_id=99,
        owner_timezone="America/Toronto",
    ).check(db_session, prompt) is PreSendDecision.READY


def test_invalid_owner_timezone_is_rejected() -> None:
    from ten_texter.config import Settings

    with pytest.raises(ValueError, match="owner timezone"):
        Settings(owner_timezone="Eastern").validate_runtime()


def _create_scheduled_counterproposal(db_session: Session) -> OutboxMessage:
    core = seed_core(db_session)
    task = core["task"]
    task.scheduled_at = datetime(2026, 9, 25, 21, 0, tzinfo=UTC)
    task.duration_minutes = 30
    db_session.flush()
    db_session.add(
        AwaitedResponse(
            task_participant_id=core["participant"].id,
            expected_response_type="availability",
            status=AwaitedResponseStatus.OPEN,
            created_at=NOW,
        )
    )
    db_session.flush()

    result = CorrelationOrchestrator(
        db_session,
        semantic=_Semantic(),
        classifier=_Classifier(
            Classification(
                kind="COUNTERPROPOSAL",
                proposals=(
                    AtomicProposal(
                        field="scheduled_at",
                        operation="SET",
                        old_value="2026-09-25T21:00:00Z",
                        proposed_value="2026-09-25T22:00:00Z",
                    ),
                ),
            )
        ),
        owner_chat_id=99,
        owner_timezone="America/Toronto",
    ).process(core["revision"].id)
    assert result.outcome == "CORRELATED"

    return db_session.scalar(
        select(OutboxMessage)
        .join(DecisionRequestPrompt)
        .join(DecisionRequest)
        .where(DecisionRequest.type == "COUNTERPROPOSAL")
    )


def _create_legacy_scheduled_counterproposal(db_session: Session) -> OutboxMessage:
    core = seed_core(db_session)
    task = core["task"]
    task.scheduled_at = datetime(2026, 9, 25, 21, 0, tzinfo=UTC)
    task.duration_minutes = 30
    db_session.flush()
    proposal = Proposal(
        task_instance_id=task.id,
        proposed_by_participant_id=core["participant"].id,
        source_message_revision_id=core["revision"].id,
        field="scheduled_at",
        operation="SET",
        old_value="2026-09-25T21:00:00Z",
        proposed_value="2026-09-25T22:00:00+00:00",
        status=ProposalStatus.PENDING,
    )
    db_session.add(proposal)
    db_session.flush()
    decision = DecisionService(db_session).create(
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
    prompt = OutboxService(db_session).create_owner(
        telegram_chat_id=99,
        task_instance_id=task.id,
        final_text=legacy_counterproposal_prompt(
            proposal.field,
            proposal.operation,
            proposal.proposed_value,
        ),
        message_kind=MessageKind.NOTIFICATION,
        idempotency_key=f"proposal:{proposal.id}:owner-prompt",
        parent_terminal_policy=ParentTerminalPolicy.TERMINATE,
    )
    db_session.add(
        DecisionRequestPrompt(
            decision_request_id=decision.id,
            outbox_message_id=prompt.id,
        )
    )
    db_session.flush()
    return prompt


class _Semantic:
    def choose(self, *_: object) -> int | None:
        return None


class _Classifier:
    def __init__(self, result: Classification) -> None:
        self.result = result

    def classify(self, *_: object) -> Classification:
        return self.result
