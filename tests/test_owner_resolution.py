from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

from ten_texter.control import ProductionOwnerCommandHandler
from ten_texter.enums import TelegramUpdateStatus
from ten_texter.model_clients import EntityResolverAssistant, TaskParseReview, TaskPlan
from ten_texter.models import DecisionRequest, OutboxMessage, TaskInstance, TelegramUpdate


def candidate(
    candidate_id: int,
    *,
    person_id: int | None = None,
    person_name: str,
    username: str | None = None,
    conversation_title: str | None = None,
    network: str = "Discord",
) -> dict[str, object]:
    return {
        "id": candidate_id,
        "person_id": person_id or candidate_id,
        "conversation_id": candidate_id + 10_000,
        "display_name": person_name,
        "person_name": person_name,
        "identity_username": username,
        "identity_display_name": person_name,
        "beeper_user_id": f"@discord_{candidate_id}:beeper",
        "conversation_title": conversation_title or f"Chat {candidate_id}",
        "beeper_conversation_id": f"!chat_{candidate_id}:beeper",
        "network": network,
    }


class RecordingResolver:
    def __init__(self, selected_id: int | None = None):
        self.selected_id = selected_id
        self.calls: list[tuple[str, list[dict[str, object]]]] = []

    def resolve(self, reference: str, candidates: list[dict[str, object]]) -> int | None:
        self.calls.append((reference, candidates))
        return self.selected_id


class RecordingBackend:
    def __init__(self, selected_id: int):
        self.selected_id = selected_id
        self.calls: list[tuple[str, dict[str, object]]] = []

    def infer(self, *, operation: str, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append((operation, payload))
        return {
            "candidate_id": self.selected_id,
            "confidence": 0.9,
            "rationale": "bounded candidate match",
        }


def handler(resolver: object, sessions: object | None = None) -> ProductionOwnerCommandHandler:
    return ProductionOwnerCommandHandler(
        sessions,  # type: ignore[arg-type]
        owner_chat_id=99,
        resolver=resolver,  # type: ignore[arg-type]
        generation=object(),  # type: ignore[arg-type]
    )


def test_exact_unique_route_bypasses_semantic_resolver() -> None:
    resolver = RecordingResolver()
    route = handler(resolver)._resolve_route(
        "Alex",
        [candidate(1, person_name="Alex"), candidate(2, person_name="Morgan")],
    )

    assert route is not None and route.person_id == 1
    assert resolver.calls == []


def test_huge_candidate_universe_is_lexically_narrowed_and_payload_is_compact() -> None:
    candidates = [candidate(index, person_name=f"Contact {index}") for index in range(1, 1001)]
    candidates.append(
        candidate(
            1001,
            person_name="Needle Person",
            username="@needle_player",
            conversation_title="Needle Tennis",
        )
    )
    backend = RecordingBackend(selected_id=1001)
    resolver = EntityResolverAssistant(backend)

    route = handler(resolver)._resolve_route("needle player tennis", candidates)

    assert route is not None and route.person_id == 1001
    assert len(backend.calls) == 1
    assert backend.calls[0][0] == "entity_resolution"
    semantic_candidates = backend.calls[0][1]["candidates"]
    assert isinstance(semantic_candidates, list)
    assert len(semantic_candidates) == 1
    assert set(semantic_candidates[0]) == {
        "id",
        "person_name",
        "identity_username",
        "identity_display_name",
        "conversation_title",
        "network",
    }


def test_semantic_resolver_candidate_input_never_exceeds_hard_bound() -> None:
    candidates = [candidate(index, person_name=f"Alex {index}") for index in range(1, 21)]
    resolver = RecordingResolver(selected_id=1)

    assert handler(resolver)._resolve_route("alex", candidates) is not None
    assert len(resolver.calls[0][1]) == 20


def test_zero_plausible_candidates_fails_closed_without_semantic_call() -> None:
    resolver = RecordingResolver(selected_id=1)

    route = handler(resolver)._resolve_route(
        "Zelda",
        [candidate(1, person_name="Alex"), candidate(2, person_name="Morgan")],
    )

    assert route is None
    assert resolver.calls == []


def test_too_many_plausible_candidates_fails_closed_without_truncation() -> None:
    resolver = RecordingResolver(selected_id=1)
    candidates = [candidate(index, person_name=f"Alex {index}") for index in range(1, 22)]

    assert handler(resolver)._resolve_route("alex", candidates) is None
    assert resolver.calls == []


def test_semantic_non_candidate_selection_is_rejected() -> None:
    resolver = RecordingResolver(selected_id=999)
    candidates = [
        candidate(1, person_name="Alex One"),
        candidate(2, person_name="Alex Two"),
    ]

    assert handler(resolver)._resolve_route("alex", candidates) is None


def test_small_ambiguous_pool_can_use_semantic_resolver() -> None:
    resolver = RecordingResolver(selected_id=2)
    candidates = [
        candidate(1, person_name="Alex One"),
        candidate(2, person_name="Alex Two"),
    ]

    route = handler(resolver)._resolve_route("alex", candidates)

    assert route is not None and route.person_id == 2
    assert [item["id"] for item in resolver.calls[0][1]] == [1, 2]


def test_prepare_command_still_rejects_duplicate_person_routes(db_session, monkeypatch) -> None:
    candidates = [
        candidate(1, person_id=7, person_name="Alex", conversation_title="Discord Tennis"),
        candidate(
            2,
            person_id=7,
            person_name="Alex",
            conversation_title="Instagram Direct",
            network="Instagram",
        ),
    ]
    monkeypatch.setattr(
        ProductionOwnerCommandHandler,
        "_route_candidates",
        staticmethod(lambda _session: candidates),
    )
    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    resolver = RecordingResolver()
    plan = TaskPlan(
        scheduled_at=datetime(2026, 9, 1, 17, tzinfo=UTC),
        duration_minutes=60,
        topic_key="tennis",
        participant_references=["Discord Tennis", "Instagram Direct"],
    )

    prepared = handler(resolver, factory).prepare_command(plan, object())  # type: ignore[arg-type]

    assert prepared.sends == ()
    assert prepared.review_reason == "Participant was resolved more than once: Instagram Direct"
    assert resolver.calls == []


def test_parse_clarification_creates_owner_review_without_task(db_session) -> None:
    factory = sessionmaker(bind=db_session.bind, expire_on_commit=False, autoflush=False)
    command_handler = handler(RecordingResolver(), factory)
    update = TelegramUpdate(
        telegram_update_id=800,
        sender_user_id=7,
        chat_id=99,
        payload_json={},
        status=TelegramUpdateStatus.PENDING,
    )
    db_session.add(update)
    db_session.flush()
    prepared = command_handler.prepare_command(
        TaskParseReview(review_reason="A precise start time and duration are required."),
        update,
    )

    command_handler.apply_command(db_session, prepared, update)
    db_session.flush()

    assert db_session.scalar(select(func.count(TaskInstance.id))) == 0
    decision = db_session.scalar(select(DecisionRequest))
    prompt = db_session.scalar(select(OutboxMessage))
    assert decision is not None and decision.type == "OWNER_COMMAND_REVIEW"
    assert decision.context_json["reason"] == "A precise start time and duration are required."
    assert prompt is not None
    assert prompt.idempotency_key == f"telegram-update:{update.id}:review"
    assert "precise start time and duration" not in prompt.final_text.lower()
    assert "telegram command 800 requires owner review" in prompt.final_text.lower()
