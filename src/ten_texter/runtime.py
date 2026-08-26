from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Callable

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session, sessionmaker

from ten_texter.beeper import BeeperDesktopAdapter
from ten_texter.control import ProductionOwnerCommandHandler
from ten_texter.correlation import (
    Classifier,
    CorrelationOrchestrator,
    SemanticCorrelator,
)
from ten_texter.domain import DomainError, TaskService, utc_now
from ten_texter.enums import (
    MessageKind,
    OutboxStatus,
    PolicyOutcome,
    ProcessingFailureType,
    ProcessingStatus,
    TaskStatus,
    Transport,
    TriggerStatus,
)
from ten_texter.health import HealthMonitor
from ten_texter.inbound import RevisionProcessor
from ten_texter.model_clients import (
    EntityResolverAssistant,
    HTTPModelBackend,
    MessageClassifier,
    MessageGenerator,
    ModelBackend,
    ModelOutputError,
    ModelUnavailable,
    SemanticCorrelationFallback,
    TaskParser,
    TriggerMessageGenerator,
)
from ten_texter.models import (
    Conversation,
    ConversationParticipant,
    Identity,
    MessageRevision,
    AwaitedResponse,
    OutboxMessage,
    Person,
    TaskDefinition,
    TaskInstance,
    TaskParticipant,
    TaskTrigger,
    TelegramUpdate,
)
from ten_texter.outbox import OutboxService, OutboxWorker
from ten_texter.policy import (
    ContactRuleResolver,
    DatabaseContextProvider,
    PolicyRevalidator,
)
from ten_texter.scheduler import RecurrenceScheduler
from ten_texter.telegram import TelegramBotAdapter, TelegramControlGateway
from ten_texter.triggers import TriggerWorker
from ten_texter.validator import (
    DatabaseValidatorContextProvider,
    IndependentMessageValidator,
    OutboxValidatorGate,
    ValidatedGenerationPipeline,
    GenerationOutcome,
    ValidatorContext,
)
from ten_texter.workflows import CoordinationWorkflow, ParticipantSendPlan


LOGGER = logging.getLogger(__name__)


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def pump_outbox_once(
    sessions: sessionmaker[Session],
    worker: OutboxWorker,
    *,
    now: datetime | None = None,
) -> None:
    timestamp = now or utc_now()
    with sessions() as session:
        expired = list(
            session.scalars(
                select(OutboxMessage.id)
                .where(
                    OutboxMessage.status == OutboxStatus.SENDING,
                    OutboxMessage.lease_expires_at <= timestamp,
                )
                .order_by(OutboxMessage.id)
            )
        )
    for outbox_id in expired:
        worker.reclaim_expired(outbox_id, at=timestamp)
    with sessions() as session:
        reconciling = list(
            session.scalars(
                select(OutboxMessage.id)
                .where(OutboxMessage.status == OutboxStatus.RECONCILING)
                .order_by(OutboxMessage.id)
            )
        )
    for outbox_id in reconciling:
        worker.reconcile(outbox_id)
    with sessions() as session:
        pending = list(
            session.scalars(
                select(OutboxMessage.id)
                .where(OutboxMessage.status == OutboxStatus.PENDING)
                .order_by(OutboxMessage.id)
            )
        )
    for outbox_id in pending:
        worker.process(outbox_id)


class DeterministicSpawnRouter:
    def __init__(self, rules: ContactRuleResolver | None = None):
        self.rules = rules or ContactRuleResolver()

    def conversation_for(
        self,
        session: Session,
        *,
        task_definition_id: int,
        person_id: int,
    ) -> int | None:
        definition = session.get(TaskDefinition, task_definition_id)
        if definition is None:
            return None
        if self.rules.resolve(
            session,
            person_id=person_id,
            task=None,
            topic_key=definition.default_topic_key,
            task_definition_id=definition.id,
        ) is not PolicyOutcome.AUTO:
            return None
        conversations = list(
            session.scalars(
                select(Conversation.id)
                .join(
                    ConversationParticipant,
                    ConversationParticipant.conversation_id == Conversation.id,
                )
                .join(Identity, Identity.id == ConversationParticipant.identity_id)
                .where(
                    Identity.person_id == person_id,
                    Conversation.archived_at.is_(None),
                )
                .distinct()
            )
        )
        return conversations[0] if len(conversations) == 1 else None


class TriggerGenerationValidator:
    def __init__(
        self,
        sessions: sessionmaker[Session],
        validator: IndependentMessageValidator,
    ):
        self.sessions = sessions
        self.validator = validator

    def validate(self, *, text: str, trigger: TaskTrigger, participant: object) -> bool:
        with self.sessions() as session:
            task = session.get(TaskInstance, trigger.task_instance_id)
            if task is None:
                return False
            claims = (
                f"topic: {task.topic_key}",
                f"scheduled_at: {task.scheduled_at.isoformat()}",
                f"duration_minutes: {task.duration_minutes}",
                *((f"location: {task.location}",) if task.location else ()),
            )
        result = self.validator.review(
            text=text,
            context=ValidatorContext(
                message_kind=MessageKind.REMINDER,
                allowed_claims=claims,
                constraints=(
                    "Use only the supplied task facts.",
                    "Do not make commitments on the owner's behalf.",
                    "Do not disclose facts from another conversation.",
                ),
            ),
        )
        return result.category.value == "VALID"


@dataclass(frozen=True, slots=True)
class RuntimeTick:
    errors: dict[str, str] = field(default_factory=dict)


class AgentRuntime:
    def __init__(
        self,
        sessions: sessionmaker[Session],
        *,
        telegram: TelegramBotAdapter,
        control: TelegramControlGateway,
        beeper: BeeperDesktopAdapter,
        revisions: RevisionProcessor,
        semantic: SemanticCorrelator,
        classifier: Classifier,
        recurrence: RecurrenceScheduler,
        triggers: TriggerWorker,
        outbox: OutboxWorker,
        health: HealthMonitor,
        owner_chat_id: int,
        recurring_generation: ValidatedGenerationPipeline,
    ):
        self.sessions = sessions
        self.telegram = telegram
        self.control = control
        self.beeper = beeper
        self.revisions = revisions
        self.semantic = semantic
        self.classifier = classifier
        self.recurrence = recurrence
        self.triggers = triggers
        self.outbox = outbox
        self.health = health
        self.owner_chat_id = owner_chat_id
        self.recurring_generation = recurring_generation
        self._recovery_pass = True

    def run_once(self, *, now: datetime | None = None) -> RuntimeTick:
        timestamp = _aware(now or utc_now())
        errors: dict[str, str] = {}
        components: tuple[tuple[str, Callable[[], None]], ...] = (
            ("telegram", self._poll_telegram),
            ("beeper", self._poll_beeper),
            ("inbound-worker", self._process_revisions),
            ("task-lifecycle", lambda: self._sweep_tasks(timestamp)),
            ("scheduler", lambda: self._poll_recurrence(timestamp)),
            ("recurring-coordination", self._initialize_recurring_tasks),
            ("trigger-worker", lambda: self._poll_triggers(timestamp)),
            ("outbox-worker", lambda: pump_outbox_once(self.sessions, self.outbox, now=timestamp)),
        )
        for name, operation in components:
            try:
                operation()
            except Exception as exc:
                errors[name] = f"{type(exc).__name__}: {exc}"
                LOGGER.exception("runtime component failed: %s", name)
                self._record_health(name, healthy=False, details=type(exc).__name__)
            else:
                self._record_health(name, healthy=True)
        self._recovery_pass = False
        return RuntimeTick(errors)

    def run_forever(self, *, interval_seconds: float = 2) -> None:
        while True:
            self.run_once()
            time.sleep(interval_seconds)

    def _record_health(self, dependency: str, *, healthy: bool, details: str = "") -> None:
        try:
            with self.sessions.begin() as session:
                self.health.record(session, dependency, healthy=healthy, details=details)
        except Exception:
            LOGGER.exception("could not persist health transition for %s", dependency)

    def _poll_telegram(self) -> None:
        with self.sessions() as session:
            last_update = session.scalar(
                select(TelegramUpdate.telegram_update_id)
                .order_by(TelegramUpdate.telegram_update_id.desc())
                .limit(1)
            )
        for raw in self.telegram.poll(
            offset=(last_update + 1) if last_update is not None else None,
            timeout=0,
        ):
            self.control.receive(raw)
        with self.sessions() as session:
            pending = list(
                session.scalars(
                    select(TelegramUpdate.id)
                    .where(TelegramUpdate.status == "PENDING")
                    .order_by(TelegramUpdate.id)
                )
            )
        for update_id in pending:
            self.control.process(update_id)

    def _poll_beeper(self) -> None:
        self.beeper.poll_inbound()

    def _process_revisions(self) -> None:
        timestamp = utc_now()
        with self.sessions() as session:
            revision_ids = list(
                session.scalars(
                    select(MessageRevision.id)
                    .where(
                        or_(
                            MessageRevision.processing_status == ProcessingStatus.PENDING,
                            (
                                (MessageRevision.processing_status == ProcessingStatus.PROCESSING)
                                & (MessageRevision.lease_expires_at <= timestamp)
                            ),
                        )
                    )
                    .order_by(MessageRevision.id)
                )
            )
        for revision_id in revision_ids:
            claim = self.revisions.claim(revision_id, now=timestamp)
            if claim is None:
                continue
            try:
                with self.sessions() as session:
                    plan = CorrelationOrchestrator(
                        session,
                        semantic=self.semantic,
                        classifier=self.classifier,
                        owner_chat_id=self.owner_chat_id,
                    ).prepare(revision_id)
            except ModelUnavailable as exc:
                self.revisions.fail(
                    claim,
                    ProcessingFailureType.MODEL_ERROR,
                    str(exc),
                    retryable=True,
                )
                raise
            except ModelOutputError as exc:
                self.revisions.fail(
                    claim,
                    ProcessingFailureType.INVALID_DATA,
                    str(exc),
                )
                continue
            except DomainError as exc:
                self.revisions.fail(
                    claim,
                    ProcessingFailureType.CORRELATION_ERROR,
                    str(exc),
                )
                continue
            except Exception as exc:
                self.revisions.fail(
                    claim,
                    ProcessingFailureType.INTERNAL_ERROR,
                    str(exc),
                )
                continue

            def apply(session: Session, _revision: MessageRevision) -> None:
                CorrelationOrchestrator(
                    session,
                    semantic=self.semantic,
                    classifier=self.classifier,
                    owner_chat_id=self.owner_chat_id,
                ).apply_prepared(revision_id, plan)

            self.revisions.commit(claim, apply)

    def _sweep_tasks(self, now: datetime) -> None:
        with self.sessions.begin() as session:
            tasks = list(
                session.scalars(
                    select(TaskInstance)
                    .where(TaskInstance.status == TaskStatus.ACTIVE)
                    .order_by(TaskInstance.id)
                )
            )
            for task in tasks:
                due = _aware(task.scheduled_at) + timedelta(
                    minutes=task.coordination_close_offset_minutes
                )
                if due <= now:
                    TaskService(session).terminalize(
                        task.id,
                        TaskStatus.LAPSED if self._recovery_pass else TaskStatus.COMPLETED,
                    )

    def _poll_recurrence(self, now: datetime) -> None:
        with self.sessions() as session:
            definition_ids = list(
                session.scalars(
                    select(TaskDefinition.id)
                    .where(
                        TaskDefinition.archived_at.is_(None),
                        TaskDefinition.next_occurrence_at.is_not(None),
                    )
                    .order_by(TaskDefinition.id)
                )
            )
        for definition_id in definition_ids:
            with self.sessions.begin() as session:
                self.recurrence.poll_definition(session, definition_id, now=now)

    def _poll_triggers(self, now: datetime) -> None:
        with self.sessions() as session:
            trigger_rows = list(
                session.execute(
                    select(TaskTrigger.id, TaskTrigger.next_run_at)
                    .where(
                        TaskTrigger.status == TriggerStatus.ACTIVE,
                        TaskTrigger.next_run_at.is_not(None),
                        TaskTrigger.next_run_at <= now,
                    )
                    .order_by(TaskTrigger.id)
                )
            )
        for trigger_id, next_run_at in trigger_rows:
            fire_key = f"scheduled:{_aware(next_run_at).isoformat()}"
            claim = self.triggers.claim(trigger_id, fire_key, now=now)
            if claim is not None:
                self.triggers.run_claim(claim)

    def _initialize_recurring_tasks(self) -> None:
        with self.sessions() as session:
            initialized = (
                select(func.count(AwaitedResponse.id))
                .join(
                    TaskParticipant,
                    TaskParticipant.id == AwaitedResponse.task_participant_id,
                )
                .where(TaskParticipant.task_instance_id == TaskInstance.id)
                .correlate(TaskInstance)
                .scalar_subquery()
            )
            task_ids = list(
                session.scalars(
                    select(TaskInstance.id)
                    .where(
                        TaskInstance.task_definition_id.is_not(None),
                        TaskInstance.status == TaskStatus.ACTIVE,
                        initialized == 0,
                    )
                    .order_by(TaskInstance.id)
                )
            )
        for task_id in task_ids:
            self._initialize_recurring_task(task_id)

    def _initialize_recurring_task(self, task_id: int) -> None:
        with self.sessions() as session:
            task = session.get(TaskInstance, task_id)
            if task is None or task.status is not TaskStatus.ACTIVE:
                return
            rows = list(
                session.execute(
                    select(TaskParticipant, Person)
                    .join(Person, Person.id == TaskParticipant.person_id)
                    .where(TaskParticipant.task_instance_id == task.id)
                    .order_by(TaskParticipant.id)
                )
            )
            task_values = (
                task.scheduled_at,
                task.duration_minutes,
                task.location,
                task.topic_key,
            )
            participants = [
                (participant.person_id, participant.conversation_id, person.display_name)
                for participant, person in rows
            ]
        scheduled_at, duration_minutes, location, topic_key = task_values
        claims = (
            f"topic: {topic_key}",
            f"scheduled_at: {scheduled_at.isoformat()}",
            f"duration_minutes: {duration_minutes}",
            *((f"location: {location}",) if location else ()),
        )
        sends: list[ParticipantSendPlan] = []
        for person_id, conversation_id, display_name in participants:
            result = self.recurring_generation.run(
                goal=f"Ask {display_name} whether they are available for the recurring activity.",
                facts=[{"claim": claim} for claim in claims],
                constraints=[
                    "Ask only about this coordination task.",
                    "Do not disclose facts from another conversation.",
                    "Do not make commitments on the owner's behalf.",
                ],
                context=ValidatorContext(
                    message_kind=MessageKind.INITIAL,
                    allowed_claims=claims + (f"participant: {display_name}",),
                    constraints=("No cross-conversation participant facts are authorized.",),
                ),
            )
            if result.outcome is not GenerationOutcome.READY or result.text is None:
                with self.sessions.begin() as session:
                    OutboxService(session).create_owner(
                        telegram_chat_id=self.owner_chat_id,
                        task_instance_id=task_id,
                        final_text=(
                            "Recurring coordination could not generate a safe initial message; "
                            "it will remain pending for recovery."
                        ),
                        message_kind=MessageKind.NOTIFICATION,
                        idempotency_key=f"task:{task_id}:initial-generation-blocked",
                    )
                return
            sends.append(
                ParticipantSendPlan(
                    person_id=person_id,
                    conversation_id=conversation_id,
                    final_text=result.text,
                )
            )
        with self.sessions.begin() as session:
            CoordinationWorkflow(session, owner_chat_id=self.owner_chat_id).start_existing(
                task_id,
                sends,
            )


def build_runtime(
    app: object,
    *,
    primary_backend: ModelBackend | None = None,
    validator_backend: ModelBackend | None = None,
    telegram: TelegramBotAdapter | None = None,
    beeper: BeeperDesktopAdapter | None = None,
) -> AgentRuntime:
    settings = app.settings
    sessions = app.sessions
    if not settings.real_transports_enabled:
        raise ValueError("agent runtime requires TEN_TEXTER_REAL_TRANSPORTS_ENABLED=true")
    if settings.owner_id is None or settings.owner_chat_id is None:
        raise ValueError("agent runtime requires owner ID and owner chat ID")

    primary_backend = primary_backend or HTTPModelBackend(settings.primary_model_url)
    validator_backend = validator_backend or HTTPModelBackend(settings.validator_model_url)
    independent_validator = IndependentMessageValidator(validator_backend)
    generator = MessageGenerator(primary_backend)
    generation = ValidatedGenerationPipeline(
        generator=generator,
        validator=independent_validator,
    )
    handler = ProductionOwnerCommandHandler(
        sessions,
        owner_chat_id=settings.owner_chat_id,
        resolver=EntityResolverAssistant(primary_backend),
        generation=generation,
    )
    telegram = telegram or TelegramBotAdapter(
        token=settings.telegram_bot_token,
        enabled=True,
    )
    beeper = beeper or BeeperDesktopAdapter(
        sessions,
        base_url=settings.beeper_base_url,
        access_token=settings.beeper_token,
        enabled=True,
    )
    control = TelegramControlGateway(
        sessions,
        owner_id=settings.owner_id,
        parser=TaskParser(primary_backend),
        handler=handler,
    )
    facts = DatabaseContextProvider()
    outbox = OutboxWorker(
        sessions,
        revalidator=PolicyRevalidator(facts=facts),
        validator=OutboxValidatorGate(
            sessions,
            validator=independent_validator,
            contexts=DatabaseValidatorContextProvider(sessions, facts=facts),
        ),
        adapters={Transport.BEEPER: beeper, Transport.TELEGRAM: telegram},
        raise_validation_errors=True,
    )
    recurrence = RecurrenceScheduler(
        router=DeterministicSpawnRouter(),
        owner_chat_id=settings.owner_chat_id,
    )
    triggers = TriggerWorker(
        sessions,
        generator=TriggerMessageGenerator(generator),
        validator=TriggerGenerationValidator(sessions, independent_validator),
        owner_chat_id=settings.owner_chat_id,
    )
    return AgentRuntime(
        sessions,
        telegram=telegram,
        control=control,
        beeper=beeper,
        revisions=RevisionProcessor(sessions),
        semantic=SemanticCorrelationFallback(primary_backend),
        classifier=MessageClassifier(primary_backend),
        recurrence=recurrence,
        triggers=triggers,
        outbox=outbox,
        health=HealthMonitor(owner_chat_id=settings.owner_chat_id),
        owner_chat_id=settings.owner_chat_id,
        recurring_generation=generation,
    )
