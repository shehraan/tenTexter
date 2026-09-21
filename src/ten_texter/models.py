from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Enum as SAEnum,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from ten_texter.db import Base
from ten_texter.enums import *


def enum_type(enum_cls: type[StrEnum], name: str) -> SAEnum:
    return SAEnum(enum_cls, native_enum=False, create_constraint=True, name=name)


def now() -> datetime:
    return datetime.now(UTC)


class ArchivedMixin:
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Person(ArchivedMixin, Base):
    __tablename__ = "person"
    id: Mapped[int] = mapped_column(primary_key=True)
    display_name: Mapped[str] = mapped_column(String(200))
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Identity(ArchivedMixin, Base):
    __tablename__ = "identity"
    id: Mapped[int] = mapped_column(primary_key=True)
    person_id: Mapped[int] = mapped_column(ForeignKey("person.id", ondelete="RESTRICT"), index=True)
    beeper_user_id: Mapped[str] = mapped_column(String(300), unique=True)
    network: Mapped[str] = mapped_column(String(100))
    username: Mapped[str | None] = mapped_column(String(200))
    display_name: Mapped[str | None] = mapped_column(String(200))
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Conversation(ArchivedMixin, Base):
    __tablename__ = "conversation"
    id: Mapped[int] = mapped_column(primary_key=True)
    beeper_conversation_id: Mapped[str] = mapped_column(String(400), unique=True)
    network: Mapped[str] = mapped_column(String(100))
    kind: Mapped[ConversationKind] = mapped_column(enum_type(ConversationKind, "conversation_kind"))
    title: Mapped[str | None] = mapped_column(String(300))
    counterparty_person_id: Mapped[int | None] = mapped_column(ForeignKey("person.id", ondelete="RESTRICT"))
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    __table_args__ = (
        CheckConstraint("kind = 'DIRECT' OR counterparty_person_id IS NULL", name="ck_conversation_counterparty_direct"),
    )


class ConversationParticipant(Base):
    __tablename__ = "conversation_participant"
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversation.id", ondelete="RESTRICT"), primary_key=True)
    identity_id: Mapped[int] = mapped_column(ForeignKey("identity.id", ondelete="RESTRICT"), primary_key=True)
    is_current: Mapped[bool] = mapped_column(Boolean, default=True)
    left_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    __table_args__ = (
        CheckConstraint(
            "(is_current = 1 AND left_at IS NULL) OR (is_current = 0 AND left_at IS NOT NULL)",
            name="ck_conversation_participant_current_left_at",
        ),
    )


class BeeperSyncCheckpoint(Base):
    """Durable progress for Beeper's global chat feed or one conversation feed."""

    __tablename__ = "beeper_sync_checkpoint"
    checkpoint_key: Mapped[str] = mapped_column(String(500), primary_key=True)
    scope: Mapped[str] = mapped_column(String(30))
    conversation_id: Mapped[int | None] = mapped_column(
        ForeignKey("conversation.id", ondelete="RESTRICT"), unique=True
    )
    newest_cursor: Mapped[str | None] = mapped_column(Text)
    backfill_cursor: Mapped[str | None] = mapped_column(Text)
    bootstrap_cutoff_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    bootstrap_complete: Mapped[bool] = mapped_column(Boolean, default=False)
    reconciliation_cursor: Mapped[str | None] = mapped_column(Text)
    reconciliation_cutoff_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_reconciled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    provider_activity_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    __table_args__ = (
        CheckConstraint(
            "(scope = 'CHAT_FEED' AND checkpoint_key = 'chat-feed' AND conversation_id IS NULL) OR "
            "(scope = 'CONVERSATION' AND checkpoint_key <> 'chat-feed' AND conversation_id IS NOT NULL)",
            name="ck_beeper_sync_checkpoint_scope",
        ),
        CheckConstraint(
            "(reconciliation_cursor IS NULL AND reconciliation_cutoff_at IS NULL) OR "
            "(reconciliation_cursor IS NOT NULL AND reconciliation_cutoff_at IS NOT NULL)",
            name="ck_beeper_sync_reconciliation_pair",
        ),
        Index(
            "ix_beeper_sync_checkpoint_reconcile",
            "scope",
            "bootstrap_complete",
            "last_reconciled_at",
        ),
    )


class TaskDefinition(ArchivedMixin, Base):
    __tablename__ = "task_definition"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(300))
    recurrence_rule: Mapped[str] = mapped_column(Text)
    default_time: Mapped[str] = mapped_column(String(8))
    timezone: Mapped[str] = mapped_column(String(100))
    default_duration_minutes: Mapped[int] = mapped_column(Integer)
    default_location: Mapped[str | None] = mapped_column(String(500))
    default_topic_key: Mapped[str] = mapped_column(String(300))
    next_occurrence_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    __table_args__ = (CheckConstraint("default_duration_minutes > 0", name="ck_definition_duration_positive"),)


class TaskDefinitionParticipant(Base):
    __tablename__ = "task_definition_participant"
    task_definition_id: Mapped[int] = mapped_column(ForeignKey("task_definition.id", ondelete="RESTRICT"), primary_key=True)
    person_id: Mapped[int] = mapped_column(ForeignKey("person.id", ondelete="RESTRICT"), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class TaskInstance(ArchivedMixin, Base):
    __tablename__ = "task_instance"
    id: Mapped[int] = mapped_column(primary_key=True)
    task_definition_id: Mapped[int | None] = mapped_column(ForeignKey("task_definition.id", ondelete="RESTRICT"))
    occurrence_key: Mapped[str | None] = mapped_column(String(100))
    scheduled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    duration_minutes: Mapped[int] = mapped_column(Integer)
    location: Mapped[str | None] = mapped_column(String(500))
    topic_key: Mapped[str] = mapped_column(String(300))
    coordination_close_offset_minutes: Mapped[int] = mapped_column(Integer, default=60)
    status: Mapped[TaskStatus] = mapped_column(enum_type(TaskStatus, "task_status"), default=TaskStatus.ACTIVE)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    __table_args__ = (
        CheckConstraint(
            "(task_definition_id IS NULL AND occurrence_key IS NULL) OR "
            "(task_definition_id IS NOT NULL AND occurrence_key IS NOT NULL)",
            name="ck_instance_definition_occurrence_pair",
        ),
        CheckConstraint("duration_minutes > 0", name="ck_instance_duration_positive"),
        CheckConstraint("coordination_close_offset_minutes >= 0", name="ck_instance_close_offset_nonnegative"),
        UniqueConstraint("task_definition_id", "occurrence_key", name="uq_instance_occurrence"),
    )


class TaskParticipant(Base):
    __tablename__ = "task_participant"
    id: Mapped[int] = mapped_column(primary_key=True)
    task_instance_id: Mapped[int] = mapped_column(ForeignKey("task_instance.id", ondelete="RESTRICT"))
    person_id: Mapped[int] = mapped_column(ForeignKey("person.id", ondelete="RESTRICT"))
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversation.id", ondelete="RESTRICT"))
    availability_status: Mapped[AvailabilityStatus] = mapped_column(
        enum_type(AvailabilityStatus, "availability_status"), default=AvailabilityStatus.UNKNOWN
    )
    availability_evidence: Mapped[AvailabilityEvidence | None] = mapped_column(
        enum_type(AvailabilityEvidence, "availability_evidence")
    )
    availability_source_revision_id: Mapped[int | None] = mapped_column(
        ForeignKey("message_revision.id", ondelete="RESTRICT")
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    __table_args__ = (
        CheckConstraint(
            "(availability_status = 'UNKNOWN' AND availability_evidence IS NULL AND availability_source_revision_id IS NULL) OR "
            "(availability_status <> 'UNKNOWN' AND availability_evidence IS NOT NULL AND availability_source_revision_id IS NOT NULL)",
            name="ck_participant_availability_evidence",
        ),
        UniqueConstraint("task_instance_id", "person_id", name="uq_task_participant_person"),
        UniqueConstraint("task_instance_id", "id", name="uq_task_participant_parent_id"),
    )


class Message(Base):
    __tablename__ = "message"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement="ignore_fk")
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversation.id", ondelete="RESTRICT"))
    provider_message_id: Mapped[str] = mapped_column(String(500))
    sender_identity_id: Mapped[int] = mapped_column(ForeignKey("identity.id", ondelete="RESTRICT"))
    provider_reply_to_message_id: Mapped[str | None] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    current_revision_id: Mapped[int | None] = mapped_column()
    __table_args__ = (
        UniqueConstraint("conversation_id", "provider_message_id", name="uq_message_provider_identity"),
        UniqueConstraint("id", "current_revision_id", name="uq_message_current_parent"),
        ForeignKeyConstraint(
            ["conversation_id", "sender_identity_id"],
            ["conversation_participant.conversation_id", "conversation_participant.identity_id"],
            ondelete="RESTRICT",
            name="fk_message_sender_membership",
        ),
        ForeignKeyConstraint(
            ["id", "current_revision_id"],
            ["message_revision.message_id", "message_revision.id"],
            ondelete="RESTRICT",
            name="fk_message_current_revision_parent",
        ),
    )


class MessageRevision(Base):
    __tablename__ = "message_revision"
    id: Mapped[int] = mapped_column(primary_key=True)
    message_id: Mapped[int] = mapped_column(ForeignKey("message.id", ondelete="RESTRICT"))
    provider_revision_key: Mapped[str] = mapped_column(String(500))
    provider_sort_key: Mapped[str | None] = mapped_column(String(500))
    provider_sequence: Mapped[int | None] = mapped_column(Integer)
    provider_event_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    content_hash: Mapped[str] = mapped_column(String(64))
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False)
    text: Mapped[str | None] = mapped_column(Text)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    content_support: Mapped[ContentSupport] = mapped_column(enum_type(ContentSupport, "content_support"))
    processing_status: Mapped[ProcessingStatus] = mapped_column(
        enum_type(ProcessingStatus, "processing_status"), default=ProcessingStatus.PENDING
    )
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    awaited_response_id: Mapped[int | None] = mapped_column(
        ForeignKey("awaited_response.id", ondelete="RESTRICT")
    )
    __table_args__ = (
        UniqueConstraint("message_id", "provider_revision_key", name="uq_revision_provider_key"),
        UniqueConstraint("message_id", "id", name="uq_revision_parent_id"),
        CheckConstraint(
            "(processing_status = 'PROCESSING' AND lease_expires_at IS NOT NULL) OR "
            "(processing_status <> 'PROCESSING' AND lease_expires_at IS NULL)",
            name="ck_revision_processing_lease",
        ),
        CheckConstraint("is_deleted = 0 OR text IS NULL", name="ck_revision_deleted_text"),
    )


class MessageProcessingAttempt(Base):
    __tablename__ = "message_processing_attempt"
    id: Mapped[int] = mapped_column(primary_key=True)
    message_revision_id: Mapped[int] = mapped_column(ForeignKey("message_revision.id", ondelete="RESTRICT"))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    result: Mapped[AttemptResult | None] = mapped_column(enum_type(AttemptResult, "processing_attempt_result"))
    failure_type: Mapped[ProcessingFailureType | None] = mapped_column(
        enum_type(ProcessingFailureType, "processing_failure_type")
    )
    error_details: Mapped[str | None] = mapped_column(Text)
    __table_args__ = (
        CheckConstraint(
            "(finished_at IS NULL AND result IS NULL) OR (finished_at IS NOT NULL AND result IS NOT NULL)",
            name="ck_processing_attempt_finished_result",
        ),
    )


class AwaitedResponse(Base):
    __tablename__ = "awaited_response"
    id: Mapped[int] = mapped_column(primary_key=True)
    task_participant_id: Mapped[int] = mapped_column(ForeignKey("task_participant.id", ondelete="RESTRICT"), index=True)
    expected_response_type: Mapped[str] = mapped_column(String(100))
    status: Mapped[AwaitedResponseStatus] = mapped_column(
        enum_type(AwaitedResponseStatus, "awaited_response_status"), default=AwaitedResponseStatus.OPEN
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Proposal(Base):
    __tablename__ = "proposal"
    id: Mapped[int] = mapped_column(primary_key=True)
    task_instance_id: Mapped[int] = mapped_column(ForeignKey("task_instance.id", ondelete="RESTRICT"))
    proposed_by_participant_id: Mapped[int | None] = mapped_column()
    source_message_revision_id: Mapped[int] = mapped_column(ForeignKey("message_revision.id", ondelete="RESTRICT"))
    field: Mapped[str] = mapped_column(String(100))
    operation: Mapped[str] = mapped_column(String(100))
    old_value: Mapped[Any] = mapped_column(JSON)
    proposed_value: Mapped[Any] = mapped_column(JSON)
    status: Mapped[ProposalStatus] = mapped_column(enum_type(ProposalStatus, "proposal_status"), default=ProposalStatus.PENDING)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    __table_args__ = (
        ForeignKeyConstraint(
            ["task_instance_id", "proposed_by_participant_id"],
            ["task_participant.task_instance_id", "task_participant.id"],
            ondelete="RESTRICT",
            name="fk_proposal_participant_parent",
        ),
        CheckConstraint(
            "(status = 'PENDING' AND resolved_at IS NULL) OR (status <> 'PENDING' AND resolved_at IS NOT NULL)",
            name="ck_proposal_resolution",
        ),
    )


class ContactRule(Base):
    __tablename__ = "contact_rule"
    id: Mapped[int] = mapped_column(primary_key=True)
    person_id: Mapped[int] = mapped_column(ForeignKey("person.id", ondelete="RESTRICT"), index=True)
    scope: Mapped[ContactRuleScope] = mapped_column(enum_type(ContactRuleScope, "contact_rule_scope"))
    type: Mapped[str] = mapped_column(String(100))
    value: Mapped[str] = mapped_column(Text)
    source: Mapped[ContactRuleSource] = mapped_column(enum_type(ContactRuleSource, "contact_rule_source"))
    strength: Mapped[RuleStrength] = mapped_column(enum_type(RuleStrength, "contact_rule_strength"))
    topic_key: Mapped[str | None] = mapped_column(String(300))
    task_definition_id: Mapped[int | None] = mapped_column(ForeignKey("task_definition.id", ondelete="RESTRICT"))
    task_instance_id: Mapped[int | None] = mapped_column(ForeignKey("task_instance.id", ondelete="RESTRICT"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_reason: Mapped[str | None] = mapped_column(String(50))
    overrides_contact_rule_id: Mapped[int | None] = mapped_column(ForeignKey("contact_rule.id", ondelete="RESTRICT"))
    __table_args__ = (
        CheckConstraint(
            "(scope = 'GLOBAL' AND topic_key IS NULL AND task_definition_id IS NULL AND task_instance_id IS NULL) OR "
            "(scope = 'TOPIC' AND topic_key IS NOT NULL AND task_definition_id IS NULL AND task_instance_id IS NULL) OR "
            "(scope = 'TASK_DEFINITION' AND topic_key IS NULL AND task_definition_id IS NOT NULL AND task_instance_id IS NULL) OR "
            "(scope = 'TASK_INSTANCE' AND topic_key IS NULL AND task_definition_id IS NULL AND task_instance_id IS NOT NULL)",
            name="ck_contact_rule_scope_target",
        ),
        CheckConstraint(
            "(revoked_at IS NULL AND revoked_reason IS NULL) OR (revoked_at IS NOT NULL AND revoked_reason IN ('REVOKED','SUPERSEDED'))",
            name="ck_contact_rule_revocation",
        ),
        CheckConstraint("overrides_contact_rule_id IS NULL OR overrides_contact_rule_id <> id", name="ck_contact_rule_no_self_override"),
    )


class DisclosureGrant(Base):
    __tablename__ = "disclosure_grant"
    id: Mapped[int] = mapped_column(primary_key=True)
    source_person_id: Mapped[int] = mapped_column(ForeignKey("person.id", ondelete="RESTRICT"))
    source_conversation_id: Mapped[int] = mapped_column(ForeignKey("conversation.id", ondelete="RESTRICT"))
    destination_conversation_id: Mapped[int] = mapped_column(ForeignKey("conversation.id", ondelete="RESTRICT"))
    task_instance_id: Mapped[int] = mapped_column(ForeignKey("task_instance.id", ondelete="RESTRICT"), index=True)
    status: Mapped[DisclosureGrantStatus] = mapped_column(
        enum_type(DisclosureGrantStatus, "disclosure_grant_status"), default=DisclosureGrantStatus.ACTIVE
    )
    inactive_reason: Mapped[DisclosureInactiveReason | None] = mapped_column(
        enum_type(DisclosureInactiveReason, "disclosure_inactive_reason")
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    __table_args__ = (
        CheckConstraint(
            "(status = 'ACTIVE' AND inactive_reason IS NULL) OR (status = 'INACTIVE' AND inactive_reason IS NOT NULL)",
            name="ck_disclosure_status_reason",
        ),
        CheckConstraint("source_conversation_id <> destination_conversation_id", name="ck_disclosure_cross_conversation"),
    )


class DisclosureGrantScope(Base):
    __tablename__ = "disclosure_grant_scope"
    disclosure_grant_id: Mapped[int] = mapped_column(ForeignKey("disclosure_grant.id", ondelete="RESTRICT"), primary_key=True)
    scope: Mapped[DisclosureScope] = mapped_column(enum_type(DisclosureScope, "disclosure_scope"), primary_key=True)


class TaskTrigger(Base):
    __tablename__ = "task_trigger"
    id: Mapped[int] = mapped_column(primary_key=True)
    task_instance_id: Mapped[int] = mapped_column(ForeignKey("task_instance.id", ondelete="RESTRICT"))
    condition_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    stop_condition_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    target_selector: Mapped[TargetSelector] = mapped_column(enum_type(TargetSelector, "target_selector"))
    target_task_participant_id: Mapped[int | None] = mapped_column()
    action_type: Mapped[TriggerActionType] = mapped_column(enum_type(TriggerActionType, "trigger_action_type"))
    action_payload_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    next_run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    status: Mapped[TriggerStatus] = mapped_column(enum_type(TriggerStatus, "trigger_status"), default=TriggerStatus.ACTIVE)
    inactive_reason: Mapped[TriggerInactiveReason | None] = mapped_column(enum_type(TriggerInactiveReason, "trigger_inactive_reason"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    __table_args__ = (
        CheckConstraint(
            "(target_selector = 'SPECIFIC_PARTICIPANT' AND target_task_participant_id IS NOT NULL) OR "
            "(target_selector <> 'SPECIFIC_PARTICIPANT' AND target_task_participant_id IS NULL)",
            name="ck_trigger_specific_target",
        ),
        CheckConstraint(
            "(status = 'ACTIVE' AND inactive_reason IS NULL) OR (status = 'INACTIVE' AND inactive_reason IS NOT NULL)",
            name="ck_trigger_status_reason",
        ),
        ForeignKeyConstraint(
            ["task_instance_id", "target_task_participant_id"],
            ["task_participant.task_instance_id", "task_participant.id"],
            ondelete="RESTRICT",
            name="fk_trigger_participant_parent",
        ),
    )


class TriggerExecution(Base):
    __tablename__ = "trigger_execution"
    id: Mapped[int] = mapped_column(primary_key=True)
    task_trigger_id: Mapped[int] = mapped_column(ForeignKey("task_trigger.id", ondelete="RESTRICT"))
    fire_key: Mapped[str] = mapped_column(String(300))
    scheduled_for: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[TriggerExecutionStatus] = mapped_column(
        enum_type(TriggerExecutionStatus, "trigger_execution_status"), default=TriggerExecutionStatus.PENDING
    )
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    __table_args__ = (
        UniqueConstraint("task_trigger_id", "fire_key", name="uq_trigger_execution_fire"),
        CheckConstraint(
            "(status = 'PROCESSING' AND lease_expires_at IS NOT NULL) OR "
            "(status <> 'PROCESSING' AND lease_expires_at IS NULL)",
            name="ck_trigger_execution_lease",
        ),
    )


class OutboxMessage(Base):
    __tablename__ = "outbox_message"
    id: Mapped[int] = mapped_column(primary_key=True)
    task_instance_id: Mapped[int | None] = mapped_column(ForeignKey("task_instance.id", ondelete="RESTRICT"))
    transport: Mapped[Transport] = mapped_column(enum_type(Transport, "transport"))
    destination_kind: Mapped[DestinationKind] = mapped_column(enum_type(DestinationKind, "destination_kind"))
    final_text: Mapped[str] = mapped_column(Text)
    message_kind: Mapped[MessageKind] = mapped_column(enum_type(MessageKind, "message_kind"))
    status: Mapped[OutboxStatus] = mapped_column(enum_type(OutboxStatus, "outbox_status"), default=OutboxStatus.PENDING)
    idempotency_key: Mapped[str] = mapped_column(String(500))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    parent_terminal_policy: Mapped[ParentTerminalPolicy] = mapped_column(enum_type(ParentTerminalPolicy, "outbox_parent_terminal_policy"))
    cancel_reason: Mapped[OutboxCancelReason | None] = mapped_column(enum_type(OutboxCancelReason, "outbox_cancel_reason"))
    trigger_execution_id: Mapped[int | None] = mapped_column(ForeignKey("trigger_execution.id", ondelete="RESTRICT"))
    corrects_outbox_message_id: Mapped[int | None] = mapped_column(ForeignKey("outbox_message.id", ondelete="RESTRICT"), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    __table_args__ = (
        UniqueConstraint("transport", "idempotency_key", name="uq_outbox_transport_idempotency"),
        CheckConstraint(
            "(transport = 'BEEPER' AND destination_kind = 'PARTICIPANT') OR "
            "(transport = 'TELEGRAM' AND destination_kind = 'OWNER')",
            name="ck_outbox_transport_destination",
        ),
        CheckConstraint(
            "(status = 'SENDING' AND lease_expires_at IS NOT NULL) OR "
            "(status <> 'SENDING' AND lease_expires_at IS NULL)",
            name="ck_outbox_sending_lease",
        ),
        CheckConstraint(
            "(status = 'CANCELLED' AND cancel_reason IS NOT NULL) OR "
            "(status <> 'CANCELLED' AND cancel_reason IS NULL)",
            name="ck_outbox_cancel_reason",
        ),
        CheckConstraint(
            "(message_kind = 'CORRECTION' AND corrects_outbox_message_id IS NOT NULL) OR "
            "(message_kind <> 'CORRECTION' AND corrects_outbox_message_id IS NULL)",
            name="ck_outbox_correction_link",
        ),
        CheckConstraint("corrects_outbox_message_id IS NULL OR corrects_outbox_message_id <> id", name="ck_outbox_no_self_correction"),
    )


class DecisionRequest(Base):
    __tablename__ = "decision_request"
    id: Mapped[int] = mapped_column(primary_key=True)
    task_instance_id: Mapped[int | None] = mapped_column(ForeignKey("task_instance.id", ondelete="RESTRICT"))
    type: Mapped[str] = mapped_column(String(100))
    status: Mapped[DecisionStatus] = mapped_column(enum_type(DecisionStatus, "decision_status"), default=DecisionStatus.PENDING)
    close_reason: Mapped[DecisionCloseReason | None] = mapped_column(enum_type(DecisionCloseReason, "decision_close_reason"))
    context_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    resolution_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    parent_terminal_policy: Mapped[ParentTerminalPolicy] = mapped_column(enum_type(ParentTerminalPolicy, "decision_parent_terminal_policy"))
    proposal_id: Mapped[int | None] = mapped_column(ForeignKey("proposal.id", ondelete="RESTRICT"))
    contact_rule_id: Mapped[int | None] = mapped_column(ForeignKey("contact_rule.id", ondelete="RESTRICT"))
    outbox_message_id: Mapped[int | None] = mapped_column(ForeignKey("outbox_message.id", ondelete="RESTRICT"))
    message_revision_id: Mapped[int | None] = mapped_column(ForeignKey("message_revision.id", ondelete="RESTRICT"))
    telegram_update_id: Mapped[int | None] = mapped_column(ForeignKey("telegram_update.id", ondelete="RESTRICT"))
    trigger_execution_id: Mapped[int | None] = mapped_column(ForeignKey("trigger_execution.id", ondelete="RESTRICT"))
    __table_args__ = (
        CheckConstraint(
            "(proposal_id IS NOT NULL) + (contact_rule_id IS NOT NULL) + (outbox_message_id IS NOT NULL) + "
            "(message_revision_id IS NOT NULL) + (telegram_update_id IS NOT NULL) + (trigger_execution_id IS NOT NULL) = 1",
            name="ck_decision_exactly_one_subject",
        ),
        CheckConstraint(
            "(status = 'PENDING' AND close_reason IS NULL AND resolved_at IS NULL AND resolution_json IS NULL) OR "
            "(status = 'CLOSED' AND close_reason IS NOT NULL AND resolved_at IS NOT NULL AND "
            "((close_reason = 'ANSWERED' AND resolution_json IS NOT NULL) OR "
            "(close_reason <> 'ANSWERED' AND resolution_json IS NULL)))",
            name="ck_decision_resolution",
        ),
    )


class TaskEvent(Base):
    __tablename__ = "task_event"
    id: Mapped[int] = mapped_column(primary_key=True)
    task_instance_id: Mapped[int] = mapped_column(ForeignKey("task_instance.id", ondelete="RESTRICT"), index=True)
    task_participant_id: Mapped[int | None] = mapped_column(ForeignKey("task_participant.id", ondelete="RESTRICT"))
    source_message_revision_id: Mapped[int | None] = mapped_column(ForeignKey("message_revision.id", ondelete="RESTRICT"))
    event_type: Mapped[str] = mapped_column(String(100))
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class OutboxMessageParticipant(Base):
    __tablename__ = "outbox_message_participant"
    outbox_message_id: Mapped[int] = mapped_column(ForeignKey("outbox_message.id", ondelete="RESTRICT"), primary_key=True)
    task_participant_id: Mapped[int] = mapped_column(ForeignKey("task_participant.id", ondelete="RESTRICT"), primary_key=True)


class BeeperOutboxDestination(Base):
    __tablename__ = "beeper_outbox_destination"
    outbox_message_id: Mapped[int] = mapped_column(ForeignKey("outbox_message.id", ondelete="RESTRICT"), primary_key=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversation.id", ondelete="RESTRICT"))


class TelegramOutboxDestination(Base):
    __tablename__ = "telegram_outbox_destination"
    outbox_message_id: Mapped[int] = mapped_column(ForeignKey("outbox_message.id", ondelete="RESTRICT"), primary_key=True)
    telegram_chat_id: Mapped[int] = mapped_column()


class OutboxDeliveryAttempt(Base):
    __tablename__ = "outbox_delivery_attempt"
    id: Mapped[int] = mapped_column(primary_key=True)
    outbox_message_id: Mapped[int] = mapped_column(ForeignKey("outbox_message.id", ondelete="RESTRICT"), index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    result: Mapped[AttemptResult | None] = mapped_column(enum_type(AttemptResult, "delivery_attempt_result"))
    failure_type: Mapped[str | None] = mapped_column(String(100))
    error_details: Mapped[str | None] = mapped_column(Text)
    provider_message_id: Mapped[str | None] = mapped_column(String(500))
    __table_args__ = (
        CheckConstraint(
            "(finished_at IS NULL AND result IS NULL) OR (finished_at IS NOT NULL AND result IS NOT NULL)",
            name="ck_delivery_attempt_finished_result",
        ),
        Index(
            "uq_delivery_attempt_unfinished",
            "outbox_message_id",
            unique=True,
            sqlite_where=text("finished_at IS NULL"),
        ),
    )


class BeeperDeliveryAttemptDetail(Base):
    __tablename__ = "beeper_delivery_attempt_detail"
    outbox_delivery_attempt_id: Mapped[int] = mapped_column(
        ForeignKey("outbox_delivery_attempt.id", ondelete="RESTRICT"), primary_key=True
    )
    pending_provider_id: Mapped[str | None] = mapped_column(String(500))


class AwaitedResponsePrompt(Base):
    __tablename__ = "awaited_response_prompt"
    awaited_response_id: Mapped[int] = mapped_column(ForeignKey("awaited_response.id", ondelete="RESTRICT"), primary_key=True)
    outbox_message_id: Mapped[int] = mapped_column(ForeignKey("outbox_message.id", ondelete="RESTRICT"), primary_key=True)


class DecisionRequestPrompt(Base):
    __tablename__ = "decision_request_prompt"
    decision_request_id: Mapped[int] = mapped_column(ForeignKey("decision_request.id", ondelete="RESTRICT"), primary_key=True)
    outbox_message_id: Mapped[int] = mapped_column(
        ForeignKey("outbox_message.id", ondelete="RESTRICT"), primary_key=True, unique=True
    )


class DecisionRequestAwaitedResponseCandidate(Base):
    __tablename__ = "decision_request_awaited_response_candidate"
    decision_request_id: Mapped[int] = mapped_column(ForeignKey("decision_request.id", ondelete="RESTRICT"), primary_key=True)
    awaited_response_id: Mapped[int] = mapped_column(ForeignKey("awaited_response.id", ondelete="RESTRICT"), primary_key=True)


class TelegramUpdate(Base):
    __tablename__ = "telegram_update"
    id: Mapped[int] = mapped_column(primary_key=True)
    telegram_update_id: Mapped[int] = mapped_column(unique=True)
    sender_user_id: Mapped[int | None] = mapped_column()
    chat_id: Mapped[int | None] = mapped_column()
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    status: Mapped[TelegramUpdateStatus] = mapped_column(
        enum_type(TelegramUpdateStatus, "telegram_update_status"), default=TelegramUpdateStatus.PENDING
    )
    error_details: Mapped[str | None] = mapped_column(Text)
