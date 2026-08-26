from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class TaskStatus(StrEnum):
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    LAPSED = "LAPSED"


class AvailabilityStatus(StrEnum):
    UNKNOWN = "UNKNOWN"
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    UNCERTAIN = "UNCERTAIN"


class AvailabilityEvidence(StrEnum):
    FIRST_PARTY = "FIRST_PARTY"
    THIRD_PARTY = "THIRD_PARTY"


class ConversationKind(StrEnum):
    DIRECT = "DIRECT"
    GROUP = "GROUP"


class ContentSupport(StrEnum):
    SUPPORTED = "SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"


class ProcessingStatus(StrEnum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    PROCESSED = "PROCESSED"
    FAILED = "FAILED"


class AttemptResult(StrEnum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    ABANDONED = "ABANDONED"


class ProcessingFailureType(StrEnum):
    TIMEOUT = "TIMEOUT"
    MODEL_ERROR = "MODEL_ERROR"
    CORRELATION_ERROR = "CORRELATION_ERROR"
    INVALID_DATA = "INVALID_DATA"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class AwaitedResponseStatus(StrEnum):
    OPEN = "OPEN"
    SATISFIED = "SATISFIED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"
    AMBIGUOUS = "AMBIGUOUS"


class ProposalStatus(StrEnum):
    PENDING = "PENDING"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    WITHDRAWN = "WITHDRAWN"
    EXPIRED = "EXPIRED"
    SUPERSEDED = "SUPERSEDED"
    PARENT_TERMINAL = "PARENT_TERMINAL"


class DecisionStatus(StrEnum):
    PENDING = "PENDING"
    CLOSED = "CLOSED"


class DecisionCloseReason(StrEnum):
    ANSWERED = "ANSWERED"
    DISMISSED = "DISMISSED"
    EXPIRED = "EXPIRED"
    PARENT_TERMINAL = "PARENT_TERMINAL"
    SUBJECT_RESOLVED = "SUBJECT_RESOLVED"


class ParentTerminalPolicy(StrEnum):
    TERMINATE = "TERMINATE"
    SURVIVE = "SURVIVE"


class ContactRuleScope(StrEnum):
    GLOBAL = "GLOBAL"
    TOPIC = "TOPIC"
    TASK_DEFINITION = "TASK_DEFINITION"
    TASK_INSTANCE = "TASK_INSTANCE"


class ContactRuleSource(StrEnum):
    USER_CONFIGURED = "USER_CONFIGURED"
    PARTICIPANT_REQUESTED = "PARTICIPANT_REQUESTED"
    SYSTEM_INFERRED = "SYSTEM_INFERRED"


class RuleStrength(StrEnum):
    WEAK = "WEAK"
    MEDIUM = "MEDIUM"
    STRONG = "STRONG"


class DisclosureGrantStatus(StrEnum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"


class DisclosureInactiveReason(StrEnum):
    EXPIRED = "EXPIRED"
    REVOKED = "REVOKED"
    TASK_TERMINAL = "TASK_TERMINAL"


class DisclosureScope(StrEnum):
    AVAILABILITY = "AVAILABILITY"
    SCHEDULING = "SCHEDULING"
    LOCATION = "LOCATION"


class TargetSelector(StrEnum):
    ALL_PARTICIPANTS = "ALL_PARTICIPANTS"
    NO_RESPONSE = "NO_RESPONSE"
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    UNCERTAIN = "UNCERTAIN"
    SPECIFIC_PARTICIPANT = "SPECIFIC_PARTICIPANT"


class TriggerActionType(StrEnum):
    SEND_MESSAGE = "SEND_MESSAGE"


class TriggerStatus(StrEnum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"


class TriggerInactiveReason(StrEnum):
    FIRED = "FIRED"
    STOP_CONDITION_MET = "STOP_CONDITION_MET"
    DISABLED = "DISABLED"
    TASK_TERMINAL = "TASK_TERMINAL"


class TriggerExecutionStatus(StrEnum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class Transport(StrEnum):
    BEEPER = "BEEPER"
    TELEGRAM = "TELEGRAM"


class DestinationKind(StrEnum):
    PARTICIPANT = "PARTICIPANT"
    OWNER = "OWNER"


class MessageKind(StrEnum):
    INITIAL = "INITIAL"
    REMINDER = "REMINDER"
    CLARIFICATION = "CLARIFICATION"
    UPDATE = "UPDATE"
    NOTIFICATION = "NOTIFICATION"
    CORRECTION = "CORRECTION"


class OutboxStatus(StrEnum):
    PENDING = "PENDING"
    SENDING = "SENDING"
    SENT = "SENT"
    FAILED = "FAILED"
    RECONCILING = "RECONCILING"
    CANCELLED = "CANCELLED"


class OutboxCancelReason(StrEnum):
    PARENT_TERMINAL = "PARENT_TERMINAL"
    STALE = "STALE"
    POLICY_BLOCKED = "POLICY_BLOCKED"


class TelegramUpdateStatus(StrEnum):
    PENDING = "PENDING"
    PROCESSED = "PROCESSED"
    FAILED = "FAILED"


class PolicyOutcome(StrEnum):
    AUTO = "AUTO"
    ASK_PARTICIPANT = "ASK_PARTICIPANT"
    ASK_ME = "ASK_ME"
    DO_NOT_ACT = "DO_NOT_ACT"


class ValidatorCategory(StrEnum):
    VALID = "VALID"
    UNSUPPORTED_CLAIM = "UNSUPPORTED_CLAIM"
    UNAUTHORIZED_COMMITMENT = "UNAUTHORIZED_COMMITMENT"
    WRONG_MESSAGE_KIND = "WRONG_MESSAGE_KIND"
    UNCLEAR_OR_AMBIGUOUS = "UNCLEAR_OR_AMBIGUOUS"
    RULE_VIOLATION = "RULE_VIOLATION"
