# tenTexter v1 — Canonical Architecture Ledger

This document is the authoritative architecture ledger for v1.

## Product goal

A local-first messaging agent that coordinates activities on the owner's existing accounts. Example: ask a Discord tennis group and an Instagram contact whether they are available at 5 PM, track replies, handle ambiguity/counterproposals safely, and notify the owner when nobody is available.

Beeper Desktop is the aggregation layer for participant-facing messaging. A dedicated Telegram bot is the owner control plane.

## Runtime topology

Processes:
- `agent-app`: one modular monolith containing orchestration, persistence, workers, scheduler, policy, adapters, and LLM clients
- primary model server: generation/parsing/classification
- validator model server: independent validation
- Beeper Desktop

v1 supports exactly one active `agent-app` instance.

## Components

- ControlGateway
- TaskParser
- EntityResolver
- Orchestrator
- Scheduler
- inbound message listener/worker
- Correlator
- Classifier
- ContextBuilder
- MessageGenerator
- independent MessageValidator
- ContactRuleResolver
- DisclosurePolicy
- OutboxWorker
- BeeperAdapter
- TelegramAdapter
- health checks
- SQLite persistence

## Deterministic vs LLM responsibilities

Deterministic:
- authorization
- entity uniqueness/integrity
- lifecycle transitions
- routing when unambiguous
- ContactRule evaluation
- DisclosurePolicy enforcement
- recurrence and scheduling
- idempotency
- leases/CAS
- terminalization
- retry/reconciliation policy

LLM-assisted:
- owner task parsing under schema validation
- semantic fallback correlation
- message classification
- phrasing generation
- independent output validation

LLM output never directly causes a transport side effect.

## Core entities

### Person
Represents a real human.

Fields include stable internal id, display metadata, archival metadata.

### Identity
Represents one platform/network account belonging to a Person.

Key invariant:
- `beeper_user_id` UNIQUE NOT NULL for Beeper-managed identities
- mutable usernames/display names are metadata only

Person 1:N Identity.

### Conversation
Represents a concrete provider/Beeper conversation.

`counterparty_person_id` may exist as optional convenience metadata for direct chats only; it is not authoritative membership.

### ConversationParticipant
Join between Conversation and Identity.

`UNIQUE(conversation_id, identity_id)`.

### BeeperSyncCheckpoint
Durable provider synchronization progress. One singleton row owns the global chat-feed cursor; one row per Conversation owns its message-feed cursor, bounded bootstrap progress, and recent reconciliation progress. Provider cursors are opaque and never become message identity or ordering state.

v1 uses REST polling only. Initial backfill and best-effort edit/deletion reconciliation cover the most recent 30 days. New-message discovery resumes from durable cursors. Because Beeper does not provide a documented replay guarantee for changes made while the app is offline, absence from a page is never treated as deletion; only an explicit provider deletion tombstone creates a deleted MessageRevision.

### TaskDefinition
Reusable/recurring coordination template.

Important fields:
- recurrence_rule
- default_time
- timezone (IANA, e.g. America/Toronto)
- default_duration_minutes
- default_location
- default_topic_key
- next_occurrence_at (derived scheduler cursor/cache)
- archival metadata

### TaskDefinitionParticipant
Default participant membership for a TaskDefinition.

`UNIQUE(task_definition_id, person_id)`.

### TaskInstance
Concrete coordination occurrence.

Important fields:
- nullable `task_definition_id`
- nullable immutable `occurrence_key`
- `scheduled_at` absolute timestamp
- duration_minutes
- location
- topic_key
- coordination_close_offset_minutes default 60
- status ACTIVE | COMPLETED | CANCELLED | LAPSED
- archival metadata

Constraint:
- `(task_definition_id IS NULL) = (occurrence_key IS NULL)`
- `UNIQUE(task_definition_id, occurrence_key)` for definition-backed occurrences

`completion_due_at` is derived from scheduled_at + coordination_close_offset_minutes.

### TaskParticipant
One Person participating in one TaskInstance, pinned to one Conversation for that occurrence.

Fields:
- task_instance_id
- person_id
- conversation_id
- availability_status UNKNOWN | AVAILABLE | UNAVAILABLE | UNCERTAIN
- availability_evidence FIRST_PARTY | THIRD_PARTY nullable only when UNKNOWN
- availability_source_revision_id nullable only when UNKNOWN

`UNIQUE(task_instance_id, person_id)`.

Conversation must contain an Identity belonging to the Person; validate transactionally.

### Message
Stable inbound provider message identity.

Fields:
- conversation_id
- provider_message_id
- sender_identity_id
- created_at
- current_revision_id

`UNIQUE(conversation_id, provider_message_id)`.

Composite integrity must ensure current_revision_id belongs to this Message.

### MessageRevision
Immutable exact content version.

Fields include:
- message_id
- provider_revision_key
- provider revision/order metadata
- content_hash
- is_deleted
- text/content
- received_at
- content_support SUPPORTED | UNSUPPORTED
- processing_status PENDING | PROCESSING | PROCESSED | FAILED
- lease_expires_at
- awaited_response_id nullable final correlation owner

`UNIQUE(message_id, provider_revision_key)`.

### MessageProcessingAttempt
Audit of processing attempts.

Result SUCCESS | FAILED | ABANDONED.

Failure type may include TIMEOUT, MODEL_ERROR, CORRELATION_ERROR, INVALID_DATA, INTERNAL_ERROR.

### AwaitedResponse
Represents an expected response from a TaskParticipant.

Fields:
- task_participant_id
- expected_response_type
- status OPEN | SATISFIED | EXPIRED | CANCELLED | AMBIGUOUS
- created_at
- expires_at

AMBIGUOUS means correlation is known and interpretation is unclear.

### AwaitedResponsePrompt
M:N join from AwaitedResponse to OutboxMessage.

Supports reminders/clarifications and group messages that prompt multiple participants.

### Proposal
One atomic participant-proposed task change.

Fields:
- task_instance_id
- proposed_by_participant_id nullable
- source_message_revision_id
- field
- operation
- old_value
- proposed_value
- status PENDING | ACCEPTED | REJECTED | WITHDRAWN | EXPIRED | SUPERSEDED | PARENT_TERMINAL
- created_at
- resolved_at

One proposal = one atomic change.

### DecisionRequest
Durable owner decision request.

Fields:
- nullable task_instance_id convenience scope
- type
- status PENDING | CLOSED
- close_reason ANSWERED | DISMISSED | EXPIRED | PARENT_TERMINAL | SUBJECT_RESOLVED
- context_json presentation only
- created_at
- expires_at
- resolved_at
- resolution_json
- parent_terminal_policy TERMINATE | SURVIVE
- exactly one typed subject FK

Typed subjects include Proposal, ContactRule, OutboxMessage, MessageRevision, TelegramUpdate, TriggerExecution.

### DecisionRequestPrompt
One DecisionRequest may have multiple owner prompt OutboxMessages; each owner Outbox prompt belongs to at most one DecisionRequest.

### DecisionRequestAwaitedResponseCandidate
Stores candidate AwaitedResponses when correlation itself is ambiguous.

### ContactRule
Persistent contact/policy rule for one Person.

Fields:
- person_id
- scope GLOBAL | TOPIC | TASK_DEFINITION | TASK_INSTANCE
- type
- value
- source USER_CONFIGURED | PARTICIPANT_REQUESTED | SYSTEM_INFERRED
- strength WEAK | MEDIUM | STRONG
- topic_key nullable
- task_definition_id nullable
- task_instance_id nullable
- created_at
- expires_at nullable
- revoked_at nullable
- revoked_reason REVOKED | SUPERSEDED nullable
- overrides_contact_rule_id nullable self-FK

Exactly the target corresponding to scope is populated.

No Topic table in v1. `topic_key` is normalized deterministic text.

### DisclosureGrant
Bounded cross-conversation disclosure permission.

Fields:
- source_person_id
- source_conversation_id
- destination_conversation_id
- task_instance_id
- status ACTIVE | INACTIVE
- inactive_reason EXPIRED | REVOKED | TASK_TERMINAL nullable when active
- expires_at
- created_at

### DisclosureGrantScope
Atomic grant scopes:
- AVAILABILITY
- SCHEDULING
- LOCATION

### TaskTrigger
Bounded condition-driven send intent.

Fields:
- task_instance_id
- condition_json
- stop_condition_json nullable
- target_selector ALL_PARTICIPANTS | NO_RESPONSE | AVAILABLE | UNAVAILABLE | UNCERTAIN | SPECIFIC_PARTICIPANT
- target_task_participant_id nullable only unless SPECIFIC_PARTICIPANT
- action_type SEND_MESSAGE
- action_payload_json generation intent/goal
- next_run_at nullable derived scheduler cache
- status ACTIVE | INACTIVE
- inactive_reason FIRED | STOP_CONDITION_MET | DISABLED | TASK_TERMINAL

No arbitrary executable condition code.

### TriggerExecution
Authoritative identity/idempotency of one trigger firing.

Fields:
- task_trigger_id
- fire_key
- scheduled_for nullable metadata
- status PENDING | PROCESSING | COMPLETED | FAILED | CANCELLED
- lease_expires_at nullable only while PROCESSING

`UNIQUE(task_trigger_id, fire_key)`.

### TaskEvent
Append-only audit/event history.

May reference TaskParticipant and exact source MessageRevision. It is not authoritative state.

### OutboxMessage
Authoritative logical external send.

Fields:
- nullable task_instance_id
- transport BEEPER | TELEGRAM
- destination_kind PARTICIPANT | OWNER
- immutable final_text
- message_kind INITIAL | REMINDER | CLARIFICATION | UPDATE | NOTIFICATION | CORRECTION
- status PENDING | SENDING | SENT | FAILED | RECONCILING | CANCELLED
- deterministic idempotency_key
- lease_expires_at nullable only while SENDING
- parent_terminal_policy TERMINATE | SURVIVE
- cancel_reason PARENT_TERMINAL | STALE | POLICY_BLOCKED nullable only when CANCELLED
- trigger_execution_id nullable
- corrects_outbox_message_id nullable only for CORRECTION

Allowed v1 pairs:
- BEEPER + PARTICIPANT
- TELEGRAM + OWNER

`UNIQUE(transport, idempotency_key)`.

### OutboxMessageParticipant
M:N logical participant targets for participant-directed sends.

### BeeperOutboxDestination
Physical Beeper destination child with one Conversation.

### TelegramOutboxDestination
Physical Telegram destination child with telegram_chat_id.

Exactly one destination child matching Outbox transport is required transactionally.

### OutboxDeliveryAttempt
Physical delivery attempt history.

Fields:
- outbox_message_id
- started_at
- finished_at
- result SUCCESS | FAILED | ABANDONED
- failure_type
- error_details
- provider_message_id

At most one unfinished attempt per OutboxMessage.

Provider-specific pending identifiers may live in transport-specific attempt detail children.

### TelegramUpdate
Durable owner-control inbound update.

Fields:
- telegram_update_id UNIQUE
- received_at
- status PENDING | PROCESSED | FAILED

No persisted PROCESSING lease needed in v1 because exactly one Telegram consumer is supported.

## Correlation

Priority:
1. provider reply linkage
2. exact pinned conversation + only one open expected response
3. recency
4. LLM semantic relevance
5. ambiguity => do not guess

Cross-conversation same-person response => ASK_ME in v1.

One revision ultimately maps to at most one AwaitedResponse.

## Contact policy

Policy outcomes:
- AUTO
- ASK_PARTICIPANT
- ASK_ME
- DO_NOT_ACT

Examples:
- ambiguous participant reply with two active requests => ASK_PARTICIPANT
- counterproposal => ASK_ME
- unauthorized new recipient => ASK_ME
- more than 25 distinct logical TaskParticipants in one TaskInstance => ASK_ME before any participant send; approval is scoped to that TaskInstance only
- participant asks not to be contacted about a topic => persistent ContactRule

Explicit owner-approved narrow exception points to exactly one original blocking ContactRule; one original may have multiple exceptions over time.

Participant boundary requests are classified independently of availability and proposals. Explicit GLOBAL, TOPIC, and TASK_INSTANCE requests become STRONG participant-requested `DO_NOT_CONTACT` rules. Ambiguous scope creates an owner DecisionRequest and a temporary send hold. TASK_INSTANCE boundaries are absolute; broader boundaries permit only an owner-approved TASK_INSTANCE `ALLOW` exception. The original Outbox remains PENDING while that exception decision is pending.

## Disclosure policy

Implicit private-DM information flowing to another conversation requires ASK_ME unless a valid DisclosureGrant exists.

An explicit owner command naming source, destination, and share intent can authorize/create a bounded grant.

Participant text is not permission to disclose unrelated private context.

## Recurrence and spawning

TaskDefinition stores local wall-clock recurrence semantics and IANA timezone. Each TaskInstance stores absolute `scheduled_at`.

Definition defaults are copied into instances and then independently editable. ContactRules remain live and are never copied.

Spawn routing is deterministic against live conversations/rules. Ambiguity means no partial instance: skip occurrence and notify owner.

## Task terminalization

When TaskInstance becomes terminal:
- OPEN or interpretation-AMBIGUOUS AwaitedResponses => CANCELLED
- ACTIVE TaskTriggers => INACTIVE/TASK_TERMINAL
- ACTIVE DisclosureGrants => INACTIVE/TASK_TERMINAL
- PENDING DecisionRequests with TERMINATE => CLOSED/PARENT_TERMINAL
- PENDING Proposals => PARENT_TERMINAL
- PENDING OutboxMessages with TERMINATE => CANCELLED/PARENT_TERMINAL
- PENDING TriggerExecutions => CANCELLED

PROCESSING TriggerExecution may finish computation but final commit rechecks parent ACTIVE; otherwise it enqueues nothing and becomes CANCELLED.

SENDING OutboxMessage crosses an external boundary and cannot be pretended away. Let the current attempt resolve; uncertain outcome becomes RECONCILING with no blind retry.

SURVIVE owner notifications/operational decisions may continue after parent terminalization.

## Outbox pre-send safety

Before PENDING -> SENDING:
- recheck task/parent lifecycle
- recheck originating object applicability
- recheck current ContactRules for logical targets
- recheck exact destination DisclosureGrant requirements
- obtain fresh destination-aware narrow context if relevant
- independently validate unchanged final_text

No regeneration inside OutboxWorker.

## Corrections

Corrections are new OutboxMessages with message_kind CORRECTION and `corrects_outbox_message_id`.

Linear correction chain only. Owner approval required in v1.

## Context retrieval

Raw local message history is source of truth; embeddings/summaries are derived caches only.

Progressive retrieval:
1. direct reply-linked
2. current TaskInstance-associated
3. recent surrounding conversation
4. semantic older context
5. long-term person facts

ContextBuilder is destination-aware and disclosure-filtered before generation.

## Health/failure behavior

Monitor Beeper, primary model, validator, Telegram, SQLite/workers/scheduler.

On dependency failure:
- ingest/store what can safely be stored
- retain pending work
- notify owner on meaningful HEALTHY -> UNHEALTHY transition
- never bypass validator/safety to keep moving
- resume on recovery

## Archival

Normal runtime deletion is archive/disable. Runtime has no hard-delete capability. Manual backend maintenance may intentionally purge data outside normal agent behavior.

## Explicitly deferred from v1

Do not add unless implementation proves a real requirement:
- multi-process app fencing
- sophisticated Topic entity/ontology
- embeddings for every message
- repeated state-edge trigger framework
- fancy missed-occurrence batching
- advanced media semantics
- generalized deadline framework
- conversation-wide ContactRule veto semantics
- exotic DST policy beyond correct IANA recurrence handling
