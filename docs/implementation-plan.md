# tenTexter v1 — Implementation Plan

Build in phases. Do not jump ahead. Each phase must preserve `docs/architecture.md` and `docs/invariants.md`.

## Phase 1 — Project skeleton and deterministic test harness

Goal: establish the Python application, configuration, SQLite plumbing, migrations, and test tooling without transports or LLMs.

Deliverables:
- Python project metadata and dependency management
- `src/ten_texter/` package
- config loading from environment with typed settings
- SQLite engine/session layer
- WAL + busy timeout initialization
- Alembic migration setup
- pytest configuration
- basic application bootstrap/CLI
- logging setup

Acceptance:
- clean install succeeds
- empty DB can migrate from zero to head
- tests run from one command
- no external network required

## Phase 2 — Full schema, enums, constraints, and indexes

Implement the complete v1 persistence model from architecture docs.

Include:
- all entities and joins
- enums
- FKs
- CHECK constraints
- UNIQUE constraints
- partial unique indexes where required
- composite FKs for same-parent integrity
- archival columns where applicable

Mandatory constraints include:
- TaskInstance definition/occurrence nullability equivalence
- TaskParticipant availability/provenance equivalence
- DecisionRequest status/close reason/resolution consistency
- exactly one typed DecisionRequest subject FK
- ContactRule scope-target consistency
- DisclosureGrant active/inactive reason consistency
- TaskTrigger specific-target consistency
- TriggerExecution processing lease consistency
- Outbox sending lease consistency
- Outbox cancellation reason consistency
- Outbox correction linkage consistency
- transport/destination pair restriction
- Proposal pending/resolved_at consistency

Acceptance:
- migration succeeds on fresh SQLite
- schema round-trip tests
- tests prove invalid combinations are rejected by DB
- tests prove uniqueness/composite-parent constraints

## Phase 3 — Domain repositories and atomic state transitions

Implement deterministic domain services/repositories before any network adapter.

Cover:
- TaskInstance creation/reschedule/terminalization
- TaskParticipant availability updates using reliably ordered evidence
- Proposal lifecycle and precondition revalidation
- DecisionRequest create/resolve/dismiss/expire/subject-resolved logic
- ContactRule create/revoke/override integrity
- DisclosureGrant lifecycle
- AwaitedResponse lifecycle

Acceptance:
- terminalization transaction tests
- stale DecisionRequest answer cannot apply side effects
- stale/older availability evidence cannot overwrite newer evidence
- ContactRule override cannot cross Person/self-cycle

## Phase 4 — Outbox and delivery state machine

Implement logical send durability independently of real transports.

Cover:
- Outbox creation helpers with deterministic logical-send idempotency keys
- destination child validation
- participant target joins
- delivery attempt lifecycle
- lease claim/reclaim
- SENT/FAILED/RECONCILING/CANCELLED transitions
- correction-chain validation
- pre-send revalidation interface hooks

Use fake transport in tests.

Acceptance:
- duplicate logical sends collapse by idempotency key
- one TriggerExecution can create multiple distinct logical sends
- uncertain external boundary never blindly retries
- terminal parent handling matches invariants
- correction chain stays linear

## Phase 5 — Scheduler, recurrence, and TriggerExecution

Implement:
- TaskDefinition recurrence cursor
- occurrence materialization
- immutable occurrence_key
- compare-and-swap cursor advancement
- missed-occurrence skip + owner alert creation
- deterministic spawn routing interface
- TaskTrigger evaluation with bounded condition DSL
- TriggerExecution unique fire keys, leases, CAS completion
- recurring permanent failure alert + advance
- one-shot permanent failure alert + deactivate

No LLM calls yet; use deterministic fake generation result where needed.

Acceptance:
- concurrent/repeated scheduler polls do not duplicate occurrence
- missed occurrence does not backfill
- unroutable occurrence creates no partial TaskInstance
- stale reclaimed trigger worker cannot commit

## Phase 6 — Message and revision ingestion core

Implement provider-agnostic inbound domain:
- Message upsert by conversation + provider message ID
- immutable MessageRevision insert/dedup
- provider ordering metadata representation
- current_revision selection
- processing leases/attempts
- stale revision optimistic commit guard
- unsupported-content handling
- deletion tombstone semantics

Acceptance:
- duplicate delivery is idempotent
- edit creates revision, not replacement
- stale revision processing cannot mutate semantic state
- conflicting same-order revisions are preserved/reconciliation-required

## Phase 7 — Correlation, AwaitedResponse, and classification orchestration

Implement deterministic correlation first:
1. reply linkage
2. pinned conversation + one open expected response
3. recency
4. pluggable semantic fallback
5. ambiguity handling

Implement:
- known-correlation interpretation ambiguity => AwaitedResponse.AMBIGUOUS
- correlation ambiguity => candidate rows + DecisionRequest; candidates stay OPEN
- distinct-participant response_count metric
- cross-conversation same-person => ASK_ME

LLM interfaces may be stubbed/faked initially.

Acceptance:
- no ambiguous correlation is guessed
- one revision maps to at most one AwaitedResponse
- response_count cannot exceed participant semantics because of multiple AR rows

## Phase 8 — Telegram owner adapter

Implement:
- Telegram polling/webhook mode chosen for local deployment, preferably simplest reliable v1 path
- OWNER_ID auth before TaskParser
- TelegramUpdate idempotency
- owner outbound messages through Outbox only
- DecisionRequest callbacks/reply correlation

Priority for owner answer correlation:
1. explicit callback/request ID
2. reply-to Outbox -> DecisionRequestPrompt
3. otherwise no guess

Acceptance:
- unauthorized sender never reaches parser/domain mutation
- crash between receipt and commit safely resumes from PENDING
- duplicate update causes no duplicate effects

## Phase 9 — Beeper adapter

Implement against current Beeper Desktop local API/SDK semantics:
- conversation/identity discovery and sync
- inbound event ingestion into Message/MessageRevision
- send through OutboxWorker
- pending/final provider message identifiers in delivery attempt/detail
- reconciliation lookup hooks

Do not assume Beeper merged conversations.

Acceptance:
- direct/group destinations route exactly as pinned
- send result uncertainty maps to RECONCILING
- username/display changes update metadata without duplicating Identity

## Phase 10 — ContactRule resolver and DisclosurePolicy integration

Wire policies into orchestration and Outbox pre-send validation.

Implement:
- deterministic ContactRule applicability/precedence
- normalized topic_key
- narrow approved overrides
- destination-aware DisclosureGrant checks
- task-relative disclosure expiry recomputation on reschedule
- destination-aware context filtering interface

Do not implement whole-group ContactRule veto. Evaluate logical targets. Disclosure still applies to exact destination conversation.

Acceptance:
- strong boundary cannot be bypassed silently
- revoked/expired rules cease applying without loss of audit history
- private facts cannot flow to another conversation without explicit permission/grant

## Phase 11 — LLM interfaces and primary model integration

Implement model-server clients behind interfaces for:
- TaskParser
- Classifier
- semantic Correlator fallback
- EntityResolver semantic assistance if needed
- MessageGenerator

All structured outputs use strict schemas and deterministic validation.

The model receives facts, goals, and constraints; it chooses phrasing, not authority or state transitions.

Acceptance:
- malformed model output fails closed
- prompt-injection text cannot invoke tools/transports
- tests run with fake model clients

## Phase 12 — Independent validator

Implement separate validator-model client/process configuration.

Validator outputs bounded categories such as:
- VALID
- UNSUPPORTED_CLAIM
- UNAUTHORIZED_COMMITMENT
- WRONG_MESSAGE_KIND
- UNCLEAR_OR_AMBIGUOUS
- RULE_VIOLATION

Repair policy:
- writing/clarity invalid => critique to generator, bounded retry
- authority invalid => ASK_ME immediately
- repair cap exhausted => ASK_ME
- validator unavailable => fail closed; keep work pending

Acceptance:
- no send can bypass validator
- validator context is minimal/relevant and independent
- authority violations never get repaired into autonomous commitments

## Phase 13 — End-to-end failure and recovery tests

Build scenarios covering the real risks:
- owner command -> participant sends -> replies -> owner notification
- nobody available trigger
- counterproposal approval
- ambiguous reply
- cross-conversation reply
- contact boundary
- disclosure grant
- message edit reversing earlier semantics
- crash during revision processing
- crash before/after external send boundary
- uncertain delivery reconciliation
- task terminalization racing workers
- scheduler duplicate poll
- missed recurrence
- permanently failed recurring trigger
- validator/model/Beeper outage and recovery

Acceptance:
- all deterministic invariants proven under restart/retry
- no duplicate external send in tested crash windows
- no stale worker/revision commits semantic state
- no known policy/privacy bypass in supported v1 scenarios

## Implementation choices intentionally left to Codex

Codex may choose reasonable details that do not alter architecture, including:
- exact Python web/CLI framework if one is needed
- SQLAlchemy organization
- repository/service module names
- test fixture structure
- HTTP client library
- structured logging package
- model-server HTTP protocol wrappers

Prefer boring, widely supported choices. Avoid introducing infrastructure that v1 does not require.

## Stop conditions

Stop and request architectural review if implementation appears to require:
- a new authoritative entity/state
- duplicated ownership of a mutable fact
- a new automatic authority/policy decision
- a changed cardinality
- bypassing Outbox or validator
- changing recurrence/terminalization semantics
- weakening disclosure/contact boundaries

Do not solve those silently in code.
