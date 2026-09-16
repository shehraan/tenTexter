# tenTexter v1 — Non-Negotiable Invariants

These invariants are implementation requirements, not suggestions.

## 1. State ownership

Each mutable fact has exactly one authoritative owner.

Examples:
- final inbound correlation: `MessageRevision.awaited_response_id`
- current provider content: `Message.current_revision_id`
- participant availability: `TaskParticipant.availability_status`
- logical send lifecycle: `OutboxMessage.status`
- physical send attempt lifecycle: `OutboxDeliveryAttempt`
- trigger firing identity/idempotency: `TriggerExecution`
- owner decision lifecycle: `DecisionRequest`

Audit rows may describe state but must not become competing authorities.

## 2. LLM boundary

LLMs may:
- parse owner instructions
- classify messages
- perform semantic fallback correlation
- generate phrasing
- validate generated messages

LLMs may not:
- send messages directly
- mutate authoritative state directly
- choose deterministic state transitions
- bypass ContactRule/DisclosurePolicy
- bypass independent validation

Friend-supplied text is untrusted data, never executable instruction context.

## 3. Owner control plane

The Telegram bot is the dedicated owner control plane.

v1 authorization is deterministic: incoming Telegram sender user ID must equal configured OWNER_ID. Optionally sanity-check private chat type. Unauthorized senders never reach TaskParser.

`TelegramUpdate.telegram_update_id` is UNIQUE and drives inbound idempotency. Persist update before side effects. State-changing effects and PROCESSED status commit atomically.

## 4. Task lifecycle

`TaskInstance.status`:
- ACTIVE
- COMPLETED
- CANCELLED
- LAPSED

Meanings:
- COMPLETED: normal coordination lifecycle reached planned close.
- CANCELLED: explicitly stopped.
- LAPSED: an existing task's useful coordination window passed before it could complete normally, typically discovered after downtime/recovery.

Terminal task state is immutable.

Late inbound messages are still stored/classified globally but may not mutate participant/proposal/trigger state for a terminal task. Actionable late messages surface to OWNER or a new ad-hoc action.

Terminalization is one transaction. TERMINATE children are closed/cancelled; SURVIVE owner/operational artifacts persist.

## 5. Recurrence

`TaskDefinition` owns recurrence semantics. `next_occurrence_at` is a derived scheduler cursor/cache only.

Definition-backed instances require both `task_definition_id` and immutable `occurrence_key`; ad-hoc instances require neither.

`occurrence_key` identifies the original recurrence slot and never changes when an instance is rescheduled.

Missed recurring occurrences are never backfilled late. Advance the recurrence cursor using compare-and-swap and create a durable owner notification in the same transaction.

If a recurring occurrence cannot be deterministically routed/spawned, create no partial TaskInstance, notify OWNER, skip that occurrence, and advance recurrence normally.

## 6. Availability evidence

`TaskParticipant.availability_status`:
- UNKNOWN
- AVAILABLE
- UNAVAILABLE
- UNCERTAIN

`availability_evidence`:
- FIRST_PARTY
- THIRD_PARTY

Latest reliably ordered evidence wins. FIRST_PARTY/THIRD_PARTY is provenance, not precedence.

Use provider/message event ordering metadata, not local `received_at`, to decide recency. If ordering cannot be established safely, do not guess; use UNCERTAIN/clarification.

Track the exact supporting `availability_source_revision_id` for non-UNKNOWN current availability.

UNKNOWN must have no evidence/source revision; non-UNKNOWN must have provenance.

## 7. Message revisions

`Message` is stable provider identity. `MessageRevision` is immutable exact content.

Inbound uniqueness: `UNIQUE(conversation_id, provider_message_id)`.

Revision dedup: `UNIQUE(message_id, provider_revision_key)`.

`provider_revision_key` is identity/dedup only, never ordering. `received_at` is audit only, never provider ordering.

If the same provider ordering position yields different content, store both and reconcile; never guess.

Before semantic mutation from a revision commits, verify `Message.current_revision_id == revision.id`. A stale revision may be successfully discarded as PROCESSED; it must not mutate current state.

All semantic effects reference exact `MessageRevision` rows.

## 8. Awaited responses and correlation

One inbound revision maps to at most one AwaitedResponse in v1.

`AwaitedResponse.AMBIGUOUS` means correlation is known but interpretation is ambiguous.

Correlation ambiguity among multiple AwaitedResponses means:
- `MessageRevision.awaited_response_id` remains NULL
- candidate AwaitedResponses remain OPEN
- a DecisionRequest plus candidate join rows is created

`response_count` must count distinct relevant TaskParticipants, not raw AwaitedResponse rows.

## 9. ContactRule

Scopes:
- GLOBAL: no target
- TOPIC: `topic_key` only
- TASK_DEFINITION: `task_definition_id` only
- TASK_INSTANCE: `task_instance_id` only

Exactly the scope-appropriate target may be populated.

Topic matching is deterministic exact matching on normalized `topic_key`; no Topic entity in v1. If topic classification is uncertain where a rule may matter, ASK_ME.

Rule source is provenance only, not precedence. Resolver precedence is:
1. applicable STRONG participant-requested boundary conflicts => ASK_ME unless an explicit approved exception overrides it
2. otherwise most specific scope
3. within same scope, stronger rule
4. unresolved tie/conflict => ASK_ME

Revocation uses `revoked_at`/`revoked_reason`; historical rows are preserved. Applicable means matching scope, not revoked, not expired, and parent scope still applicable.

Approved exceptions use `overrides_contact_rule_id`, must not self-reference, must refer to the same Person, must be narrower/applicable to the approved context, and must not form cycles.

An explicit participant-requested boundary is a STRONG `DO_NOT_CONTACT` rule. A TASK_INSTANCE participant boundary is absolute for that task. A broader participant boundary may be overridden only by an explicit owner-approved TASK_INSTANCE `ALLOW` rule referencing the blocking rule. While that decision is pending, the original immutable Outbox remains PENDING and is neither validated nor sent.

Boundary classification is a bounded semantic effect separate from availability/proposal classification. Ambiguous scope creates a typed MessageRevision DecisionRequest and holds sends to that Person for the uniquely attributable task, or globally when no unique task is attributable. Editing a message never silently revokes an established boundary.

ContactRules apply to logical targets, not every incidental member of a group conversation. Do not let one group member implicitly veto a whole group merely because they are present.

## 10. Disclosure

Cross-conversation disclosure is destination-aware.

`DisclosureGrant` includes source person, source conversation, destination conversation, task instance, atomic scopes, status, expiry.

Atomic scopes:
- AVAILABILITY
- SCHEDULING
- LOCATION

ContextBuilder must filter context for the actual destination before generation. The validator independently rechecks disclosure constraints.

A group destination does not create a blanket ContactRule veto, but disclosure authorization must still be valid for that exact destination conversation.

## 11. DecisionRequest

Statuses: PENDING, CLOSED.

Close reasons:
- ANSWERED
- DISMISSED
- EXPIRED
- PARENT_TERMINAL
- SUBJECT_RESOLVED

Exactly one typed subject FK is populated.

Before applying an owner answer, atomically revalidate that the subject still requires a decision. If not, apply no side effect and close as SUBJECT_RESOLVED.

`parent_terminal_policy` is immutable TERMINATE or SURVIVE behavior configuration.

## 12. Proposal

One Proposal represents one atomic change.

Statuses:
- PENDING
- ACCEPTED
- REJECTED
- WITHDRAWN
- EXPIRED
- SUPERSEDED
- PARENT_TERMINAL

Approval must revalidate the operation-specific precondition. If it no longer holds, mark SUPERSEDED and do not apply.

## 13. Trigger execution

`TriggerExecution` owns firing identity/idempotency. `UNIQUE(task_trigger_id, fire_key)`.

Generation/validation must happen outside long SQLite transactions.

Flow:
1. short tx create/claim execution
2. generate/validate outside DB txn
3. short tx compare-and-swap expected execution state, recheck parent ACTIVE, commit Outbox rows, mark COMPLETED, and advance/deactivate trigger atomically

A reclaimed/stale worker must not be able to commit later.

Permanent recurring execution failure:
- keep execution FAILED
- notify OWNER
- advance trigger to next valid firing

Permanent one-shot execution failure:
- keep execution FAILED
- notify OWNER
- mark trigger INACTIVE

## 14. Outbox

Only external action type is SEND_MESSAGE.

Allowed transport/destination pairs in v1:
- BEEPER + PARTICIPANT
- TELEGRAM + OWNER

`OutboxMessage.final_text` is immutable after creation.

Statuses:
- PENDING
- SENDING
- SENT
- FAILED
- RECONCILING
- CANCELLED

All sends go through Outbox. No adapter or LLM bypass.

Each individual logical send gets its own deterministic idempotency key. `UNIQUE(transport, idempotency_key)`. A trigger execution may create multiple distinct logical sends with distinct keys.

Immediately before PENDING -> SENDING, revalidate:
- parent/task lifecycle
- originating object still applicable
- current ContactRules
- current DisclosureGrant
- fresh narrow destination-aware ContextBuilder result as needed
- independent validator on unchanged final text

If facts stale => CANCELLED/STale. If policy changed => CANCELLED/POLICY_BLOCKED. Validator unavailable => fail closed/keep pending.

No regeneration inside OutboxWorker.

## 15. Delivery attempts and uncertainty

`OutboxDeliveryAttempt` owns physical attempt details.

At most one unfinished attempt per OutboxMessage.

If definitely no external send began, logical Outbox may safely return to PENDING.

If the external boundary was crossed and outcome is unknown, logical Outbox becomes RECONCILING. Never blindly retry uncertain sends.

Reconcile strongest-first:
1. final provider message ID
2. pending provider ID
3. exact conversation search using sender/text/timestamp evidence
4. unresolved uncertainty => durable owner decision/information

For terminal parent + uncertain participant delivery, do not retry the old send.

## 16. Corrections

CORRECTION messages require `corrects_outbox_message_id`.

One message may be directly corrected at most once, producing a linear correction chain. Correction target must be SENT, same destination, same TaskInstance, non-self, acyclic, and owner-approved in v1.

## 17. Archival and hard deletion

Runtime API may archive/restore but performs no hard delete.

Historical rows and joins are preserved. Foreign keys should RESTRICT manual hard deletion by default unless an intentional maintenance purge removes dependents.

## 18. Runtime assumptions

v1 supports exactly one active `agent-app` instance.

SQLite implementation should use WAL, busy timeout, migrations, short transactions, and compare-and-swap where specified.

Multi-process fencing is deferred until the product actually requires multiple app instances.
