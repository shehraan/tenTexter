# tenTexter v1 — One-Shot Codex Evaluation Contract

This document exists only to evaluate whether a coding agent can implement the frozen v1 architecture in one continuous run. It does not supersede `docs/architecture.md` or `docs/invariants.md`.

## Authority order

When documents disagree, use this order:

1. `docs/invariants.md`
2. `docs/architecture.md`
3. `docs/implementation-plan.md`
4. this document
5. existing implementation details

Do not infer architecture from old conversations or invent replacements for explicit invariants.

## Objective

Implement the entire v1 application described by all 13 phases in `docs/implementation-plan.md` in sequence during one run.

The run is allowed to continue automatically from one phase to the next only after the current phase's relevant tests pass. Do not wait for human approval between phases unless a stop condition below is reached.

## Required workflow

Before writing code:

1. Read `AGENTS.md`, `docs/architecture.md`, `docs/invariants.md`, and `docs/implementation-plan.md` completely.
2. Inspect the repository and existing work before choosing implementation details.
3. Build a private phase checklist mapping every implementation-plan deliverable and acceptance criterion to intended code/tests.
4. Preserve the architecture exactly; implementation freedom applies only where the docs explicitly leave details open.

Then implement phases 1 through 13 in order.

For each phase:

1. implement the smallest complete slice satisfying that phase;
2. add or update tests proving its acceptance criteria and relevant invariants;
3. run the narrow tests for that phase;
4. run the full test suite before proceeding if the phase changes shared domain or persistence behavior;
5. fix failures before continuing;
6. do not weaken tests or constraints merely to make them pass.

## Definition of done

A one-shot run is successful only if all of the following are true:

- all 13 phases are implemented or an explicitly external integration limitation is clearly isolated and reported;
- the deterministic core works without external network access in tests;
- fresh SQLite database migration from zero to head succeeds;
- all database invariants that can be enforced in SQLite are actually enforced and tested;
- domain transitions are covered by deterministic tests;
- fake/stub transports and model clients permit end-to-end tests without sending real messages;
- every real external send path still flows through Outbox and independent pre-send validation;
- crash/retry/idempotency/reconciliation tests cover the major failure windows listed in Phase 13;
- no known policy, disclosure, or ContactRule bypass is introduced;
- the full automated test suite passes;
- the application has a documented local startup/configuration path and an example environment file with no secrets;
- no credentials, tokens, local databases, logs, or generated runtime state are committed.

## External integrations

### Telegram

Use the simplest reliable local v1 integration consistent with the architecture. Authentication must occur before parsing/domain mutation. Real credentials are configuration only and must never be required for unit/integration tests.

### Beeper Desktop

Beeper API details are external and may change. Verify current local API/SDK semantics from authoritative documentation available during implementation before depending on endpoint or identifier behavior.

If current Beeper semantics cannot be verified or exercised in the environment:

- keep the adapter behind the defined interface;
- implement all deterministic surrounding behavior and contract tests using a fake adapter;
- do not fabricate provider guarantees;
- clearly report the exact unverified integration surface at the end.

Do not assume merged conversations.

### Local model servers

Keep primary and validator clients independently configurable and replaceable. Tests must use fakes. The validator must remain a separate logical dependency and may not be silently bypassed when unavailable.

## Side-effect safety during development

Automated tests and default development commands must not send real Telegram/Beeper messages.

Real transport execution must require explicit configuration and should fail closed when required configuration is missing.

Do not use real participant conversations as test fixtures.

## Quality bar

Prefer straightforward Python with explicit domain services and strong typing over abstractions that obscure state ownership.

Avoid:

- microservices;
- background infrastructure not required by v1;
- generic workflow engines;
- arbitrary-code condition evaluation;
- hidden retries across uncertain external boundaries;
- duplicated state ownership;
- model-driven side effects;
- weakening SQLite constraints in favor of comments alone when SQLite can enforce the invariant.

Keep transactions short around SQLite. Generation/model calls and network calls must not occur while holding long write transactions.

## Stop conditions

Stop the one-shot run and report the contradiction instead of improvising if implementation truly requires any of the following:

- a new authoritative entity or lifecycle state;
- two owners for the same mutable fact;
- a changed cardinality;
- a new autonomous authority/policy decision;
- bypassing Outbox;
- bypassing independent validation;
- weakening ContactRule or disclosure policy;
- changing terminalization, recurrence, or uncertain-delivery semantics;
- an interpretation of the docs that would permit a wrong-recipient or duplicate-send risk not already resolved by the existing invariants.

An ordinary library/API choice, module layout choice, test organization choice, or adapter implementation detail is not a reason to stop.

## Required final report

At the end of the run, report:

1. phase-by-phase completion status (1–13);
2. test commands run and results;
3. migration validation performed;
4. external integrations that are real vs fake/contract-tested;
5. any architecture ambiguity encountered and how it was handled;
6. any remaining manual setup required for Telegram, Beeper, primary model server, or validator model server;
7. files changed;
8. repository status and whether any unrelated/user files were intentionally left untouched.

Follow the commit/push approval rules in `AGENTS.md`; completing the one-shot implementation does not itself authorize a commit or push.
