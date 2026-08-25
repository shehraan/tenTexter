# AGENTS.md

This repository implements the frozen v1 architecture for tenTexter.

## Source of truth

Read these files completely before changing architecture-sensitive code:

1. `docs/architecture.md`
2. `docs/invariants.md`
3. `docs/implementation-plan.md`

Treat them as authoritative. Do not infer design from old chat history, commit history, comments, or partially implemented code when those conflict with the docs.

## Architectural change policy

Do not silently add or change:

- entities
- lifecycle states
- cardinalities
- authoritative state ownership
- retry semantics
- disclosure/contact policy
- automatic approval behavior
- terminalization behavior
- idempotency semantics

If implementation exposes a real contradiction or missing capability, stop and report it. Do not redesign around it implicitly.

## Core implementation rule

Every mutable fact has exactly one authoritative owner. Derived caches, audit rows, and convenience fields must never become competing authorities.

## Safety boundary

LLMs may parse, classify, correlate semantically, generate text, and validate text. They must never directly invoke transports, mutate authoritative state, bypass policy, or decide deterministic lifecycle transitions.

All external sends go through the durable Outbox path and pre-send revalidation.

## Implementation discipline

- Prefer small deterministic domain services over clever generalized frameworks.
- Use SQLite v1 and preserve transactional invariants.
- Keep model-server integrations behind interfaces until deterministic core phases are complete.
- Add tests for every lifecycle/constraint invariant before integrating real transports.
- Fail closed when policy, routing, correlation, or validation is ambiguous.

## Scope

Implement one modular monolith `agent-app` for v1. Exactly one active app instance is supported. Model servers and Beeper Desktop are separate processes.
