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

## One-shot evaluation mode

Normal implementation follows `docs/implementation-plan.md` phase-by-phase with human review between meaningful phases.

When the user explicitly requests the **one-shot Codex evaluation**, read `docs/one-shot-eval.md` in addition to the normal source-of-truth files. In that mode, Codex is authorized to implement phases 1 through 13 sequentially in one continuous run, advancing only after the current phase's relevant tests pass. The one-shot contract does not authorize architectural redesign, real-message side effects during tests, commits, or pushes.

If `docs/one-shot-eval.md` conflicts with `docs/invariants.md` or `docs/architecture.md`, the invariant/architecture documents win.

## Commit and push approval

- After completing and verifying every repository change, provide the user with an appropriate, scoped commit message for approval.
- Do not create a commit or push anything until the user explicitly approves the proposed commit message or otherwise clearly authorizes committing and pushing.
- After approval, review the repository status and diff again, stage only the files required for the completed task, create the approved commit, and push the current intended branch.
- Never use broad staging commands such as `git add .` or `git add -A`; add the approved task files explicitly.
- Preserve and exclude unrelated user changes and untracked files. In particular, do not stage or push user-authored guidance, issue notes, logs, transcripts, local environment files, credentials, or similar workspace artifacts.
- Treat Markdown files as excluded by default unless they were created or intentionally modified as a deliverable for the current task and the user has approved including them. Files such as `issues.md`, `issue-*.md`, `logs.md`, `*-log.md`, transcripts, scratch notes, and agent-direction files must not be pushed merely because they are present in the worktree.
- Before committing, tell the user which repository and files will be included. After pushing, report the commit hash and remote branch.

## Pull requests

- Before reviewing a pull request, read the existing review comments and avoid duplicating feedback that another reviewer has already raised.
- When a repository has a `staging` branch, always target new pull requests to `staging`. Use the repository's default branch only when `staging` does not exist, unless the user explicitly requests another base branch.
- Keep pull request titles concise and outcome-focused, using plain language such as `Add provider-configurable session types` or `Stop sensitive data logging in user lookup`.
- Structure every pull request description with these sections in this order at the beginning: `Summary`, `Why`, `What changed`, and `Impact`.
- End every pull request description with `Deployment/dependency notes`, followed by `Validation` as the final section.
- Additional task-specific sections may appear between `Impact` and `Deployment/dependency notes` when useful.
