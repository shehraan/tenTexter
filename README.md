# tenTexter

Local-first messaging coordination agent for arranging activities across Beeper-connected conversations, with a dedicated Telegram owner control plane.

## Status

Architecture v1 is frozen. Phase 1 (project skeleton and deterministic test
harness) is implemented; domain persistence begins in Phase 2.

The authoritative implementation documents are:

- `AGENTS.md` — instructions for Codex and other coding agents.
- `docs/architecture.md` — canonical architecture ledger and data-flow design.
- `docs/invariants.md` — non-negotiable ownership, lifecycle, integrity, privacy, and crash-recovery invariants.
- `docs/implementation-plan.md` — phased build order and acceptance criteria.

Do not infer architecture from commit history or old discussions. If implementation exposes a contradiction in the documents above, stop and surface the contradiction instead of silently redesigning the system.

## Development

Python 3.12 or newer is required.

```console
python -m pip install -e '.[dev]'
pytest
```

Configuration is loaded from the environment:

- `TEN_TEXTER_DATABASE_PATH` (default: `ten_texter.db`)
- `TEN_TEXTER_SQLITE_BUSY_TIMEOUT_MS` (default: `5000`)
- `TEN_TEXTER_LOG_LEVEL` (default: `INFO`)

Initialize or upgrade the local database and check connectivity with:

```console
ten-texter db upgrade
ten-texter check
```
