# tenTexter

Local-first messaging coordination agent for arranging activities across Beeper-connected conversations, with a dedicated Telegram owner control plane.

## Status

Architecture v1 is frozen. Phase 1 provides the application skeleton, typed
configuration, SQLite/Alembic plumbing, CLI, logging, and deterministic test
harness. It intentionally contains no domain schema.

The authoritative implementation documents are:

- `AGENTS.md` — instructions for Codex and other coding agents.
- `docs/architecture.md` — canonical architecture ledger and data-flow design.
- `docs/invariants.md` — non-negotiable ownership, lifecycle, integrity, privacy, and crash-recovery invariants.
- `docs/implementation-plan.md` — phased build order and acceptance criteria.

Do not infer architecture from commit history or old discussions. If implementation exposes a contradiction in the documents above, stop and surface the contradiction instead of silently redesigning the system.

## Development setup

Python 3.14 and [uv](https://docs.astral.sh/uv/) are required. Install the exact
locked development environment:

```bash
uv sync --locked
```

Run checks with:

```bash
uv run --locked ruff check .
uv run --locked pytest
```

The test suite blocks outbound network connections. Dependency acquisition is
the only setup step that may require a package index when the uv cache is empty.

## Configuration

Configuration is read from environment variables when a command starts:

| Variable | Default |
| --- | --- |
| `TENTEXTER_DATABASE_URL` | `sqlite+pysqlite:///./ten_texter.db` |
| `TENTEXTER_SQLITE_BUSY_TIMEOUT_MS` | `5000` |
| `TENTEXTER_LOG_LEVEL` | `INFO` |

The database URL must identify a file-backed SQLite database. Configuration can
be checked without creating the database:

```bash
uv run --locked ten-texter config check
```

## Database migrations

Migrations are the only supported schema installation path:

```bash
uv run --locked ten-texter db upgrade
uv run --locked ten-texter db current
```

Database bootstrap establishes and verifies WAL mode. Foreign-key enforcement
and the configured busy timeout are applied and verified for every connection.
The Phase 1 migration head is an empty baseline; v1 domain tables belong to
Phase 2.
