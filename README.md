# tenTexter (v1)

Local-first messaging coordination agent for arranging activities across Beeper-connected conversations, with a dedicated Telegram owner control plane.

## Status

The frozen v1 architecture is implemented as a Python modular monolith with SQLite persistence, Alembic migrations, deterministic domain services, durable workers, policy enforcement, model boundaries, and fake-driven end-to-end recovery tests.

## Local setup

Python 3.12 or newer and `uv` are recommended:

```bash
uv sync --extra dev
cp .env.example .env
```

Environment variables are read directly by the process; load `.env` with your shell or service manager. At minimum, configure `TEN_TEXTER_DATABASE_URL`, `TEN_TEXTER_OWNER_ID`, and `TEN_TEXTER_OWNER_CHAT_ID`. Real sends additionally require both transport credentials and `TEN_TEXTER_REAL_TRANSPORTS_ENABLED=true`. The checked-in example contains placeholders only.

Initialize and verify the application database:

```bash
uv run ten-texter migrate
uv run ten-texter check
```

Run the complete agent after Telegram, Beeper Desktop, and both model servers are available:

```bash
uv run ten-texter run
```

Use `uv run ten-texter run --once` for one complete polling pass. The runtime polls and processes Telegram owner updates and Beeper inbound revisions, advances recurrence and triggers, sweeps task lifecycle deadlines, records dependency health transitions, and drains every Outbox recovery state. It refuses to start unless real transports are explicitly enabled.

`uv run ten-texter worker` remains available when only the outbound delivery pump is needed. It reclaims expired `SENDING` leases, reconciles uncertain deliveries, and processes new `PENDING` sends. Every participant and owner send is freshly policy-checked with destination-aware facts, independently validated against concrete allowed claims, and then passed to the configured adapter.

## External services

- Telegram uses Bot API long polling for owner updates and `sendMessage` only through Outbox. Create a bot with BotFather, set the owner’s numeric user/chat IDs, and provide the token.
- Beeper uses the local Desktop REST v1 API at `http://127.0.0.1:23373` by default. Enable Desktop API access, provide its token, and keep Beeper Desktop running. Chats remain distinct; tenTexter never assumes merged conversations.
- The primary and validator model servers are independent llama.cpp servers. Configure each server's base URL (or its full `/v1/chat/completions` URL); tenTexter sends non-streaming OpenAI-compatible chat requests with schema-constrained JSON output and then strictly validates the returned `choices[0].message.content`. Tests replace both with fakes.

The Beeper adapter follows the documented v1 chat/message endpoints. A successful send request returns a pending message ID and therefore enters reconciliation until a final successful provider message is observed. WebSocket delivery remains optional/experimental; provider event ingestion is exposed through the adapter’s deterministic sync/ingestion boundary.

## Tests

```bash
uv run pytest
```

Tests use temporary SQLite databases plus fake transports/model clients. They do not require network access or send real messages.
