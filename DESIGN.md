# JARVIS System Design Document

This document is the authoritative reference for understanding, extending, and maintaining the JARVIS project. Read this before making any architectural change.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Project Layout](#2-project-layout)
3. [Core Architecture — The Agentic Loop](#3-core-architecture--the-agentic-loop)
4. [Agent Hierarchy](#4-agent-hierarchy)
5. [Tool System](#5-tool-system)
6. [Memory & Persistence](#6-memory--persistence)
7. [Voice Interface](#7-voice-interface)
8. [Production API (FastAPI)](#8-production-api-fastapi)
9. [Async Task Queue (RabbitMQ)](#9-async-task-queue-rabbitmq)
10. [Observability](#10-observability)
11. [Configuration Reference](#11-configuration-reference)
12. [Dependency Map](#12-dependency-map)
13. [Infrastructure (Docker)](#13-infrastructure-docker)
14. [Extension Guide](#14-extension-guide)
15. [Known Constraints & Design Decisions](#15-known-constraints--design-decisions)

---

## 1. System Overview

JARVIS is a production-grade agentic AI system powered by Claude (Anthropic). It is **not** a chatbot wrapper — it is a tool-calling research agent that can search the web, remember context, run OS commands, automate browsers, delegate to sub-agents, and expose its capabilities over HTTP/WebSocket with full observability.

```
             ┌──────────────────────────────────────────┐
             │               Entry Points                │
             │   CLI (main.py)   │   API (server.py)    │
             └──────────┬────────┴──────────┬───────────┘
                        │                   │
             ┌──────────▼───────────────────▼───────────┐
             │            Agent Layer                    │
             │  PlannerAgent → ResearcherAgent / Coder  │
             │                  / QA (sub-agents)        │
             └──────────┬────────────────────────────────┘
                        │ tool dispatch
             ┌──────────▼────────────────────────────────┐
             │             Tool Layer                     │
             │  web_search │ browser │ memory │ os_cmd   │
             │  report_writer │ url_reader │ delegation  │
             └───────────────────────────────────────────┘
                        │
             ┌──────────▼────────────────────────────────┐
             │          Infrastructure                     │
             │  RabbitMQ  │  Prometheus  │  ChromaDB     │
             │  Grafana   │  structlog                   │
             └───────────────────────────────────────────┘
```

---

## 2. Project Layout

```
jarvis-project/
├── src/jarvis/
│   ├── config.py                  # All settings (pydantic-settings + env vars)
│   ├── main.py                    # CLI entrypoint — thin shim, no logic
│   │
│   ├── agents/
│   │   ├── base_agent.py          # Abstract loop: token tracking, streaming, tool dispatch
│   │   ├── researcher.py          # Researcher persona, search-quota enforcement
│   │   ├── planner.py             # Orchestrator — delegates to sub-agents
│   │   ├── coder.py               # Python/AI coder sub-agent
│   │   └── qa.py                  # Code reviewer sub-agent
│   │
│   ├── tools/
│   │   ├── registry.py            # Tool catalog + dispatch map factory
│   │   ├── web_search.py          # Tavily web search
│   │   ├── report_writer.py       # Save / update markdown reports
│   │   ├── url_reader.py          # Fetch + clean arbitrary URLs / arXiv papers
│   │   ├── conversation_export.py # Export conversation history to JSON
│   │   ├── memory.py              # ChromaDB semantic memory (search + index)
│   │   ├── os_command.py          # Sandboxed OS command runner
│   │   ├── browser.py             # Playwright browser automation
│   │   └── delegation.py          # Spawn sub-agents as tools
│   │
│   ├── prompts/
│   │   ├── loader.py              # load_prompt(name, **vars) reads .md templates
│   │   ├── researcher.md          # JARVIS researcher system prompt
│   │   ├── planner.md             # Planner/orchestrator system prompt
│   │   ├── coder.md               # Python AI coder system prompt
│   │   └── qa.md                  # Code reviewer system prompt
│   │
│   ├── voice/
│   │   ├── stt.py                 # Whisper speech-to-text (PyAudio mic input)
│   │   └── tts.py                 # pyttsx3 (local) / ElevenLabs (cloud) TTS
│   │
│   ├── api/
│   │   ├── server.py              # FastAPI: POST /api/chat, WS /api/ws/{session}
│   │   ├── metrics.py             # Prometheus counters / gauges / histograms
│   │   └── models.py              # Pydantic request/response models
│   │
│   ├── queue/
│   │   ├── producer.py            # Publish task to RabbitMQ
│   │   ├── consumer.py            # Process a single task (build agent, run turn)
│   │   └── worker.py              # Long-running consumer process with signal handling
│   │
│   └── utils/
│       └── console.py             # Rich terminal rendering helpers (pure, no agent imports)
│
├── tests/
│   ├── test_tools.py
│   └── test_researcher.py
│
├── Dockerfile                     # python:3.11-slim, uv, non-root jarvis user
├── docker-compose.yml             # jarvis-api, jarvis-worker, rabbitmq, prometheus, grafana
├── prometheus.yml                 # Scrape config: jarvis-api:8000/metrics every 15s
├── grafana/provisioning/          # Auto-provision Prometheus datasource + JARVIS dashboard
│   ├── datasources/prometheus.yml
│   └── dashboards/
│       ├── dashboard.yml
│       └── jarvis.json            # 10-panel Grafana dashboard
│
├── pyproject.toml                 # Dependencies, scripts, pytest config
├── .env.example                   # Template — copy to .env and fill in keys
├── CLAUDE.md                      # Coding rules and conventions (read by Claude Code)
└── DESIGN.md                      # ← this file
```

---

## 3. Core Architecture — The Agentic Loop

Everything flows through `BaseAgent.run_turn()`. This is the most important method in the codebase.

### Flow Diagram

```
run_turn(messages, on_chunk=None)
  │
  ├─ _make_api_params(messages)
  │    └─ system prompt with cache_control: ephemeral
  │
  ├─ if on_chunk:  messages.stream()  → on_chunk(chunk) per text piece
  │  else:         messages.create()
  │
  ├─ _accumulate_usage(response.usage)
  │
  ├─ stop_reason == "end_turn"
  │    └─ extract text → return (text, updated_messages)
  │
  └─ stop_reason == "tool_use"
       ├─ for each tool_use block:
       │    ├─ _before_dispatch(name, input)   ← override hook
       │    └─ _dispatch(name, input)          → handler(input) → str
       ├─ append assistant turn + tool_results to messages
       └─ recurse: run_turn(updated, on_chunk)  ← handles chains
```

### Key Properties

- **Recursive** — tool chains resolve naturally; no explicit loop needed.
- **Stateless** — messages list is passed in and a new list is returned. The agent never mutates shared state.
- **Streaming-compatible** — `on_chunk` callback works through the entire recursive chain.
- **Token-tracked** — `_accumulate_usage()` accumulates across all recursive calls including cache hits.

### Prompt Caching

Every API call wraps the system prompt with `cache_control: {"type": "ephemeral"}`. This tells Anthropic to cache the prompt prefix, saving ~90% of input token cost for repeated turns on the same session.

```python
system=[{"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}}]
```

---

## 4. Agent Hierarchy

```
BaseAgent (abstract)
  ├── ResearcherAgent     — JARVIS research persona, search-quota tracking
  ├── PlannerAgent        — Orchestrator, has delegate_task tool
  ├── CoderAgent          — Python/AI code generation sub-agent
  └── QAAgent             — Code review sub-agent
```

### ResearcherAgent

- Tracks `_search_calls_used` / `_max_search_calls`, injects quota note into system prompt each call.
- Overrides `_before_dispatch()` to call `on_tool_call()` for quota accounting.
- `get_messages()` exposes `_messages` so the `export_conversation` tool can access them.
- `run_conversation()` drives the interactive REPL loop.

### PlannerAgent

- Default agent mode in CLI and API.
- Uses the planner system prompt (`prompts/planner.md`).
- Has `delegate_task` in its tool registry (from `build_planner_registry()`).
- Sub-agents spawned by `delegate_task` receive only the base registry — **no delegation tool** — preventing infinite recursion.

### CoderAgent / QAAgent

- Sub-agents only; they have no `run_conversation()` method.
- Spawned by `PlannerAgent` via the `delegate_task` tool.
- Use `prompts/coder.md` and `prompts/qa.md`.

### Adding a New Agent

1. Create `src/jarvis/agents/my_agent.py`, subclass `BaseAgent`.
2. Implement `get_system_prompt()` — use `load_prompt("my_agent")`.
3. Create `src/jarvis/prompts/my_agent.md`.
4. If it needs to be a delegatable sub-agent, add it to `delegation.py`'s `AGENT_REGISTRY`.

---

## 5. Tool System

### Registry Pattern

Tools are never imported by agents directly. Instead, `build_registry()` in `registry.py` is the single factory that:
1. Assembles a `schemas` list (tool definitions for the Anthropic API).
2. Returns a `registry` dict mapping `tool_name → handler(input) → str`.

Handlers are closures that capture config (API keys, paths) at construction time.

```python
schemas, registry = build_registry(
    tavily_api_key=...,
    reports_dir=...,
    get_messages=...,    # optional, for export_conversation
    allowed_commands=...,
)
```

### Tool Catalog

| Tool name | File | What it does |
|-----------|------|-------------|
| `web_search` | `web_search.py` | Tavily semantic web search → list of results |
| `save_report` | `report_writer.py` | Save markdown report + auto-index into ChromaDB |
| `update_report` | `report_writer.py` | Append to or replace section in existing report |
| `read_url` | `url_reader.py` | Fetch URL → cleaned text (handles arXiv abs redirect) |
| `export_conversation` | `conversation_export.py` | Dump messages to JSON file |
| `search_memory` | `memory.py` | ChromaDB semantic search over past reports |
| `run_command` | `os_command.py` | Run OS command (allowlist-sandboxed) |
| `browse` | `browser.py` | Playwright: navigate / click / type / get_text / screenshot |
| `delegate_task` | `delegation.py` | Spawn a sub-agent (researcher/coder/qa) with a task |

### Tool Contract

- **Never raise exceptions.** Every handler wraps in `try/except` and returns `"ERROR: {msg}"`.
- Input/output are typed (dataclasses or Pydantic).
- Schemas in `registry.py` are the source of truth — written as explicit Python dicts.

### Planner Registry

`build_planner_registry()` layers `delegate_task` on top of the base registry:

```python
planner_schemas, planner_registry = build_planner_registry(
    base_schemas=base_schemas,
    base_registry=base_registry,
    client=client, model=model, max_tokens=max_tokens,
)
```

The `build_delegation_handler()` factory captures `base_schemas` and `base_registry` in the closure — sub-agents never see `delegate_task`, preventing recursive loops.

---

## 6. Memory & Persistence

### Short-term Memory

The `messages: list[dict]` passed into `run_turn()` is the conversation context. `ResearcherAgent.run_conversation()` keeps it in `self._messages` and trims to the last 20 turns when it exceeds 40 entries.

### Long-term Memory (ChromaDB)

`src/jarvis/tools/memory.py` provides a persistent semantic index over all saved reports.

- **Storage**: `reports/.chroma/` directory (PersistentClient).
- **Collection**: `jarvis_reports`, cosine distance.
- **Indexing**: `index_new_report(reports_dir, filename)` — called automatically by `_save_report_and_index()` wrapper in `registry.py` whenever a report is saved.
- **Query**: `handle_search_memory(input, reports_dir)` — returns top-k matching chunks with metadata (filename, source).

### Reports Directory

`reports/` is the shared volume in Docker. Reports are markdown files. ChromaDB index lives at `reports/.chroma/`.

---

## 7. Voice Interface

Optional feature — requires installing the `[voice]` or `[voice-cloud]` extras.

```bash
uv sync --extra voice         # pyttsx3 (local, offline)
uv sync --extra voice-cloud   # ElevenLabs (realistic, costs per char)
```

### STT — `voice/stt.py`

1. Open PyAudio stream (16 kHz, mono).
2. Capture frames until RMS amplitude drops below threshold (silence detection).
3. Write WAV buffer to memory.
4. Pass to `whisper.load_model(config.whisper_model).transcribe()`.
5. Return transcript string.

### TTS — `voice/tts.py`

- `_clean_text()` strips markdown before speaking.
- `speak(text)` dispatches based on `JARVIS_TTS_ENGINE`:
  - `"local"` → pyttsx3 (offline, robotic but free)
  - `"elevenlabs"` → ElevenLabs `eleven_turbo_v2_5`, streams audio

### Voice Loop (`main.py`)

```
_run_voice(agent):
  loop:
    record_and_transcribe()  →  user_text
    agent.run_turn(messages, on_chunk=None)
    speak(response_text)
```

Enable with `--voice` flag: `uv run python -m jarvis.main --voice`

---

## 8. Production API (FastAPI)

`src/jarvis/api/server.py` exposes JARVIS over HTTP and WebSocket.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Returns `{"status": "ok", "sessions_active": N}` |
| `GET` | `/metrics` | Prometheus text metrics |
| `POST` | `/api/chat` | Synchronous single-turn chat (blocks until reply) |
| `WS` | `/api/ws/{session_id}` | Streaming WebSocket — session persists across reconnects |

### Session Store

`_sessions: dict[str, dict]` maps `session_id → {agent, messages, created_at}`.

Sessions are in-memory (not persisted across restarts). TTL cleanup is not yet implemented — see `JARVIS_SESSION_TTL_MINUTES` config for future use.

### Async/Sync Bridge

The Anthropic SDK is synchronous. FastAPI is async. The bridge pattern:

```
WebSocket handler (async coroutine)
  │
  ├─ asyncio.Queue  ← thread-safe bridge
  │
  ├─ ThreadPoolExecutor.submit(run_turn)
  │    └─ on_chunk(text) → loop.call_soon_threadsafe(queue.put_nowait, WsChunk(...))
  │    └─ on_tool_event(name) → loop.call_soon_threadsafe(queue.put_nowait, WsToolCall(...))
  │    └─ finally: queue.put_nowait(None)  ← sentinel
  │
  └─ async while: msg = await queue.get() → websocket.send_json(msg)
       until sentinel → send WsDone
```

### WebSocket Message Protocol

| Type | Direction | Fields |
|------|-----------|--------|
| `WsIncoming` | Client → Server | `message`, `researcher_mode` |
| `WsThinking` | Server → Client | _(signal)_ |
| `WsChunk` | Server → Client | `text` — streaming fragment |
| `WsToolCall` | Server → Client | `tool` — tool name being called |
| `WsDone` | Server → Client | `text`, `usage` — final reply + token stats |
| `WsError` | Server → Client | `message` — error description |

---

## 9. Async Task Queue (RabbitMQ)

For fire-and-forget / background task processing.

### Flow

```
Client → POST /api/chat (or custom publisher)
  └─ producer.publish_task(QueueTask)
       └─ RabbitMQ queue: jarvis.tasks
            └─ worker._on_message() → consumer.process_task(QueueTask)
                 └─ builds fresh agent → agent.run_turn(messages)
                 └─ publishes QueueResult to jarvis.results
```

### Queue Config

- Queue name: `RABBITMQ_TASK_QUEUE` (default `jarvis.tasks`)
- Durable: yes (survives broker restarts)
- `prefetch_count=1` — worker processes one task at a time
- On success: `basic_ack(delivery_tag)`
- On failure: `basic_nack(delivery_tag, requeue=False)` — message dropped, not re-queued

### Worker

Start with: `uv run jarvis-worker` or `python -m jarvis.queue.worker`

Handles `SIGTERM`/`SIGINT` gracefully (stops consuming, closes connection).

---

## 10. Observability

### Logging — structlog

All log output is structured JSON. Format:

```json
{"event": "chat_complete", "session_id": "abc", "duration_s": 1.23, "level": "info", "timestamp": "2026-04-12T..."}
```

Add logging anywhere:
```python
import structlog
log = structlog.get_logger()
log.info("my_event", key="value")
```

### Metrics — Prometheus

Defined in `src/jarvis/api/metrics.py`:

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `jarvis_requests_total` | Counter | `mode` (http/websocket) | Total requests |
| `jarvis_request_duration_seconds` | Histogram | `mode` | Request latency |
| `jarvis_active_websocket_connections` | Gauge | — | Live WS connections |
| `jarvis_tokens_total` | Counter | `token_type` (input/output/cache_read/cache_write) | Token usage |
| `jarvis_cost_usd_total` | Counter | — | Estimated cost in USD |
| `jarvis_tool_calls_total` | Counter | `tool_name` | Tool invocation count |
| `jarvis_queue_tasks_published_total` | Counter | — | Tasks sent to RabbitMQ |
| `jarvis_queue_tasks_processed_total` | Counter | `status` (success/error) | Tasks processed by worker |

Metrics are exposed at `GET /metrics` (Prometheus text format).

### Grafana Dashboard

Pre-built 10-panel dashboard at `grafana/provisioning/dashboards/jarvis.json`. Auto-provisioned on `docker compose up`.

Access: `http://localhost:3000` (admin / admin)

Panels: Total Requests, Active WS Connections, Estimated Cost, Input Tokens, Requests/min (timeseries), Request p95 latency, Tool Calls by Type, Token Usage Over Time, Queue Tasks Processed, Queue Errors.

---

## 11. Configuration Reference

All settings live in `src/jarvis/config.py` as a `pydantic-settings BaseSettings` class.

| Env Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | _(required)_ | Anthropic API key |
| `TAVILY_API_KEY` | _(required)_ | Tavily search API key |
| `JARVIS_MODEL` | `claude-sonnet-4-6` | LLM model identifier |
| `JARVIS_MAX_TOKENS` | `8096` | Max tokens per API response |
| `JARVIS_REPORTS_DIR` | `reports` | Path to reports + ChromaDB storage |
| `JARVIS_MAX_SEARCH_CALLS` | `20` | Max Tavily calls per ResearcherAgent session |
| `JARVIS_ALLOWED_COMMANDS` | `ls,dir,cat,echo,python,python3,git,pwd,whoami,date` | Allowlist for `run_command` tool |
| `JARVIS_TTS_ENGINE` | `local` | TTS engine: `local` (pyttsx3) or `elevenlabs` |
| `JARVIS_WHISPER_MODEL` | `base` | Whisper model size: `tiny/base/small/medium/large` |
| `ELEVENLABS_API_KEY` | — | Required if `JARVIS_TTS_ENGINE=elevenlabs` |
| `JARVIS_ELEVENLABS_VOICE` | `Rachel` | ElevenLabs voice name or ID |
| `JARVIS_API_HOST` | `0.0.0.0` | FastAPI bind host |
| `JARVIS_API_PORT` | `8000` | FastAPI bind port |
| `JARVIS_SESSION_TTL_MINUTES` | `60` | Session TTL (cleanup not yet implemented) |
| `RABBITMQ_URL` | `amqp://guest:guest@localhost:5672/` | RabbitMQ connection URL |
| `RABBITMQ_TASK_QUEUE` | `jarvis.tasks` | Queue name for async tasks |

---

## 12. Dependency Map

Module-level import rules (enforced by convention, not tooling):

```
main.py
  → agents/  (builds agents, wires registries)
  → voice/   (STT/TTS, only if --voice)

agents/
  → tools/registry.py    (gets schemas + registry)
  → prompts/loader.py    (gets system prompt text)
  → anthropic SDK

tools/
  → external libs only   (tavily, chromadb, playwright, httpx, etc.)
  → NO agents/ imports   (would be circular)

utils/console.py
  → rich only
  → NO agents/ or tools/ imports (pure rendering helpers)

api/
  → agents/
  → tools/registry.py
  → api/metrics.py, models.py

queue/
  → tools/registry.py
  → agents/
  → api/metrics.py
```

**Rule**: `agents/` imports `tools/` and `utils/`. `tools/` never imports `agents/`. `utils/` imports nothing from this project.

---

## 13. Infrastructure (Docker)

### Services

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| `jarvis-api` | `./Dockerfile` | 8000 | FastAPI server |
| `jarvis-worker` | `./Dockerfile` | — | RabbitMQ task consumer |
| `rabbitmq` | `rabbitmq:3.13-management-alpine` | 5672, 15672 | Message broker |
| `prometheus` | `prom/prometheus:v2.52.0` | 9090 | Metrics scraper |
| `grafana` | `grafana/grafana:10.4.2` | 3000 | Dashboards |

### Shared Volume

`reports` named volume is mounted into both `jarvis-api` and `jarvis-worker` at `/app/reports`. This ensures ChromaDB index and saved reports are shared across both containers.

### Quickstart

```bash
cp .env.example .env        # fill in API keys
docker compose up -d        # start all services
curl localhost:8000/api/health
open http://localhost:3000  # Grafana (admin/admin)
open http://localhost:15672 # RabbitMQ Management (guest/guest)
```

### Dockerfile Notes

- Base: `python:3.11-slim`
- Package manager: `uv` (fast installs)
- Non-root user: `jarvis` (uid 1000)
- Entrypoint: `jarvis-api` script (maps to `jarvis.api.server:main`)

---

## 14. Extension Guide

### Add a New Tool

1. Create `src/jarvis/tools/my_tool.py`:
   ```python
   SCHEMA = {"name": "my_tool", "description": "...", "input_schema": {...}}

   def handle_my_tool(inp: dict) -> str:
       try:
           # implementation
           return result_string
       except Exception as e:
           return f"ERROR: {e}"
   ```
2. Import and wire in `registry.py`'s `build_registry()`:
   ```python
   from jarvis.tools import my_tool
   schemas.append(my_tool.SCHEMA)
   registry["my_tool"] = lambda inp: my_tool.handle_my_tool(inp)
   ```

### Add a New Agent Type for Delegation

1. Create agent file + system prompt (see [Agent Hierarchy](#4-agent-hierarchy)).
2. Add to `delegation.py`'s `AGENT_REGISTRY` dict.
3. Add the new type to the `delegate_task` schema's `agent_type` enum.

### Add a New API Endpoint

1. Add Pydantic models to `api/models.py`.
2. Add route handler to `api/server.py`.
3. Add relevant Prometheus metrics to `api/metrics.py` if needed.

### Add a New Metric

1. Define in `api/metrics.py` using `prometheus_client` (`Counter`, `Gauge`, `Histogram`).
2. Instrument the relevant code path.
3. Optionally add a panel to `grafana/provisioning/dashboards/jarvis.json`.

### Add a System Prompt Template

1. Create `src/jarvis/prompts/my_name.md` with the prompt text.
2. Use `{variable}` syntax for substitution (optional).
3. Load with `load_prompt("my_name", variable="value")`.

---

## 15. Known Constraints & Design Decisions

### Synchronous Anthropic SDK

The `anthropic` Python SDK is synchronous. All agent logic runs synchronously. The FastAPI layer bridges this with `ThreadPoolExecutor` + `asyncio.Queue`. This is intentional — simpler and sufficient for current load. If needed, switch to `AsyncAnthropic` client and rewrite `run_turn` as `async def`.

### In-memory Session Store

`_sessions` in `server.py` is a plain dict. Sessions are lost on restart. For multi-instance deployments, replace with Redis. The `JARVIS_SESSION_TTL_MINUTES` config exists but TTL eviction is not yet implemented.

### ChromaDB Not Pre-indexed

The vector index is not pre-warmed on startup. The first `search_memory` call after writing a report is the first time those chunks appear. Reports existing before the ChromaDB collection was created won't be indexed until a new report is saved to the same directory. To re-index manually, delete `reports/.chroma/` and call `memory.index_new_report()` for each existing report.

### No Tool Output Size Limit

Tool handlers can return arbitrarily long strings. Long browser page dumps or large file reads can bloat the context window. Consider truncating at the tool layer if this becomes a problem.

### Delegation Depth = 1

Sub-agents spawned by `delegate_task` do **not** have the `delegate_task` tool themselves. This is deliberate — it prevents runaway recursion and keeps execution predictable. If you need deeper delegation, refactor with explicit depth tracking.

### Playwright is Sync

`browser.py` uses the synchronous Playwright API (not `async_playwright`). The singleton browser/page pattern means concurrent browser calls from multiple sessions would serialize. For production multi-user browser automation, use one page per session or switch to async Playwright.

### Cost Tracking is Estimated

Token costs in `get_usage_summary()` use hardcoded rates for `claude-sonnet-4-6` as of early 2026:
- Input: $3.00 / 1M tokens
- Output: $15.00 / 1M tokens
- Cache write: $3.75 / 1M tokens
- Cache read: $0.30 / 1M tokens

Update these in `base_agent.py:get_usage_summary()` if Anthropic changes pricing.

### Voice Requires Platform Audio Support

`pyaudio` requires PortAudio system library. In Docker this would need `libportaudio2`. Voice mode is not included in the default Docker image — it's a local-only feature.
