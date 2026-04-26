# JARVIS Project

## Vision

JARVIS (Just A Rather Very Intelligent System) is a Claude-powered AI research agent inspired by Iron Man. It has two purposes:

1. **Research tool** — autonomously search and synthesize information on how frontier AI models (transformers, RLHF, constitutional AI, multimodal systems) work
2. **Learning codebase** — a clean, readable example of a real agentic system built with the Anthropic SDK

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11+ |
| Package manager | `uv` |
| AI SDK | `anthropic>=0.40.0` |
| Default model | `claude-sonnet-4-6` |
| Web search | `tavily-python` |
| Terminal UI | `rich` |
| Config | `python-dotenv` + `pydantic` |
| Vector memory | `chromadb` |
| API server | `fastapi` + `uvicorn` |
| Task queue | `pika` (RabbitMQ) |
| Scheduler | `apscheduler` |
| Tracing | `opentelemetry` → Grafana Tempo |
| Tests | `pytest` |

## Project Structure

```
src/jarvis/
├── config.py              # All settings from environment (single source of truth)
├── main.py                # CLI entrypoint — arg parsing + wiring only
│
├── agents/
│   ├── base_agent.py      # Generic agentic loop: tool dispatch, OTel spans, failure logging
│   ├── researcher.py      # ResearcherAgent — JARVIS persona + conversation REPL
│   └── planner.py         # PlannerAgent — orchestrates sub-agents via delegate_task
│
├── tools/
│   ├── registry.py        # Central tool schema list + dispatch map (source of truth)
│   ├── web_search.py      # Tavily search
│   ├── report_writer.py   # Save/update research reports to reports/
│   ├── memory.py          # Hybrid BM25+semantic search (ChromaDB + forgetting curve)
│   ├── url_reader.py      # Fetch and parse web pages
│   ├── browser.py         # Playwright browser automation
│   ├── os_command.py      # Allowlisted shell commands
│   ├── delegation.py      # delegate_task tool for PlannerAgent sub-agents
│   ├── conversation_export.py
│   ├── plugin_loader.py   # Auto-discovers tools from tools/plugins/
│   └── plugins/           # Drop-in tool plugins (SCHEMA + handle())
│       ├── example_weather.py
│       ├── tool_generator.py    # Scaffold new plugin files (self-improvement)
│       ├── filesystem_search.py # Glob + content search in local files
│       ├── git_context.py       # git log/diff/status on any repo
│       ├── local_model.py       # Ollama local model integration
│       └── document_ingestion.py # PDF/DOCX/TXT → ChromaDB index
│
├── memory/
│   ├── episodic.py        # Timestamped conversation log (SQLite + FTS5), user_id isolated
│   ├── graph.py           # Entity-relationship knowledge graph (SQLite), user_id namespaced
│   ├── failures.py        # Tool error pattern analysis (SQLite)
│   └── feedback.py        # User ratings on responses (SQLite), record_feedback tool
│
├── models/
│   └── router.py          # ModelRouter: smart Haiku/Sonnet routing, drop-in for Anthropic client
│
├── api/
│   ├── server.py          # FastAPI: HTTP chat, WebSocket streaming, schedules, auth, budget
│   ├── models.py          # Pydantic request/response models
│   ├── metrics.py         # Prometheus counters/histograms
│   └── budget.py          # Per-user monthly USD spend tracking + BudgetExceededError
│
├── scheduler/
│   └── core.py            # APScheduler: research + monitor cron jobs, SQLite job store
│
├── queue/
│   └── worker.py          # RabbitMQ consumer — runs agent turns asynchronously
│
├── auth/
│   └── core.py            # JWT tokens, PBKDF2 passwords, roles (admin/user/readonly)
│
├── telemetry/
│   └── tracing.py         # OTel spans for agent turns + tool calls; _NoopTracer fallback
│
├── evals/
│   ├── suite.py           # EvalCase dataclass + built-in baseline suite
│   ├── runner.py          # run_suite(), summarize(), Claude-as-judge scoring
│   └── main.py            # jarvis-eval CLI entrypoint
│
├── vision/
│   ├── capture.py         # YOLOv8 object detection (lazy import)
│   └── face.py            # OpenCV Haar cascade face detection (lazy import)
│
├── voice/
│   ├── stt.py             # Whisper speech-to-text
│   ├── tts.py             # pyttsx3 / ElevenLabs TTS
│   ├── wake_word.py       # Picovoice Porcupine wake-word (+ Enter fallback)
│   └── ambient.py         # Continuous wake-word loop + daily episodic briefing
│
├── edge/
│   ├── agent.py           # EdgeAgent: relay to cloud JARVIS via MQTT or offline
│   ├── mqtt_transport.py  # MQTTTransport: pub/sub with request-response correlation
│   ├── sync.py            # Knowledge graph delta export/import
│   └── main.py            # jarvis-edge CLI entrypoint
│
└── utils/
    └── console.py         # Rich terminal rendering helpers (no agent-specific imports)
```

## Dev Commands

```bash
# Install dependencies
uv sync

# Run the interactive JARVIS agent (PlannerAgent + all tools)
uv run jarvis

# Use ResearcherAgent directly
uv run jarvis --researcher

# Research a topic non-interactively
uv run jarvis --topic "RLHF in large language models"

# Ambient voice mode (wake-word activated)
uv run jarvis --ambient

# Run the API server
uv run jarvis-api

# Run the queue worker
uv run jarvis-worker

# Run the edge agent
uv run jarvis-edge --mqtt-host 10.0.0.5
uv run jarvis-edge --no-cloud    # offline mode

# Run eval suite
uv run jarvis-eval
uv run jarvis-eval --judge       # enable Claude-as-judge scoring
uv run jarvis-eval --tags ml     # filter by tag
uv run jarvis-eval --output results.json

# Run tests
uv run pytest
uv run pytest -v -s

# Docker Compose (full stack: API + worker + RabbitMQ + Prometheus + Grafana + Tempo)
docker compose up -d

# Kubernetes (full stack)
kubectl apply -k k8s/
```

## Architecture: The Agentic Loop

JARVIS uses a **recursive tool-dispatch loop**:

```
User input
  → append to messages
  → client.messages.create(model, system, tools, messages)
  → if stop_reason == "end_turn":   render text, wait for next input
  → if stop_reason == "tool_use":
        for each tool_use block:
          OTel span start → dispatch → log failure if ERROR → span end
        append assistant turn + tool_results to messages
        call API again  ← (repeat until end_turn)
```

`ModelRouter` sits between `BaseAgent` and the Anthropic client. With `strategy=smart`, it automatically uses the fast model (Haiku) for tool-dispatch turns and the primary model (Sonnet) for synthesis turns — transparent to the agent.

## Coding Rules

### Tools
- **Tools must never raise exceptions.** Wrap all handlers in try/except and return `"ERROR: {str(e)}"` — Claude will handle it gracefully in the next turn.
- Tool input/output must be typed. Use `dataclasses` or Pydantic models for inputs.
- Tool schemas in `registry.py` are the **source of truth** — written as Python dicts, not generated from annotations (keeps them explicit and readable).
- **Plugins** live in `tools/plugins/`. Each plugin must export `SCHEMA: dict` and `handle(tool_input: dict) -> str`. They are auto-discovered on startup.

### API Calls
- **Always use prompt caching** on system prompts:
  ```python
  system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]
  ```
- Never hardcode API keys — always read from environment via `config.py`.
- Model is defined once in `config.py` as `MODEL`. Never hardcode `"claude-sonnet-4-6"` elsewhere.

### Structure
- `main.py` is a thin CLI shim — argument parsing and wiring only.
- No business logic in `utils/` — only pure helpers with no agent-specific imports.
- `agents/` imports from `tools/` and `utils/`, never the reverse.
- All new SQLite tables live in `reports_dir/jarvis.db` — use `CREATE TABLE IF NOT EXISTS`.

### Style
- No unnecessary comments — code should be self-explanatory.
- Keep functions short and single-purpose.
- Prefer explicit over clever.

## Environment Variables

Copy `.env.example` to `.env` and fill in your keys:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | Yes | — | Anthropic API key |
| `TAVILY_API_KEY` | Yes | — | Tavily search API key |
| `JARVIS_MODEL` | No | `claude-sonnet-4-6` | Primary Claude model |
| `JARVIS_MAX_TOKENS` | No | `8096` | Max tokens per response |
| `JARVIS_REPORTS_DIR` | No | `reports` | Reports + DB directory |
| `JARVIS_MAX_SEARCH_CALLS` | No | `20` | Web searches per session |
| `JARVIS_ROUTING_STRATEGY` | No | `always_primary` | `always_primary` or `smart` |
| `JARVIS_FAST_MODEL` | No | `claude-haiku-4-5-20251001` | Fast model for tool-dispatch turns |
| `OLLAMA_BASE_URL` | No | `http://localhost:11434` | Ollama API base URL |
| `OLLAMA_MODEL` | No | `llama3.2` | Default Ollama model |
| `JARVIS_AUTH_ENABLED` | No | `false` | Enable JWT auth on API |
| `JARVIS_JWT_SECRET` | No | `change-me-in-production` | JWT signing secret |
| `JARVIS_JWT_EXPIRE_MINUTES` | No | `1440` | Token expiry (24h) |
| `JARVIS_OTEL_ENABLED` | No | `false` | Enable OpenTelemetry tracing |
| `OTL_EXPORTER_OTLP_ENDPOINT` | No | `http://localhost:4317` | Tempo OTLP gRPC endpoint |
| `PICOVOICE_ACCESS_KEY` | No | — | Wake-word detection (Porcupine) |
| `JARVIS_TTS_ENGINE` | No | `local` | `local` or `elevenlabs` |
| `JARVIS_WHISPER_MODEL` | No | `base` | Whisper model size |
| `ELEVENLABS_API_KEY` | No | — | ElevenLabs API key |
| `RABBITMQ_URL` | No | `amqp://guest:guest@localhost:5672/` | RabbitMQ connection |
| `JARVIS_SESSION_TTL_MINUTES` | No | `60` | Session eviction TTL |

## Research Topics JARVIS Knows

- **Transformer architecture** — attention, positional encoding, scaling laws
- **RLHF** — reward modeling, PPO in language models, preference datasets
- **Constitutional AI** — Anthropic's self-critique training, harmlessness
- **Multimodal systems** — vision encoders, cross-attention fusion, CLIP
- **Memory systems** — RAG, vector databases, long-context strategies
- **Voice synthesis** — TTS architectures, voice cloning
- **Model evaluation** — benchmarks, evals methodology, red-teaming
