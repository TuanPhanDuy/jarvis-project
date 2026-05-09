# JARVIS Architecture

## Overview

JARVIS is a **fully local, multi-agent AI system** powered by Ollama. No cloud APIs, no API keys, no external dependencies. Everything runs on your machine.

```
User ──→ CLI / API Server ──→ PlannerAgent ──→ Tools + Memory
                                    │
                         ┌──────────┼──────────┐
                         ▼          ▼          ▼
                    Researcher    Coder        QA
                    Sub-agent   Sub-agent   Sub-agent
                         │          │          │
                         └──────────┴──────────┘
                                    │
                               Ollama LLM
                           (qwen2.5:14b local)
```

---

## System Components

### 1. Local LLM (Ollama)
- **Model**: `qwen2.5:14b` (default) — runs on your M5 Pro, no internet required
- **Vision model**: `llava:13b` — for camera/image analysis
- **Interface**: `ollama.chat()` with native tool/function calling
- **Cost**: $0.00 — free, unlimited, private

### 2. Agent Layer

```
┌─────────────────────────────────────────────────────────────┐
│                       PlannerAgent                          │
│  The primary brain. Routes requests, orchestrates work,     │
│  synthesizes results. Has access to delegate_task and       │
│  create_plan to coordinate specialist sub-agents.           │
│                                                             │
│  System prompt: prompts/planner.md                          │
└──────────────────┬──────────────────────────────────────────┘
                   │  delegates via delegate_task / create_plan
       ┌───────────┼───────────┐
       ▼           ▼           ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│Researcher│ │  Coder   │ │   QA     │
│  Agent   │ │  Agent   │ │  Agent   │
│          │ │          │ │          │
│Web search│ │Write code│ │Review    │
│Synthesis │ │Run code  │ │Test code │
│Reports   │ │Debug     │ │Audit     │
└──────────┘ └──────────┘ └──────────┘
       │           │           │
       └───────────┴───────────┘
                   │
            Shared Tool Registry
```

**Sub-agents are isolated** — they have the base tool set but NOT `delegate_task`, preventing infinite recursion.

### 3. Agentic Loop (BaseAgent)

Every agent follows the same recursive loop:

```
┌─────────────────────────────────────────────────────────────┐
│                    Agentic Loop                             │
│                                                             │
│  messages = [system_prompt] + conversation_history          │
│                    │                                        │
│                    ▼                                        │
│          ollama.chat(model, messages, tools)                │
│                    │                                        │
│         ┌──────────┴──────────┐                            │
│         ▼                     ▼                            │
│    text response         tool_calls                        │
│         │                     │                            │
│         ▼                     ▼                            │
│    return to user     for each tool_call:                  │
│                         dispatch(name, args)               │
│                               │                            │
│                               ▼                            │
│                       append tool result                   │
│                       to messages                          │
│                               │                            │
│                               └──→ loop back               │
└─────────────────────────────────────────────────────────────┘
```

### 4. Tool System

27 built-in tools + auto-discovered plugins:

| Category | Tools |
|----------|-------|
| **Search** | `web_search` (DuckDuckGo, free) |
| **Content** | `read_url`, `browse` (Playwright) |
| **Files** | `save_report`, `update_report`, `filesystem_search` |
| **Code** | `run_command`, `execute_python` |
| **Memory** | `search_memory`, `search_episodic_memory`, `update_knowledge_graph`, `query_knowledge_graph` |
| **Vision** | `capture_camera` (YOLO), `describe_scene` (Ollama vision), `recognize_face` |
| **Agents** | `delegate_task`, `create_plan` (Planner only) |
| **System** | `git_context`, `ingest_document`, `record_feedback` |
| **Plugins** | Auto-discovered from `tools/plugins/` |

### 5. Memory System

Four layers of memory, all stored locally in SQLite + ChromaDB:

```
┌─────────────────────────────────────────────┐
│              Memory Layers                  │
│                                             │
│  [1] Episodic Memory (SQLite + FTS5)        │
│      Timestamped conversation log           │
│      Searchable by keyword or time          │
│                                             │
│  [2] Knowledge Graph (SQLite)               │
│      Entity-relationship store              │
│      Who/what/when connections              │
│                                             │
│  [3] Vector Memory (ChromaDB)               │
│      Semantic search over reports           │
│      BM25 + embedding hybrid search         │
│                                             │
│  [4] User Preferences (SQLite)              │
│      Persistent per-user settings           │
│      Loaded at session start                │
└─────────────────────────────────────────────┘
```

---

## Rule Flow

### PlannerAgent Decision Rules

```
User Message
      │
      ▼
┌─────────────────────────────────────────────────┐
│  Is it a greeting or simple factual question?   │
│  e.g. "hello", "what is 2+2", "what time is it"│
└──────────────┬───────────────────────┬──────────┘
               │ YES                   │ NO
               ▼                       ▼
        Answer directly         ┌──────────────────────────────┐
                                │  Does it need web research?  │
                                │  e.g. "find info on X",      │
                                │  "how does Y work",          │
                                │  "latest news on Z"          │
                                └──────┬──────────────┬────────┘
                                       │ YES          │ NO
                                       ▼              ▼
                               delegate: researcher   ┌──────────────────────────┐
                                                      │  Does it need code?      │
                                                      │  e.g. "write X", "build  │
                                                      │  Y", "implement Z"       │
                                                      └──────┬───────────┬───────┘
                                                             │ YES       │ NO
                                                             ▼           ▼
                                                     delegate: coder   ┌────────────────────────┐
                                                                       │  Does it need review?  │
                                                                       │  e.g. "check this",    │
                                                                       │  "find bugs", "test X" │
                                                                       └──────┬─────────┬───────┘
                                                                              │ YES     │ NO
                                                                              ▼         ▼
                                                                      delegate: qa   ┌──────────────────────┐
                                                                                     │ Multi-step task with │
                                                                                     │ dependencies?        │
                                                                                     └──────┬──────────┬────┘
                                                                                            │ YES      │ NO
                                                                                            ▼          ▼
                                                                                     create_plan   Answer directly
```

### Tool Dispatch Flow

```
agent calls tool(name, args)
          │
          ▼
   approval_gate.requires_approval(name)?
          │
     ┌────┴────┐
     │ YES     │ NO
     ▼         ▼
  prompt    dispatch immediately
  user          │
     │          │
     ▼          │
  approved? ────┘
     │
  ┌──┴──┐
  │ NO  │ YES
  ▼     ▼
DENIED  run handler(args)
            │
            ▼
        timeout? (60s default)
            │
     ┌──────┴──────┐
     │ YES         │ NO
     ▼             ▼
  "ERROR:       result string
  timed out"        │
                    ▼
              log to audit DB
                    │
                    ▼
            return to agent
```

### create_plan Execution Flow

```
create_plan(goal, steps)
          │
          ▼
   topological sort steps
   (respect depends_on)
          │
          ▼
   for each step (in order):
          │
          ▼
   gather context from
   completed dependencies
          │
          ▼
   run_step(agent_type, task + context)
          │
          ▼
   critic.critique(task, result)
   score 1-5
          │
     ┌────┴────────────┐
     │ score <= 2?     │ score >= 3
     ▼                 ▼
  retry once       accept result
  (revised task)        │
          │             │
          └──────┬──────┘
                 ▼
          store result
          in context map
                 │
                 ▼
         next step...
                 │
                 ▼
   all steps done → aggregate results
```

---

## Data Flow

```
User Types Message
        │
        ▼
CLI (main.py)
  get_user_input()
        │
        ▼
PlannerAgent.run_conversation()
  messages.append({role: user, ...})
        │
        ▼
BaseAgent.run_turn()
  ollama.chat(model, [system] + messages, tools)
        │
        ├─── text response ──→ on_response(text) ──→ print to terminal
        │
        └─── tool_calls ──→ for each call:
                                _dispatch(name, args)
                                    │
                                    ▼
                               tool_registry[name](args)
                                    │
                                    ▼
                               result string
                                    │
                                    ▼
                               messages.append({role: tool, content: result})
                                    │
                                    └──→ ollama.chat() again
```

---

## File Structure

```
src/jarvis/
├── config.py              # All settings from environment
├── main.py                # CLI entrypoint
│
├── agents/
│   ├── base_agent.py      # Core agentic loop (Ollama-native)
│   ├── planner.py         # PlannerAgent — orchestrator
│   ├── researcher.py      # ResearcherAgent — web research
│   ├── coder.py           # CoderAgent — code writing/running
│   ├── qa.py              # QAAgent — code review/testing
│   ├── executor.py        # ExecutorAgent — runs multi-step plans
│   ├── critic.py          # CriticAgent — scores step outputs
│   └── team_agent.py      # TeamAgent — role-based coordination
│
├── tools/
│   ├── registry.py        # Tool schemas + dispatch map
│   ├── web_search.py      # DuckDuckGo search (free, no key)
│   ├── report_writer.py   # Save/update reports
│   ├── url_reader.py      # Fetch and parse web pages
│   ├── memory.py          # ChromaDB semantic search
│   ├── os_command.py      # Allowlisted shell commands
│   ├── delegation.py      # delegate_task tool
│   ├── plan_tool.py       # create_plan tool
│   └── plugins/           # Auto-discovered drop-in tools
│
├── prompts/
│   ├── planner.md         # PlannerAgent system prompt + rules
│   ├── researcher.md      # ResearcherAgent system prompt
│   ├── coder.md           # CoderAgent system prompt
│   ├── qa.md              # QAAgent system prompt
│   └── critic.md          # CriticAgent evaluation prompt
│
├── memory/
│   ├── episodic.py        # Conversation log (SQLite + FTS5)
│   ├── graph.py           # Knowledge graph (SQLite)
│   ├── failures.py        # Tool error analysis
│   ├── feedback.py        # User ratings
│   └── preferences.py     # Per-user persistent preferences
│
├── models/
│   └── router.py          # (reserved for future multi-model routing)
│
├── vision/
│   ├── capture.py         # YOLO object detection + Ollama vision
│   └── face.py            # OpenCV face recognition
│
└── api/
    └── server.py          # FastAPI HTTP + WebSocket server
```

---

## Technology Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| LLM | Ollama + qwen2.5:14b | Local, free, fast on Apple Silicon |
| Vision | Ollama + llava:13b | Local vision model |
| Object detection | YOLO v8 | Best local object detection |
| Web search | DuckDuckGo (ddgs) | Free, no API key, privacy |
| Vector memory | ChromaDB | Local vector database |
| Episodic memory | SQLite + FTS5 | Fast full-text search |
| Terminal UI | Rich | Beautiful terminal output |
| API server | FastAPI + uvicorn | Production-ready HTTP/WebSocket |
| Package manager | uv | Fast Python package management |
