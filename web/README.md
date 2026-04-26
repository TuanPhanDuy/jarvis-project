# JARVIS Web UI

React + Vite frontend for the JARVIS AI Research Agent.

## Quick Start

```bash
cd web
npm install
npm run dev        # → http://localhost:3000
```

Requires the JARVIS API server running on port 8000:
```bash
# From project root
uv run jarvis-api
```

## Features

- **Streaming chat** via WebSocket — real-time token streaming
- **Tool call visualization** — see which tools JARVIS is calling
- **Approval gate** — modal prompt when JARVIS needs permission for risky actions
- **Agent mode toggle** — switch between Planner and Researcher agents
- **Session management** — new session button, session ID display
- **Token usage & cost** — expandable per-message stats
- **Feedback** — thumbs up/down on each response
- **Proactive notifications** — toast alerts when JARVIS pushes autonomous events
- **Schedule panel** — create/delete cron research jobs
- **Audit log** — view tool-call history with risk levels

## Build for Production

```bash
npm run build     # outputs to dist/
```

Serve `dist/` from any static host or directly via the FastAPI server by mounting:
```python
app.mount("/", StaticFiles(directory="web/dist", html=True))
```
