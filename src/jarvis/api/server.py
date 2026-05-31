"""JARVIS FastAPI server.

Provides:
  POST  /api/chat         — synchronous single-turn chat
  WS    /api/ws/{session} — streaming bidirectional WebSocket chat
  GET   /api/health       — health check
  GET   /metrics          — Prometheus text metrics

Run with:
    python -m jarvis.api.server
    # or via uvicorn:
    uvicorn jarvis.api.server:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio
import collections
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from jarvis.api.metrics import (
    ACTIVE_WS_CONNECTIONS,
    REQUEST_DURATION,
    REQUESTS_TOTAL,
    TOOL_CALLS_TOTAL,
    record_usage,
)

from jarvis.api.models import (
    BudgetRequest,
    BudgetStatusResponse,
    ChatRequest,
    ChatResponse,
    StructuredChatRequest,
    StructuredChatResponse,
    ComponentStatus,
    EvalRunRequest,
    EvalRunResponse,
    EvalResultItem,
    FeedbackRequest,
    ParallelMapRequest,
    FeedbackStatsResponse,
    HealthResponse,
    ScheduleItem,
    ScheduleRequest,
    ScheduleResponse,
    SessionInfo,
    TokenRequest,
    TokenResponse,
    UserCreate,
    UsageSummary,
    WsApprovalRequest,
    WsApprovalResponse,
    WsChunk,
    WsDone,
    WsError,
    WsIncoming,
    WsPing,
    WsProactive,
    WsThinking,
    WsToolCall,
)
from jarvis.config import get_settings
from jarvis.agents.planner import PlannerAgent
from jarvis.agents.researcher import ResearcherAgent
from jarvis.agents.team_agent import TeamAgent
from jarvis.tools.registry import build_planner_registry, build_registry

# ── Logging ──────────────────────────────────────────────────────────────────

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.PrintLoggerFactory(),
)
log = structlog.get_logger()

# ── Session store ─────────────────────────────────────────────────────────────
# Maps session_id → {"agent": agent, "messages": list[dict], "created_at": float}
_sessions: dict[str, dict] = {}
_executor = ThreadPoolExecutor(max_workers=4)

# Active WebSocket connections for proactive push: session_id → WebSocket
_active_websockets: dict[str, "WebSocket"] = {}

# Last activity timestamp per session — updated on each incoming WS message
_session_activity: dict[str, float] = {}

# ── Rate limiting helpers ─────────────────────────────────────────────────────

def _parse_rate(rate_str: str) -> tuple[int, float]:
    """Parse '30/minute' → (30, 60.0). Also accepts /second and /hour."""
    count_s, _, period_s = rate_str.partition("/")
    count = int(count_s.strip())
    period_map = {"second": 1.0, "minute": 60.0, "hour": 3600.0}
    window = period_map.get(period_s.strip().lower(), 60.0)
    return count, window


class _RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiter applied to chat endpoints.

    When per_user=True and auth is enabled, buckets are keyed by user_id extracted
    from the JWT Authorization header.  Falls back to per-IP keying when auth is
    disabled or the token is missing/invalid.
    """

    _RATE_LIMITED_PATHS = {"/api/chat", "/api/chat/stream"}

    def __init__(
        self,
        app,
        max_calls: int,
        window_seconds: float,
        enabled: bool,
        per_user: bool = False,
    ) -> None:
        super().__init__(app)
        self._max_calls = max_calls
        self._window = window_seconds
        self._enabled = enabled
        self._per_user = per_user
        self._buckets: dict[str, collections.deque] = collections.defaultdict(collections.deque)

    def _bucket_key(self, request: Request) -> str:
        if self._per_user:
            auth = request.headers.get("authorization", "")
            if auth.lower().startswith("bearer "):
                token = auth[7:]
                try:
                    from jarvis.auth.core import verify_token
                    settings = get_settings()
                    user = verify_token(token, settings.jwt_secret)
                    if user:
                        return f"user:{user.username}"
                except Exception:
                    pass
        return f"ip:{request.client.host if request.client else 'unknown'}"

    async def dispatch(self, request: Request, call_next):
        if not self._enabled or request.url.path not in self._RATE_LIMITED_PATHS:
            return await call_next(request)

        key = self._bucket_key(request)
        now = time.monotonic()
        bucket = self._buckets[key]

        while bucket and bucket[0] < now - self._window:
            bucket.popleft()

        if len(bucket) >= self._max_calls:
            retry_after = int(self._window - (now - bucket[0])) + 1
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Please slow down."},
                headers={"Retry-After": str(retry_after)},
            )

        bucket.append(now)
        return await call_next(request)


# ── Request access-log middleware ─────────────────────────────────────────────

_ACCESS_LOG_SKIP = frozenset(["/api/events/stream", "/api/health", "/api/ready"])


class _AccessLogMiddleware(BaseHTTPMiddleware):
    """Records every non-streaming request to the SQLite access_log table."""

    async def dispatch(self, request: Request, call_next):
        if request.url.path in _ACCESS_LOG_SKIP:
            return await call_next(request)
        t0 = time.perf_counter()
        response = await call_next(request)
        latency_ms = (time.perf_counter() - t0) * 1000
        try:
            from jarvis.api.access_log import record_request
            db_path = get_settings().reports_dir / "jarvis.db"
            user_id = "anonymous"
            auth = request.headers.get("authorization", "")
            if auth.lower().startswith("bearer "):
                try:
                    from jarvis.auth.core import verify_token
                    user = verify_token(auth[7:], get_settings().jwt_secret)
                    if user:
                        user_id = user.username
                except Exception:
                    pass
            record_request(db_path, request.method, request.url.path,
                           response.status_code, latency_ms, user_id)
        except Exception:
            pass
        return response


# ── App factory ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _require_auth
    settings = get_settings()
    _pre_index_reports(settings)
    asyncio.create_task(_evict_stale_sessions())
    from jarvis.scheduler.core import start_scheduler
    start_scheduler(settings.reports_dir / "scheduler.db")
    if settings.otel_enabled:
        from jarvis.telemetry.tracing import instrument_fastapi, setup_tracing
        setup_tracing(settings.otel_endpoint)
        instrument_fastapi(app)
    if settings.auth_enabled:
        from jarvis.auth.core import ensure_admin_exists
        ensure_admin_exists(settings.reports_dir / "jarvis.db")

    if settings.rate_limit_enabled:
        log.info("rate_limiting_enabled", limit=settings.chat_rate_limit)

    from jarvis.auth.core import make_auth_dependency
    _require_auth = make_auth_dependency(
        db_path=settings.reports_dir / "jarvis.db",
        jwt_secret=settings.jwt_secret,
        auth_enabled=settings.auth_enabled,
    )

    # Restore sessions persisted before shutdown
    _restore_persisted_sessions(settings)

    if settings.proactive_enabled:
        await _start_event_bus(settings)

    if settings.peer_enabled:
        await _start_peer_coordinator(settings)

    try:
        from jarvis.tools.plugins.reminder_manager import set_event_loop as _set_reminder_loop
        _set_reminder_loop(asyncio.get_running_loop())
    except Exception:
        pass

    yield

    from jarvis.scheduler.core import stop_scheduler
    stop_scheduler()
    from jarvis.events.bus import get_event_bus
    await get_event_bus().shutdown()


app = FastAPI(
    title="JARVIS API",
    description="Just A Rather Very Intelligent System — powered by Claude",
    version="0.1.0",
    lifespan=lifespan,
)

import os as _os
app.add_middleware(
    CORSMiddleware,
    allow_origins=_os.getenv("JARVIS_CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

_rl_enabled = _os.getenv("JARVIS_RATE_LIMIT_ENABLED", "false").lower() == "true"
_rl_per_user = _os.getenv("JARVIS_RATE_LIMIT_PER_USER", "false").lower() == "true"
_rl_max, _rl_window = _parse_rate(_os.getenv("JARVIS_CHAT_RATE_LIMIT", "30/minute"))
app.add_middleware(
    _RateLimitMiddleware,
    max_calls=_rl_max,
    window_seconds=_rl_window,
    enabled=_rl_enabled,
    per_user=_rl_per_user,
)

app.add_middleware(_AccessLogMiddleware)


_require_auth = None  # set in lifespan; callable FastAPI dependency
_limiter = None  # unused sentinel — kept for backwards compatibility


async def _start_event_bus(settings) -> None:
    """Initialize the event bus, register handlers, and start monitors."""
    from jarvis.events.bus import get_event_bus
    from jarvis.events.triggers import SystemMonitor, IdleDetector, FileWatcher
    from jarvis.events.autonomous_agent import handle_event

    bus = get_event_bus()
    loop = asyncio.get_running_loop()

    def _build_autonomous_agent():
        return _build_agent_for_session(settings, researcher_mode=False, session_id="autonomous")

    async def _on_any_event(event) -> None:
        await handle_event(
            event=event,
            active_websockets=_active_websockets,
            build_agent_fn=_build_autonomous_agent,
            loop=loop,
        )

    bus.subscribe("system_alert", _on_any_event)
    bus.subscribe("user_event", _on_any_event)
    bus.subscribe("external_event", _on_any_event)

    asyncio.create_task(bus._dispatch_loop())
    asyncio.create_task(SystemMonitor(bus).run())

    # Idle detector — reads from module-level _session_activity, updated per WS message
    def _get_session_activity() -> dict[str, float]:
        return dict(_session_activity)

    asyncio.create_task(
        IdleDetector(bus, _get_session_activity, idle_minutes=settings.idle_minutes).run()
    )
    asyncio.create_task(FileWatcher(bus, settings.reports_dir).run())

    log.info("event_bus_initialized", proactive=True)


async def _start_peer_coordinator(settings) -> None:
    """Start peer discovery and graph sync coordinator."""
    global _peer_coordinator
    from jarvis.peer.coordinator import PeerCoordinator
    import uuid

    device_id = str(uuid.uuid4())[:8]
    db_path = settings.reports_dir / "jarvis.db"
    _peer_coordinator = PeerCoordinator(
        db_path=db_path,
        device_id=device_id,
        http_port=settings.peer_port,
    )
    asyncio.create_task(_peer_coordinator.start())
    log.info("peer_coordinator_started", device_id=device_id, port=settings.peer_port)


def _count_user_sessions(settings, user_id: str | None) -> int:
    """Count prior distinct sessions for a user (for personality adaptation). Best-effort."""
    if not user_id or user_id == "anonymous":
        return 0
    try:
        import sqlite3
        db_path = settings.reports_dir / "jarvis.db"
        if not db_path.exists():
            return 0
        conn = sqlite3.connect(str(db_path))
        try:
            row = conn.execute(
                "SELECT COUNT(DISTINCT session_id) FROM episodes WHERE user_id = ?", (user_id,)
            ).fetchone()
        finally:
            conn.close()
        return row[0] if row else 0
    except Exception:
        return 0


def _restore_persisted_sessions(settings) -> None:
    """Load previously persisted sessions back into _sessions on startup."""
    try:
        from jarvis.memory.sessions import load_sessions
        db_path = settings.reports_dir / "jarvis.db"
        rows = load_sessions(db_path, ttl_minutes=settings.api_session_ttl_minutes)
        for row in rows:
            sid = row["session_id"]
            if sid in _sessions:
                continue
            researcher_mode = row["agent_type"] == "ResearcherAgent"
            team_mode = row["agent_type"] == "TeamAgent"
            try:
                agent = _build_agent_for_session(
                    settings,
                    researcher_mode=researcher_mode,
                    team_mode=team_mode,
                    session_id=sid,
                    user_id=row.get("user_id"),
                )
                _sessions[sid] = {
                    "agent": agent,
                    "messages": row["messages"],
                    "created_at": row["created_at"],
                    "user_id": row.get("user_id"),
                    "approval_gate": None,
                    "fork_of": row.get("fork_of"),
                    "forked_at": row.get("updated_at"),
                }
            except Exception as exc:
                log.warning("session_restore_failed", session_id=sid, error=str(exc))
        if rows:
            log.info("sessions_restored", count=len(rows))
    except Exception as exc:
        log.warning("session_restore_error", error=str(exc))


def _persist_session(session_id: str, session: dict, settings=None) -> None:
    """Save a session's message history to SQLite. Best-effort — never raises."""
    try:
        if settings is None:
            settings = get_settings()
        from jarvis.memory.sessions import save_session
        agent = session.get("agent")
        save_session(
            db_path=settings.reports_dir / "jarvis.db",
            session_id=session_id,
            messages=session.get("messages", []),
            agent_type=type(agent).__name__ if agent else "PlannerAgent",
            user_id=session.get("user_id"),
            fork_of=session.get("fork_of"),
            created_at=session.get("created_at"),
        )
    except Exception as exc:
        log.warning("session_persist_failed", session_id=session_id, error=str(exc))


def _pre_index_reports(settings) -> None:
    from jarvis.tools.memory import index_new_report
    from pathlib import Path

    reports_dir = Path(settings.reports_dir)
    if not reports_dir.exists():
        return
    for md_file in sorted(reports_dir.glob("*.md")):
        index_new_report(reports_dir, md_file.name)
    log.info("memory_pre_indexed", reports_dir=str(reports_dir))


_last_memory_prune: float = 0.0


async def _evict_stale_sessions() -> None:
    global _last_memory_prune
    settings = get_settings()
    ttl_seconds = settings.api_session_ttl_minutes * 60
    while True:
        await asyncio.sleep(60)
        now = time.time()
        stale = [sid for sid, s in _sessions.items() if now - s["created_at"] > ttl_seconds]
        for sid in stale:
            del _sessions[sid]
            log.info("session_evicted", session_id=sid)
        if stale:
            log.info("session_eviction_complete", evicted=len(stale), remaining=len(_sessions))

        # Prune old memory rows once per hour
        if now - _last_memory_prune > 3600:
            _last_memory_prune = now
            db_path = settings.reports_dir / "jarvis.db"
            retention = settings.memory_retention_days
            try:
                from jarvis.memory.episodic import prune_old_episodes
                from jarvis.memory.feedback import prune_old_feedback
                from jarvis.memory.failures import prune_old_failures
                from jarvis.memory.preferences import prune_old_preferences
                from jarvis.memory.turns import prune_old_turns
                from jarvis.security.audit import prune_old_audit
                ep = prune_old_episodes(db_path, retention)
                fb = prune_old_feedback(db_path, retention)
                fa = prune_old_failures(db_path, retention)
                pr = prune_old_preferences(db_path, retention)
                tu = prune_old_turns(db_path, retention)
                au = prune_old_audit(db_path, retention)
                if ep + fb + fa + pr + tu + au > 0:
                    log.info("memory_pruned", episodes=ep, feedback=fb, failures=fa, preferences=pr,
                             turns=tu, audit=au, retention_days=retention)
            except Exception as exc:
                log.warning("memory_prune_error", error=str(exc))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_agent_for_session(
    settings,
    researcher_mode: bool = False,
    team_mode: bool = False,
    session_id: str = "",
    user_id: str | None = None,
    approval_gate=None,
) -> PlannerAgent | ResearcherAgent | TeamAgent:
    base_schemas, base_registry = build_registry(
        reports_dir=settings.reports_dir,
        allowed_commands=settings.allowed_commands,
        user_id=user_id or "anonymous",
        vision_model=settings.vision_model,
    )
    if researcher_mode:
        return ResearcherAgent(
            model=settings.model,
            max_tokens=settings.max_tokens,
            tool_schemas=base_schemas,
            tool_registry=base_registry,
            max_search_calls=settings.max_search_calls,
            session_id=session_id,
            user_id=user_id,
            approval_gate=approval_gate,
        )
    if team_mode:
        return TeamAgent(
            model=settings.model,
            max_tokens=settings.max_tokens,
            tool_schemas=base_schemas,
            tool_registry=base_registry,
            role="manager",
            session_id=session_id,
            user_id=user_id,
            approval_gate=approval_gate,
        )
    planner_schemas, planner_registry = build_planner_registry(
        base_schemas=base_schemas,
        base_registry=base_registry,
        model=settings.model,
        max_tokens=settings.max_tokens,
        session_id=session_id,
        user_id=user_id,
    )
    session_count = _count_user_sessions(settings, user_id)
    return PlannerAgent(
        model=settings.model,
        max_tokens=settings.max_tokens,
        tool_schemas=planner_schemas,
        tool_registry=planner_registry,
        session_id=session_id,
        user_id=user_id,
        approval_gate=approval_gate,
        session_count=session_count,
    )


def _get_session(
    session_id: str,
    researcher_mode: bool = False,
    team_mode: bool = False,
    user_id: str | None = None,
    approval_gate=None,
) -> dict:
    if session_id not in _sessions:
        settings = get_settings()
        agent = _build_agent_for_session(
            settings,
            researcher_mode=researcher_mode,
            team_mode=team_mode,
            session_id=session_id,
            user_id=user_id,
            approval_gate=approval_gate,
        )
        _sessions[session_id] = {
            "agent": agent,
            "messages": [],
            "created_at": time.time(),
            "user_id": user_id,
            "approval_gate": approval_gate,
        }
        log.info("session_created", session_id=session_id)
    return _sessions[session_id]


def _instrument_tool_dispatch(
    agent: PlannerAgent | ResearcherAgent | TeamAgent,
    on_tool_event: Callable[[str], None] | None = None,
) -> None:
    """Monkey-patch _before_dispatch to emit metrics and optional callback."""
    original = agent._before_dispatch

    def instrumented(name: str, tool_input: dict) -> None:
        TOOL_CALLS_TOTAL.labels(tool_name=name).inc()
        if on_tool_event:
            on_tool_event(name)
        original(name, tool_input)

    agent._before_dispatch = instrumented  # type: ignore[method-assign]


# ── HTTP endpoints ─────────────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    components: dict[str, ComponentStatus] = {}

    # DB check
    try:
        import sqlite3
        db_path = get_settings().reports_dir / "jarvis.db"
        if db_path.exists():
            conn = sqlite3.connect(str(db_path), timeout=2)
            try:
                conn.execute("SELECT 1").fetchone()
            finally:
                conn.close()
            components["db"] = ComponentStatus(ok=True)
        else:
            components["db"] = ComponentStatus(ok=True, detail="not yet created")
    except Exception as exc:
        components["db"] = ComponentStatus(ok=False, detail=str(exc))

    # Scheduler check
    try:
        from jarvis.scheduler.core import get_scheduler
        sched = get_scheduler()
        components["scheduler"] = ComponentStatus(ok=sched is not None and sched.running)
    except Exception as exc:
        components["scheduler"] = ComponentStatus(ok=False, detail=str(exc))

    pending_approvals = sum(
        len(session["agent"]._approval_gate.get_pending())
        for session in _sessions.values()
        if session.get("agent") is not None
        and session["agent"]._approval_gate is not None
    )

    all_ok = all(c.ok for c in components.values())
    return HealthResponse(
        status="ok" if all_ok else "degraded",
        sessions_active=len(_sessions),
        ws_connections=len(_active_websockets),
        pending_approvals=pending_approvals,
        components=components,
    )


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> PlainTextResponse:
    return PlainTextResponse(
        generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )


def _auth_dep(request: Request):
    if _require_auth:
        return _require_auth(request)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, _user=Depends(_auth_dep)) -> ChatResponse:
    """Synchronous single-turn chat. Blocks until JARVIS replies."""
    from jarvis.api.budget import BudgetExceededError, check_budget

    session_id = req.session_id or str(uuid.uuid4())
    session = _get_session(session_id, req.researcher_mode, team_mode=req.team_mode)
    agent = session["agent"]
    messages = session["messages"]

    db_path = get_settings().reports_dir / "jarvis.db"
    user_id = session.get("user_id") or "anonymous"
    try:
        check_budget(db_path, user_id)
    except BudgetExceededError as exc:
        raise HTTPException(status_code=402, detail=str(exc))

    _instrument_tool_dispatch(agent)
    _check_injection(req.message, session_id, user_id, db_path)

    messages.append({"role": "user", "content": req.message})
    t0 = time.perf_counter()

    loop = asyncio.get_running_loop()
    result_holder: list = []

    def run():
        try:
            text, updated = agent.run_turn(messages)
            result_holder.append(("ok", text, updated))
        except Exception as exc:
            result_holder.append(("err", str(exc), messages))

    _timeout = get_settings().agent_turn_timeout_seconds
    try:
        await asyncio.wait_for(loop.run_in_executor(_executor, run), timeout=_timeout)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"Agent turn timed out after {_timeout}s")

    duration = time.perf_counter() - t0
    status, text_or_err, updated_messages = result_holder[0]

    REQUESTS_TOTAL.labels(mode="http").inc()
    REQUEST_DURATION.labels(mode="http").observe(duration)

    if status == "err":
        log.error("chat_error", session_id=session_id, error=text_or_err)
        raise RuntimeError(text_or_err)

    session["messages"] = updated_messages
    usage = agent.get_usage_summary()
    record_usage(usage)

    _log_episodes(session_id, req.message, text_or_err, user_id=user_id)
    _record_spend(user_id, usage["estimated_cost_usd"])
    _persist_session(session_id, session)
    _maybe_set_auto_title(session_id, updated_messages)
    log.info("chat_complete", session_id=session_id, duration_s=round(duration, 2))
    return ChatResponse(
        session_id=session_id,
        response=text_or_err,
        usage=UsageSummary(**usage),
    )


# ── Batch chat ────────────────────────────────────────────────────────────────

@app.post("/api/chat/batch", status_code=200)
async def chat_batch(body: dict, _user=Depends(_auth_dep)) -> list[dict]:
    """Run multiple chat turns concurrently and return all results.

    Body:
      requests      list of {session_id?, message, researcher_mode?}
      max_concurrent int (default 4) — semaphore cap for parallel execution

    Each result: {session_id, response, usage, error?}
    Failed items include an "error" key; others include "response" and "usage".
    """
    from jarvis.api.budget import BudgetExceededError, check_budget

    requests_list = body.get("requests") or []
    if not requests_list:
        raise HTTPException(status_code=422, detail="'requests' list is required")
    max_concurrent = max(1, int(body.get("max_concurrent") or 4))
    semaphore = asyncio.Semaphore(max_concurrent)
    loop = asyncio.get_running_loop()
    settings = get_settings()

    async def _one(req_dict: dict) -> dict:
        session_id = req_dict.get("session_id") or str(uuid.uuid4())
        message = str(req_dict.get("message") or "")
        if not message:
            return {"session_id": session_id, "error": "message is required"}
        researcher_mode = bool(req_dict.get("researcher_mode", False))
        async with semaphore:
            try:
                session = _get_session(session_id, researcher_mode)
                agent = session["agent"]
                messages = session["messages"]
                db_path = settings.reports_dir / "jarvis.db"
                user_id = session.get("user_id") or "anonymous"
                check_budget(db_path, user_id)
                _instrument_tool_dispatch(agent)
                _check_injection(message, session_id, user_id, db_path)
                messages.append({"role": "user", "content": message})
                result_holder: list = []

                def _run():
                    try:
                        text, updated = agent.run_turn(messages)
                        result_holder.append(("ok", text, updated))
                    except Exception as exc:
                        result_holder.append(("err", str(exc), messages))

                await loop.run_in_executor(_executor, _run)
                status, text_or_err, updated_messages = result_holder[0]
                if status == "err":
                    return {"session_id": session_id, "error": text_or_err}
                session["messages"] = updated_messages
                usage = agent.get_usage_summary()
                _record_spend(user_id, usage["estimated_cost_usd"])
                _persist_session(session_id, session)
                _maybe_set_auto_title(session_id, updated_messages)
                return {
                    "session_id": session_id,
                    "response": text_or_err,
                    "usage": usage,
                }
            except BudgetExceededError as exc:
                return {"session_id": session_id, "error": str(exc)}
            except HTTPException as exc:
                return {"session_id": session_id, "error": exc.detail}
            except Exception as exc:
                return {"session_id": session_id, "error": str(exc)}

    return await asyncio.gather(*[_one(r) for r in requests_list])


# ── SSE streaming chat ────────────────────────────────────────────────────────

@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest, _user=Depends(_auth_dep)) -> StreamingResponse:
    """Streaming chat via Server-Sent Events.

    Emits a stream of JSON-encoded SSE events:
      data: {"type":"chunk","text":"..."}    — partial response text
      data: {"type":"tool","name":"..."}     — tool invocation notification
      data: {"type":"usage","input_tokens":N,"output_tokens":N,"cost_usd":N,"latency_ms":N}
      data: {"type":"done","session_id":"...","usage":{...}}
      data: {"type":"error","message":"..."}
    """
    import json as _json
    from jarvis.api.budget import BudgetExceededError, check_budget

    session_id = req.session_id or str(uuid.uuid4())
    session = _get_session(session_id, req.researcher_mode, team_mode=req.team_mode)
    agent = session["agent"]
    messages = session["messages"]

    db_path = get_settings().reports_dir / "jarvis.db"
    user_id = session.get("user_id") or "anonymous"
    try:
        check_budget(db_path, user_id)
    except BudgetExceededError as exc:
        _err_msg = str(exc)  # capture before Python 3 deletes the except-as variable
        async def _budget_err():
            yield f"data: {_json.dumps({'type': 'error', 'message': _err_msg})}\n\n"
        return StreamingResponse(_budget_err(), media_type="text/event-stream")

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _on_chunk(text: str) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, ("chunk", text))

    def _on_tool(name: str) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, ("tool", name))

    _instrument_tool_dispatch(agent, on_tool_event=_on_tool)
    _check_injection(req.message, session_id, user_id, db_path)
    messages.append({"role": "user", "content": req.message})

    def _run_agent():
        try:
            text, updated = agent.run_turn(messages, on_chunk=_on_chunk)
            loop.call_soon_threadsafe(queue.put_nowait, ("done", (text, updated)))
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))

    _timeout = get_settings().agent_turn_timeout_seconds
    t0 = time.perf_counter()
    future = loop.run_in_executor(_executor, _run_agent)

    async def _event_gen():
        try:
            while True:
                try:
                    kind, payload = await asyncio.wait_for(queue.get(), timeout=_timeout + 5)
                except asyncio.TimeoutError:
                    yield f"data: {_json.dumps({'type': 'error', 'message': 'stream timed out'})}\n\n"
                    return

                if kind == "chunk":
                    yield f"data: {_json.dumps({'type': 'chunk', 'text': payload})}\n\n"
                elif kind == "tool":
                    yield f"data: {_json.dumps({'type': 'tool', 'name': payload})}\n\n"
                    TOOL_CALLS_TOTAL.labels(tool_name=payload).inc()
                elif kind == "done":
                    text, updated_messages = payload
                    session["messages"] = updated_messages
                    usage = agent.get_usage_summary()
                    record_usage(usage)
                    _log_episodes(session_id, req.message, text, user_id=user_id)
                    _record_spend(user_id, usage["estimated_cost_usd"])
                    _persist_session(session_id, session)
                    _maybe_set_auto_title(session_id, updated_messages)
                    REQUESTS_TOTAL.labels(mode="sse").inc()
                    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
                    yield f"data: {_json.dumps({'type': 'usage', 'input_tokens': usage['input_tokens'], 'output_tokens': usage['output_tokens'], 'cost_usd': usage['estimated_cost_usd'], 'latency_ms': latency_ms})}\n\n"
                    yield f"data: {_json.dumps({'type': 'done', 'session_id': session_id, 'usage': usage})}\n\n"
                    return
                elif kind == "error":
                    log.error("sse_chat_error", session_id=session_id, error=payload)
                    yield f"data: {_json.dumps({'type': 'error', 'message': payload})}\n\n"
                    return
        finally:
            future.cancel()

    return StreamingResponse(
        _event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Structured output ─────────────────────────────────────────────────────────

@app.post("/api/chat/structured", response_model=StructuredChatResponse)
async def chat_structured(req: StructuredChatRequest, _user=Depends(_auth_dep)) -> StructuredChatResponse:
    """Single-turn chat that returns a JSON object matching the provided JSON Schema.

    The model is instructed via system prompt + Ollama's native format parameter
    to respond with valid JSON.  Returns 422 if the model output cannot be parsed.
    """
    from jarvis.api.budget import BudgetExceededError, check_budget

    session_id = req.session_id or str(uuid.uuid4())
    session = _get_session(session_id, req.researcher_mode, team_mode=req.team_mode)
    agent = session["agent"]
    messages = session["messages"]

    db_path = get_settings().reports_dir / "jarvis.db"
    user_id = session.get("user_id") or "anonymous"
    try:
        check_budget(db_path, user_id)
    except BudgetExceededError as exc:
        raise HTTPException(status_code=402, detail=str(exc))

    _instrument_tool_dispatch(agent)
    messages.append({"role": "user", "content": req.message})

    loop = asyncio.get_running_loop()
    result_holder: list = []

    def run():
        try:
            parsed, updated = agent.run_turn_structured(messages, req.json_schema)
            result_holder.append(("ok", parsed, updated))
        except ValueError as exc:
            result_holder.append(("parse_error", str(exc), messages))
        except Exception as exc:
            result_holder.append(("err", str(exc), messages))

    _timeout = get_settings().agent_turn_timeout_seconds
    try:
        await asyncio.wait_for(loop.run_in_executor(_executor, run), timeout=_timeout)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"Agent turn timed out after {_timeout}s")

    status, payload, updated_messages = result_holder[0]
    if status == "parse_error":
        raise HTTPException(status_code=422, detail=payload)
    if status == "err":
        raise HTTPException(status_code=500, detail=payload)

    session["messages"] = updated_messages
    usage = agent.get_usage_summary()
    record_usage(usage)
    _log_episodes(session_id, req.message, str(payload), user_id=user_id)
    _record_spend(user_id, usage["estimated_cost_usd"])
    _persist_session(session_id, session)
    return StructuredChatResponse(
        session_id=session_id,
        result=payload,
        usage=UsageSummary(**usage),
    )


# ── WebSocket ──────────────────────────────────────────────────────────────────

@app.websocket("/api/ws/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str, token: str | None = None) -> None:
    """Streaming WebSocket chat. Session persists across reconnections.

    When auth is enabled pass ?token=<jwt> as a query parameter.
    """
    await websocket.accept()

    settings = get_settings()
    if settings.auth_enabled:
        from jarvis.auth.core import verify_token
        user = verify_token(token or "", settings.jwt_secret) if token else None
        if user is None:
            await websocket.send_json(WsError(message="Unauthorized").model_dump())
            await websocket.close(code=4001)
            return

    ACTIVE_WS_CONNECTIONS.inc()
    _active_websockets[session_id] = websocket
    log.info("ws_connected", session_id=session_id)

    # Build an approval gate that pushes WsApprovalRequest over this WebSocket.
    loop = asyncio.get_running_loop()
    from jarvis.security.approval import ApprovalGate, RiskLevel

    threshold_name = getattr(settings, "approval_threshold", "medium").upper()
    threshold = RiskLevel[threshold_name] if threshold_name in RiskLevel.__members__ else RiskLevel.MEDIUM
    timeout_s = getattr(settings, "approval_timeout_seconds", 60)

    def _push_approval_request(req) -> None:
        expires_in = max(0, int(req.expires_at - time.time()))
        msg = WsApprovalRequest(
            request_id=req.request_id,
            tool_name=req.tool_name,
            description=req.description,
            risk_level=req.risk_level.name,
            expires_in=expires_in,
        ).model_dump()
        loop.call_soon_threadsafe(
            asyncio.ensure_future,
            websocket.send_json(msg),
        )

    approval_gate = ApprovalGate(
        threshold=threshold,
        timeout_seconds=timeout_s,
        request_callback=_push_approval_request,
        session_id=session_id,
    )

    _heartbeat_interval = get_settings().ws_heartbeat_seconds

    async def _heartbeat() -> None:
        while True:
            await asyncio.sleep(_heartbeat_interval)
            try:
                await websocket.send_json(WsPing().model_dump())
            except Exception:
                break

    heartbeat_task = asyncio.create_task(_heartbeat())
    try:
        while True:
            raw = await websocket.receive_json()

            # Track last activity for idle detection
            _session_activity[session_id] = time.time()

            # Handle approval responses without starting a new agent turn
            if raw.get("type") == "approval_response":
                try:
                    resp = WsApprovalResponse(**raw)
                    approval_gate.resolve(resp.request_id, resp.approved)
                except Exception:
                    pass
                continue

            try:
                req = WsIncoming(**raw)
            except Exception:
                await websocket.send_json(WsError(message="Invalid message format").model_dump())
                continue

            session = _get_session(
                session_id,
                researcher_mode=req.researcher_mode,
                team_mode=req.team_mode,
                approval_gate=approval_gate,
            )

            # Enforce monthly budget before running a turn
            from jarvis.api.budget import BudgetExceededError, check_budget as _check_budget
            _db_path = get_settings().reports_dir / "jarvis.db"
            _user_id = session.get("user_id") or "anonymous"
            try:
                _check_budget(_db_path, _user_id)
            except BudgetExceededError as _exc:
                await websocket.send_json(WsError(message=str(_exc)).model_dump())
                continue

            agent = session["agent"]
            # Keep approval gate in sync in case session was pre-existing
            agent._approval_gate = approval_gate
            messages = list(session["messages"])  # snapshot
            messages.append({"role": "user", "content": req.message})

            # Bridge: sync agent → async WebSocket via asyncio.Queue
            chunk_queue: asyncio.Queue = asyncio.Queue()

            def on_chunk(text: str) -> None:
                loop.call_soon_threadsafe(
                    chunk_queue.put_nowait,
                    WsChunk(text=text).model_dump(),
                )

            def on_tool_event(tool_name: str) -> None:
                loop.call_soon_threadsafe(
                    chunk_queue.put_nowait,
                    WsToolCall(tool=tool_name).model_dump(),
                )

            _instrument_tool_dispatch(agent, on_tool_event=on_tool_event)

            result_holder: list = []
            t0 = time.perf_counter()

            def run_turn() -> None:
                try:
                    text, updated = agent.run_turn(messages, on_chunk=on_chunk)
                    result_holder.append(("ok", text, updated))
                except Exception as exc:
                    result_holder.append(("err", str(exc), messages))
                finally:
                    loop.call_soon_threadsafe(chunk_queue.put_nowait, None)  # sentinel

            # Send thinking indicator, then start agent in thread
            await websocket.send_json(WsThinking().model_dump())
            _executor.submit(run_turn)

            # Drain chunks until sentinel (with agent turn timeout)
            _ws_timeout = get_settings().agent_turn_timeout_seconds
            _loop = asyncio.get_running_loop()
            _deadline = _loop.time() + _ws_timeout
            while True:
                remaining = _deadline - _loop.time()
                if remaining <= 0:
                    result_holder.append(("err", f"Agent turn timed out after {_ws_timeout}s", messages))
                    break
                try:
                    msg = await asyncio.wait_for(chunk_queue.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    result_holder.append(("err", f"Agent turn timed out after {_ws_timeout}s", messages))
                    break
                if msg is None:
                    break
                await websocket.send_json(msg)

            duration = time.perf_counter() - t0
            REQUESTS_TOTAL.labels(mode="websocket").inc()
            REQUEST_DURATION.labels(mode="websocket").observe(duration)

            if result_holder and result_holder[0][0] == "ok":
                _, response_text, updated_messages = result_holder[0]
                session["messages"] = updated_messages
                usage = agent.get_usage_summary()
                record_usage(usage)
                _log_episodes(session_id, req.message, response_text, user_id=_user_id)
                _record_spend(_user_id, usage["estimated_cost_usd"])
                await websocket.send_json(
                    WsDone(text=response_text, usage=UsageSummary(**usage)).model_dump()
                )
                log.info("ws_turn_complete", session_id=session_id, duration_s=round(duration, 2))
            else:
                err = result_holder[0][1] if result_holder else "Unknown error"
                log.error("ws_turn_error", session_id=session_id, error=err)
                await websocket.send_json(WsError(message=err).model_dump())

    except WebSocketDisconnect:
        log.info("ws_disconnected", session_id=session_id)
    finally:
        heartbeat_task.cancel()
        _active_websockets.pop(session_id, None)
        _session_activity.pop(session_id, None)
        ACTIVE_WS_CONNECTIONS.dec()


# ── Episodic logging helper ────────────────────────────────────────────────────

def _log_episodes(session_id: str, user_msg: str, assistant_reply: str, user_id: str = "anonymous") -> None:
    """Persist user + assistant turn to episodic memory. Best-effort."""
    try:
        from jarvis.memory.episodic import log_episode
        db_path = get_settings().reports_dir / "jarvis.db"
        log_episode(db_path, session_id, "user", user_msg, user_id=user_id)
        log_episode(db_path, session_id, "assistant", assistant_reply, user_id=user_id)
    except Exception:
        pass


def _record_spend(user_id: str, cost_usd: float) -> None:
    """Record per-user token spend for budget tracking. Best-effort."""
    try:
        from jarvis.api.budget import record_spend
        db_path = get_settings().reports_dir / "jarvis.db"
        record_spend(db_path, user_id, cost_usd)
    except Exception:
        pass


def _maybe_set_auto_title(session_id: str, messages: list[dict]) -> None:
    """Set an auto-generated title on first exchange if none exists. Best-effort."""
    try:
        from jarvis.memory.sessions import generate_title, get_metadata, set_title
        db_path = get_settings().reports_dir / "jarvis.db"
        meta = get_metadata(db_path, session_id)
        if meta and not meta.get("title"):
            title = generate_title(messages)
            set_title(db_path, session_id, title)
    except Exception:
        pass


def _check_injection(message: str, session_id: str, user_id: str, db_path: Path) -> None:
    """Scan message for prompt injection; log and optionally block. Best-effort."""
    try:
        from jarvis.security.injection import log_detection, max_severity, scan
        findings = scan(message)
        if not findings:
            return
        severity = max_severity(findings)
        blocked = severity == "high" and get_settings().injection_block_high
        log_detection(db_path, session_id, user_id, findings, blocked, snippet=message[:200])
        if blocked:
            raise HTTPException(
                status_code=400,
                detail=f"Message blocked: prompt injection detected ({findings[0]['label']})",
            )
    except HTTPException:
        raise
    except Exception:
        pass


# ── Feedback endpoints ────────────────────────────────────────────────────────

@app.post("/api/feedback", status_code=201)
async def submit_feedback(req: FeedbackRequest) -> dict:
    """Record user rating on a JARVIS response."""
    from jarvis.memory.feedback import log_feedback
    db_path = get_settings().reports_dir / "jarvis.db"
    log_feedback(db_path, req.session_id, req.response_snippet, req.rating, req.comment, rating_type=req.rating_type)
    return {"status": "recorded"}


@app.get("/api/feedback/{session_id}", response_model=FeedbackStatsResponse)
async def get_feedback(session_id: str) -> FeedbackStatsResponse:
    """Retrieve feedback statistics for a session."""
    from jarvis.memory.feedback import get_feedback_stats
    db_path = get_settings().reports_dir / "jarvis.db"
    stats = get_feedback_stats(db_path, session_id)
    return FeedbackStatsResponse(**stats)


@app.get("/api/feedback")
async def list_feedback(
    limit: int = 50,
    offset: int = 0,
    session_id: str | None = None,
    user_id: str | None = None,
    _user=Depends(_auth_dep),
) -> list[dict]:
    """Return paginated raw feedback entries, newest first. Filter by session_id or user_id."""
    from jarvis.memory.feedback import get_feedback_list
    db_path = get_settings().reports_dir / "jarvis.db"
    return get_feedback_list(db_path, limit=limit, offset=offset, session_id=session_id, user_id=user_id)


# ── Tool quota endpoints ──────────────────────────────────────────────────────

@app.get("/api/quotas/{user_id}")
async def get_user_quotas(user_id: str, _user=Depends(_auth_dep)) -> list[dict]:
    """Return all tool quotas for a user, including current usage in the window."""
    from jarvis.tools.quotas import get_quotas
    db_path = get_settings().reports_dir / "jarvis.db"
    return get_quotas(db_path, user_id)


@app.put("/api/quotas/{user_id}/{tool_name}", status_code=200)
async def set_user_tool_quota(
    user_id: str,
    tool_name: str,
    body: dict,
    _user=Depends(_auth_dep),
) -> dict:
    """Set a tool call quota for a user. Body: {limit: int, window_seconds: int}"""
    from jarvis.tools.quotas import set_quota
    limit = int(body.get("limit") or 0)
    window = int(body.get("window_seconds") or 0)
    if limit <= 0 or window <= 0:
        raise HTTPException(status_code=422, detail="'limit' and 'window_seconds' must be positive integers")
    db_path = get_settings().reports_dir / "jarvis.db"
    set_quota(db_path, user_id, tool_name, limit, window)
    return {"user_id": user_id, "tool_name": tool_name, "limit": limit, "window_seconds": window}


@app.delete("/api/quotas/{user_id}/{tool_name}", status_code=200)
async def delete_user_tool_quota(
    user_id: str,
    tool_name: str,
    _user=Depends(_auth_dep),
) -> dict:
    """Remove a tool call quota for a user."""
    from jarvis.tools.quotas import delete_quota
    db_path = get_settings().reports_dir / "jarvis.db"
    removed = delete_quota(db_path, user_id, tool_name)
    if not removed:
        raise HTTPException(status_code=404, detail=f"No quota found for user '{user_id}', tool '{tool_name}'")
    return {"user_id": user_id, "tool_name": tool_name, "deleted": True}


# ── Budget endpoints ───────────────────────────────────────────────────────────

@app.put("/api/budget/{user_id}", response_model=BudgetStatusResponse)
async def set_budget(user_id: str, req: BudgetRequest, _user=Depends(_auth_dep)) -> BudgetStatusResponse:
    """Set monthly USD spending budget for a user (0 = unlimited)."""
    from jarvis.api.budget import get_budget_status, set_budget as _set_budget
    db_path = get_settings().reports_dir / "jarvis.db"
    _set_budget(db_path, user_id, req.monthly_budget_usd)
    return BudgetStatusResponse(**get_budget_status(db_path, user_id))


@app.get("/api/budget/{user_id}", response_model=BudgetStatusResponse)
async def get_budget(user_id: str, _user=Depends(_auth_dep)) -> BudgetStatusResponse:
    """Get current spending and remaining budget for a user."""
    from jarvis.api.budget import get_budget_status
    db_path = get_settings().reports_dir / "jarvis.db"
    return BudgetStatusResponse(**get_budget_status(db_path, user_id))


@app.get("/api/budget")
async def list_all_budgets(_user=Depends(_auth_dep)) -> list[dict]:
    """Return budget status for all users. Admin view."""
    from jarvis.api.budget import get_all_budget_statuses
    db_path = get_settings().reports_dir / "jarvis.db"
    if not db_path.exists():
        return []
    return get_all_budget_statuses(db_path)


# ── Auth endpoints ────────────────────────────────────────────────────────────

@app.post("/api/auth/register", response_model=TokenResponse)
async def register(req: UserCreate) -> TokenResponse:
    settings = get_settings()
    if not settings.auth_enabled:
        raise HTTPException(status_code=404, detail="Auth not enabled.")
    from jarvis.auth.core import create_token, create_user
    db_path = settings.reports_dir / "jarvis.db"
    user = create_user(db_path, req.username, req.password, req.role)
    token = create_token(user, settings.jwt_secret, settings.jwt_expire_minutes)
    return TokenResponse(access_token=token, expires_in=settings.jwt_expire_minutes * 60)


@app.post("/api/auth/token", response_model=TokenResponse)
async def login(req: TokenRequest) -> TokenResponse:
    settings = get_settings()
    if not settings.auth_enabled:
        raise HTTPException(status_code=404, detail="Auth not enabled.")
    from jarvis.auth.core import authenticate, create_token
    db_path = settings.reports_dir / "jarvis.db"
    user = authenticate(db_path, req.username, req.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid credentials.")
    token = create_token(user, settings.jwt_secret, settings.jwt_expire_minutes)
    return TokenResponse(access_token=token, expires_in=settings.jwt_expire_minutes * 60)


# ── User management endpoints ─────────────────────────────────────────────────

@app.get("/api/users")
async def list_users(_user=Depends(_auth_dep)) -> list[dict]:
    """List all registered users. Admin endpoint."""
    from jarvis.auth.core import list_users as _list_users
    db_path = get_settings().reports_dir / "jarvis.db"
    if not db_path.exists():
        return []
    return _list_users(db_path)


@app.delete("/api/users/{username}", status_code=204)
async def delete_user(username: str, _user=Depends(_auth_dep)) -> None:
    """Delete a user account by username."""
    from jarvis.auth.core import delete_user as _delete_user
    db_path = get_settings().reports_dir / "jarvis.db"
    if not _delete_user(db_path, username):
        raise HTTPException(status_code=404, detail=f"User '{username}' not found.")


@app.patch("/api/users/{username}/role")
async def update_user_role(username: str, body: dict, _user=Depends(_auth_dep)) -> dict:
    """Update a user's role. Valid roles: admin, user, readonly."""
    from jarvis.auth.core import update_user_role as _update_role
    role = body.get("role", "")
    if role not in ("admin", "user", "readonly"):
        raise HTTPException(status_code=422, detail="role must be one of: admin, user, readonly")
    db_path = get_settings().reports_dir / "jarvis.db"
    if not _update_role(db_path, username, role):
        raise HTTPException(status_code=404, detail=f"User '{username}' not found.")
    return {"username": username, "role": role}


# ── Schedule endpoints ─────────────────────────────────────────────────────────

@app.post("/api/schedules", response_model=ScheduleResponse)
async def create_schedule(req: ScheduleRequest, _user=Depends(_auth_dep)) -> ScheduleResponse:
    """Create a recurring proactive agent job (research or monitor)."""
    from apscheduler.triggers.cron import CronTrigger
    from jarvis.scheduler.core import JOB_FUNCTIONS, get_scheduler

    scheduler = get_scheduler()
    if scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler not running.")

    if req.job_type not in JOB_FUNCTIONS:
        raise HTTPException(status_code=400, detail=f"Unknown job_type '{req.job_type}'. Use 'research' or 'monitor'.")

    parts = req.cron.strip().split()
    if len(parts) != 5:
        raise HTTPException(status_code=400, detail="cron must have 5 fields: minute hour day month weekday")

    minute, hour, day, month, day_of_week = parts
    trigger = CronTrigger(minute=minute, hour=hour, day=day, month=month, day_of_week=day_of_week, timezone="UTC")

    settings = get_settings()
    job = scheduler.add_job(
        JOB_FUNCTIONS[req.job_type],
        trigger,
        kwargs={
            **req.params,
            "session_id": req.session_id or str(uuid.uuid4()),
            "rabbitmq_url": settings.rabbitmq_url,
            "queue_name": settings.rabbitmq_task_queue,
        },
        id=str(uuid.uuid4()),
    )
    log.info("schedule_created", job_id=job.id, job_type=req.job_type, cron=req.cron)
    return ScheduleResponse(job_id=job.id, message=f"'{req.job_type}' job scheduled (cron: {req.cron} UTC).")


@app.get("/api/schedules", response_model=list[ScheduleItem])
async def list_schedules(_user=Depends(_auth_dep)) -> list[ScheduleItem]:
    """List all active scheduled jobs."""
    from jarvis.scheduler.core import JOB_FUNCTIONS, get_scheduler

    scheduler = get_scheduler()
    if scheduler is None:
        return []

    func_to_type = {v.__name__: k for k, v in JOB_FUNCTIONS.items()}
    items = []
    for job in scheduler.get_jobs():
        job_type = func_to_type.get(job.func.__name__, job.func.__name__)
        kwargs = job.kwargs or {}
        subject = kwargs.get("topic") or kwargs.get("query") or ""
        next_run = job.next_run_time.isoformat() if job.next_run_time else None
        cron_str = str(job.trigger)
        items.append(ScheduleItem(job_id=job.id, job_type=job_type, subject=subject, cron=cron_str, next_run=next_run))
    return items


@app.get("/api/schedules/{job_id}", response_model=ScheduleItem)
async def get_schedule(job_id: str, _user=Depends(_auth_dep)) -> ScheduleItem:
    """Return details for a single scheduled job by ID."""
    from jarvis.scheduler.core import JOB_FUNCTIONS, get_scheduler

    scheduler = get_scheduler()
    if scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler not running.")

    job = scheduler.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    func_to_type = {v.__name__: k for k, v in JOB_FUNCTIONS.items()}
    job_type = func_to_type.get(job.func.__name__, job.func.__name__)
    kwargs = job.kwargs or {}
    subject = kwargs.get("topic") or kwargs.get("query") or ""
    next_run = job.next_run_time.isoformat() if job.next_run_time else None
    return ScheduleItem(job_id=job.id, job_type=job_type, subject=subject,
                        cron=str(job.trigger), next_run=next_run)


@app.delete("/api/schedules/{job_id}", response_model=ScheduleResponse)
async def delete_schedule(job_id: str, _user=Depends(_auth_dep)) -> ScheduleResponse:
    """Remove a scheduled job by ID."""
    from apscheduler.jobstores.base import JobLookupError
    from jarvis.scheduler.core import get_scheduler

    scheduler = get_scheduler()
    if scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler not running.")
    try:
        scheduler.remove_job(job_id)
        log.info("schedule_deleted", job_id=job_id)
        return ScheduleResponse(job_id=job_id, message="Job removed.")
    except JobLookupError:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")


@app.patch("/api/schedules/{job_id}", response_model=ScheduleResponse)
async def reschedule_job(job_id: str, body: dict, _user=Depends(_auth_dep)) -> ScheduleResponse:
    """Update the cron trigger of an existing scheduled job."""
    from apscheduler.jobstores.base import JobLookupError
    from jarvis.scheduler.core import _parse_cron, get_scheduler

    cron = body.get("cron", "")
    if not cron:
        raise HTTPException(status_code=422, detail="'cron' field is required (5-field cron expression)")

    scheduler = get_scheduler()
    if scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler not running.")

    try:
        trigger = _parse_cron(cron)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    try:
        scheduler.reschedule_job(job_id, trigger=trigger)
        log.info("schedule_updated", job_id=job_id, cron=cron)
        return ScheduleResponse(job_id=job_id, message=f"Job rescheduled (cron: {cron} UTC).")
    except JobLookupError:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")


# ── Audit endpoints ────────────────────────────────────────────────────────────

@app.get("/api/feedback/stats")
async def get_feedback_stats(session_id: str | None = None, _user=Depends(_auth_dep)) -> dict:
    """Return aggregate feedback statistics."""
    from jarvis.memory.feedback import get_feedback_stats as _get_stats
    db_path = get_settings().reports_dir / "jarvis.db"
    return _get_stats(db_path, session_id=session_id)


@app.get("/api/improvement-report")
async def get_improvement_report(_user=Depends(_auth_dep)) -> dict:
    """Return the latest self-improvement analysis report content."""
    settings = get_settings()
    report_path = settings.reports_dir / "improvement_suggestions.md"
    if not report_path.exists():
        return {"content": None}
    return {"content": report_path.read_text(encoding="utf-8")}


@app.get("/api/audit")
async def get_audit(
    limit: int = 50, offset: int = 0, session_id: str | None = None,
    _user=Depends(_auth_dep),
) -> list[dict]:
    """Return paginated tool-call audit log entries, newest first."""
    from jarvis.security.audit import get_recent_audit
    db_path = get_settings().reports_dir / "jarvis.db"
    return get_recent_audit(db_path, limit=limit, offset=offset, session_id=session_id)


@app.get("/api/audit/stats")
async def get_audit_stats_endpoint(
    since_ts: float | None = None,
    _user=Depends(_auth_dep),
) -> dict:
    """Return aggregated audit statistics: total calls, approval rate, top tools, risk breakdown."""
    from jarvis.security.audit import get_audit_stats
    db_path = get_settings().reports_dir / "jarvis.db"
    return get_audit_stats(db_path, since_ts=since_ts)


@app.get("/api/turns")
async def get_agent_turns(
    limit: int = 50, session_id: str | None = None,
    _user=Depends(_auth_dep),
) -> list[dict]:
    """Return recent per-agent-turn records (tokens, latency, model, tool calls)."""
    from jarvis.memory.turns import get_turn_stats
    db_path = get_settings().reports_dir / "jarvis.db"
    return get_turn_stats(db_path, session_id=session_id, limit=limit)


@app.get("/api/failures")
async def get_failure_patterns_endpoint(
    tool_name: str | None = None,
    limit: int = 50,
    _user=Depends(_auth_dep),
) -> list[dict]:
    """Return tool failure patterns grouped by tool + error message, sorted by frequency."""
    from jarvis.memory.failures import get_failure_patterns
    db_path = get_settings().reports_dir / "jarvis.db"
    return get_failure_patterns(db_path, tool_name=tool_name, limit=limit)


@app.get("/api/stats")
async def system_stats(_user=Depends(_auth_dep)) -> dict:
    """Return a comprehensive system snapshot.

    Includes: session counts, total tokens & estimated cost, top tools,
    memory sizes (episodes, graph entities, preferences), job statuses, DB size.
    """
    from jarvis.api.stats import get_system_stats
    db_path = get_settings().reports_dir / "jarvis.db"
    return get_system_stats(db_path, sessions_active=len(_sessions))


# ── API access log ───────────────────────────────────────────────────────────

@app.get("/api/access-log")
async def get_access_log_endpoint(
    path: str | None = None,
    status: int | None = None,
    since_ts: float | None = None,
    limit: int = 100,
    _user=Depends(_auth_dep),
) -> list[dict]:
    """Return recent API access-log entries, newest first.

    Filter by ?path= (substring), ?status= (exact HTTP status), ?since_ts= (Unix timestamp).
    """
    from jarvis.api.access_log import get_access_log
    db_path = get_settings().reports_dir / "jarvis.db"
    return get_access_log(db_path, path_filter=path, status_filter=status,
                          since_ts=since_ts, limit=min(limit, 500))


@app.get("/api/access-log/stats")
async def get_access_log_stats(
    since_ts: float | None = None,
    _user=Depends(_auth_dep),
) -> dict:
    """Return aggregated API access stats: top paths, error rate, avg latency."""
    from jarvis.api.access_log import get_access_log_stats
    db_path = get_settings().reports_dir / "jarvis.db"
    return get_access_log_stats(db_path, since_ts=since_ts)


# ── Database maintenance ──────────────────────────────────────────────────────

@app.get("/api/maintenance/stats")
async def maintenance_db_stats(_user=Depends(_auth_dep)) -> dict:
    """Return table row counts and DB file size."""
    from jarvis.api.maintenance import get_db_stats
    db_path = get_settings().reports_dir / "jarvis.db"
    return get_db_stats(db_path)


@app.post("/api/maintenance/vacuum", status_code=200)
async def maintenance_vacuum(_user=Depends(_auth_dep)) -> dict:
    """Run SQLite VACUUM to reclaim disk space. Returns size_before, size_after, reclaimed_bytes."""
    from jarvis.api.maintenance import vacuum_db
    db_path = get_settings().reports_dir / "jarvis.db"
    loop = asyncio.get_running_loop()
    result_holder: list = []
    await loop.run_in_executor(_executor, lambda: result_holder.append(vacuum_db(db_path)))
    return result_holder[0] if result_holder else {}


@app.post("/api/maintenance/prune", status_code=200)
async def maintenance_prune(body: dict, _user=Depends(_auth_dep)) -> dict:
    """Delete old records from specified tables.

    Body:
      target         (str) — "turns"|"episodes"|"audit"|"checkpoints"|"jobs"|"all"
      older_than_days (int) — delete records older than this many days

    Returns {deleted_counts: {table: N}}.
    """
    from jarvis.api.maintenance import prune_data
    target = str(body.get("target") or "all").lower()
    valid_targets = {"turns", "episodes", "audit", "checkpoints", "jobs", "all"}
    if target not in valid_targets:
        raise HTTPException(status_code=422, detail=f"'target' must be one of {sorted(valid_targets)}")
    older_than_days = int(body.get("older_than_days") or 0)
    if older_than_days <= 0:
        raise HTTPException(status_code=422, detail="'older_than_days' must be a positive integer")
    db_path = get_settings().reports_dir / "jarvis.db"
    loop = asyncio.get_running_loop()
    result_holder: list = []
    await loop.run_in_executor(
        _executor,
        lambda: result_holder.append(prune_data(db_path, target, older_than_days)),
    )
    return result_holder[0] if result_holder else {}


@app.get("/api/analytics/agents")
async def get_agent_analytics(
    agent_type: str | None = None,
    hours: float = 24,
    _user=Depends(_auth_dep),
) -> list[dict]:
    """Return per-agent performance stats: latency percentiles, token usage, call counts."""
    import time
    from jarvis.memory.analytics import get_agent_performance
    db_path = get_settings().reports_dir / "jarvis.db"
    since_ts = time.time() - hours * 3600
    return get_agent_performance(db_path, agent_type=agent_type, since_ts=since_ts)


@app.get("/api/analytics/tools")
async def get_tool_analytics(
    hours: float = 24,
    _user=Depends(_auth_dep),
) -> list[dict]:
    """Return per-tool performance stats derived from audit_log: error rate, latency."""
    import time
    from jarvis.memory.analytics import get_tool_performance
    db_path = get_settings().reports_dir / "jarvis.db"
    since_ts = time.time() - hours * 3600
    return get_tool_performance(db_path, since_ts=since_ts)


@app.get("/api/memory/stats")
async def get_memory_stats(_user=Depends(_auth_dep)) -> dict:
    """Return row counts for each memory subsystem (episodes, feedback, preferences, failures)."""
    import sqlite3
    db_path = get_settings().reports_dir / "jarvis.db"
    if not db_path.exists():
        return {"episodes": 0, "feedback": 0, "preferences": 0, "failures": 0}
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            def _count(table: str) -> int:
                try:
                    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                except Exception:
                    return 0
            return {
                "episodes": _count("episodes"),
                "feedback": _count("feedback"),
                "preferences": _count("user_preferences"),
                "failures": _count("tool_failures"),
            }
        finally:
            conn.close()
    except Exception:
        return {"episodes": 0, "feedback": 0, "preferences": 0, "failures": 0}


# ── Preferences endpoints ────────────────────────────────────────────────────

@app.get("/api/preferences/{user_id}")
async def get_user_preferences(user_id: str, _user=Depends(_auth_dep)) -> list[dict]:
    """Return all stored preferences for a user with full metadata."""
    from jarvis.memory.preferences import get_preferences_with_metadata
    db_path = get_settings().reports_dir / "jarvis.db"
    return get_preferences_with_metadata(db_path, user_id)


@app.delete("/api/preferences/{user_id}/{category}/{key}", status_code=204)
async def delete_user_preference(
    user_id: str, category: str, key: str, _user=Depends(_auth_dep)
) -> None:
    """Delete a single preference entry. Returns 404 if the entry does not exist."""
    from jarvis.memory.preferences import delete_preference
    db_path = get_settings().reports_dir / "jarvis.db"
    deleted = delete_preference(db_path, user_id, category, key)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Preference '{category}/{key}' not found for user '{user_id}'.")


# ── Knowledge graph entity browser ───────────────────────────────────────────

@app.get("/api/knowledge-graph/entities")
async def list_kg_entities(
    user_id: str = "shared",
    entity_type: str | None = None,
    limit: int = 50,
    offset: int = 0,
    _user=Depends(_auth_dep),
) -> list[dict]:
    """List knowledge graph entities with metadata, newest first.

    Optionally filter by entity_type (e.g. "concept", "person", "tool").
    """
    from jarvis.memory.graph import list_entities
    db_path = get_settings().reports_dir / "jarvis.db"
    return list_entities(db_path, user_id=user_id, entity_type=entity_type,
                         limit=limit, offset=offset)


@app.get("/api/knowledge-graph/entities/{name}")
async def get_kg_entity(
    name: str,
    user_id: str = "shared",
    _user=Depends(_auth_dep),
) -> dict:
    """Return a single entity with its outgoing relationships."""
    from jarvis.memory.graph import get_entity
    db_path = get_settings().reports_dir / "jarvis.db"
    entity = get_entity(db_path, name, user_id=user_id)
    if not entity:
        raise HTTPException(status_code=404, detail=f"Entity '{name}' not found")
    return entity


@app.get("/api/knowledge-graph/entities/{name}/neighbors")
async def get_kg_entity_neighbors(
    name: str,
    user_id: str = "shared",
    depth: int = 1,
    _user=Depends(_auth_dep),
) -> dict:
    """Return a BFS subgraph of neighbors up to `depth` hops from entity `name`.

    Returns {nodes: [{name, type, description}], edges: [{from, relation, to}]}.
    depth is capped at 3.
    """
    from jarvis.memory.graph import get_entity_neighbors
    db_path = get_settings().reports_dir / "jarvis.db"
    return get_entity_neighbors(db_path, name, user_id=user_id, depth=depth)


# ── Knowledge graph export ───────────────────────────────────────────────────

@app.get("/api/knowledge-graph/export")
async def export_knowledge_graph(
    user_id: str = "shared",
    limit: int = 500,
    _user=Depends(_auth_dep),
) -> dict:
    """Export the full knowledge graph as {nodes, edges} for D3/Cytoscape visualisation."""
    from jarvis.memory.graph import export_graph
    db_path = get_settings().reports_dir / "jarvis.db"
    return export_graph(db_path, user_id=user_id, limit=limit)


# ── Cache management endpoints ───────────────────────────────────────────────

@app.get("/api/cache/stats")
async def get_cache_stats_endpoint(_user=Depends(_auth_dep)) -> dict:
    """Return tool-cache statistics: live entry count, expired count, per-tool breakdown."""
    from jarvis.tools.cache import get_cache_stats
    db_path = get_settings().reports_dir / "jarvis.db"
    return get_cache_stats(db_path)


@app.delete("/api/cache")
async def clear_cache_endpoint(_user=Depends(_auth_dep)) -> dict:
    """Flush all cached tool results. Returns the number of entries deleted."""
    from jarvis.tools.cache import clear_cache
    db_path = get_settings().reports_dir / "jarvis.db"
    deleted = clear_cache(db_path)
    return {"deleted": deleted}


@app.get("/api/tools/cache/ttls")
async def get_cache_ttls_endpoint(_user=Depends(_auth_dep)) -> dict:
    """Return the current per-tool cache TTL map (seconds). Tools absent are uncached."""
    from jarvis.tools.cache import get_cache_ttls
    return get_cache_ttls()


@app.put("/api/tools/cache/ttls/{tool_name}")
async def set_cache_ttl_endpoint(tool_name: str, body: dict, _user=Depends(_auth_dep)) -> dict:
    """Set the cache TTL (seconds) for a specific tool. Pass ttl_seconds=0 to disable caching."""
    from jarvis.tools.cache import set_cache_ttl
    ttl = body.get("ttl_seconds")
    if ttl is None:
        raise HTTPException(status_code=422, detail="'ttl_seconds' is required")
    if not isinstance(ttl, int) or ttl < 0:
        raise HTTPException(status_code=422, detail="'ttl_seconds' must be a non-negative integer")
    return set_cache_ttl(tool_name, ttl)


# ── Circuit breaker endpoints ─────────────────────────────────────────────────

@app.get("/api/tools/circuit-breakers")
async def list_circuit_breakers(_user=Depends(_auth_dep)) -> list[dict]:
    """Return the current state of every tool circuit breaker."""
    from jarvis.tools.circuit_breaker import get_all_states
    return get_all_states()


@app.delete("/api/tools/circuit-breakers/{tool_name}")
async def reset_circuit_breaker(tool_name: str, _user=Depends(_auth_dep)) -> dict:
    """Reset a tool's circuit breaker back to CLOSED state."""
    from jarvis.tools.circuit_breaker import reset_breaker
    found = reset_breaker(tool_name)
    if not found:
        raise HTTPException(status_code=404, detail=f"No breaker found for tool '{tool_name}'")
    return {"tool": tool_name, "state": "closed"}


@app.patch("/api/tools/circuit-breakers/{tool_name}")
async def update_circuit_breaker(tool_name: str, body: dict, _user=Depends(_auth_dep)) -> dict:
    """Update failure_threshold and/or reset_timeout_s for a tool's circuit breaker."""
    from jarvis.tools.circuit_breaker import update_breaker_config
    failure_threshold = body.get("failure_threshold")
    reset_timeout_s = body.get("reset_timeout_s")
    if failure_threshold is None and reset_timeout_s is None:
        raise HTTPException(status_code=422, detail="Provide at least one of: failure_threshold, reset_timeout_s")
    if failure_threshold is not None and (not isinstance(failure_threshold, int) or failure_threshold < 1):
        raise HTTPException(status_code=422, detail="failure_threshold must be a positive integer")
    if reset_timeout_s is not None and (not isinstance(reset_timeout_s, (int, float)) or reset_timeout_s <= 0):
        raise HTTPException(status_code=422, detail="reset_timeout_s must be a positive number")
    return update_breaker_config(tool_name, failure_threshold=failure_threshold, reset_timeout_s=reset_timeout_s)


@app.get("/api/approval/pending")
async def get_pending_approvals(session_id: str, _user=Depends(_auth_dep)) -> list[dict]:
    """Return pending approval requests for a session."""
    session = _sessions.get(session_id)
    if not session or not session.get("approval_gate"):
        return []
    return session["approval_gate"].get_pending()


# ── Session management endpoints ─────────────────────────────────────────────

def _session_info(sid: str, s: dict) -> SessionInfo:
    agent = s.get("agent")
    try:
        raw = agent.get_usage_summary() if agent else {}
        usage = dict(raw) if isinstance(raw, dict) else {}
    except Exception:
        usage = {}
    return SessionInfo(
        session_id=sid,
        created_at=s.get("created_at", 0.0),
        message_count=len(s.get("messages", [])),
        user_id=s.get("user_id"),
        agent_type=type(agent).__name__ if agent else "unknown",
        last_turn_tools=list(getattr(agent, "_turn_tool_calls", [])) if agent else [],
        usage=usage,
        fork_of=s.get("fork_of"),
        forked_at=s.get("forked_at"),
    )


@app.get("/api/sessions", response_model=list[SessionInfo])
async def list_sessions(_user=Depends(_auth_dep)) -> list[SessionInfo]:
    """List all currently active in-memory sessions."""
    return sorted(
        [_session_info(sid, s) for sid, s in _sessions.items()],
        key=lambda x: x.created_at,
        reverse=True,
    )


@app.get("/api/sessions/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str, _user=Depends(_auth_dep)) -> SessionInfo:
    """Return metadata and usage for a specific session."""
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return _session_info(session_id, session)


@app.delete("/api/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str, _user=Depends(_auth_dep)) -> None:
    """Evict a session from memory, freeing its agent and message history."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    del _sessions[session_id]
    _active_websockets.pop(session_id, None)
    _session_activity.pop(session_id, None)
    log.info("session_deleted", session_id=session_id)


# ── Cross-session context injection ──────────────────────────────────────────

@app.post("/api/sessions/{session_id}/inject-context", status_code=200)
async def inject_context(
    session_id: str,
    body: dict,
    _user=Depends(_auth_dep),
) -> dict:
    """Prepend context from other sessions into this session's message history.

    Body:
      sources        list of {session_id: str, max_chars?: int (default 2000)}
      label?         str — heading for the injected block (default: "Injected context")

    Returns: {injected_chars, source_count, session_id}

    The context is injected as a system message at the front of the target
    session's messages so it is available on the next agent turn.
    """
    from jarvis.agents.context_inject import build_context_block, inject_into_messages
    from jarvis.memory.sessions import load_sessions

    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    sources_spec = body.get("sources") or []
    if not sources_spec:
        raise HTTPException(status_code=422, detail="'sources' list is required")

    label = str(body.get("label") or "Injected context from prior sessions")
    db_path = get_settings().reports_dir / "jarvis.db"
    all_sessions = {r["session_id"]: r for r in load_sessions(db_path, ttl_minutes=99999)}

    sources: list[tuple[str, list[dict]]] = []
    for spec in sources_spec:
        src_id = str(spec.get("session_id") or "")
        if not src_id:
            continue
        max_chars = int(spec.get("max_chars") or 2000)
        # Check in-memory sessions first, then persisted
        if src_id in _sessions:
            msgs = list(_sessions[src_id].get("messages", []))
        elif src_id in all_sessions:
            msgs = all_sessions[src_id]["messages"]
        else:
            continue
        sources.append((src_id, msgs))

    if not sources:
        raise HTTPException(status_code=422, detail="No valid source sessions found")

    result = build_context_block(sources, label=label)
    if result.injected_chars == 0:
        return {"injected_chars": 0, "source_count": 0, "session_id": session_id}

    _sessions[session_id]["messages"] = inject_into_messages(
        _sessions[session_id].get("messages", []),
        result.context_block,
    )
    return {
        "injected_chars": result.injected_chars,
        "source_count": result.source_count,
        "session_id": session_id,
    }


# ── Session notes ────────────────────────────────────────────────────────────

@app.get("/api/sessions/{session_id}/notes")
async def get_session_notes(session_id: str, _user=Depends(_auth_dep)) -> list[dict]:
    """Return all notes for a session, oldest first."""
    from jarvis.memory.notes import list_notes
    db_path = get_settings().reports_dir / "jarvis.db"
    return list_notes(db_path, session_id)


@app.post("/api/sessions/{session_id}/notes", status_code=201)
async def add_session_note(
    session_id: str,
    body: dict,
    _user=Depends(_auth_dep),
) -> dict:
    """Add a free-text note to a session. Body: {content: str, author?: str}"""
    from jarvis.memory.notes import add_note
    content = (body.get("content") or "").strip()
    if not content:
        raise HTTPException(status_code=422, detail="'content' is required")
    author = str(body.get("author") or "user")
    db_path = get_settings().reports_dir / "jarvis.db"
    return add_note(db_path, session_id, content, author=author)


@app.delete("/api/sessions/{session_id}/notes/{note_id}", status_code=200)
async def delete_session_note(
    session_id: str,
    note_id: str,
    _user=Depends(_auth_dep),
) -> dict:
    """Delete a session note by ID."""
    from jarvis.memory.notes import delete_note
    db_path = get_settings().reports_dir / "jarvis.db"
    removed = delete_note(db_path, note_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Note '{note_id}' not found")
    return {"note_id": note_id, "deleted": True}


# ── Session metadata and title ────────────────────────────────────────────────

@app.get("/api/sessions/{session_id}/metadata")
async def get_session_metadata(session_id: str, _user=Depends(_auth_dep)) -> dict:
    """Return metadata (title, tags, agent_type, timestamps) for a session.

    Does not include the full message history.
    """
    from jarvis.memory.sessions import get_metadata
    db_path = get_settings().reports_dir / "jarvis.db"
    meta = get_metadata(db_path, session_id)
    if not meta:
        # Fall back to in-memory session
        session = _sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
        agent = session.get("agent")
        return {
            "session_id": session_id,
            "agent_type": type(agent).__name__ if agent else "Unknown",
            "user_id": session.get("user_id"),
            "fork_of": session.get("fork_of"),
            "title": session.get("title"),
            "tags": [],
            "created_at": None,
            "updated_at": None,
        }
    return meta


@app.patch("/api/sessions/{session_id}/title", status_code=200)
async def update_session_title(
    session_id: str,
    body: dict,
    _user=Depends(_auth_dep),
) -> dict:
    """Manually set the title for a session. Body: {title: str}"""
    from jarvis.memory.sessions import set_title
    title = (body.get("title") or "").strip()
    if not title:
        raise HTTPException(status_code=422, detail="'title' is required")
    db_path = get_settings().reports_dir / "jarvis.db"
    ok = set_title(db_path, session_id, title)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    # Also update in-memory session if active
    if session_id in _sessions:
        _sessions[session_id]["title"] = title
    return {"session_id": session_id, "title": title}


# ── Session tags ──────────────────────────────────────────────────────────────

@app.get("/api/sessions/{session_id}/tags")
async def get_session_tags(session_id: str, _user=Depends(_auth_dep)) -> list[str]:
    """Return all tags for a session."""
    from jarvis.memory.sessions import get_tags
    db_path = get_settings().reports_dir / "jarvis.db"
    return get_tags(db_path, session_id)


@app.post("/api/sessions/{session_id}/tags", status_code=201)
async def add_session_tag(
    session_id: str,
    body: dict,
    _user=Depends(_auth_dep),
) -> dict:
    """Add a tag to a session. Body: {tag: str}"""
    from jarvis.memory.sessions import add_tag
    tag = (body.get("tag") or "").strip().lower()
    if not tag:
        raise HTTPException(status_code=422, detail="'tag' is required")
    db_path = get_settings().reports_dir / "jarvis.db"
    ok = add_tag(db_path, session_id, tag)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return {"session_id": session_id, "tag": tag}


@app.delete("/api/sessions/{session_id}/tags/{tag}", status_code=200)
async def remove_session_tag(
    session_id: str,
    tag: str,
    _user=Depends(_auth_dep),
) -> dict:
    """Remove a tag from a session."""
    from jarvis.memory.sessions import remove_tag
    db_path = get_settings().reports_dir / "jarvis.db"
    removed = remove_tag(db_path, session_id, tag.lower())
    if not removed:
        raise HTTPException(status_code=404, detail=f"Tag '{tag}' not found on session")
    return {"session_id": session_id, "tag": tag.lower(), "removed": True}


# ── Session full-text search ───────────────────────────────────────────────────

@app.get("/api/sessions/search")
async def search_sessions_endpoint(
    q: str,
    tag: str | None = None,
    _user=Depends(_auth_dep),
) -> list[dict]:
    """Full-text search across persisted session message content.

    Returns [{session_id, snippet, rank}] ordered by relevance.
    Optionally filter to sessions with a given tag.
    """
    if not q.strip():
        raise HTTPException(status_code=422, detail="'q' query parameter is required")
    from jarvis.memory.sessions import search_sessions
    db_path = get_settings().reports_dir / "jarvis.db"
    return search_sessions(db_path, q, tag=tag)


# ── Plan history endpoints ────────────────────────────────────────────────────

@app.get("/api/plans")
async def list_plans(
    session_id: str | None = None,
    user_id: str | None = None,
    limit: int = 50,
    _user=Depends(_auth_dep),
) -> list[dict]:
    """Return plan execution records, newest first. Filter by session_id or user_id."""
    from jarvis.agents.executor import get_plans
    db_path = get_settings().reports_dir / "jarvis.db"
    return get_plans(db_path, session_id=session_id, user_id=user_id, limit=limit)


@app.get("/api/plans/{plan_id}")
async def get_plan_by_id(plan_id: str, _user=Depends(_auth_dep)) -> dict:
    """Return a single plan record by ID including all steps."""
    from jarvis.agents.executor import get_plan
    db_path = get_settings().reports_dir / "jarvis.db"
    plan = get_plan(db_path, plan_id)
    if plan is None:
        raise HTTPException(status_code=404, detail=f"Plan '{plan_id}' not found.")
    return plan


# ── Peer coordination endpoints ───────────────────────────────────────────────

_peer_coordinator = None


@app.get("/api/peer/list")
async def get_peer_list() -> list[dict]:
    """Return discovered peer JARVIS nodes."""
    if _peer_coordinator is None:
        return []
    return _peer_coordinator.get_peer_list()


@app.post("/api/peer/sync")
async def receive_peer_sync(request: Request) -> dict:
    """Accept an incoming knowledge graph delta from a peer node."""
    try:
        from jarvis.peer.protocol import merge_incoming_delta

        body = await request.json()
        settings = get_settings()
        db_path = settings.reports_dir / "jarvis.db"
        count = merge_incoming_delta(body, db_path)
        log.info("peer_sync_received", items=count)
        return {"status": "ok", "merged": count}
    except Exception as exc:
        log.error("peer_sync_receive_error", error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/peer/delta")
async def get_peer_delta(since_ts: float = 0.0, _user=Depends(_auth_dep)) -> dict:
    """Serve local graph delta to a requesting peer (since the given timestamp)."""
    from jarvis.edge.sync import export_delta

    settings = get_settings()
    db_path = settings.reports_dir / "jarvis.db"
    return export_delta(db_path, since_ts=since_ts)


# ── Reports endpoints ──────────────────────────────────────────────────────────

@app.get("/api/reports")
async def list_reports(limit: int = 50, offset: int = 0) -> list[dict]:
    """List saved research reports, newest first. Supports pagination."""
    settings = get_settings()
    reports_dir = settings.reports_dir
    if not reports_dir.exists():
        return []
    files = sorted(reports_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    page = files[offset: offset + limit]
    return [
        {
            "name": p.name,
            "size_bytes": p.stat().st_size,
            "modified": p.stat().st_mtime,
        }
        for p in page
    ]


@app.get("/api/reports/{filename}")
async def get_report(filename: str) -> dict:
    """Return the content of a single report file."""
    settings = get_settings()
    report_path = (settings.reports_dir / filename).resolve()
    if not str(report_path).startswith(str(settings.reports_dir.resolve())):
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not report_path.exists() or not report_path.suffix == ".md":
        raise HTTPException(status_code=404, detail="Report not found")
    return {"name": filename, "content": report_path.read_text(encoding="utf-8")}


# ── Voice status endpoint ──────────────────────────────────────────────────────

_voice_active: bool = False


@app.get("/api/voice/status")
async def voice_status() -> dict:
    """Return whether voice mode is currently active."""
    return {"active": _voice_active}


@app.get("/api/reminders")
async def list_reminders(_user=Depends(_auth_dep)) -> list[dict]:
    """Return all pending reminders with their next scheduled fire time."""
    from jarvis.tools.plugins.reminder_manager import get_reminders
    return get_reminders()


# ── Config introspection endpoint ─────────────────────────────────────────────

@app.get("/api/config")
async def get_config(_user=Depends(_auth_dep)) -> dict:
    """Return the current runtime configuration (feature flags and model settings). Secrets are redacted."""
    s = get_settings()
    return {
        "auth_enabled": s.auth_enabled,
        "proactive_enabled": s.proactive_enabled,
        "peer_enabled": s.peer_enabled,
        "rate_limit_enabled": s.rate_limit_enabled,
        "otel_enabled": s.otel_enabled,
        "model": s.model,
        "fast_model": s.fast_model,
        "routing_strategy": s.routing_strategy,
        "memory_retention_days": s.memory_retention_days,
        "auto_training_enabled": getattr(s, "auto_training_enabled", False),
        "max_tokens": s.max_tokens,
        "max_search_calls": s.max_search_calls,
    }


# ── Memory browser endpoints ───────────────────────────────────────────────────

@app.get("/api/memory/episodes")
async def get_memory_episodes(
    user_id: str | None = None,
    session_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
    _user=Depends(_auth_dep),
) -> list[dict]:
    """Return paginated episodic memory entries, newest first.

    Filter by ?user_id= and/or ?session_id=. Includes importance score.
    """
    from jarvis.memory.episodic import list_episodes
    db_path = get_settings().reports_dir / "jarvis.db"
    return list_episodes(db_path, user_id=user_id, session_id=session_id,
                         limit=min(limit, 200), offset=offset)


@app.get("/api/memory/episodes/{episode_id}")
async def get_memory_episode(episode_id: int, _user=Depends(_auth_dep)) -> dict:
    """Return a single episodic memory entry by ID."""
    from jarvis.memory.episodic import get_episode
    db_path = get_settings().reports_dir / "jarvis.db"
    ep = get_episode(db_path, episode_id)
    if not ep:
        raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")
    return ep


@app.post("/api/memory/episodes/{episode_id}/boost", status_code=200)
async def boost_episode_importance(
    episode_id: int,
    body: dict = {},
    _user=Depends(_auth_dep),
) -> dict:
    """Manually boost the importance of an episodic memory entry.

    Body (optional): {delta: float (default 0.2)}
    Higher importance = surfaced more often in proactive memory injection.
    """
    from jarvis.memory.episodic import boost_importance, get_episode
    delta = float(body.get("delta") or 0.2)
    if delta <= 0:
        raise HTTPException(status_code=422, detail="'delta' must be positive")
    db_path = get_settings().reports_dir / "jarvis.db"
    boost_importance(db_path, episode_id, delta)
    ep = get_episode(db_path, episode_id)
    if not ep:
        raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")
    return ep


@app.delete("/api/memory/episodes/{episode_id}", status_code=200)
async def delete_episode_endpoint(episode_id: int, _user=Depends(_auth_dep)) -> dict:
    """Delete a single episodic memory entry."""
    from jarvis.memory.episodic import delete_episode
    db_path = get_settings().reports_dir / "jarvis.db"
    removed = delete_episode(db_path, episode_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")
    return {"episode_id": episode_id, "deleted": True}


@app.get("/api/memory/graph")
async def get_memory_graph(entity: str | None = None, limit: int = 50, _user=Depends(_auth_dep)) -> list[dict]:
    """Return knowledge graph entities and their relationships."""
    db_path = get_settings().reports_dir / "jarvis.db"
    if not db_path.exists():
        return []
    try:
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            if entity:
                ents = conn.execute(
                    "SELECT name, type, description, user_id, created_at FROM entities "
                    "WHERE name LIKE ? LIMIT ?",
                    (f"%{entity}%", limit),
                ).fetchall()
                rels = conn.execute(
                    "SELECT from_entity, relation, to_entity, notes, user_id, created_at "
                    "FROM relationships WHERE from_entity LIKE ? OR to_entity LIKE ? LIMIT ?",
                    (f"%{entity}%", f"%{entity}%", limit),
                ).fetchall()
            else:
                ents = conn.execute(
                    "SELECT name, type, description, user_id, created_at FROM entities "
                    "ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
                rels = conn.execute(
                    "SELECT from_entity, relation, to_entity, notes, user_id, created_at "
                    "FROM relationships ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        finally:
            conn.close()
        return [
            {"record_type": "entity", **dict(r)} for r in ents
        ] + [
            {"record_type": "relationship", **dict(r)} for r in rels
        ]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/memory/episodes/search")
async def search_memory_episodes(
    q: str,
    user_id: str | None = None,
    limit: int = 20,
    _user=Depends(_auth_dep),
) -> list[dict]:
    """Full-text search episodic memory by keyword."""
    from jarvis.memory.episodic import search_episodes
    db_path = get_settings().reports_dir / "jarvis.db"
    return search_episodes(db_path, q, limit=limit, user_id=user_id)


@app.delete("/api/memory/episodes", status_code=200)
async def delete_memory_episodes(
    session_id: str | None = None,
    user_id: str | None = None,
    _user=Depends(_auth_dep),
) -> dict:
    """Delete episodes filtered by session_id and/or user_id. At least one filter required."""
    if not session_id and not user_id:
        raise HTTPException(status_code=400, detail="Provide at least one of: session_id, user_id")
    from jarvis.memory.episodic import delete_episodes
    db_path = get_settings().reports_dir / "jarvis.db"
    deleted = delete_episodes(db_path, session_id=session_id, user_id=user_id)
    return {"deleted": deleted}


@app.get("/api/memory/summaries/{user_id}")
async def get_session_summaries(
    user_id: str,
    limit: int = 20,
    _user=Depends(_auth_dep),
) -> list[dict]:
    """Return session summaries for a user, newest first."""
    from jarvis.memory.preferences import get_session_summaries_full
    db_path = get_settings().reports_dir / "jarvis.db"
    return get_session_summaries_full(db_path, user_id, limit=limit)


@app.post("/api/knowledge-graph/deduplicate", status_code=200)
async def deduplicate_knowledge_graph(
    body: dict = {},
    _user=Depends(_auth_dep),
) -> dict:
    """Run an entity deduplication pass on the knowledge graph.

    Body (optional):
      user_id             (str, default "shared")
      similarity_threshold (float, default 0.85) — Jaccard similarity cutoff
      dry_run             (bool, default false) — find pairs but do not merge

    Returns: {pairs_found, merged_count, duration_ms, pairs (only when dry_run=true)}
    """
    from jarvis.memory.graph_dedup import find_duplicate_pairs, deduplicate_entities

    user_id = str(body.get("user_id") or "shared")
    threshold = float(body.get("similarity_threshold") or 0.85)
    dry_run = bool(body.get("dry_run", False))
    db_path = get_settings().reports_dir / "jarvis.db"

    loop = asyncio.get_running_loop()
    result_holder: list = []

    def _run():
        t0 = time.perf_counter()
        pairs = find_duplicate_pairs(db_path, user_id=user_id, similarity_threshold=threshold)
        if dry_run:
            duration_ms = round((time.perf_counter() - t0) * 1000, 1)
            result_holder.append({
                "pairs_found": len(pairs),
                "merged_count": 0,
                "duration_ms": duration_ms,
                "pairs": [{"entity_a": a, "entity_b": b} for a, b in pairs],
            })
        else:
            from jarvis.memory.graph_dedup import merge_entities
            merged = 0
            for canonical, duplicate in pairs:
                if merge_entities(db_path, canonical, duplicate, user_id) >= 0:
                    merged += 1
            duration_ms = round((time.perf_counter() - t0) * 1000, 1)
            result_holder.append({
                "pairs_found": len(pairs),
                "merged_count": merged,
                "duration_ms": duration_ms,
            })

    await loop.run_in_executor(_executor, _run)
    return result_holder[0] if result_holder else {}


@app.delete("/api/knowledge-graph/entities/{name}", status_code=204)
async def delete_knowledge_graph_entity(
    name: str,
    user_id: str = "shared",
    _user=Depends(_auth_dep),
) -> None:
    """Delete a knowledge graph entity (and its relationships) by name."""
    from jarvis.memory.graph import delete_entity
    db_path = get_settings().reports_dir / "jarvis.db"
    if not delete_entity(db_path, name, user_id=user_id):
        raise HTTPException(status_code=404, detail=f"Entity '{name}' not found for user '{user_id}'.")


@app.delete("/api/knowledge-graph/relationships", status_code=204)
async def delete_knowledge_graph_relationship(body: dict, _user=Depends(_auth_dep)) -> None:
    """Delete a specific relationship triple. Body: {from, relation, to, user_id?}."""
    from jarvis.memory.graph import delete_relationship
    frm = body.get("from", "")
    relation = body.get("relation", "")
    to = body.get("to", "")
    if not frm or not relation or not to:
        raise HTTPException(status_code=422, detail="'from', 'relation', and 'to' are required")
    user_id = body.get("user_id", "shared")
    db_path = get_settings().reports_dir / "jarvis.db"
    if not delete_relationship(db_path, frm, relation, to, user_id=user_id):
        raise HTTPException(status_code=404, detail="Relationship not found.")


# ── Metrics summary endpoint ───────────────────────────────────────────────────

@app.get("/api/metrics/summary")
async def get_metrics_summary(_user=Depends(_auth_dep)) -> dict:
    """Return a JSON summary of key Prometheus counters."""
    from prometheus_client import REGISTRY

    summary: dict = {
        "requests_total": {"http": 0, "websocket": 0},
        "tool_calls": {},
        "active_ws_connections": 0,
    }
    try:
        for metric in REGISTRY.collect():
            if metric.name == "jarvis_requests_total":
                for sample in metric.samples:
                    mode = sample.labels.get("mode", "unknown")
                    summary["requests_total"][mode] = int(sample.value)
            elif metric.name == "jarvis_tool_calls_total":
                for sample in metric.samples:
                    tool = sample.labels.get("tool_name", "unknown")
                    summary["tool_calls"][tool] = int(sample.value)
            elif metric.name == "jarvis_active_ws_connections":
                for sample in metric.samples:
                    summary["active_ws_connections"] = int(sample.value)
    except Exception:
        pass
    summary["active_sessions"] = len(_sessions)
    return summary


# ── Memory consolidation trigger ─────────────────────────────────────────────

@app.post("/api/memory/consolidate/{user_id}", status_code=202)
async def trigger_memory_consolidation(
    user_id: str,
    lookback_hours: int = 24,
    _user=Depends(_auth_dep),
) -> dict:
    """Trigger memory consolidation for a user in the background.

    Reads recent episodes, extracts preferences via LLM, and upserts them.
    Returns immediately with status 202; consolidation runs asynchronously.
    """
    settings = get_settings()
    db_path = settings.reports_dir / "jarvis.db"
    model = settings.model

    def _run() -> None:
        try:
            from jarvis.memory.consolidator import consolidate_user_memory
            count = consolidate_user_memory(db_path, user_id, model, lookback_hours=lookback_hours)
            log.info("consolidation_complete", user_id=user_id, preferences_written=count)
        except Exception as exc:
            log.error("consolidation_failed", user_id=user_id, error=str(exc))

    asyncio.get_running_loop().run_in_executor(_executor, _run)
    return {"status": "started", "user_id": user_id, "lookback_hours": lookback_hours}


# ── Autonomous events endpoint ─────────────────────────────────────────────────

@app.get("/api/autonomous/events")
async def get_autonomous_events(limit: int = 20, _user=Depends(_auth_dep)) -> list[dict]:
    """Return recent autonomous agent events from the database."""
    db_path = get_settings().reports_dir / "jarvis.db"
    try:
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS autonomous_events "
                "(id INTEGER PRIMARY KEY AUTOINCREMENT, event_type TEXT, summary TEXT, timestamp REAL)"
            )
            rows = conn.execute(
                "SELECT * FROM autonomous_events ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
        finally:
            conn.close()
        return [dict(r) for r in rows]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/autonomous/events", status_code=202)
async def inject_autonomous_event(
    body: dict,
    _user=Depends(_auth_dep),
) -> dict:
    """Inject an ExternalEvent into the live event bus to trigger the autonomous agent.

    Body: {"sub_type": str, "payload": dict}  (payload is optional)
    Returns 503 when the proactive event bus is not running.
    """
    settings = get_settings()
    if not settings.proactive_enabled:
        raise HTTPException(status_code=503, detail="Proactive event bus is not enabled.")

    sub_type = body.get("sub_type", "").strip()
    if not sub_type:
        raise HTTPException(status_code=422, detail="sub_type is required.")

    from jarvis.events.bus import get_event_bus
    from jarvis.events.types import ExternalEvent
    event = ExternalEvent(sub_type=sub_type, payload=body.get("payload") or {})
    bus = get_event_bus()
    await bus.publish(event)
    log.info("autonomous_event_injected", sub_type=sub_type)
    return {"status": "published", "sub_type": sub_type}


# ── Training pipeline API ──────────────────────────────────────────────────────

@app.get("/api/training/status")
async def get_training_status(_user=Depends(_auth_dep)) -> dict:
    """Return current auto-training config and last run info."""
    settings = get_settings()
    db_path = settings.reports_dir / "jarvis.db"
    try:
        from jarvis.training.tracking import get_last_run
        last_crawl = get_last_run(db_path, "crawl")
        last_ft = get_last_run(db_path, "finetune")
    except Exception:
        last_crawl = last_ft = None

    from jarvis.scheduler.core import get_scheduler
    scheduler = get_scheduler()
    next_crawl = next_ft = None
    if scheduler:
        crawl_job = scheduler.get_job("builtin_auto_crawl")
        ft_job = scheduler.get_job("builtin_auto_finetune")
        if crawl_job and crawl_job.next_run_time:
            next_crawl = crawl_job.next_run_time.isoformat()
        if ft_job and ft_job.next_run_time:
            next_ft = ft_job.next_run_time.isoformat()

    return {
        "auto_training_enabled": settings.auto_training_enabled,
        "topics": settings.auto_training_topics,
        "crawl_cron": settings.auto_crawl_cron,
        "finetune_cron": settings.auto_finetune_cron,
        "model_name": settings.auto_training_model_name,
        "min_new_docs": settings.auto_training_min_new_docs,
        "next_crawl": next_crawl,
        "next_finetune": next_ft,
        "last_crawl": {
            "status": last_crawl.status,
            "completed_at": last_crawl.completed_at,
            "docs_crawled": last_crawl.docs_crawled,
        } if last_crawl else None,
        "last_finetune": {
            "status": last_ft.status,
            "completed_at": last_ft.completed_at,
            "pairs_generated": last_ft.pairs_generated,
            "model_name": last_ft.model_name,
        } if last_ft else None,
    }


@app.get("/api/training/history")
async def get_training_history(limit: int = 20, _user=Depends(_auth_dep)) -> list[dict]:
    """Return recent training run records."""
    db_path = get_settings().reports_dir / "jarvis.db"
    try:
        from jarvis.training.tracking import get_history
        runs = get_history(db_path, limit=limit)
        return [
            {
                "id": r.id, "run_type": r.run_type, "status": r.status,
                "started_at": r.started_at, "completed_at": r.completed_at,
                "docs_crawled": r.docs_crawled, "pairs_generated": r.pairs_generated,
                "model_name": r.model_name, "notes": r.notes,
            }
            for r in runs
        ]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/training/runs/{run_id}")
async def get_training_run(run_id: int, _user=Depends(_auth_dep)) -> dict:
    """Return a specific training run record by ID."""
    from jarvis.training.tracking import get_run_by_id
    db_path = get_settings().reports_dir / "jarvis.db"
    if not db_path.exists():
        raise HTTPException(status_code=404, detail=f"Training run {run_id} not found.")
    run = get_run_by_id(db_path, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Training run {run_id} not found.")
    return {
        "id": run.id, "run_type": run.run_type, "status": run.status,
        "started_at": run.started_at, "completed_at": run.completed_at,
        "docs_crawled": run.docs_crawled, "pairs_generated": run.pairs_generated,
        "model_name": run.model_name, "notes": run.notes,
    }


@app.delete("/api/training/runs/{run_id}", status_code=204)
async def delete_training_run(run_id: int, _user=Depends(_auth_dep)) -> None:
    """Delete a training run record by ID."""
    from jarvis.training.tracking import delete_run
    db_path = get_settings().reports_dir / "jarvis.db"
    if not db_path.exists() or not delete_run(db_path, run_id):
        raise HTTPException(status_code=404, detail=f"Training run {run_id} not found.")


@app.post("/api/training/crawl")
async def trigger_crawl(_user=Depends(_auth_dep)) -> dict:
    """Trigger an immediate research crawl in the background."""
    settings = get_settings()
    db_path = settings.reports_dir / "jarvis.db"
    reports_dir = settings.reports_dir

    def _run():
        from jarvis.scheduler.core import _auto_crawl_job
        _auto_crawl_job(str(db_path), str(reports_dir))

    asyncio.get_running_loop().run_in_executor(_executor, _run)
    return {"status": "crawl started", "topics": settings.auto_training_topics}


@app.post("/api/training/finetune")
async def trigger_finetune(_user=Depends(_auth_dep)) -> dict:
    """Trigger an immediate fine-tuning run in the background."""
    settings = get_settings()
    db_path = settings.reports_dir / "jarvis.db"
    reports_dir = settings.reports_dir

    def _run():
        from jarvis.scheduler.core import _auto_finetune_job
        _auto_finetune_job(str(db_path), str(reports_dir))

    asyncio.get_running_loop().run_in_executor(_executor, _run)
    return {"status": "finetune started", "model_name": settings.auto_training_model_name}


# ── Evals ─────────────────────────────────────────────────────────────────────

@app.post("/api/evals", response_model=EvalRunResponse)
async def run_evals(
    body: EvalRunRequest,
    _user=Depends(_auth_dep),
) -> EvalRunResponse:
    """Run the eval suite and return aggregated results. Runs in background thread."""
    import uuid as _uuid
    settings = get_settings()

    def _run() -> EvalRunResponse:
        from jarvis.evals.suite import BASELINE_SUITE
        from jarvis.evals.runner import run_suite, summarize, persist_results

        results = run_suite(
            cases=BASELINE_SUITE,
            settings=settings,
            use_judge=body.use_judge,
            tags_filter=body.tags or None,
        )
        summary = summarize(results)
        persist_results(results, summary, settings.reports_dir)

        run_id = str(_uuid.uuid4())[:8]
        result_dicts = [
            {
                "case_id": r.case_id, "overall_pass": r.overall_pass,
                "contains_pass": r.contains_pass, "forbidden_pass": r.forbidden_pass,
                "latency_s": r.latency_s, "cost_usd": r.cost_usd,
                "judge_score": r.judge_score, "error": r.error,
            }
            for r in results
        ]
        # Persist to trend table
        from jarvis.evals.trend import record_run
        record_run(
            db_path=settings.reports_dir / "jarvis.db",
            run_id=run_id,
            total=summary["total"],
            passed=summary["passed"],
            failed=summary["failed"],
            pass_rate=summary["pass_rate"],
            avg_latency_s=summary["avg_latency_s"],
            total_cost_usd=summary["total_cost_usd"],
            avg_judge_score=summary.get("avg_judge_score"),
            tags=body.tags,
            results=result_dicts,
        )
        response = EvalRunResponse(
            run_id=run_id,
            total=summary["total"],
            passed=summary["passed"],
            failed=summary["failed"],
            pass_rate=summary["pass_rate"],
            avg_latency_s=summary["avg_latency_s"],
            total_cost_usd=summary["total_cost_usd"],
            avg_judge_score=summary.get("avg_judge_score"),
            results=[EvalResultItem(**d) for d in result_dicts],
        )
        # Fire notification + webhooks
        try:
            from jarvis.events.notifications import push_notification
            from jarvis.events.webhooks import fire_event
            _db = settings.reports_dir / "jarvis.db"
            _title = f"Eval run {run_id}: {summary['passed']}/{summary['total']} passed ({summary['pass_rate']:.0%})"
            _sev = "info" if summary["pass_rate"] >= 0.8 else "warning"
            push_notification(_db, event="eval.complete", title=_title, severity=_sev)
            fire_event(_db, "eval.complete", {
                "run_id": run_id, "pass_rate": summary["pass_rate"],
                "passed": summary["passed"], "total": summary["total"],
            })
        except Exception:
            pass
        return response

    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(_executor, _run)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/evals/results")
async def get_eval_results(
    limit: int = 10,
    _user=Depends(_auth_dep),
) -> list[dict]:
    """Return recent eval run summaries from eval_history.jsonl."""
    import json as _json
    settings = get_settings()
    history_path = settings.reports_dir / "eval_history.jsonl"
    if not history_path.exists():
        return []
    try:
        lines = history_path.read_text(encoding="utf-8").splitlines()
        records = [_json.loads(line) for line in lines if line.strip()]
        return [{"timestamp": r["timestamp"], "git_hash": r.get("git_hash", ""), **r["summary"]}
                for r in records[-limit:]]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Eval case CRUD ─────────────────────────────────────────────────────────────

def _eval_cases_path() -> "Path":
    from pathlib import Path as _Path
    return get_settings().reports_dir / "eval_cases.json"


def _load_eval_cases() -> list:
    import dataclasses
    from jarvis.evals.suite import BASELINE_SUITE, load_suite
    path = _eval_cases_path()
    if path.exists():
        try:
            return [dataclasses.asdict(c) for c in load_suite(path)]
        except Exception:
            pass
    return [dataclasses.asdict(c) for c in BASELINE_SUITE]


def _save_eval_cases(cases_dicts: list) -> None:
    import dataclasses
    from jarvis.evals.suite import EvalCase, save_suite
    cases = [EvalCase(**d) for d in cases_dicts]
    path = _eval_cases_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    save_suite(cases, path)


@app.get("/api/evals/cases")
async def list_eval_cases(_user=Depends(_auth_dep)) -> list[dict]:
    """Return all eval cases (custom + baseline)."""
    return _load_eval_cases()


@app.post("/api/evals/cases", status_code=201)
async def create_eval_case(body: dict, _user=Depends(_auth_dep)) -> dict:
    """Add a new eval case. Requires 'id' and 'prompt' fields."""
    case_id = str(body.get("id", "")).strip()
    prompt = str(body.get("prompt", "")).strip()
    if not case_id:
        raise HTTPException(status_code=422, detail="'id' is required")
    if not prompt:
        raise HTTPException(status_code=422, detail="'prompt' is required")

    cases = _load_eval_cases()
    if any(c["id"] == case_id for c in cases):
        raise HTTPException(status_code=409, detail=f"Eval case '{case_id}' already exists")

    new_case = {
        "id": case_id,
        "prompt": prompt,
        "expected_contains": body.get("expected_contains", []),
        "forbidden": body.get("forbidden", []),
        "judge_rubric": body.get("judge_rubric", ""),
        "tags": body.get("tags", []),
        "timeout_seconds": body.get("timeout_seconds", 120),
    }
    cases.append(new_case)
    _save_eval_cases(cases)
    log.info("eval_case_created", case_id=case_id)
    return new_case


@app.delete("/api/evals/cases/{case_id}", status_code=204)
async def delete_eval_case(case_id: str, _user=Depends(_auth_dep)) -> None:
    """Delete an eval case by ID."""
    cases = _load_eval_cases()
    remaining = [c for c in cases if c["id"] != case_id]
    if len(remaining) == len(cases):
        raise HTTPException(status_code=404, detail=f"Eval case '{case_id}' not found")
    _save_eval_cases(remaining)
    log.info("eval_case_deleted", case_id=case_id)


# ── Unified memory search ─────────────────────────────────────────────────────

@app.get("/api/memory/search")
async def memory_search(
    q: str,
    type: str | None = None,
    limit: int = 20,
    user_id: str | None = None,
    _user=Depends(_auth_dep),
) -> dict:
    """Fan-out search across all memory subsystems.

    type: "episodic" | "graph" | "reports" | None (all three)
    Returns {episodic: [...], graph: [...], reports: [...]}
    """
    if not q.strip():
        raise HTTPException(status_code=422, detail="q must not be empty")

    settings = get_settings()
    db_path = settings.reports_dir / "jarvis.db"
    result: dict[str, list] = {"episodic": [], "graph": [], "reports": []}
    wanted = {type} if type else {"episodic", "graph", "reports"}

    if "episodic" in wanted:
        try:
            from jarvis.memory.episodic import search_episodes
            rows = search_episodes(db_path, q, limit=limit, user_id=user_id)
            result["episodic"] = [
                {
                    "id": r.get("id"), "role": r.get("role"),
                    "content": str(r.get("content", ""))[:400],
                    "timestamp": r.get("timestamp"),
                    "session_id": r.get("session_id"),
                    "importance": r.get("importance", 1.0),
                }
                for r in rows
            ]
        except Exception:
            pass

    if "graph" in wanted:
        try:
            from jarvis.memory.graph import handle_query_knowledge_graph
            raw = handle_query_knowledge_graph(
                {"entity": q.split()[0] if q.split() else q, "depth": 2}, db_path
            )
            result["graph"] = [{"text": raw}] if raw and not raw.startswith("No ") else []
        except Exception:
            pass

    if "reports" in wanted:
        try:
            from jarvis.tools.memory import handle_search_memory
            raw = handle_search_memory({"query": q, "limit": limit}, settings.reports_dir)
            result["reports"] = [{"text": raw}] if raw and not raw.startswith("No research") else []
        except Exception:
            pass

    return result


# ── Per-tool configuration ────────────────────────────────────────────────────

@app.get("/api/tools/{tool_name}/config")
async def get_tool_config_endpoint(tool_name: str, _user=Depends(_auth_dep)) -> dict:
    """Return the per-tool runtime configuration (timeout, retries, cache TTL).

    Fields with null values use the global setting / default.
    """
    from jarvis.tools.tool_config import get_tool_config
    db_path = get_settings().reports_dir / "jarvis.db"
    return get_tool_config(db_path, tool_name)


@app.put("/api/tools/{tool_name}/config", status_code=200)
async def set_tool_config_endpoint(
    tool_name: str,
    body: dict,
    _user=Depends(_auth_dep),
) -> dict:
    """Set per-tool runtime overrides.

    Body (all optional):
      timeout_seconds   (int) — tool-level timeout, overrides global
      max_retries       (int) — retry count, overrides global
      cache_ttl_seconds (int) — cache entry lifetime for this tool
    """
    from jarvis.tools.tool_config import set_tool_config, get_tool_config
    timeout = body.get("timeout_seconds")
    retries = body.get("max_retries")
    cache_ttl = body.get("cache_ttl_seconds")
    if timeout is None and retries is None and cache_ttl is None:
        raise HTTPException(status_code=422, detail="At least one field is required")
    db_path = get_settings().reports_dir / "jarvis.db"
    set_tool_config(db_path, tool_name,
                    timeout_seconds=int(timeout) if timeout is not None else None,
                    max_retries=int(retries) if retries is not None else None,
                    cache_ttl_seconds=int(cache_ttl) if cache_ttl is not None else None)
    return get_tool_config(db_path, tool_name)


@app.delete("/api/tools/{tool_name}/config", status_code=200)
async def delete_tool_config_endpoint(tool_name: str, _user=Depends(_auth_dep)) -> dict:
    """Remove per-tool overrides, reverting this tool to global defaults."""
    from jarvis.tools.tool_config import delete_tool_config
    db_path = get_settings().reports_dir / "jarvis.db"
    removed = delete_tool_config(db_path, tool_name)
    if not removed:
        raise HTTPException(status_code=404, detail=f"No config override for tool '{tool_name}'")
    return {"tool_name": tool_name, "reset": True}


# ── Tool registry introspection ───────────────────────────────────────────────

@app.get("/api/tools")
async def list_tools(_user=Depends(_auth_dep)) -> list[dict]:
    """List all registered tools (core + plugins) with name, description, and enabled status."""
    from jarvis.tools.plugin_loader import _disabled as _disabled_plugins
    settings = get_settings()
    schemas, _ = build_registry(
        reports_dir=settings.reports_dir,
        allowed_commands=settings.allowed_commands,
    )
    return [
        {
            "name": s["name"],
            "description": s.get("description", ""),
            "enabled": s["name"] not in _disabled_plugins,
            "input_schema": s.get("input_schema", {}),
        }
        for s in schemas
    ]


@app.get("/api/tools/metrics")
async def get_tool_metrics(
    since_hours: float = 24.0,
    _user=Depends(_auth_dep),
) -> list[dict]:
    """Return per-tool stats from the audit log for the last since_hours hours.

    Each entry: {tool_name, call_count, avg_duration_ms, error_rate, last_called_at}
    """
    import time as _time
    since_ts = _time.time() - since_hours * 3600
    db_path = get_settings().reports_dir / "jarvis.db"
    if not db_path.exists():
        return []
    try:
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT tool_name,
                      COUNT(*) AS call_count,
                      AVG(duration_ms) AS avg_duration_ms,
                      SUM(CASE WHEN result_ok = 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS error_rate,
                      MAX(timestamp) AS last_called_at
               FROM audit_log
               WHERE timestamp >= ?
               GROUP BY tool_name
               ORDER BY call_count DESC""",
            (since_ts,),
        ).fetchall()
        conn.close()
        return [
            {
                "tool_name": r["tool_name"],
                "call_count": r["call_count"],
                "avg_duration_ms": round(r["avg_duration_ms"] or 0, 1),
                "error_rate": round(r["error_rate"] or 0, 4),
                "last_called_at": r["last_called_at"],
            }
            for r in rows
        ]
    except Exception:
        return []


@app.get("/api/tools/{tool_name}")
async def get_tool_detail(tool_name: str, _user=Depends(_auth_dep)) -> dict:
    """Return the full JSON Schema for a single tool."""
    settings = get_settings()
    schemas, _ = build_registry(
        reports_dir=settings.reports_dir,
        allowed_commands=settings.allowed_commands,
    )
    schema = next((s for s in schemas if s["name"] == tool_name), None)
    if not schema:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
    return schema


# ── Eval scheduling ──────────────────────────────────────────────────────────

@app.post("/api/evals/schedule", status_code=201)
async def schedule_eval_run(body: dict, _user=Depends(_auth_dep)) -> dict:
    """Schedule a recurring eval suite run via cron.

    Body:
      cron     (str, required) — 5-field cron expression (minute hour day month weekday, UTC)
      tags     (list[str])     — only run cases with these tags; empty = all cases
      judge    (bool)          — enable Claude-as-judge scoring (default false)

    Returns {job_id, cron, message}.
    """
    from apscheduler.triggers.cron import CronTrigger
    from jarvis.scheduler.core import JOB_FUNCTIONS, get_scheduler
    import json as _json

    scheduler = get_scheduler()
    if scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler not running")

    cron = (body.get("cron") or "").strip()
    parts = cron.split()
    if len(parts) != 5:
        raise HTTPException(status_code=422, detail="cron must have 5 fields: minute hour day month weekday")

    tags = body.get("tags") or []
    use_judge = bool(body.get("judge", False))
    settings = get_settings()

    minute, hour, day, month, dow = parts
    trigger = CronTrigger(minute=minute, hour=hour, day=day, month=month,
                          day_of_week=dow, timezone="UTC")
    job_id = f"eval-{str(uuid.uuid4())[:8]}"
    scheduler.add_job(
        JOB_FUNCTIONS["eval_run"],
        trigger,
        kwargs={
            "db_path_str": str(settings.reports_dir / "jarvis.db"),
            "reports_dir_str": str(settings.reports_dir),
            "tags_json": _json.dumps(tags),
            "use_judge": use_judge,
        },
        id=job_id,
        replace_existing=True,
    )
    return {"job_id": job_id, "cron": cron, "tags": tags, "judge": use_judge,
            "message": f"Eval run scheduled (cron: {cron} UTC)"}


@app.get("/api/evals/schedule")
async def list_eval_schedules(_user=Depends(_auth_dep)) -> list[dict]:
    """List all scheduled eval run jobs."""
    from jarvis.scheduler.core import get_scheduler
    scheduler = get_scheduler()
    if scheduler is None:
        return []
    jobs = []
    for job in scheduler.get_jobs():
        if str(job.id).startswith("eval-"):
            next_run = job.next_run_time.isoformat() if job.next_run_time else None
            jobs.append({"job_id": job.id, "next_run": next_run, "func": job.func.__name__})
    return jobs


@app.delete("/api/evals/schedule/{job_id}", status_code=200)
async def delete_eval_schedule(job_id: str, _user=Depends(_auth_dep)) -> dict:
    """Remove a scheduled eval run job."""
    from jarvis.scheduler.core import get_scheduler
    scheduler = get_scheduler()
    if scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler not running")
    try:
        scheduler.remove_job(job_id)
        return {"job_id": job_id, "deleted": True}
    except Exception:
        raise HTTPException(status_code=404, detail=f"Scheduled eval job '{job_id}' not found")


# ── Eval trend ────────────────────────────────────────────────────────────────

@app.get("/api/evals/trend")
async def get_eval_trend(last_n: int = 10, _user=Depends(_auth_dep)) -> list[dict]:
    """Return the last N eval run summaries ordered newest-first for trend analysis."""
    from jarvis.evals.trend import get_trend
    db_path = get_settings().reports_dir / "jarvis.db"
    return get_trend(db_path, last_n=last_n)


@app.get("/api/evals/runs/{run_id}")
async def get_eval_run(run_id: str, _user=Depends(_auth_dep)) -> dict:
    """Return full details (including per-case results) for a single eval run."""
    from jarvis.evals.trend import get_run
    db_path = get_settings().reports_dir / "jarvis.db"
    record = get_run(db_path, run_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Eval run '{run_id}' not found")
    return record


@app.get("/api/evals/compare")
async def compare_eval_runs(
    run_a: str,
    run_b: str,
    _user=Depends(_auth_dep),
) -> dict:
    """Compare two eval runs side-by-side.

    Returns {run_a, run_b, delta_pass_rate, improved, regressed,
             unchanged_pass, unchanged_fail}.

    improved  — case_ids that failed in run_a but passed in run_b
    regressed — case_ids that passed in run_a but failed in run_b
    delta_pass_rate — run_b.pass_rate − run_a.pass_rate (positive = improvement)
    """
    from jarvis.evals.trend import compare_runs
    db_path = get_settings().reports_dir / "jarvis.db"
    result = compare_runs(db_path, run_a, run_b)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"One or both eval runs not found: '{run_a}', '{run_b}'",
        )
    return result


# ── Parallel map SSE ─────────────────────────────────────────────────────────

@app.post("/api/parallel-map/stream")
async def parallel_map_stream(
    body: ParallelMapRequest,
    _user=Depends(_auth_dep),
) -> StreamingResponse:
    """Run parallel_map and stream one SSE event per topic as it completes.

    Event format (text/event-stream):
      data: {"type": "topic", "topic": "...", "result": "..."}
      data: {"type": "synthesis", "result": "..."}   (only when synthesize=True)
      data: {"type": "done"}
    """
    import json as _json
    from concurrent.futures import ThreadPoolExecutor, as_completed

    settings = get_settings()
    base_schemas, base_registry = build_registry(settings)

    from jarvis.agents.researcher import ResearcherAgent
    from jarvis.agents.coder import CoderAgent
    from jarvis.agents.qa import QAAgent
    from jarvis.agents.data_analyst import DataAnalystAgent
    from jarvis.agents.devops import DevOpsAgent

    _agent_classes = {
        "researcher": ResearcherAgent,
        "coder": CoderAgent,
        "qa": QAAgent,
        "analyst": DataAnalystAgent,
        "devops": DevOpsAgent,
    }
    AgentClass = _agent_classes.get(body.agent_type)
    if AgentClass is None:
        raise HTTPException(status_code=422, detail=f"Unknown agent_type '{body.agent_type}'")

    def _run_topic(topic: str) -> tuple[str, str]:
        task = body.task_template.replace("{topic}", topic)
        try:
            agent = AgentClass(
                model=settings.model,
                max_tokens=settings.max_tokens,
                tool_schemas=base_schemas,
                tool_registry=base_registry,
            )
            result, _ = agent.run_turn([{"role": "user", "content": task}])
            return topic, result
        except Exception as exc:
            return topic, f"ERROR: {exc}"

    async def _event_generator():
        loop = asyncio.get_running_loop()
        n_workers = min(len(body.topics), 8)
        topic_results: dict[str, str] = {}

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            future_to_topic = {loop.run_in_executor(pool, _run_topic, t): t for t in body.topics}
            pending = set(future_to_topic)

            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for fut in done:
                    topic, result = await fut
                    topic_results[topic] = result
                    payload = _json.dumps({"type": "topic", "topic": topic, "result": result})
                    yield f"data: {payload}\n\n"

        if body.synthesize and len(body.topics) > 1:
            combined = "\n\n".join(
                f"[{t}]:\n{topic_results.get(t, '')[:1500]}" for t in body.topics
            )
            synthesis_prompt = (
                f"Synthesise research findings on {len(body.topics)} topics. "
                "Highlight common themes, key differences, and cross-cutting insights.\n\n"
                + combined
            )
            try:
                synth_agent = ResearcherAgent(
                    model=settings.model,
                    max_tokens=settings.max_tokens,
                    tool_schemas=base_schemas,
                    tool_registry=base_registry,
                )
                synthesis, _ = await loop.run_in_executor(
                    _executor,
                    lambda: synth_agent.run_turn([{"role": "user", "content": synthesis_prompt}]),
                )
                yield f"data: {_json.dumps({'type': 'synthesis', 'result': synthesis})}\n\n"
            except Exception as exc:
                yield f"data: {_json.dumps({'type': 'synthesis', 'result': f'ERROR: {exc}'})}\n\n"

        yield f"data: {_json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(_event_generator(), media_type="text/event-stream")


# ── Webhooks ──────────────────────────────────────────────────────────────────

@app.post("/api/webhooks", status_code=201)
async def create_webhook(body: dict, _user=Depends(_auth_dep)) -> dict:
    """Register an HTTP callback for system events.

    Body: {"url": "https://...", "events": ["schedule.complete", ...], "secret": "optional"}
    Valid events: schedule.complete, eval.complete, training.complete, tool.error, chat.complete
    """
    from jarvis.events.webhooks import register_webhook
    url = body.get("url", "").strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=422, detail="url must start with http:// or https://")
    events = body.get("events", [])
    if not events:
        raise HTTPException(status_code=422, detail="events list must not be empty")
    secret = body.get("secret")
    db_path = get_settings().reports_dir / "jarvis.db"
    try:
        return register_webhook(db_path, url, events, secret)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/api/webhooks")
async def list_webhooks_endpoint(event: str | None = None, _user=Depends(_auth_dep)) -> list[dict]:
    """List all active webhooks, optionally filtered by event type."""
    from jarvis.events.webhooks import list_webhooks
    db_path = get_settings().reports_dir / "jarvis.db"
    return list_webhooks(db_path, event=event)


@app.delete("/api/webhooks/{webhook_id}", status_code=204)
async def delete_webhook_endpoint(webhook_id: str, _user=Depends(_auth_dep)) -> None:
    """Deactivate a webhook by ID."""
    from jarvis.events.webhooks import delete_webhook
    db_path = get_settings().reports_dir / "jarvis.db"
    found = delete_webhook(db_path, webhook_id)
    if not found:
        raise HTTPException(status_code=404, detail=f"Webhook '{webhook_id}' not found")


@app.get("/api/webhooks/{webhook_id}/deliveries")
async def get_webhook_deliveries(
    webhook_id: str,
    limit: int = 20,
    _user=Depends(_auth_dep),
) -> list[dict]:
    """Return recent delivery history for a webhook."""
    from jarvis.events.webhooks import get_deliveries
    db_path = get_settings().reports_dir / "jarvis.db"
    return get_deliveries(db_path, webhook_id, limit=limit)


# ── Plugin management ─────────────────────────────────────────────────────────

@app.get("/api/plugins")
async def list_plugins(_user=Depends(_auth_dep)) -> list[dict]:
    """List all discovered plugins with enabled/disabled status."""
    from jarvis.tools.plugin_loader import list_plugin_info
    return list_plugin_info()


@app.post("/api/plugins/reload", status_code=200)
async def reload_plugins_endpoint(_user=Depends(_auth_dep)) -> dict:
    """Force-reimport all plugin modules and rebuild the tool registry.

    Returns the count of successfully loaded plugins.
    """
    from jarvis.tools.plugin_loader import reload_plugins

    loop = asyncio.get_running_loop()
    schemas, registry = await loop.run_in_executor(
        _executor, reload_plugins
    )
    log.info("plugins_hot_reloaded", count=len(schemas))
    return {"reloaded": len(schemas), "tools": [s["name"] for s in schemas]}


@app.post("/api/plugins/{tool_name}/disable", status_code=200)
async def disable_plugin_endpoint(tool_name: str, _user=Depends(_auth_dep)) -> dict:
    """Disable a plugin by tool name (takes effect on next registry rebuild)."""
    from jarvis.tools.plugin_loader import disable_plugin
    found = disable_plugin(tool_name)
    if not found:
        raise HTTPException(status_code=404, detail=f"Plugin '{tool_name}' not found")
    return {"tool_name": tool_name, "enabled": False}


@app.post("/api/plugins/{tool_name}/enable", status_code=200)
async def enable_plugin_endpoint(tool_name: str, _user=Depends(_auth_dep)) -> dict:
    """Re-enable a previously disabled plugin."""
    from jarvis.tools.plugin_loader import enable_plugin, list_plugin_info
    known = {i["tool_name"] for i in list_plugin_info() if i.get("tool_name")}
    if tool_name not in known:
        raise HTTPException(status_code=404, detail=f"Plugin '{tool_name}' not found")
    enable_plugin(tool_name)
    return {"tool_name": tool_name, "enabled": True}


# ── Audit timeline ────────────────────────────────────────────────────────────

@app.get("/api/sessions/{session_id}/timeline")
async def get_session_timeline(session_id: str, _user=Depends(_auth_dep)) -> list[dict]:
    """Return the ordered tool-call timeline for a session from the audit log."""
    from jarvis.security.audit import get_session_timeline
    db_path = get_settings().reports_dir / "jarvis.db"
    return get_session_timeline(db_path, session_id)


@app.get("/api/audit/slow-tools")
async def get_slow_tools(
    threshold_ms: float = 5000.0,
    _user=Depends(_auth_dep),
) -> list[dict]:
    """Return tools whose average execution time exceeds threshold_ms (default 5000ms)."""
    from jarvis.security.audit import get_slow_tools
    db_path = get_settings().reports_dir / "jarvis.db"
    return get_slow_tools(db_path, threshold_ms=threshold_ms)


@app.get("/api/security/injection-stats")
async def injection_stats(
    since_ts: float | None = None,
    _user=Depends(_auth_dep),
) -> dict:
    """Return aggregated prompt-injection detection statistics.

    Optionally filter to events after since_ts (Unix timestamp).
    """
    from jarvis.security.injection import get_stats
    db_path = get_settings().reports_dir / "jarvis.db"
    return get_stats(db_path, since_ts=since_ts)


# ── Session fork ──────────────────────────────────────────────────────────────

@app.post("/api/sessions/{session_id}/fork", status_code=201)
async def fork_session(
    session_id: str,
    body: dict = {},
    _user=Depends(_auth_dep),
) -> dict:
    """Fork a session, copying its message history into a new independent session.

    Body (optional):
      message_index: int — copy only messages[:message_index] (default: all)
      new_session_id: str — explicit ID for the fork (default: auto-generated UUID)
    """
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    source = _sessions[session_id]
    messages = list(source.get("messages", []))
    idx = body.get("message_index")
    if idx is not None:
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail="message_index must be an integer")
        messages = messages[:idx]

    new_sid = body.get("new_session_id") or str(uuid.uuid4())
    if new_sid in _sessions:
        raise HTTPException(status_code=409, detail=f"Session '{new_sid}' already exists")

    settings = get_settings()
    researcher_mode = isinstance(source.get("agent"), ResearcherAgent)
    team_mode = isinstance(source.get("agent"), TeamAgent)
    new_agent = _build_agent_for_session(
        settings,
        researcher_mode=researcher_mode,
        team_mode=team_mode,
        session_id=new_sid,
        user_id=source.get("user_id"),
        approval_gate=source.get("approval_gate"),
    )
    _sessions[new_sid] = {
        "agent": new_agent,
        "messages": messages,
        "created_at": time.time(),
        "user_id": source.get("user_id"),
        "approval_gate": source.get("approval_gate"),
        "fork_of": session_id,
        "forked_at": time.time(),
    }
    _persist_session(new_sid, _sessions[new_sid])
    log.info("session_forked", source=session_id, fork=new_sid, messages_copied=len(messages))
    return {
        "session_id": new_sid,
        "fork_of": session_id,
        "message_count": len(messages),
        "forked_at": _sessions[new_sid]["forked_at"],
    }


# ── Agent checkpoints ─────────────────────────────────────────────────────────

@app.get("/api/sessions/{session_id}/checkpoints")
async def list_checkpoints(
    session_id: str,
    _user=Depends(_auth_dep),
) -> list[dict]:
    """List all saved checkpoints for a session, oldest first."""
    from jarvis.agents.checkpoint import list_checkpoints as _list_cp
    db_path = get_settings().reports_dir / "jarvis.db"
    return _list_cp(db_path, session_id)


@app.post("/api/sessions/{session_id}/checkpoints", status_code=201)
async def create_checkpoint(
    session_id: str,
    _user=Depends(_auth_dep),
) -> dict:
    """Manually save a checkpoint for the current in-memory session state."""
    from jarvis.agents.checkpoint import save_checkpoint
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    db_path = get_settings().reports_dir / "jarvis.db"
    agent = session.get("agent")
    turn_count = getattr(agent, "_turn_count", 0) if agent else 0
    agent_type = type(agent).__name__ if agent else "Unknown"
    messages = session.get("messages", [])
    cp_id = save_checkpoint(db_path, session_id, turn_count, agent_type, messages)
    return {"checkpoint_id": cp_id, "session_id": session_id, "turn_count": turn_count,
            "message_count": len(messages)}


@app.post("/api/sessions/{session_id}/checkpoints/{checkpoint_id}/restore", status_code=201)
async def restore_checkpoint(
    session_id: str,
    checkpoint_id: str,
    _user=Depends(_auth_dep),
) -> dict:
    """Fork a new session from a saved checkpoint, restoring that message state."""
    from jarvis.agents.checkpoint import load_checkpoint
    db_path = get_settings().reports_dir / "jarvis.db"
    cp = load_checkpoint(db_path, checkpoint_id)
    if cp is None or cp["session_id"] != session_id:
        raise HTTPException(status_code=404, detail=f"Checkpoint '{checkpoint_id}' not found")
    new_sid = str(uuid.uuid4())
    _sessions[new_sid] = {
        "messages": list(cp["messages"]),
        "agent": None,
        "user_id": None,
        "fork_of": session_id,
        "forked_at": time.time(),
        "restored_from_checkpoint": checkpoint_id,
    }
    return {
        "session_id": new_sid,
        "fork_of": session_id,
        "checkpoint_id": checkpoint_id,
        "turn_count": cp["turn_count"],
        "message_count": len(cp["messages"]),
    }


@app.delete("/api/sessions/{session_id}/checkpoints", status_code=200)
async def delete_checkpoints(
    session_id: str,
    _user=Depends(_auth_dep),
) -> dict:
    """Delete all checkpoints for a session."""
    from jarvis.agents.checkpoint import delete_checkpoints as _del_cp
    db_path = get_settings().reports_dir / "jarvis.db"
    deleted = _del_cp(db_path, session_id)
    return {"deleted": deleted}


# ── Agent capability registry ────────────────────────────────────────────────

@app.get("/api/agents")
async def list_agent_types(_user=Depends(_auth_dep)) -> list[dict]:
    """List all known agent types with their capabilities.

    Returns [{name, description, allowed_tools, tool_count, prompt_file, prompt_source}].
    allowed_tools is null for agents that accept all tools (planner, team_agent).
    """
    from jarvis.agents.registry import list_agents
    return list_agents()


@app.get("/api/agents/{agent_name}")
async def get_agent_type(agent_name: str, _user=Depends(_auth_dep)) -> dict:
    """Return capability info for a specific agent type."""
    from jarvis.agents.registry import get_agent_info
    info = get_agent_info(agent_name)
    if not info:
        raise HTTPException(status_code=404, detail=f"Agent type '{agent_name}' not found")
    return info


# ── Prompt management ────────────────────────────────────────────────────────

_KNOWN_AGENT_TYPES = frozenset([
    "researcher", "planner", "coder", "qa", "critic", "data_analyst",
    "devops", "team_agent", "backend", "frontend", "manager", "team_lead",
])


@app.get("/api/prompts")
async def list_prompts(_user=Depends(_auth_dep)) -> list[dict]:
    """List all agent prompts with their source (file or in-memory override).

    Returns [{agent_type, source: "file"|"override", length_chars}].
    """
    from jarvis.prompts.overrides import list_overrides
    from jarvis.prompts.loader import _PROMPTS_DIR
    overrides = list_overrides()
    result = []
    for md_file in sorted(_PROMPTS_DIR.glob("*.md")):
        name = md_file.stem
        source = "override" if name in overrides else "file"
        try:
            from jarvis.prompts.loader import load_prompt
            text = load_prompt(name)
        except Exception:
            text = ""
        result.append({"agent_type": name, "source": source, "length_chars": len(text)})
    # Also include overrides for types without a .md file
    for name in overrides:
        if not (_PROMPTS_DIR / f"{name}.md").exists():
            result.append({"agent_type": name, "source": "override", "length_chars": len(overrides[name])})
    return result


@app.get("/api/prompts/{agent_type}")
async def get_prompt(agent_type: str, _user=Depends(_auth_dep)) -> dict:
    """Return the current effective prompt for an agent type.

    Returns {agent_type, prompt, source: "file"|"override"}.
    """
    from jarvis.prompts.overrides import get_override
    from jarvis.prompts.loader import load_prompt, _PROMPTS_DIR
    override = get_override(agent_type)
    source = "override" if override else "file"
    try:
        prompt = load_prompt(agent_type)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No prompt found for '{agent_type}'")
    return {"agent_type": agent_type, "prompt": prompt, "source": source}


@app.put("/api/prompts/{agent_type}", status_code=200)
async def set_prompt_override(
    agent_type: str,
    body: dict,
    _user=Depends(_auth_dep),
) -> dict:
    """Set an in-memory prompt override for an agent type.

    Body: {prompt: str}
    The override is active immediately and persists until reset or server restart.
    """
    from jarvis.prompts.overrides import get_override, set_override
    from jarvis.prompts.history import record_version
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="'prompt' is required")
    # Save the current value to history before overwriting
    current = get_override(agent_type)
    if current is None:
        try:
            from jarvis.prompts.loader import load_prompt
            current = load_prompt(agent_type)
        except Exception:
            current = ""
    if current:
        record_version(get_settings().reports_dir / "jarvis.db", agent_type, current)
    set_override(agent_type, prompt)
    return {"agent_type": agent_type, "source": "override", "length_chars": len(prompt)}


@app.delete("/api/prompts/{agent_type}", status_code=200)
async def reset_prompt_override(agent_type: str, _user=Depends(_auth_dep)) -> dict:
    """Remove the in-memory override for an agent type, reverting to the .md file."""
    from jarvis.prompts.overrides import clear_override
    existed = clear_override(agent_type)
    if not existed:
        raise HTTPException(status_code=404, detail=f"No override set for '{agent_type}'")
    return {"agent_type": agent_type, "source": "file", "reset": True}


@app.get("/api/prompts/{agent_type}/history")
async def get_prompt_history(agent_type: str, _user=Depends(_auth_dep)) -> list[dict]:
    """Return saved prompt versions for an agent type, newest first.

    Does not include the full prompt text — use the rollback endpoint to
    retrieve a specific version's content.
    Returns [{version_id, agent_type, set_at, length_chars}].
    """
    from jarvis.prompts.history import get_history
    db_path = get_settings().reports_dir / "jarvis.db"
    return get_history(db_path, agent_type)


@app.post("/api/prompts/{agent_type}/rollback/{version_id}", status_code=200)
async def rollback_prompt(
    agent_type: str,
    version_id: str,
    _user=Depends(_auth_dep),
) -> dict:
    """Restore a saved prompt version as the current in-memory override."""
    from jarvis.prompts.history import get_version, record_version
    from jarvis.prompts.overrides import get_override, set_override
    db_path = get_settings().reports_dir / "jarvis.db"
    version = get_version(db_path, version_id)
    if not version or version["agent_type"] != agent_type.lower():
        raise HTTPException(status_code=404, detail=f"Version '{version_id}' not found for '{agent_type}'")
    # Archive current before rolling back
    current = get_override(agent_type)
    if current:
        record_version(db_path, agent_type, current)
    set_override(agent_type, version["prompt"])
    return {"agent_type": agent_type, "version_id": version_id,
            "source": "override", "length_chars": len(version["prompt"])}


# ── Config hot-reload ─────────────────────────────────────────────────────────

@app.post("/api/config/reload", status_code=200)
async def reload_config(_user=Depends(_auth_dep)) -> dict:
    """Re-read environment variables and update running app settings.

    Propagates changes to:
      - Auth dependency (_require_auth)
      - Rate limiter buckets (cleared so new limits apply immediately)
      - Scheduler cron jobs (eval + training auto-schedules)

    Returns {changed: {key: {old, new}}, reloaded_at}.
    """
    global _require_auth
    old_settings = get_settings()

    # Force re-read env vars by reloading dotenv
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except Exception:
        pass

    new_settings = get_settings()
    reloaded_at = time.time()

    # Diff settings to report what changed
    changed: dict[str, dict] = {}
    for field in old_settings.model_fields:
        old_val = getattr(old_settings, field, None)
        new_val = getattr(new_settings, field, None)
        if old_val != new_val:
            changed[field] = {"old": str(old_val), "new": str(new_val)}

    # Propagate: auth dependency
    try:
        from jarvis.auth.core import make_auth_dependency
        _require_auth = make_auth_dependency(
            db_path=new_settings.reports_dir / "jarvis.db",
            jwt_secret=new_settings.jwt_secret,
            auth_enabled=new_settings.auth_enabled,
        )
    except Exception as exc:
        log.warning("config_reload_auth_failed", error=str(exc))

    # Propagate: clear rate-limiter buckets so new limits apply immediately
    try:
        for mw in app.middleware_stack.app.middleware_stack.app.middleware_stack.app.__dict__.get(  # type: ignore
            "_middleware", []
        ):
            pass
        # Walk middleware stack looking for _RateLimitMiddleware
        node = app.middleware_stack
        while hasattr(node, "app"):
            if hasattr(node, "_buckets") and hasattr(node, "_max_calls"):
                node._buckets.clear()
                new_max, new_window = _parse_rate(new_settings.chat_rate_limit)
                node._max_calls = new_max
                node._window = new_window
                break
            node = node.app
    except Exception:
        pass

    log.info("config_reloaded", changed_keys=list(changed.keys()))
    return {"changed": changed, "reloaded_at": reloaded_at}


# ── Live event bus SSE stream ─────────────────────────────────────────────────

@app.get("/api/events/stream")
async def event_stream(
    filter: str | None = None,
    _user=Depends(_auth_dep),
) -> StreamingResponse:
    """Stream all JARVIS event-bus events as Server-Sent Events.

    Each event: data: {"event_type": "...", "event_id": "...", "timestamp": ..., ...}

    Query params:
      filter (str) — comma-separated list of event_type values to include.
                     Omit to receive all events.

    Connect with EventSource in the browser or `curl -N /api/events/stream`.
    The connection stays open until the client disconnects.
    """
    import json as _json
    from jarvis.events.bus import get_event_bus

    allowed_types: set[str] | None = None
    if filter:
        allowed_types = {t.strip() for t in filter.split(",") if t.strip()}

    bus = get_event_bus()
    q = bus.add_sse_listener()

    async def _gen():
        try:
            # Send a keep-alive comment immediately so the client knows it's connected
            yield ": connected\n\n"
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=30)
                    if allowed_types and event.event_type not in allowed_types:
                        continue
                    payload = event.to_dict() if hasattr(event, "to_dict") else {"event_type": event.event_type}
                    yield f"data: {_json.dumps(payload)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            bus.remove_sse_listener(q)

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Notification center ────────────────────────────────────────────────────────

@app.get("/api/notifications")
async def list_notifications(
    unread_only: bool = False,
    limit: int = 50,
    offset: int = 0,
    _user=Depends(_auth_dep),
) -> list[dict]:
    """Return system notifications, newest-first."""
    from jarvis.events.notifications import list_notifications as _list
    db_path = get_settings().reports_dir / "jarvis.db"
    return _list(db_path, unread_only=unread_only, limit=limit, offset=offset)


@app.get("/api/notifications/unread-count")
async def get_unread_count(_user=Depends(_auth_dep)) -> dict:
    """Return the number of unread notifications."""
    from jarvis.events.notifications import unread_count
    db_path = get_settings().reports_dir / "jarvis.db"
    return {"unread": unread_count(db_path)}


@app.patch("/api/notifications/{notification_id}/read", status_code=200)
async def mark_notification_read(notification_id: str, _user=Depends(_auth_dep)) -> dict:
    """Mark a notification as read."""
    from jarvis.events.notifications import mark_read
    db_path = get_settings().reports_dir / "jarvis.db"
    found = mark_read(db_path, notification_id)
    if not found:
        raise HTTPException(status_code=404, detail=f"Notification '{notification_id}' not found")
    return {"id": notification_id, "read": True}


@app.delete("/api/notifications", status_code=200)
async def clear_notifications(_user=Depends(_auth_dep)) -> dict:
    """Delete all read notifications. Returns count deleted."""
    from jarvis.events.notifications import clear_read
    db_path = get_settings().reports_dir / "jarvis.db"
    deleted = clear_read(db_path)
    return {"deleted": deleted}


@app.post("/api/notifications", status_code=201)
async def create_notification(body: dict, _user=Depends(_auth_dep)) -> dict:
    """Push a manual system notification (useful for testing or admin alerts)."""
    from jarvis.events.notifications import push_notification
    title = body.get("title", "").strip()
    if not title:
        raise HTTPException(status_code=422, detail="title must not be empty")
    db_path = get_settings().reports_dir / "jarvis.db"
    return push_notification(
        db_path,
        event=body.get("event", "system.info"),
        title=title,
        body=body.get("body", ""),
        severity=body.get("severity", "info"),
    )


# ── Async job queue ───────────────────────────────────────────────────────────

@app.post("/api/jobs", status_code=202)
async def enqueue_job(body: dict, _user=Depends(_auth_dep)) -> dict:
    """Enqueue a background agent task. Returns immediately with a job ID.

    Body: {message: str, agent_type?: str (planner|researcher|coder|...), user_id?: str}
    Poll GET /api/jobs/{id} for status and result.
    """
    message = (body.get("message") or "").strip()
    if not message:
        raise HTTPException(status_code=422, detail="'message' is required")
    agent_type = str(body.get("agent_type") or "planner").lower()
    user_id = str(body.get("user_id") or (_user.username if hasattr(_user, "username") else "anonymous"))

    from jarvis.api.jobs import create_job, mark_done, mark_failed, mark_running
    db_path = get_settings().reports_dir / "jarvis.db"
    job_id = create_job(db_path, message, agent_type=agent_type, user_id=user_id)

    loop = asyncio.get_running_loop()

    def _run_job():
        mark_running(db_path, job_id)
        try:
            settings = get_settings()
            session_id = f"job-{job_id[:8]}"
            session = _get_session(session_id, researcher_mode=(agent_type == "researcher"))
            agent = session["agent"]
            messages = [{"role": "user", "content": message}]
            result_text, _ = agent.run_turn(messages)
            usage = agent.get_usage_summary()
            _record_spend(user_id, usage["estimated_cost_usd"])
            mark_done(db_path, job_id, result_text, usage)
        except Exception as exc:
            mark_failed(db_path, job_id, str(exc))

    loop.run_in_executor(_executor, _run_job)
    return {"job_id": job_id, "status": "pending"}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str, _user=Depends(_auth_dep)) -> dict:
    """Return status and result for a background job."""
    from jarvis.api.jobs import get_job as _get_job
    db_path = get_settings().reports_dir / "jarvis.db"
    job = _get_job(db_path, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job


@app.get("/api/jobs")
async def list_jobs(
    user_id: str | None = None,
    status: str | None = None,
    limit: int = 50,
    _user=Depends(_auth_dep),
) -> list[dict]:
    """List recent background jobs, newest first."""
    from jarvis.api.jobs import list_jobs as _list_jobs
    db_path = get_settings().reports_dir / "jarvis.db"
    return _list_jobs(db_path, user_id=user_id, status=status, limit=limit)


@app.delete("/api/jobs/{job_id}", status_code=204)
async def cancel_job(job_id: str, _user=Depends(_auth_dep)) -> None:
    """Cancel a pending job (no-op if already running or done)."""
    import sqlite3 as _sq
    db_path = get_settings().reports_dir / "jarvis.db"
    from jarvis.api.jobs import _conn as _jobs_conn
    conn = _jobs_conn(db_path)
    try:
        cur = conn.execute(
            "UPDATE agent_jobs SET status='cancelled', finished_at=? WHERE id=? AND status='pending'",
            (time.time(), job_id),
        )
        conn.commit()
        found = conn.execute("SELECT 1 FROM agent_jobs WHERE id=?", (job_id,)).fetchone()
    finally:
        conn.close()
    if not found:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")


# ── Agent debate ──────────────────────────────────────────────────────────────

@app.post("/api/debate", status_code=200)
async def run_debate_endpoint(body: dict, _user=Depends(_auth_dep)) -> dict:
    """Run a researcher→critic debate on a question.

    Body: {question: str, model?: str, max_tokens?: int}

    Returns: {question, research_summary, critique_issues, verdict,
              confidence_score, critic_score, retry_recommended, revised_question}

    verdict values: "well_supported" | "partially_supported" | "poorly_supported"
    confidence_score: 0.0 – 1.0 (derived from critic score 1-5)
    """
    question = (body.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=422, detail="'question' is required")

    settings = get_settings()
    model = body.get("model") or settings.model
    max_tokens = int(body.get("max_tokens") or min(settings.max_tokens, 2048))

    from jarvis.agents.debate import run_debate
    from jarvis.agents.researcher import ResearcherAgent
    from jarvis.agents.critic import build_critic
    from jarvis.tools.registry import build_registry, get_tool_schemas

    schemas = get_tool_schemas()
    registry = build_registry()

    researcher = ResearcherAgent(
        model=model,
        max_tokens=max_tokens,
        tool_schemas=schemas,
        tool_registry=registry,
        max_search_calls=5,
    )
    critic = build_critic(model=model, max_tokens=512)

    loop = asyncio.get_running_loop()
    result_holder: list = []

    def _run():
        try:
            result_holder.append(run_debate(question, researcher, critic))
        except Exception as exc:
            result_holder.append(exc)

    await loop.run_in_executor(_executor, _run)
    result = result_holder[0] if result_holder else None
    if isinstance(result, Exception):
        raise HTTPException(status_code=500, detail=str(result))
    return result or {}


# ── Agent pipeline ────────────────────────────────────────────────────────────

_PIPELINE_AGENT_TYPES = frozenset(["planner", "researcher", "coder", "qa", "analyst", "devops"])


@app.post("/api/pipeline", status_code=200)
async def run_pipeline(body: dict, _user=Depends(_auth_dep)) -> dict:
    """Run a multi-agent pipeline where each step's output feeds the next step's input.

    Body:
      prompt:  str           — initial user message
      steps:   list[{agent_type: str, instructions?: str}]
      session_id?: str       — optional session to attach the result to
      timeout_seconds?: float (default: agent_turn_timeout_seconds × len(steps))

    Each step receives:
      "[Previous output]\\n{prior_step_output}\\n\\n[Task]\\n{instructions or prompt}"

    Returns:
      {steps: [{agent_type, output, usage}], final_output: str, total_usage: dict}
    """
    prompt: str = body.get("prompt", "").strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="prompt must not be empty")

    steps_spec: list[dict] = body.get("steps", [])
    if not steps_spec or not isinstance(steps_spec, list):
        raise HTTPException(status_code=422, detail="steps must be a non-empty list")

    unknown = [s["agent_type"] for s in steps_spec if s.get("agent_type") not in _PIPELINE_AGENT_TYPES]
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown agent_type(s): {unknown}. Valid: {sorted(_PIPELINE_AGENT_TYPES)}",
        )

    settings = get_settings()
    per_step_timeout = settings.agent_turn_timeout_seconds
    total_timeout = float(body.get("timeout_seconds") or per_step_timeout * len(steps_spec))

    loop = asyncio.get_running_loop()
    step_results: list[dict] = []
    current_input = prompt
    total_input_tokens = 0
    total_output_tokens = 0

    def _run_step(agent_type: str, instructions: str | None, user_input: str) -> tuple[str, dict]:
        researcher_mode = agent_type == "researcher"
        team_mode = False
        coder_mode = agent_type == "coder"
        analyst_mode = agent_type == "analyst"
        devops_mode = agent_type == "devops"
        qa_mode = agent_type == "qa"

        step_sid = str(uuid.uuid4())
        agent = _build_agent_for_session(
            settings,
            researcher_mode=researcher_mode,
            team_mode=team_mode,
            session_id=step_sid,
        )
        msg_content = (
            f"[Previous output]\n{user_input}\n\n[Task]\n{instructions}"
            if instructions and user_input != prompt
            else (instructions or user_input)
        )
        msgs = [{"role": "user", "content": msg_content}]
        output, _ = agent.run_turn(msgs)
        usage = agent.get_usage_summary()
        return output, usage

    async def _run_pipeline() -> None:
        nonlocal current_input, total_input_tokens, total_output_tokens
        for spec in steps_spec:
            agent_type = spec["agent_type"]
            instructions = spec.get("instructions")
            output, usage = await asyncio.wait_for(
                loop.run_in_executor(_executor, _run_step, agent_type, instructions, current_input),
                timeout=per_step_timeout,
            )
            step_results.append({"agent_type": agent_type, "output": output, "usage": usage})
            total_input_tokens += usage.get("input_tokens", 0)
            total_output_tokens += usage.get("output_tokens", 0)
            current_input = output  # chain: this step's output → next step's input

    try:
        await asyncio.wait_for(_run_pipeline(), timeout=total_timeout)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Pipeline timed out")

    return {
        "steps": step_results,
        "final_output": current_input,
        "total_usage": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "estimated_cost_usd": 0.0,
        },
    }


# ── Conversation export / import ──────────────────────────────────────────────

@app.get("/api/sessions/{session_id}/export")
async def export_session(session_id: str, _user=Depends(_auth_dep)) -> dict:
    """Export a session as a self-contained JSON bundle.

    The bundle can be imported with POST /api/sessions/import on any instance.
    Bundle schema: {session_id, agent_type, user_id, messages, fork_of, exported_at}
    """
    session = _sessions.get(session_id)
    if session is None:
        # Try loading from persistence
        from jarvis.memory.sessions import load_sessions
        db_path = get_settings().reports_dir / "jarvis.db"
        rows = load_sessions(db_path, ttl_minutes=99999)
        row = next((r for r in rows if r["session_id"] == session_id), None)
        if not row:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
        return {
            "session_id": row["session_id"],
            "agent_type": row["agent_type"],
            "user_id": row["user_id"],
            "messages": row["messages"],
            "fork_of": row["fork_of"],
            "exported_at": time.time(),
        }
    from jarvis.memory.turns import get_session_cost
    db_path = get_settings().reports_dir / "jarvis.db"
    from jarvis.memory.notes import list_notes
    agent = session.get("agent")
    return {
        "session_id": session_id,
        "agent_type": type(agent).__name__ if agent else "PlannerAgent",
        "user_id": session.get("user_id"),
        "messages": session.get("messages", []),
        "fork_of": session.get("fork_of"),
        "cost_summary": get_session_cost(db_path, session_id)["totals"],
        "notes": list_notes(db_path, session_id),
        "exported_at": time.time(),
    }


@app.post("/api/sessions/import", status_code=201)
async def import_session(bundle: dict, _user=Depends(_auth_dep)) -> dict:
    """Import a conversation bundle exported by GET /api/sessions/{id}/export.

    Creates (or replaces) a session with the provided message history.
    The session_id from the bundle is reused unless new_session_id is provided in the body.
    """
    messages = bundle.get("messages", [])
    if not isinstance(messages, list):
        raise HTTPException(status_code=422, detail="messages must be a list")

    orig_sid = bundle.get("session_id", "")
    new_sid = bundle.get("new_session_id") or orig_sid or str(uuid.uuid4())
    agent_type = bundle.get("agent_type", "PlannerAgent")
    user_id = bundle.get("user_id")

    settings = get_settings()
    researcher_mode = agent_type == "ResearcherAgent"
    team_mode = agent_type == "TeamAgent"
    agent = _build_agent_for_session(
        settings,
        researcher_mode=researcher_mode,
        team_mode=team_mode,
        session_id=new_sid,
        user_id=user_id,
    )
    _sessions[new_sid] = {
        "agent": agent,
        "messages": messages,
        "created_at": time.time(),
        "user_id": user_id,
        "approval_gate": None,
        "fork_of": bundle.get("fork_of"),
        "forked_at": None,
    }
    _persist_session(new_sid, _sessions[new_sid])
    log.info("session_imported", session_id=new_sid, messages=len(messages))
    return {
        "session_id": new_sid,
        "agent_type": agent_type,
        "message_count": len(messages),
        "imported_at": time.time(),
    }


# ── Session activity timeline ────────────────────────────────────────────────

@app.get("/api/sessions/{session_id}/activity")
async def get_session_activity(
    session_id: str,
    _user=Depends(_auth_dep),
) -> list[dict]:
    """Return a merged chronological activity timeline for a session.

    Combines: messages (user/assistant turns), tool calls, checkpoints, notes, tags.
    Each item: {timestamp, type, summary, detail?}

    type values: "message" | "tool_call" | "checkpoint" | "note" | "tag"
    """
    from jarvis.memory.activity import get_activity
    from jarvis.memory.sessions import load_sessions

    db_path = get_settings().reports_dir / "jarvis.db"

    # Resolve messages: in-memory first, then persisted
    session = _sessions.get(session_id)
    if session is not None:
        messages = list(session.get("messages", []))
    else:
        rows = load_sessions(db_path, ttl_minutes=99999)
        row = next((r for r in rows if r["session_id"] == session_id), None)
        if not row:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
        messages = row["messages"]

    return get_activity(db_path, session_id, messages=messages)


# ── Session summarizer ────────────────────────────────────────────────────────

@app.post("/api/sessions/{session_id}/summarize", status_code=200)
async def summarize_session_endpoint(
    session_id: str,
    body: dict = {},
    _user=Depends(_auth_dep),
) -> dict:
    """Summarize a session's conversation history.

    Body (optional):
      model (str) — Ollama model to use; omit for fast heuristic mode

    Returns: {summary, key_topics, action_items, message_count}
    """
    from jarvis.agents.summarize import summarize_session
    from jarvis.memory.sessions import load_sessions

    session = _sessions.get(session_id)
    if session is not None:
        messages = list(session.get("messages", []))
    else:
        rows = load_sessions(get_settings().reports_dir / "jarvis.db", ttl_minutes=99999)
        row = next((r for r in rows if r["session_id"] == session_id), None)
        if not row:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
        messages = row["messages"]

    model = str(body.get("model") or "")
    loop = asyncio.get_running_loop()
    result_holder: list = []

    def _run():
        result_holder.append(summarize_session(messages, model=model))

    await loop.run_in_executor(_executor, _run)
    s = result_holder[0]
    return {
        "summary": s.summary,
        "key_topics": s.key_topics,
        "action_items": s.action_items,
        "message_count": s.message_count,
    }


# ── Session replay ───────────────────────────────────────────────────────────

@app.post("/api/sessions/{session_id}/replay", status_code=200)
async def replay_session_endpoint(
    session_id: str,
    body: dict = {},
    _user=Depends(_auth_dep),
) -> list[dict]:
    """Replay all user turns from a saved session through the current agent config.

    Returns per-turn diffs: [{turn_index, user_message, original_response,
    replayed_response, changed}]

    Body (optional):
      dry_run (bool, default false) — skip real model calls, return "<stub>" responses
      model   (str)                 — override model for replay
    """
    from jarvis.agents.replay import replay_session
    from jarvis.memory.sessions import load_sessions

    dry_run = bool(body.get("dry_run", False))
    db_path = get_settings().reports_dir / "jarvis.db"

    # Resolve messages: in-memory first, then persisted
    session = _sessions.get(session_id)
    if session is not None:
        messages = list(session.get("messages", []))
    else:
        rows = load_sessions(db_path, ttl_minutes=99999)
        row = next((r for r in rows if r["session_id"] == session_id), None)
        if not row:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
        messages = row["messages"]

    if not messages:
        return []

    if dry_run:
        turns = replay_session(messages, run_turn_fn=lambda m: ("<stub>", m), dry_run=True)
        return [
            {
                "turn_index": t.turn_index,
                "user_message": t.user_message,
                "original_response": t.original_response,
                "replayed_response": t.replayed_response,
                "changed": t.changed,
            }
            for t in turns
        ]

    # Real replay: build a fresh agent then run each turn in a thread
    model = body.get("model") or get_settings().model
    replay_session_id = str(uuid.uuid4())
    tmp_session = _get_session(replay_session_id, researcher_mode=False)
    agent = tmp_session["agent"]

    def _run_turn(ctx: list[dict]) -> tuple[str, list[dict]]:
        return agent.run_turn(ctx)

    loop = asyncio.get_running_loop()
    result_holder: list = []

    def _run():
        try:
            result_holder.append(replay_session(messages, _run_turn))
        except Exception as exc:
            result_holder.append(exc)

    await loop.run_in_executor(_executor, _run)
    raw = result_holder[0] if result_holder else []
    if isinstance(raw, Exception):
        raise HTTPException(status_code=500, detail=str(raw))

    return [
        {
            "turn_index": t.turn_index,
            "user_message": t.user_message,
            "original_response": t.original_response,
            "replayed_response": t.replayed_response,
            "changed": t.changed,
        }
        for t in raw
    ]


# ── Markdown session export ───────────────────────────────────────────────────

@app.get("/api/sessions/{session_id}/export/markdown")
async def export_session_markdown(
    session_id: str,
    _user=Depends(_auth_dep),
) -> Response:
    """Export a session as a Markdown document.

    Returns text/markdown content with conversation formatted as headings,
    tool calls in fenced code blocks, and a metadata footer.
    """
    from jarvis.tools.markdown_export import session_to_markdown
    from jarvis.memory.sessions import load_sessions, get_metadata

    session = _sessions.get(session_id)
    if session is not None:
        messages = list(session.get("messages", []))
        title = session.get("title", "")
    else:
        db_path = get_settings().reports_dir / "jarvis.db"
        rows = load_sessions(db_path, ttl_minutes=99999)
        row = next((r for r in rows if r["session_id"] == session_id), None)
        if not row:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
        messages = row["messages"]
        meta = get_metadata(db_path, session_id)
        title = (meta or {}).get("title") or ""

    md = session_to_markdown(messages, session_id=session_id, title=title or "")
    filename = f"session-{session_id[:8]}.md"
    return Response(
        content=md,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── Session cost breakdown ────────────────────────────────────────────────────

@app.get("/api/sessions/{session_id}/cost")
async def get_session_cost_breakdown(
    session_id: str,
    _user=Depends(_auth_dep),
) -> dict:
    """Return per-turn token usage and estimated USD cost for a session."""
    from jarvis.memory.turns import get_session_cost
    db_path = get_settings().reports_dir / "jarvis.db"
    data = get_session_cost(db_path, session_id)
    if not data["turns"] and session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return data


# ── Session history ────────────────────────────────────────────────────────────

@app.get("/api/sessions/{session_id}/history")
async def get_session_history(
    session_id: str,
    limit: int = 50,
    offset: int = 0,
    _user=Depends(_auth_dep),
) -> list[dict]:
    """Return paginated message history for a session (newest-first).

    Falls back to the in-memory session if the session has not been persisted yet.
    """
    from jarvis.memory.sessions import get_session_history as _get_history
    db_path = get_settings().reports_dir / "jarvis.db"
    rows = _get_history(db_path, session_id, limit=limit, offset=offset)
    if not rows and session_id in _sessions:
        msgs = list(reversed(_sessions[session_id].get("messages", [])))
        rows = msgs[offset: offset + limit]
    if not rows and session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return rows


# ── Message editing ───────────────────────────────────────────────────────────

def _get_session_or_404(session_id: str) -> dict:
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return _sessions[session_id]


@app.patch("/api/sessions/{session_id}/messages/{index}", status_code=200)
async def edit_message(
    session_id: str,
    index: int,
    body: dict,
    _user=Depends(_auth_dep),
) -> dict:
    """Edit the content of a single message in a session's history.

    Body: {"content": "new text", "role": "user|assistant"  (optional)}
    Index is 0-based; negative indices count from the end.
    """
    session = _get_session_or_404(session_id)
    messages = session.get("messages", [])
    try:
        real_idx = index if index >= 0 else len(messages) + index
        if real_idx < 0 or real_idx >= len(messages):
            raise IndexError
    except (IndexError, TypeError):
        raise HTTPException(status_code=422, detail=f"Message index {index} out of range (len={len(messages)})")

    if "content" not in body:
        raise HTTPException(status_code=422, detail="body must include 'content'")

    updated_msg = dict(messages[real_idx])
    updated_msg["content"] = body["content"]
    if "role" in body:
        updated_msg["role"] = body["role"]
    messages[real_idx] = updated_msg
    session["messages"] = messages
    _persist_session(session_id, session)
    return {"index": real_idx, "message": updated_msg}


@app.delete("/api/sessions/{session_id}/messages/{index}", status_code=200)
async def delete_message(
    session_id: str,
    index: int,
    _user=Depends(_auth_dep),
) -> dict:
    """Remove a single message from a session's history by index.

    Returns the removed message and the new message count.
    """
    session = _get_session_or_404(session_id)
    messages = session.get("messages", [])
    try:
        real_idx = index if index >= 0 else len(messages) + index
        if real_idx < 0 or real_idx >= len(messages):
            raise IndexError
    except (IndexError, TypeError):
        raise HTTPException(status_code=422, detail=f"Message index {index} out of range (len={len(messages)})")

    removed = messages.pop(real_idx)
    session["messages"] = messages
    _persist_session(session_id, session)
    return {"removed": removed, "message_count": len(messages)}


@app.post("/api/sessions/{session_id}/messages", status_code=201)
async def insert_message(
    session_id: str,
    body: dict,
    _user=Depends(_auth_dep),
) -> dict:
    """Insert a message into a session's history.

    Body: {"role": "user|assistant|system", "content": "...", "position": int (optional, default: append)}
    """
    session = _get_session_or_404(session_id)
    role = body.get("role", "user")
    content = body.get("content", "")
    if not content:
        raise HTTPException(status_code=422, detail="content must not be empty")

    messages = session.get("messages", [])
    new_msg = {"role": role, "content": content}
    position = body.get("position")
    if position is None:
        messages.append(new_msg)
        real_idx = len(messages) - 1
    else:
        try:
            real_idx = int(position)
            messages.insert(real_idx, new_msg)
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail="position must be an integer")

    session["messages"] = messages
    _persist_session(session_id, session)
    return {"index": real_idx, "message": new_msg, "message_count": len(messages)}


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    import uvicorn

    settings = get_settings()
    log.info("starting_server", host=settings.api_host, port=settings.api_port)
    uvicorn.run(
        "jarvis.api.server:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )


if __name__ == "__main__":
    main()
