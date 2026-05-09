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
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
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
    ComponentStatus,
    FeedbackRequest,
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
_executor = ThreadPoolExecutor(max_workers=20)

# Active WebSocket connections for proactive push: session_id → WebSocket
_active_websockets: dict[str, "WebSocket"] = {}

# Last activity timestamp per session — updated on each incoming WS message
_session_activity: dict[str, float] = {}

# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="JARVIS API",
    description="Just A Rather Very Intelligent System — powered by Claude",
    version="0.1.0",
)

import os as _os
app.add_middleware(
    CORSMiddleware,
    allow_origins=_os.getenv("JARVIS_CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


_require_auth = None  # set in startup; callable FastAPI dependency
_limiter = None  # slowapi Limiter, set in startup if rate_limit_enabled


@app.on_event("startup")
async def startup() -> None:
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
        global _limiter
        try:
            from slowapi import Limiter, _rate_limit_exceeded_handler
            from slowapi.errors import RateLimitExceeded
            from slowapi.util import get_remote_address
            _limiter = Limiter(key_func=get_remote_address, default_limits=[settings.chat_rate_limit])
            app.state.limiter = _limiter
            app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
            log.info("rate_limiting_enabled", limit=settings.chat_rate_limit)
        except ImportError:
            log.warning("slowapi_not_installed", hint="pip install slowapi to enable rate limiting")

    from jarvis.auth.core import make_auth_dependency
    _require_auth = make_auth_dependency(
        db_path=settings.reports_dir / "jarvis.db",
        jwt_secret=settings.jwt_secret,
        auth_enabled=settings.auth_enabled,
    )

    if settings.proactive_enabled:
        await _start_event_bus(settings)

    if settings.peer_enabled:
        await _start_peer_coordinator(settings)


@app.on_event("shutdown")
async def shutdown() -> None:
    from jarvis.scheduler.core import stop_scheduler
    stop_scheduler()
    from jarvis.events.bus import get_event_bus
    await get_event_bus().shutdown()


async def _start_event_bus(settings) -> None:
    """Initialize the event bus, register handlers, and start monitors."""
    from jarvis.events.bus import get_event_bus
    from jarvis.events.triggers import SystemMonitor, IdleDetector, FileWatcher
    from jarvis.events.autonomous_agent import handle_event

    bus = get_event_bus()
    loop = asyncio.get_event_loop()

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
        row = conn.execute(
            "SELECT COUNT(DISTINCT session_id) FROM episodes WHERE user_id = ?", (user_id,)
        ).fetchone()
        conn.close()
        return row[0] if row else 0
    except Exception:
        return 0


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
                ep = prune_old_episodes(db_path, retention)
                fb = prune_old_feedback(db_path, retention)
                fa = prune_old_failures(db_path, retention)
                pr = prune_old_preferences(db_path, retention)
                if ep + fb + fa + pr > 0:
                    log.info("memory_pruned", episodes=ep, feedback=fb, failures=fa, preferences=pr, retention_days=retention)
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
            conn.execute("SELECT 1").fetchone()
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

    all_ok = all(c.ok for c in components.values())
    return HealthResponse(
        status="ok" if all_ok else "degraded",
        sessions_active=len(_sessions),
        ws_connections=len(_active_websockets),
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

    messages.append({"role": "user", "content": req.message})
    t0 = time.perf_counter()

    loop = asyncio.get_event_loop()
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

    _log_episodes(session_id, req.message, text_or_err)
    _record_spend(session_id, usage["estimated_cost_usd"])
    log.info("chat_complete", session_id=session_id, duration_s=round(duration, 2))
    return ChatResponse(
        session_id=session_id,
        response=text_or_err,
        usage=UsageSummary(**usage),
    )


# ── WebSocket ──────────────────────────────────────────────────────────────────

@app.websocket("/api/ws/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str) -> None:
    """Streaming WebSocket chat. Session persists across reconnections."""
    await websocket.accept()
    ACTIVE_WS_CONNECTIONS.inc()
    _active_websockets[session_id] = websocket
    log.info("ws_connected", session_id=session_id)

    settings = get_settings()

    # Build an approval gate that pushes WsApprovalRequest over this WebSocket.
    loop = asyncio.get_event_loop()
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
            _deadline = asyncio.get_event_loop().time() + _ws_timeout
            while True:
                remaining = _deadline - asyncio.get_event_loop().time()
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
                _log_episodes(session_id, req.message, response_text)
                _record_spend(session_id, usage["estimated_cost_usd"])
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

def _log_episodes(session_id: str, user_msg: str, assistant_reply: str) -> None:
    """Persist user + assistant turn to episodic memory. Best-effort."""
    try:
        from jarvis.memory.episodic import log_episode
        db_path = get_settings().reports_dir / "jarvis.db"
        log_episode(db_path, session_id, "user", user_msg)
        log_episode(db_path, session_id, "assistant", assistant_reply)
    except Exception:
        pass


def _record_spend(session_id: str, cost_usd: float) -> None:
    """Record per-session token spend for budget tracking. Best-effort."""
    try:
        from jarvis.api.budget import record_spend
        db_path = get_settings().reports_dir / "jarvis.db"
        record_spend(db_path, session_id, cost_usd)
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


# ── Budget endpoints ───────────────────────────────────────────────────────────

@app.put("/api/budget/{user_id}", response_model=BudgetStatusResponse)
async def set_budget(user_id: str, req: BudgetRequest) -> BudgetStatusResponse:
    """Set monthly USD spending budget for a user (0 = unlimited)."""
    from jarvis.api.budget import get_budget_status, set_budget as _set_budget
    db_path = get_settings().reports_dir / "jarvis.db"
    _set_budget(db_path, user_id, req.monthly_budget_usd)
    return BudgetStatusResponse(**get_budget_status(db_path, user_id))


@app.get("/api/budget/{user_id}", response_model=BudgetStatusResponse)
async def get_budget(user_id: str) -> BudgetStatusResponse:
    """Get current spending and remaining budget for a user."""
    from jarvis.api.budget import get_budget_status
    db_path = get_settings().reports_dir / "jarvis.db"
    return BudgetStatusResponse(**get_budget_status(db_path, user_id))


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


# ── Schedule endpoints ─────────────────────────────────────────────────────────

@app.post("/api/schedules", response_model=ScheduleResponse)
async def create_schedule(req: ScheduleRequest) -> ScheduleResponse:
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
async def list_schedules() -> list[ScheduleItem]:
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


@app.delete("/api/schedules/{job_id}", response_model=ScheduleResponse)
async def delete_schedule(job_id: str) -> ScheduleResponse:
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


# ── Audit endpoints ────────────────────────────────────────────────────────────

@app.get("/api/feedback/stats")
async def get_feedback_stats(session_id: str | None = None) -> dict:
    """Return aggregate feedback statistics."""
    from jarvis.memory.feedback import get_feedback_stats as _get_stats
    db_path = get_settings().reports_dir / "jarvis.db"
    return _get_stats(db_path, session_id=session_id)


@app.get("/api/improvement-report")
async def get_improvement_report() -> dict:
    """Return the latest self-improvement analysis report content."""
    settings = get_settings()
    report_path = settings.reports_dir / "improvement_suggestions.md"
    if not report_path.exists():
        return {"content": None}
    return {"content": report_path.read_text(encoding="utf-8")}


@app.get("/api/audit")
async def get_audit(
    limit: int = 50, offset: int = 0, session_id: str | None = None
) -> list[dict]:
    """Return paginated tool-call audit log entries, newest first."""
    from jarvis.security.audit import get_recent_audit
    db_path = get_settings().reports_dir / "jarvis.db"
    return get_recent_audit(db_path, limit=limit, offset=offset, session_id=session_id)


@app.get("/api/approval/pending")
async def get_pending_approvals(session_id: str) -> list[dict]:
    """Return pending approval requests for a session."""
    session = _sessions.get(session_id)
    if not session or not session.get("approval_gate"):
        return []
    return session["approval_gate"].get_pending()


# ── Session management endpoints ─────────────────────────────────────────────

@app.get("/api/sessions", response_model=list[SessionInfo])
async def list_sessions() -> list[SessionInfo]:
    """List all currently active in-memory sessions."""
    return [
        SessionInfo(
            session_id=sid,
            created_at=s["created_at"],
            message_count=len(s.get("messages", [])),
            user_id=s.get("user_id"),
        )
        for sid, s in _sessions.items()
    ]


@app.get("/api/sessions/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str) -> SessionInfo:
    """Return metadata for a specific session."""
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return SessionInfo(
        session_id=session_id,
        created_at=session["created_at"],
        message_count=len(session.get("messages", [])),
        user_id=session.get("user_id"),
    )


@app.delete("/api/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str) -> None:
    """Evict a session from memory, freeing its agent and message history."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    del _sessions[session_id]
    _active_websockets.pop(session_id, None)
    _session_activity.pop(session_id, None)
    log.info("session_deleted", session_id=session_id)


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


# ── Memory browser endpoints ───────────────────────────────────────────────────

@app.get("/api/memory/episodes")
async def get_memory_episodes(
    session_id: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> list[dict]:
    """Return paginated episodic memory entries."""
    from jarvis.memory.episodic import _get_conn
    db_path = get_settings().reports_dir / "jarvis.db"
    try:
        conn = _get_conn(db_path)
        if session_id:
            rows = conn.execute(
                "SELECT session_id, user_id, role, content, timestamp FROM episodes "
                "WHERE session_id = ? ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                (session_id, limit, offset),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT session_id, user_id, role, content, timestamp FROM episodes "
                "ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/memory/graph")
async def get_memory_graph(entity: str | None = None, limit: int = 50) -> list[dict]:
    """Return knowledge graph entities and their relationships."""
    db_path = get_settings().reports_dir / "jarvis.db"
    try:
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        if entity:
            rows = conn.execute(
                "SELECT * FROM knowledge_graph WHERE subject LIKE ? OR object LIKE ? LIMIT ?",
                (f"%{entity}%", f"%{entity}%", limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM knowledge_graph ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Metrics summary endpoint ───────────────────────────────────────────────────

@app.get("/api/metrics/summary")
async def get_metrics_summary() -> dict:
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


# ── Autonomous events endpoint ─────────────────────────────────────────────────

@app.get("/api/autonomous/events")
async def get_autonomous_events(limit: int = 20) -> list[dict]:
    """Return recent autonomous agent events from the database."""
    db_path = get_settings().reports_dir / "jarvis.db"
    try:
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.execute(
            "CREATE TABLE IF NOT EXISTS autonomous_events "
            "(id INTEGER PRIMARY KEY AUTOINCREMENT, event_type TEXT, summary TEXT, timestamp REAL)"
        )
        rows = conn.execute(
            "SELECT * FROM autonomous_events ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


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
