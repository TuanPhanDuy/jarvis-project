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
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

import anthropic

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
    FeedbackRequest,
    FeedbackStatsResponse,
    HealthResponse,
    ScheduleItem,
    ScheduleRequest,
    ScheduleResponse,
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
    WsProactive,
    WsThinking,
    WsToolCall,
)
from jarvis.config import get_settings
from jarvis.agents.planner import PlannerAgent
from jarvis.agents.researcher import ResearcherAgent
from jarvis.agents.team_agent import TeamAgent
from jarvis.tools.registry import build_planner_registry, build_registry, build_team_registry

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup() -> None:
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


async def _evict_stale_sessions() -> None:
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_agent_for_session(
    settings,
    researcher_mode: bool = False,
    team_mode: bool = False,
    session_id: str = "",
    user_id: str | None = None,
    approval_gate=None,
) -> PlannerAgent | ResearcherAgent | TeamAgent:
    base_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    if settings.routing_strategy == "always_primary":
        client = base_client
    else:
        from jarvis.models.router import ModelRouter
        client = ModelRouter(  # type: ignore[assignment]
            primary=base_client,
            primary_model=settings.model,
            fast_model=settings.fast_model,
            strategy=settings.routing_strategy,
        )
    base_schemas, base_registry = build_registry(
        tavily_api_key=settings.tavily_api_key,
        reports_dir=settings.reports_dir,
        allowed_commands=settings.allowed_commands,
        user_id=user_id or "anonymous",
        anthropic_api_key=settings.anthropic_api_key,
    )
    if researcher_mode:
        return ResearcherAgent(
            client=client,
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
        team_schemas, team_registry = build_team_registry(
            base_schemas=base_schemas,
            base_registry=base_registry,
            client=client,
            model=settings.model,
            max_tokens=settings.max_tokens,
        )
        return TeamAgent(
            client=client,
            model=settings.model,
            max_tokens=settings.max_tokens,
            tool_schemas=team_schemas,
            tool_registry=team_registry,
            role="manager",
            session_id=session_id,
            user_id=user_id,
            approval_gate=approval_gate,
        )
    planner_schemas, planner_registry = build_planner_registry(
        base_schemas=base_schemas,
        base_registry=base_registry,
        client=client,
        model=settings.model,
        max_tokens=settings.max_tokens,
        session_id=session_id,
        user_id=user_id,
    )
    session_count = _count_user_sessions(settings, user_id)
    return PlannerAgent(
        client=client,
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
    return HealthResponse(sessions_active=len(_sessions))


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> PlainTextResponse:
    return PlainTextResponse(
        generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """Synchronous single-turn chat. Blocks until JARVIS replies."""
    session_id = req.session_id or str(uuid.uuid4())
    session = _get_session(session_id, req.researcher_mode)
    agent = session["agent"]
    messages = session["messages"]

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

    await loop.run_in_executor(_executor, run)

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

            # Drain chunks until sentinel
            while True:
                msg = await chunk_queue.get()
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
    log_feedback(db_path, req.session_id, req.response_snippet, req.rating, req.comment)
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
async def get_audit(limit: int = 50, session_id: str | None = None) -> list[dict]:
    """Return recent tool-call audit log entries."""
    from jarvis.security.audit import get_recent_audit
    db_path = get_settings().reports_dir / "jarvis.db"
    return get_recent_audit(db_path, limit=limit, session_id=session_id)


@app.get("/api/approval/pending")
async def get_pending_approvals(session_id: str) -> list[dict]:
    """Return pending approval requests for a session."""
    session = _sessions.get(session_id)
    if not session or not session.get("approval_gate"):
        return []
    return session["approval_gate"].get_pending()


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
