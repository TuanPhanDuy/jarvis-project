"""Pydantic models for JARVIS API request/response payloads."""
from __future__ import annotations

from pydantic import BaseModel, Field


# ── HTTP Chat ────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message to send to JARVIS")
    session_id: str | None = Field(None, description="Session ID for conversation continuity")
    researcher_mode: bool = Field(False, description="Use ResearcherAgent instead of PlannerAgent")
    team_mode: bool = Field(False, description="Use TeamAgent (multi-agent collaboration)")


class UsageSummary(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_write_tokens: int
    cache_read_tokens: int
    estimated_cost_usd: float


class ChatResponse(BaseModel):
    session_id: str
    response: str
    usage: UsageSummary


# ── WebSocket messages ────────────────────────────────────────────────────────

class WsIncoming(BaseModel):
    """Message sent from client to server over WebSocket."""
    message: str
    researcher_mode: bool = False
    team_mode: bool = False


class WsChunk(BaseModel):
    """Streaming text chunk sent server → client."""
    type: str = "chunk"
    text: str


class WsThinking(BaseModel):
    """Thinking/processing indicator sent server → client."""
    type: str = "thinking"
    message: str = "JARVIS is thinking..."


class WsToolCall(BaseModel):
    """Tool invocation notification sent server → client."""
    type: str = "tool_call"
    tool: str


class WsDone(BaseModel):
    """Final message sent when response is complete."""
    type: str = "done"
    text: str
    usage: UsageSummary


class WsError(BaseModel):
    """Error notification."""
    type: str = "error"
    message: str


class WsApprovalRequest(BaseModel):
    """Server → client: JARVIS wants to execute a risky tool and needs permission."""
    type: str = "approval_request"
    request_id: str
    tool_name: str
    description: str
    risk_level: str
    expires_in: int


class WsApprovalResponse(BaseModel):
    """Client → server: user approves or denies a tool call."""
    type: str = "approval_response"
    request_id: str
    approved: bool


class WsProactive(BaseModel):
    """Server → client: JARVIS-initiated message (not in response to user input)."""
    type: str = "proactive"
    trigger: str
    text: str
    severity: str = "info"


class WsPing(BaseModel):
    """Server → client: heartbeat keep-alive ping."""
    type: str = "ping"


# ── Queue tasks ───────────────────────────────────────────────────────────────

class QueueTask(BaseModel):
    """Task published to RabbitMQ for async processing."""
    task_id: str
    message: str
    session_id: str
    researcher_mode: bool = False
    reply_to: str | None = None  # RabbitMQ reply-to queue


class QueueResult(BaseModel):
    """Result returned by the worker via RabbitMQ."""
    task_id: str
    session_id: str
    response: str
    usage: UsageSummary
    error: str | None = None


# ── Scheduler ─────────────────────────────────────────────────────────────────

class ScheduleRequest(BaseModel):
    """Create a recurring proactive agent job."""
    job_type: str = Field(..., description="'research' to save periodic reports, 'monitor' to watch for updates")
    params: dict[str, str] = Field(
        ...,
        description="For research: {'topic': 'RLHF'}. For monitor: {'query': 'GPT-5 release'}.",
    )
    cron: str = Field("0 9 * * *", description="Cron expression (UTC): minute hour day month weekday")
    session_id: str | None = Field(None, description="Session to associate results with")


class ScheduleItem(BaseModel):
    """A scheduled job entry."""
    job_id: str
    job_type: str
    subject: str
    cron: str
    next_run: str | None = None


class ScheduleResponse(BaseModel):
    job_id: str
    message: str


# ── Feedback ──────────────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    session_id: str
    response_snippet: str = ""
    rating: int = Field(..., ge=-1, le=5, description="-1/+1 thumbs or 1-5 stars")
    comment: str = ""
    rating_type: str = Field("thumbs", description="'thumbs' for -1/+1, 'stars' for 1-5")


class FeedbackStatsResponse(BaseModel):
    total: int
    avg_rating: float
    recent: list[dict]


# ── Budget ────────────────────────────────────────────────────────────────────

class BudgetRequest(BaseModel):
    monthly_budget_usd: float = Field(..., ge=0, description="0 = unlimited")


class BudgetStatusResponse(BaseModel):
    user_id: str
    monthly_budget_usd: float
    spent_usd: float
    remaining_usd: float | None
    period: str


# ── Auth ──────────────────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    username: str
    password: str
    role: str = "user"


class TokenRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


# ── Sessions ──────────────────────────────────────────────────────────────────

class SessionInfo(BaseModel):
    session_id: str
    created_at: float
    message_count: int
    user_id: str | None = None


# ── Health ────────────────────────────────────────────────────────────────────

class ComponentStatus(BaseModel):
    ok: bool
    detail: str = ""


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    sessions_active: int = 0
    ws_connections: int = 0
    components: dict[str, ComponentStatus] = {}
