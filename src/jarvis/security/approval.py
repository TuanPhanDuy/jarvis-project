"""Action approval gate — classifies tool risk and requests user confirmation.

Tools at or above the configured approval_threshold block until the user
approves or denies via the WebSocket approval flow (WsApprovalRequest /
WsApprovalResponse).  The gate uses threading.Event so it works inside the
ThreadPoolExecutor that runs synchronous agent turns.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class RiskLevel(Enum):
    SAFE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


# Explicit risk map — every tool is classified here.
# Tools not in this map default to LOW.
TOOL_RISK_MAP: dict[str, RiskLevel] = {
    # Read-only / informational
    "web_search": RiskLevel.SAFE,
    "search_memory": RiskLevel.SAFE,
    "search_episodic_memory": RiskLevel.SAFE,
    "query_knowledge_graph": RiskLevel.SAFE,
    "read_url": RiskLevel.SAFE,
    "analyze_failures": RiskLevel.SAFE,
    "recall_user_preferences": RiskLevel.SAFE,
    "query_plan_history": RiskLevel.SAFE,
    # Side effects — write data locally
    "save_report": RiskLevel.LOW,
    "update_report": RiskLevel.LOW,
    "update_knowledge_graph": RiskLevel.LOW,
    "update_user_preference": RiskLevel.LOW,
    "record_feedback": RiskLevel.LOW,
    "export_conversation": RiskLevel.LOW,
    # Medium — external interactions or hardware
    "browse": RiskLevel.MEDIUM,
    "capture_camera": RiskLevel.MEDIUM,
    "recognize_face": RiskLevel.MEDIUM,
    "delegate_task": RiskLevel.MEDIUM,
    "create_plan": RiskLevel.MEDIUM,
    # High — OS-level, external APIs with auth
    "run_command": RiskLevel.HIGH,
    "run_os_command": RiskLevel.HIGH,
    # New plugin tools
    "system_info": RiskLevel.SAFE,
    "query_database": RiskLevel.LOW,
    "manage_reminder": RiskLevel.LOW,
}


@dataclass
class ApprovalRequest:
    request_id: str
    tool_name: str
    tool_input: dict
    risk_level: RiskLevel
    description: str
    session_id: str
    user_id: str | None
    timestamp: float
    expires_at: float
    _event: threading.Event = field(default_factory=threading.Event, repr=False)
    _approved: bool = field(default=False, repr=False)


class ApprovalGate:
    """Blocks tool dispatch for high-risk tools until user responds.

    Args:
        threshold: Tools at or above this level require explicit approval.
        timeout_seconds: Seconds to wait before auto-denying.
        request_callback: Called with ApprovalRequest when approval is needed.
            Use this to push WsApprovalRequest over WebSocket.
    """

    def __init__(
        self,
        threshold: RiskLevel = RiskLevel.MEDIUM,
        timeout_seconds: int = 60,
        request_callback: Callable[[ApprovalRequest], None] | None = None,
        session_id: str = "",
        user_id: str | None = None,
    ) -> None:
        self._threshold = threshold
        self._timeout = timeout_seconds
        self._callback = request_callback
        self._session_id = session_id
        self._user_id = user_id
        self._pending: dict[str, ApprovalRequest] = {}

    def requires_approval(self, tool_name: str) -> bool:
        level = TOOL_RISK_MAP.get(tool_name, RiskLevel.MEDIUM)
        return level.value >= self._threshold.value

    def check_sync(self, tool_name: str, tool_input: dict) -> bool:
        """Block until the user approves/denies, or timeout elapses.

        Returns True if approved, False if denied or timed out.
        This is called from the agent thread (ThreadPoolExecutor).
        """
        if not self.requires_approval(tool_name):
            return True

        level = TOOL_RISK_MAP.get(tool_name, RiskLevel.MEDIUM)
        description = _describe_tool_call(tool_name, tool_input)
        now = time.time()
        req = ApprovalRequest(
            request_id=str(uuid.uuid4()),
            tool_name=tool_name,
            tool_input=tool_input,
            risk_level=level,
            description=description,
            session_id=self._session_id,
            user_id=self._user_id,
            timestamp=now,
            expires_at=now + self._timeout,
        )
        self._pending[req.request_id] = req

        if self._callback:
            try:
                self._callback(req)
            except Exception:
                pass

        approved = req._event.wait(timeout=self._timeout)
        self._pending.pop(req.request_id, None)
        return approved and req._approved

    def resolve(self, request_id: str, approved: bool) -> None:
        """Called from the async WebSocket handler when the user responds."""
        req = self._pending.get(request_id)
        if req:
            req._approved = approved
            req._event.set()

    def get_pending(self) -> list[dict]:
        """Return serializable list of pending requests."""
        now = time.time()
        return [
            {
                "request_id": r.request_id,
                "tool_name": r.tool_name,
                "description": r.description,
                "risk_level": r.risk_level.name,
                "expires_in": max(0, int(r.expires_at - now)),
            }
            for r in self._pending.values()
            if r.expires_at > now
        ]


def _describe_tool_call(tool_name: str, tool_input: dict) -> str:
    """Human-readable description of what a tool call will do."""
    if tool_name == "run_command":
        return f"Run shell command: {tool_input.get('command', '?')}"
    if tool_name == "browse":
        return f"Open browser at: {tool_input.get('url', '?')}"
    if tool_name == "capture_camera":
        return "Take a photo from your webcam"
    if tool_name == "recognize_face":
        return "Scan webcam for faces"
    if tool_name == "delegate_task":
        agent = tool_input.get("agent_type", "?")
        task = tool_input.get("task", "")[:80]
        return f"Delegate to {agent} agent: {task}"
    if tool_name == "create_plan":
        goal = tool_input.get("goal", "")[:80]
        steps = tool_input.get("steps", [])
        return f"Execute {len(steps)}-step plan: {goal}"
    # Fallback
    parts = [f"{k}={str(v)[:40]}" for k, v in tool_input.items()]
    return f"{tool_name}({', '.join(parts[:3])})"


class ToolDeniedException(Exception):
    """Raised when the user explicitly denies a tool call approval."""
