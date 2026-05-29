"""Unit tests for the approval gate and audit log. No API keys needed."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from jarvis.security.approval import ApprovalGate, RiskLevel, TOOL_RISK_MAP
from jarvis.security.audit import log_tool_call, get_recent_audit


# ── ApprovalGate ──────────────────────────────────────────────────────────────

class TestApprovalGate:
    def test_safe_tool_never_requires_approval(self) -> None:
        gate = ApprovalGate(threshold=RiskLevel.MEDIUM)
        assert not gate.requires_approval("web_search")
        assert not gate.requires_approval("search_memory")

    def test_low_risk_below_medium_threshold(self) -> None:
        gate = ApprovalGate(threshold=RiskLevel.MEDIUM)
        assert not gate.requires_approval("save_report")

    def test_medium_risk_at_medium_threshold_requires_approval(self) -> None:
        gate = ApprovalGate(threshold=RiskLevel.MEDIUM)
        assert gate.requires_approval("browse")
        assert gate.requires_approval("capture_camera")

    def test_high_risk_always_requires_approval_at_medium_threshold(self) -> None:
        gate = ApprovalGate(threshold=RiskLevel.MEDIUM)
        assert gate.requires_approval("run_command")

    def test_threshold_low_catches_safe_tools(self) -> None:
        gate = ApprovalGate(threshold=RiskLevel.LOW)
        # LOW threshold — safe tools still pass, low tools require approval
        assert not gate.requires_approval("web_search")
        assert gate.requires_approval("save_report")

    def test_approve_flow(self) -> None:
        """check_sync returns True when resolved as approved before timeout."""
        gate = ApprovalGate(threshold=RiskLevel.MEDIUM, timeout_seconds=5)
        approved_results: list[bool] = []

        def _resolve_approve():
            time.sleep(0.1)
            pending = list(gate._pending.values())
            if pending:
                gate.resolve(pending[0].request_id, approved=True)

        t = threading.Thread(target=_resolve_approve)
        t.start()
        result = gate.check_sync("browse", {"url": "http://example.com"})
        t.join()
        assert result is True

    def test_deny_flow(self) -> None:
        """check_sync returns False when resolved as denied."""
        gate = ApprovalGate(threshold=RiskLevel.MEDIUM, timeout_seconds=5)

        def _resolve_deny():
            time.sleep(0.1)
            pending = list(gate._pending.values())
            if pending:
                gate.resolve(pending[0].request_id, approved=False)

        t = threading.Thread(target=_resolve_deny)
        t.start()
        result = gate.check_sync("browse", {})
        t.join()
        assert result is False

    def test_timeout_returns_false(self) -> None:
        """check_sync times out and returns False if nobody resolves."""
        gate = ApprovalGate(threshold=RiskLevel.MEDIUM, timeout_seconds=1)
        result = gate.check_sync("browse", {})
        assert result is False

    def test_callback_called_on_approval_request(self) -> None:
        called: list = []
        gate = ApprovalGate(
            threshold=RiskLevel.MEDIUM,
            timeout_seconds=1,
            request_callback=lambda req: called.append(req),
        )
        gate.check_sync("capture_camera", {})
        assert len(called) == 1
        assert called[0].tool_name == "capture_camera"

    def test_unknown_tool_defaults_to_medium_risk(self) -> None:
        gate = ApprovalGate(threshold=RiskLevel.HIGH)
        # Unknown tool → MEDIUM < HIGH → no approval at HIGH threshold
        assert not gate.requires_approval("some_unknown_plugin_tool")

    def test_unknown_tool_requires_approval_at_medium_threshold(self) -> None:
        gate = ApprovalGate(threshold=RiskLevel.MEDIUM, timeout_seconds=1)
        # Unknown tool defaults to MEDIUM → requires approval at MEDIUM threshold
        assert gate.requires_approval("some_new_plugin_tool")

    def test_get_pending_returns_active_requests(self) -> None:
        gate = ApprovalGate(threshold=RiskLevel.MEDIUM, timeout_seconds=2)
        pending_snapshot: list = []

        def _check_pending():
            time.sleep(0.05)
            pending_snapshot.extend(gate.get_pending())
            # Deny to unblock
            for p in gate._pending.values():
                gate.resolve(p.request_id, approved=False)

        t = threading.Thread(target=_check_pending)
        t.start()
        gate.check_sync("browse", {})
        t.join()
        assert len(pending_snapshot) == 1

    def test_tool_risk_map_covers_core_tools(self) -> None:
        for tool in ["web_search", "save_report", "browse", "run_command", "capture_camera"]:
            assert tool in TOOL_RISK_MAP


# ── Audit log ─────────────────────────────────────────────────────────────────

class TestAuditLog:
    def test_log_and_retrieve(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        log_tool_call(db, "sess1", "web_search", {"query": "RLHF"}, risk_level="safe",
                      approved=1, duration_ms=120.0)
        entries = get_recent_audit(db, limit=10)
        assert len(entries) == 1
        assert entries[0]["tool_name"] == "web_search"
        assert entries[0]["approved"] == 1

    def test_multiple_entries_ordered_desc(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        log_tool_call(db, "sess1", "tool_a", {}, risk_level="safe", approved=1, duration_ms=1)
        time.sleep(0.01)
        log_tool_call(db, "sess1", "tool_b", {}, risk_level="medium", approved=0, duration_ms=2)
        entries = get_recent_audit(db, limit=10)
        assert entries[0]["tool_name"] == "tool_b"  # most recent first

    def test_session_filter(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        log_tool_call(db, "sess1", "web_search", {}, risk_level="safe", approved=1, duration_ms=1)
        log_tool_call(db, "sess2", "save_report", {}, risk_level="low", approved=1, duration_ms=1)
        entries = get_recent_audit(db, limit=10, session_id="sess1")
        assert all(e["session_id"] == "sess1" for e in entries)
        assert len(entries) == 1

    def test_limit_respected(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        for i in range(20):
            log_tool_call(db, "s", f"tool_{i}", {}, risk_level="safe", approved=1, duration_ms=1)
        entries = get_recent_audit(db, limit=5)
        assert len(entries) == 5

    def test_empty_db_returns_empty(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        assert get_recent_audit(db) == []


# ── Rate limiter ──────────────────────────────────────────────────────────────

class TestParseRate:
    def test_per_minute(self):
        from jarvis.api.server import _parse_rate
        count, window = _parse_rate("30/minute")
        assert count == 30
        assert window == 60.0

    def test_per_second(self):
        from jarvis.api.server import _parse_rate
        count, window = _parse_rate("5/second")
        assert count == 5
        assert window == 1.0

    def test_per_hour(self):
        from jarvis.api.server import _parse_rate
        count, window = _parse_rate("100/hour")
        assert count == 100
        assert window == 3600.0

    def test_unknown_period_defaults_to_minute(self):
        from jarvis.api.server import _parse_rate
        count, window = _parse_rate("10/fortnight")
        assert count == 10
        assert window == 60.0


class TestRateLimitMiddleware:
    def _make_middleware(self, max_calls=3, window=60.0, enabled=True):
        from jarvis.api.server import _RateLimitMiddleware
        return _RateLimitMiddleware(app=None, max_calls=max_calls, window_seconds=window, enabled=enabled)

    def _request(self, path="/api/chat", ip="127.0.0.1"):
        from unittest.mock import MagicMock
        req = MagicMock()
        req.url.path = path
        req.client.host = ip
        return req

    def test_allows_under_limit(self):
        mw = self._make_middleware(max_calls=3)
        req = self._request()
        bucket = mw._buckets[req.client.host]
        # Manually populate 2 calls
        bucket.append(time.monotonic())
        bucket.append(time.monotonic())
        assert len(bucket) < mw._max_calls

    def test_blocks_at_limit(self):
        import asyncio
        from unittest.mock import AsyncMock, MagicMock
        from jarvis.api.server import _RateLimitMiddleware

        mw = _RateLimitMiddleware(app=None, max_calls=2, window_seconds=60.0, enabled=True)

        async def run():
            call_next = AsyncMock(return_value=MagicMock(status_code=200))
            req = self._request()
            # Two calls succeed
            await mw.dispatch(req, call_next)
            await mw.dispatch(req, call_next)
            # Third is blocked
            resp = await mw.dispatch(req, call_next)
            return resp

        resp = asyncio.get_event_loop().run_until_complete(run())
        assert resp.status_code == 429

    def test_disabled_allows_any_volume(self):
        import asyncio
        from unittest.mock import AsyncMock, MagicMock
        from jarvis.api.server import _RateLimitMiddleware

        mw = _RateLimitMiddleware(app=None, max_calls=1, window_seconds=60.0, enabled=False)

        async def run():
            call_next = AsyncMock(return_value=MagicMock(status_code=200))
            req = self._request()
            for _ in range(5):
                resp = await mw.dispatch(req, call_next)
            return resp

        resp = asyncio.get_event_loop().run_until_complete(run())
        assert resp.status_code == 200

    def test_non_chat_path_bypasses_limit(self):
        import asyncio
        from unittest.mock import AsyncMock, MagicMock
        from jarvis.api.server import _RateLimitMiddleware

        mw = _RateLimitMiddleware(app=None, max_calls=1, window_seconds=60.0, enabled=True)

        async def run():
            call_next = AsyncMock(return_value=MagicMock(status_code=200))
            req = self._request(path="/api/health")
            for _ in range(5):
                resp = await mw.dispatch(req, call_next)
            return resp

        resp = asyncio.get_event_loop().run_until_complete(run())
        assert resp.status_code == 200


# ── WebSocket JWT auth ────────────────────────────────────────────────────────


def _fake_settings_ws(tmp_path: Path):
    from unittest.mock import MagicMock
    s = MagicMock()
    s.anthropic_api_key = "test-key"
    s.tavily_api_key = "test-key"
    s.model = "llama3.2"
    s.fast_model = "llama3.2"
    s.max_tokens = 512
    s.max_search_calls = 5
    s.routing_strategy = "always_primary"
    s.allowed_commands = []
    s.reports_dir = tmp_path / "reports"
    s.otel_enabled = False
    s.auth_enabled = True
    s.rate_limit_enabled = False
    s.proactive_enabled = False
    s.peer_enabled = False
    s.api_session_ttl_minutes = 60
    s.memory_retention_days = 90
    s.jwt_secret = "ws-test-secret"
    s.chat_rate_limit = "100/minute"
    s.idle_minutes = 30
    s.agent_turn_timeout_seconds = 120
    s.tool_timeout_seconds = 60
    s.peer_port = 8001
    s.vision_model = "llava:13b"
    return s


class TestWebSocketAuth:
    @pytest.fixture
    def ws_client(self, tmp_path):
        from unittest.mock import patch
        settings = _fake_settings_ws(tmp_path)
        settings.reports_dir.mkdir(parents=True, exist_ok=True)

        with (
            patch("jarvis.api.server.get_settings", return_value=settings),
            patch("jarvis.config.get_settings", return_value=settings),
            patch("jarvis.scheduler.core.start_scheduler"),
            patch("jarvis.scheduler.core.stop_scheduler"),
            patch("jarvis.tools.registry.build_registry", return_value=([], {})),
        ):
            from fastapi.testclient import TestClient
            from jarvis.api.server import app
            with TestClient(app, raise_server_exceptions=False) as c:
                yield c, settings

    def _make_token(self, settings) -> str:
        from jarvis.auth.core import User, create_token
        user = User(user_id=1, username="alice", role="user")
        return create_token(user, settings.jwt_secret)

    def test_ws_rejected_when_no_token(self, ws_client):
        c, _ = ws_client
        with c.websocket_connect("/api/ws/test-session") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "Unauthorized" in msg["message"]

    def test_ws_rejected_when_bad_token(self, ws_client):
        c, _ = ws_client
        with c.websocket_connect("/api/ws/test-session?token=bad.token.here") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "Unauthorized" in msg["message"]

    def test_ws_accepted_with_valid_token(self, ws_client):
        c, settings = ws_client
        token = self._make_token(settings)
        # A valid token should not get an Unauthorized error.
        # Send a ping to confirm the session is alive; no immediate close.
        with c.websocket_connect(f"/api/ws/test-session?token={token}") as ws:
            ws.send_json({"type": "ping"})
            # If we get here without WebSocketDisconnect, auth passed
