"""Tests for Phase 6 additions: tool timeout, health endpoint, audit pagination,
tool metrics, and WebSocket heartbeat model."""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jarvis.security.audit import get_recent_audit, log_tool_call


# ── 6.1 Per-tool timeout ──────────────────────────────────────────────────────

class TestToolTimeout:
    def test_slow_tool_returns_error(self, tmp_path: Path) -> None:
        """A handler that sleeps past the timeout should return an ERROR string."""
        from jarvis.agents.researcher import ResearcherAgent
        from jarvis.tools.registry import build_registry

        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        schemas, registry = build_registry(tavily_api_key="fake", reports_dir=reports_dir)

        # Inject a slow tool
        def _slow(_input: dict) -> str:
            time.sleep(999)
            return "should not reach here"

        registry["slow_tool"] = _slow

        client = MagicMock()
        agent = ResearcherAgent(
            client=client,
            model="claude-sonnet-4-6",
            max_tokens=1024,
            tool_schemas=schemas,
            tool_registry=registry,
        )

        with patch("jarvis.config.get_settings") as mock_settings:
            s = MagicMock()
            s.tool_timeout_seconds = 1
            s.reports_dir = tmp_path
            mock_settings.return_value = s
            result = agent._dispatch("slow_tool", {})

        assert result.startswith("ERROR:")
        assert "timed out" in result.lower()

    def test_fast_tool_completes_normally(self, tmp_path: Path) -> None:
        from jarvis.agents.researcher import ResearcherAgent
        from jarvis.tools.registry import build_registry

        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        schemas, registry = build_registry(tavily_api_key="fake", reports_dir=reports_dir)
        registry["fast_tool"] = lambda _: "done"

        client = MagicMock()
        agent = ResearcherAgent(
            client=client, model="m", max_tokens=100,
            tool_schemas=schemas, tool_registry=registry,
        )

        with patch("jarvis.config.get_settings") as mock_settings:
            s = MagicMock()
            s.tool_timeout_seconds = 5
            s.reports_dir = tmp_path
            mock_settings.return_value = s
            result = agent._dispatch("fast_tool", {})

        assert result == "done"

    def test_unknown_tool_returns_error(self, tmp_path: Path) -> None:
        from jarvis.agents.researcher import ResearcherAgent
        from jarvis.tools.registry import build_registry

        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        schemas, registry = build_registry(tavily_api_key="fake", reports_dir=reports_dir)
        client = MagicMock()
        agent = ResearcherAgent(
            client=client, model="m", max_tokens=100,
            tool_schemas=schemas, tool_registry=registry,
        )
        result = agent._dispatch("nonexistent_tool", {})
        assert result.startswith("ERROR:")
        assert "unknown tool" in result


# ── 6.4 Audit pagination ──────────────────────────────────────────────────────

class TestAuditPagination:
    def test_offset_returns_correct_page(self, db_path: Path) -> None:
        for i in range(10):
            log_tool_call(db_path, "sess", f"tool_{i}", {}, risk_level="safe", approved=1, duration_ms=1)

        page1 = get_recent_audit(db_path, limit=5, offset=0)
        page2 = get_recent_audit(db_path, limit=5, offset=5)

        assert len(page1) == 5
        assert len(page2) == 5
        # Pages should not overlap
        names1 = {r["tool_name"] for r in page1}
        names2 = {r["tool_name"] for r in page2}
        assert names1.isdisjoint(names2)

    def test_offset_beyond_end_returns_empty(self, db_path: Path) -> None:
        log_tool_call(db_path, "s", "tool_a", {}, risk_level="safe", approved=1, duration_ms=1)
        result = get_recent_audit(db_path, limit=10, offset=999)
        assert result == []

    def test_limit_respected_with_offset(self, db_path: Path) -> None:
        for i in range(20):
            log_tool_call(db_path, "s", f"t{i}", {}, risk_level="safe", approved=1, duration_ms=1)
        result = get_recent_audit(db_path, limit=3, offset=5)
        assert len(result) == 3


# ── 6.5 Tool metrics ──────────────────────────────────────────────────────────

class TestToolMetrics:
    def test_tool_duration_recorded(self, tmp_path: Path) -> None:
        from jarvis.agents.researcher import ResearcherAgent
        from jarvis.tools.registry import build_registry

        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        schemas, registry = build_registry(tavily_api_key="fake", reports_dir=reports_dir)
        registry["metric_tool"] = lambda _: "ok"

        client = MagicMock()
        agent = ResearcherAgent(
            client=client, model="m", max_tokens=100,
            tool_schemas=schemas, tool_registry=registry,
        )

        recorded: list[tuple] = []

        with (
            patch("jarvis.config.get_settings") as mock_settings,
            patch("jarvis.api.metrics.TOOL_DURATION") as mock_hist,
        ):
            s = MagicMock()
            s.tool_timeout_seconds = 5
            s.reports_dir = tmp_path
            mock_settings.return_value = s

            mock_hist.labels.return_value.observe = lambda d: recorded.append(("metric_tool", d))
            agent._dispatch("metric_tool", {})

        assert len(recorded) == 1
        assert recorded[0][0] == "metric_tool"
        assert recorded[0][1] >= 0


# ── 6.3 Enhanced health response model ───────────────────────────────────────

class TestHealthModel:
    def test_component_status_model(self) -> None:
        from jarvis.api.models import ComponentStatus, HealthResponse
        cs = ComponentStatus(ok=True, detail="running")
        assert cs.ok is True
        assert cs.detail == "running"

    def test_health_response_with_components(self) -> None:
        from jarvis.api.models import ComponentStatus, HealthResponse
        hr = HealthResponse(
            status="degraded",
            sessions_active=2,
            ws_connections=1,
            components={"db": ComponentStatus(ok=False, detail="timeout")},
        )
        assert hr.status == "degraded"
        assert hr.ws_connections == 1
        assert hr.components["db"].ok is False


# ── 6.2 WsPing model ─────────────────────────────────────────────────────────

class TestWsPing:
    def test_ws_ping_type(self) -> None:
        from jarvis.api.models import WsPing
        ping = WsPing()
        assert ping.type == "ping"
        assert ping.model_dump()["type"] == "ping"
