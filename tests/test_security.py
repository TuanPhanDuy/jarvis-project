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

    def test_unknown_tool_defaults_to_low_risk(self) -> None:
        gate = ApprovalGate(threshold=RiskLevel.HIGH)
        # Unknown tool → LOW < HIGH → no approval needed
        assert not gate.requires_approval("some_unknown_plugin_tool")

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
