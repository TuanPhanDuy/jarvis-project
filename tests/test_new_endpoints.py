"""Tests for all new endpoints and library functions added in this round:
  - prune_old_audit / prune_old_turns
  - episodic search_episodes / delete_episodes
  - preferences get_session_summaries_full
  - auth list_users / delete_user / update_user_role
  - graph delete_entity / delete_relationship
  - feedback get_feedback_list
  - audit get_audit_stats
  - training get_run_by_id / delete_run
  - TOOL_RISK_MAP new plugin classifications
  - API: GET /api/config, GET|DELETE /api/memory/episodes/search|episodes,
         GET /api/memory/summaries/{user_id},
         GET/DELETE/PATCH /api/users,
         GET /api/schedules/{job_id},
         GET/DELETE /api/training/runs/{run_id},
         DELETE /api/knowledge-graph/entities/{name},
         DELETE /api/knowledge-graph/relationships,
         GET /api/feedback (list),
         GET /api/audit/stats
"""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Helpers / fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _fake_settings(tmp_path: Path) -> MagicMock:
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
    s.auth_enabled = False
    s.rate_limit_enabled = False
    s.proactive_enabled = False
    s.peer_enabled = False
    s.api_session_ttl_minutes = 60
    s.memory_retention_days = 90
    s.jwt_secret = "test-secret"
    s.chat_rate_limit = "100/minute"
    s.idle_minutes = 30
    s.agent_turn_timeout_seconds = 120
    s.tool_timeout_seconds = 60
    s.peer_port = 8001
    s.vision_model = "llava:13b"
    s.auto_training_enabled = False
    return s


@pytest.fixture
def client(tmp_path: Path):
    settings = _fake_settings(tmp_path)
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
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c, settings.reports_dir


# ─────────────────────────────────────────────────────────────────────────────
# Unit: prune_old_audit / prune_old_turns
# ─────────────────────────────────────────────────────────────────────────────

class TestPruneOldAudit:
    def test_returns_zero_for_missing_db(self, tmp_path: Path) -> None:
        from jarvis.security.audit import prune_old_audit
        assert prune_old_audit(tmp_path / "missing.db", 90) == 0

    def test_deletes_old_rows(self, tmp_path: Path) -> None:
        from jarvis.security.audit import _get_conn, prune_old_audit
        db = tmp_path / "jarvis.db"
        conn = _get_conn(db)
        old_ts = time.time() - 100 * 86400
        conn.execute(
            "INSERT INTO audit_log (timestamp, session_id, tool_name, tool_input, risk_level, approved) "
            "VALUES (?,?,?,?,?,?)", (old_ts, "s1", "web_search", "{}", "SAFE", 1)
        )
        conn.commit()
        conn.close()
        deleted = prune_old_audit(db, 90)
        assert deleted == 1

    def test_keeps_recent_rows(self, tmp_path: Path) -> None:
        from jarvis.security.audit import _get_conn, prune_old_audit
        db = tmp_path / "jarvis.db"
        conn = _get_conn(db)
        conn.execute(
            "INSERT INTO audit_log (timestamp, session_id, tool_name, tool_input, risk_level, approved) "
            "VALUES (?,?,?,?,?,?)", (time.time(), "s1", "web_search", "{}", "SAFE", 1)
        )
        conn.commit()
        conn.close()
        deleted = prune_old_audit(db, 90)
        assert deleted == 0


class TestPruneOldTurns:
    def test_returns_zero_for_missing_db(self, tmp_path: Path) -> None:
        from jarvis.memory.turns import prune_old_turns
        assert prune_old_turns(tmp_path / "missing.db", 90) == 0

    def test_deletes_old_rows(self, tmp_path: Path) -> None:
        from jarvis.memory.turns import _get_conn, log_turn, prune_old_turns
        db = tmp_path / "jarvis.db"
        conn = _get_conn(db)
        old_ts = time.time() - 100 * 86400
        conn.execute(
            "INSERT INTO agent_turns (id, session_id, agent_type, model, input_tokens, "
            "output_tokens, tool_calls_json, latency_ms, timestamp) VALUES (?,?,?,?,?,?,?,?,?)",
            ("t1", "s1", "researcher", "llama3.2", 100, 50, "[]", 200.0, old_ts)
        )
        conn.commit()
        conn.close()
        deleted = prune_old_turns(db, 90)
        assert deleted == 1

    def test_keeps_recent_rows(self, tmp_path: Path) -> None:
        from jarvis.memory.turns import _get_conn, prune_old_turns
        db = tmp_path / "jarvis.db"
        conn = _get_conn(db)
        conn.execute(
            "INSERT INTO agent_turns (id, session_id, agent_type, model, input_tokens, "
            "output_tokens, tool_calls_json, latency_ms, timestamp) VALUES (?,?,?,?,?,?,?,?,?)",
            ("t2", "s1", "researcher", "llama3.2", 100, 50, "[]", 200.0, time.time())
        )
        conn.commit()
        conn.close()
        assert prune_old_turns(db, 90) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Unit: episodic search_episodes / delete_episodes
# ─────────────────────────────────────────────────────────────────────────────

class TestSearchEpisodes:
    def test_returns_empty_for_missing_db(self, tmp_path: Path) -> None:
        from jarvis.memory.episodic import search_episodes
        assert search_episodes(tmp_path / "missing.db", "RLHF") == []

    def test_finds_matching_episode(self, tmp_path: Path) -> None:
        from jarvis.memory.episodic import log_episode, search_episodes
        db = tmp_path / "jarvis.db"
        log_episode(db, "s1", "user", "Tell me about RLHF and reward models", "alice")
        results = search_episodes(db, "RLHF")
        assert len(results) >= 1
        assert any("RLHF" in r["content"] for r in results)

    def test_does_not_find_non_matching(self, tmp_path: Path) -> None:
        from jarvis.memory.episodic import log_episode, search_episodes
        db = tmp_path / "jarvis.db"
        log_episode(db, "s1", "user", "Tell me about transformers", "alice")
        results = search_episodes(db, "RLHF")
        assert results == []

    def test_user_id_filter(self, tmp_path: Path) -> None:
        from jarvis.memory.episodic import log_episode, search_episodes
        db = tmp_path / "jarvis.db"
        log_episode(db, "s1", "user", "RLHF for alice", "alice")
        log_episode(db, "s2", "user", "RLHF for bob", "bob")
        results = search_episodes(db, "RLHF", user_id="alice")
        assert all("alice" in r["content"] for r in results)


class TestDeleteEpisodes:
    def test_raises_if_no_filter(self, tmp_path: Path) -> None:
        from jarvis.memory.episodic import delete_episodes
        db = tmp_path / "jarvis.db"
        with pytest.raises(ValueError):
            delete_episodes(db)

    def test_deletes_by_session_id(self, tmp_path: Path) -> None:
        from jarvis.memory.episodic import delete_episodes, log_episode
        db = tmp_path / "jarvis.db"
        log_episode(db, "sess-A", "user", "hello", "alice")
        log_episode(db, "sess-B", "user", "world", "alice")
        deleted = delete_episodes(db, session_id="sess-A")
        assert deleted == 1

    def test_deletes_by_user_id(self, tmp_path: Path) -> None:
        from jarvis.memory.episodic import delete_episodes, log_episode
        db = tmp_path / "jarvis.db"
        log_episode(db, "s1", "user", "alice msg", "alice")
        log_episode(db, "s2", "user", "bob msg", "bob")
        deleted = delete_episodes(db, user_id="alice")
        assert deleted == 1

    def test_deletes_with_both_filters(self, tmp_path: Path) -> None:
        from jarvis.memory.episodic import delete_episodes, log_episode
        db = tmp_path / "jarvis.db"
        log_episode(db, "s1", "user", "alice in s1", "alice")
        log_episode(db, "s2", "user", "alice in s2", "alice")
        deleted = delete_episodes(db, session_id="s1", user_id="alice")
        assert deleted == 1


# ─────────────────────────────────────────────────────────────────────────────
# Unit: get_session_summaries_full
# ─────────────────────────────────────────────────────────────────────────────

class TestGetSessionSummariesFull:
    def test_returns_empty_for_missing_db(self, tmp_path: Path) -> None:
        from jarvis.memory.preferences import get_session_summaries_full
        assert get_session_summaries_full(tmp_path / "missing.db", "alice") == []

    def test_returns_saved_summary(self, tmp_path: Path) -> None:
        from jarvis.memory.preferences import get_session_summaries_full, save_session_summary
        db = tmp_path / "jarvis.db"
        save_session_summary(db, "sess1", "alice", "Talked about RLHF", ["RLHF", "reward"])
        results = get_session_summaries_full(db, "alice")
        assert len(results) == 1
        assert results[0]["summary"] == "Talked about RLHF"
        assert "RLHF" in results[0]["key_topics"]

    def test_includes_all_fields(self, tmp_path: Path) -> None:
        from jarvis.memory.preferences import get_session_summaries_full, save_session_summary
        db = tmp_path / "jarvis.db"
        save_session_summary(db, "sess1", "alice", "Summary", [])
        r = get_session_summaries_full(db, "alice")[0]
        for key in ("session_id", "user_id", "summary", "key_topics", "created_at"):
            assert key in r

    def test_user_isolated(self, tmp_path: Path) -> None:
        from jarvis.memory.preferences import get_session_summaries_full, save_session_summary
        db = tmp_path / "jarvis.db"
        save_session_summary(db, "s1", "alice", "Alice summary", [])
        save_session_summary(db, "s2", "bob", "Bob summary", [])
        assert len(get_session_summaries_full(db, "alice")) == 1
        assert len(get_session_summaries_full(db, "bob")) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Unit: auth list_users / delete_user / update_user_role
# ─────────────────────────────────────────────────────────────────────────────

class TestAuthUserManagement:
    def test_list_users_empty(self, tmp_path: Path) -> None:
        from jarvis.auth.core import list_users, _get_conn
        db = tmp_path / "jarvis.db"
        _get_conn(db).close()
        assert list_users(db) == []

    def test_list_users_returns_created_user(self, tmp_path: Path) -> None:
        from jarvis.auth.core import create_user, list_users
        db = tmp_path / "jarvis.db"
        create_user(db, "alice", "pass", "user")
        users = list_users(db)
        assert len(users) == 1
        assert users[0]["username"] == "alice"

    def test_list_users_no_password_in_result(self, tmp_path: Path) -> None:
        from jarvis.auth.core import create_user, list_users
        db = tmp_path / "jarvis.db"
        create_user(db, "alice", "secret", "user")
        users = list_users(db)
        assert "hashed_password" not in users[0]
        assert "salt" not in users[0]

    def test_delete_user_returns_true(self, tmp_path: Path) -> None:
        from jarvis.auth.core import create_user, delete_user
        db = tmp_path / "jarvis.db"
        create_user(db, "alice", "pass", "user")
        assert delete_user(db, "alice") is True

    def test_delete_user_returns_false_for_unknown(self, tmp_path: Path) -> None:
        from jarvis.auth.core import delete_user, _get_conn
        db = tmp_path / "jarvis.db"
        _get_conn(db).close()
        assert delete_user(db, "nobody") is False

    def test_delete_user_removes_user(self, tmp_path: Path) -> None:
        from jarvis.auth.core import create_user, delete_user, get_user
        db = tmp_path / "jarvis.db"
        create_user(db, "alice", "pass", "user")
        delete_user(db, "alice")
        assert get_user(db, "alice") is None

    def test_update_role_returns_true(self, tmp_path: Path) -> None:
        from jarvis.auth.core import create_user, update_user_role
        db = tmp_path / "jarvis.db"
        create_user(db, "alice", "pass", "user")
        assert update_user_role(db, "alice", "admin") is True

    def test_update_role_changes_role(self, tmp_path: Path) -> None:
        from jarvis.auth.core import create_user, get_user, update_user_role
        db = tmp_path / "jarvis.db"
        create_user(db, "alice", "pass", "user")
        update_user_role(db, "alice", "readonly")
        assert get_user(db, "alice").role == "readonly"

    def test_update_role_returns_false_for_unknown(self, tmp_path: Path) -> None:
        from jarvis.auth.core import update_user_role, _get_conn
        db = tmp_path / "jarvis.db"
        _get_conn(db).close()
        assert update_user_role(db, "nobody", "admin") is False


# ─────────────────────────────────────────────────────────────────────────────
# Unit: graph delete_entity / delete_relationship
# ─────────────────────────────────────────────────────────────────────────────

class TestGraphDelete:
    def _add_entity(self, db, name: str, user_id: str = "shared") -> None:
        from jarvis.memory.graph import handle_update_knowledge_graph
        handle_update_knowledge_graph(
            {"entities": [{"name": name, "type": "concept"}], "user_id": user_id}, db
        )

    def _add_rel(self, db, frm: str, rel: str, to: str, user_id: str = "shared") -> None:
        from jarvis.memory.graph import handle_update_knowledge_graph
        handle_update_knowledge_graph(
            {"relationships": [{"from": frm, "relation": rel, "to": to}], "user_id": user_id}, db
        )

    def test_delete_entity_returns_false_for_missing_db(self, tmp_path: Path) -> None:
        from jarvis.memory.graph import delete_entity
        assert delete_entity(tmp_path / "missing.db", "RLHF") is False

    def test_delete_entity_returns_true_when_found(self, tmp_path: Path) -> None:
        from jarvis.memory.graph import delete_entity
        db = tmp_path / "jarvis.db"
        self._add_entity(db, "RLHF")
        assert delete_entity(db, "RLHF") is True

    def test_delete_entity_returns_false_when_not_found(self, tmp_path: Path) -> None:
        from jarvis.memory.graph import _get_conn, delete_entity
        db = tmp_path / "jarvis.db"
        _get_conn(db).close()
        assert delete_entity(db, "NonExistent") is False

    def test_delete_entity_removes_relationships(self, tmp_path: Path) -> None:
        import sqlite3
        from jarvis.memory.graph import delete_entity
        db = tmp_path / "jarvis.db"
        self._add_entity(db, "RLHF")
        self._add_entity(db, "PPO")
        self._add_rel(db, "RLHF", "uses", "PPO")
        delete_entity(db, "RLHF")
        conn = sqlite3.connect(str(db))
        count = conn.execute(
            "SELECT COUNT(*) FROM relationships WHERE from_entity='RLHF' OR to_entity='RLHF'"
        ).fetchone()[0]
        conn.close()
        assert count == 0

    def test_delete_relationship_returns_true_when_found(self, tmp_path: Path) -> None:
        from jarvis.memory.graph import delete_relationship
        db = tmp_path / "jarvis.db"
        self._add_entity(db, "RLHF")
        self._add_entity(db, "PPO")
        self._add_rel(db, "RLHF", "uses", "PPO")
        assert delete_relationship(db, "RLHF", "uses", "PPO") is True

    def test_delete_relationship_returns_false_when_not_found(self, tmp_path: Path) -> None:
        from jarvis.memory.graph import _get_conn, delete_relationship
        db = tmp_path / "jarvis.db"
        _get_conn(db).close()
        assert delete_relationship(db, "A", "rel", "B") is False


# ─────────────────────────────────────────────────────────────────────────────
# Unit: feedback get_feedback_list
# ─────────────────────────────────────────────────────────────────────────────

class TestGetFeedbackList:
    def test_returns_empty_for_no_data(self, tmp_path: Path) -> None:
        from jarvis.memory.feedback import get_feedback_list, _get_conn
        db = tmp_path / "jarvis.db"
        _get_conn(db).close()
        assert get_feedback_list(db) == []

    def test_returns_feedback_row(self, tmp_path: Path) -> None:
        from jarvis.memory.feedback import get_feedback_list, log_feedback
        db = tmp_path / "jarvis.db"
        log_feedback(db, "s1", "response", 4, "good", user_id="alice")
        rows = get_feedback_list(db)
        assert len(rows) == 1
        assert rows[0]["rating"] == 4

    def test_session_id_filter(self, tmp_path: Path) -> None:
        from jarvis.memory.feedback import get_feedback_list, log_feedback
        db = tmp_path / "jarvis.db"
        log_feedback(db, "s1", "r1", 4, user_id="alice")
        log_feedback(db, "s2", "r2", 2, user_id="alice")
        rows = get_feedback_list(db, session_id="s1")
        assert len(rows) == 1
        assert rows[0]["session_id"] == "s1"

    def test_user_id_filter(self, tmp_path: Path) -> None:
        from jarvis.memory.feedback import get_feedback_list, log_feedback
        db = tmp_path / "jarvis.db"
        log_feedback(db, "s1", "r1", 4, user_id="alice")
        log_feedback(db, "s2", "r2", 2, user_id="bob")
        rows = get_feedback_list(db, user_id="alice")
        assert len(rows) == 1

    def test_limit_respected(self, tmp_path: Path) -> None:
        from jarvis.memory.feedback import get_feedback_list, log_feedback
        db = tmp_path / "jarvis.db"
        for i in range(5):
            log_feedback(db, f"s{i}", "r", 3, user_id="u")
        rows = get_feedback_list(db, limit=2)
        assert len(rows) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Unit: audit get_audit_stats
# ─────────────────────────────────────────────────────────────────────────────

class TestGetAuditStats:
    def test_returns_zeros_for_missing_db(self, tmp_path: Path) -> None:
        from jarvis.security.audit import get_audit_stats
        stats = get_audit_stats(tmp_path / "missing.db")
        assert stats["total_calls"] == 0

    def test_counts_total_calls(self, tmp_path: Path) -> None:
        from jarvis.security.audit import _get_conn, get_audit_stats, log_tool_call
        db = tmp_path / "jarvis.db"
        log_tool_call(db, "s1", "web_search", {}, "SAFE", approved=1)
        log_tool_call(db, "s1", "web_search", {}, "SAFE", approved=1)
        stats = get_audit_stats(db)
        assert stats["total_calls"] == 2

    def test_approved_count(self, tmp_path: Path) -> None:
        from jarvis.security.audit import get_audit_stats, log_tool_call
        db = tmp_path / "jarvis.db"
        log_tool_call(db, "s1", "browse", {}, "MEDIUM", approved=1)
        log_tool_call(db, "s1", "browse", {}, "MEDIUM", approved=0)
        stats = get_audit_stats(db)
        assert stats["approved"] == 1
        assert stats["denied"] == 1

    def test_top_tools_sorted(self, tmp_path: Path) -> None:
        from jarvis.security.audit import get_audit_stats, log_tool_call
        db = tmp_path / "jarvis.db"
        for _ in range(3):
            log_tool_call(db, "s1", "web_search", {}, "SAFE", approved=1)
        log_tool_call(db, "s1", "browse", {}, "MEDIUM", approved=1)
        stats = get_audit_stats(db)
        assert stats["top_tools"][0]["tool_name"] == "web_search"
        assert stats["top_tools"][0]["count"] == 3

    def test_risk_breakdown_populated(self, tmp_path: Path) -> None:
        from jarvis.security.audit import get_audit_stats, log_tool_call
        db = tmp_path / "jarvis.db"
        log_tool_call(db, "s1", "web_search", {}, "SAFE", approved=1)
        log_tool_call(db, "s1", "browse", {}, "MEDIUM", approved=1)
        stats = get_audit_stats(db)
        assert "SAFE" in stats["risk_breakdown"]
        assert "MEDIUM" in stats["risk_breakdown"]

    def test_since_ts_filters(self, tmp_path: Path) -> None:
        from jarvis.security.audit import _get_conn, get_audit_stats
        db = tmp_path / "jarvis.db"
        conn = _get_conn(db)
        old_ts = time.time() - 7200
        conn.execute(
            "INSERT INTO audit_log (timestamp, session_id, tool_name, tool_input, risk_level, approved) "
            "VALUES (?,?,?,?,?,?)", (old_ts, "s1", "web_search", "{}", "SAFE", 1)
        )
        conn.commit()
        conn.close()
        stats = get_audit_stats(db, since_ts=time.time() - 3600)
        assert stats["total_calls"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# Unit: training get_run_by_id / delete_run
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainingRunById:
    def test_returns_none_for_missing_db(self, tmp_path: Path) -> None:
        from jarvis.training.tracking import get_run_by_id
        assert get_run_by_id(tmp_path / "missing.db", 1) is None

    def test_returns_none_for_unknown_id(self, tmp_path: Path) -> None:
        from jarvis.training.tracking import _conn, get_run_by_id
        db = tmp_path / "jarvis.db"
        _conn(db).close()
        assert get_run_by_id(db, 9999) is None

    def test_returns_run_by_id(self, tmp_path: Path) -> None:
        from jarvis.training.tracking import complete_run, get_run_by_id, start_run
        db = tmp_path / "jarvis.db"
        run_id = start_run(db, "crawl")
        complete_run(db, run_id, docs_crawled=5)
        run = get_run_by_id(db, run_id)
        assert run is not None
        assert run.id == run_id
        assert run.docs_crawled == 5

    def test_delete_run_returns_true(self, tmp_path: Path) -> None:
        from jarvis.training.tracking import delete_run, start_run
        db = tmp_path / "jarvis.db"
        run_id = start_run(db, "crawl")
        assert delete_run(db, run_id) is True

    def test_delete_run_returns_false_for_unknown(self, tmp_path: Path) -> None:
        from jarvis.training.tracking import _conn, delete_run
        db = tmp_path / "jarvis.db"
        _conn(db).close()
        assert delete_run(db, 9999) is False

    def test_delete_run_removes_record(self, tmp_path: Path) -> None:
        from jarvis.training.tracking import delete_run, get_run_by_id, start_run
        db = tmp_path / "jarvis.db"
        run_id = start_run(db, "crawl")
        delete_run(db, run_id)
        assert get_run_by_id(db, run_id) is None


# ─────────────────────────────────────────────────────────────────────────────
# Unit: TOOL_RISK_MAP classifications
# ─────────────────────────────────────────────────────────────────────────────

class TestToolRiskMap:
    def test_calendar_is_low(self) -> None:
        from jarvis.security.approval import TOOL_RISK_MAP, RiskLevel
        assert TOOL_RISK_MAP["calendar"] == RiskLevel.LOW

    def test_execute_code_is_high(self) -> None:
        from jarvis.security.approval import TOOL_RISK_MAP, RiskLevel
        assert TOOL_RISK_MAP["execute_code"] == RiskLevel.HIGH

    def test_image_analysis_is_safe(self) -> None:
        from jarvis.security.approval import TOOL_RISK_MAP, RiskLevel
        assert TOOL_RISK_MAP["image_analysis"] == RiskLevel.SAFE

    def test_analyze_text_is_safe(self) -> None:
        from jarvis.security.approval import TOOL_RISK_MAP, RiskLevel
        assert TOOL_RISK_MAP["analyze_text"] == RiskLevel.SAFE

    def test_youtube_summary_is_safe(self) -> None:
        from jarvis.security.approval import TOOL_RISK_MAP, RiskLevel
        assert TOOL_RISK_MAP["youtube_summary"] == RiskLevel.SAFE

    def test_filesystem_search_is_safe(self) -> None:
        from jarvis.security.approval import TOOL_RISK_MAP, RiskLevel
        assert TOOL_RISK_MAP["filesystem_search"] == RiskLevel.SAFE

    def test_git_context_is_safe(self) -> None:
        from jarvis.security.approval import TOOL_RISK_MAP, RiskLevel
        assert TOOL_RISK_MAP["git_context"] == RiskLevel.SAFE

    def test_generate_tool_is_high(self) -> None:
        from jarvis.security.approval import TOOL_RISK_MAP, RiskLevel
        assert TOOL_RISK_MAP["generate_tool"] == RiskLevel.HIGH

    def test_local_model_is_low(self) -> None:
        from jarvis.security.approval import TOOL_RISK_MAP, RiskLevel
        assert TOOL_RISK_MAP["local_model"] == RiskLevel.LOW

    def test_notebooklm_is_medium(self) -> None:
        from jarvis.security.approval import TOOL_RISK_MAP, RiskLevel
        assert TOOL_RISK_MAP["notebooklm"] == RiskLevel.MEDIUM


# ─────────────────────────────────────────────────────────────────────────────
# API endpoint tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGetConfig:
    def test_returns_200(self, client) -> None:
        c, _ = client
        assert c.get("/api/config").status_code == 200

    def test_returns_feature_flags(self, client) -> None:
        c, _ = client
        data = c.get("/api/config").json()
        for key in ("auth_enabled", "proactive_enabled", "peer_enabled", "otel_enabled"):
            assert key in data

    def test_returns_model_info(self, client) -> None:
        c, _ = client
        data = c.get("/api/config").json()
        assert "model" in data
        assert "routing_strategy" in data

    def test_no_secrets_in_response(self, client) -> None:
        c, _ = client
        data = c.get("/api/config").json()
        for key in ("jwt_secret", "anthropic_api_key", "tavily_api_key"):
            assert key not in data


class TestEpisodeSearchEndpoint:
    def test_returns_200(self, client) -> None:
        c, _ = client
        assert c.get("/api/memory/episodes/search?q=RLHF").status_code == 200

    def test_returns_list(self, client) -> None:
        c, _ = client
        data = c.get("/api/memory/episodes/search?q=RLHF").json()
        assert isinstance(data, list)

    def test_finds_inserted_episode(self, client) -> None:
        c, reports_dir = client
        from jarvis.memory.episodic import log_episode
        log_episode(reports_dir / "jarvis.db", "s1", "user", "RLHF is great", "alice")
        data = c.get("/api/memory/episodes/search?q=RLHF").json()
        assert any("RLHF" in r["content"] for r in data)


class TestEpisodeDeleteEndpoint:
    def test_returns_400_without_filter(self, client) -> None:
        c, _ = client
        assert c.delete("/api/memory/episodes").status_code == 400

    def test_deletes_by_session_id(self, client) -> None:
        c, reports_dir = client
        from jarvis.memory.episodic import log_episode
        log_episode(reports_dir / "jarvis.db", "del-me", "user", "text", "alice")
        resp = c.delete("/api/memory/episodes?session_id=del-me")
        assert resp.status_code == 200
        assert resp.json()["deleted"] == 1

    def test_deletes_by_user_id(self, client) -> None:
        c, reports_dir = client
        from jarvis.memory.episodic import log_episode
        log_episode(reports_dir / "jarvis.db", "s1", "user", "text", "del-user")
        resp = c.delete("/api/memory/episodes?user_id=del-user")
        assert resp.json()["deleted"] == 1


class TestSessionSummariesEndpoint:
    def test_returns_200(self, client) -> None:
        c, _ = client
        assert c.get("/api/memory/summaries/alice").status_code == 200

    def test_returns_empty_for_no_summaries(self, client) -> None:
        c, _ = client
        assert c.get("/api/memory/summaries/nobody").json() == []

    def test_returns_saved_summary(self, client) -> None:
        c, reports_dir = client
        from jarvis.memory.preferences import save_session_summary
        save_session_summary(reports_dir / "jarvis.db", "s1", "alice", "Talked about AI", [])
        data = c.get("/api/memory/summaries/alice").json()
        assert len(data) == 1
        assert data[0]["summary"] == "Talked about AI"


class TestUserManagementEndpoints:
    def test_list_users_returns_200(self, client) -> None:
        c, _ = client
        assert c.get("/api/users").status_code == 200

    def test_list_users_returns_list(self, client) -> None:
        c, _ = client
        assert isinstance(c.get("/api/users").json(), list)

    def test_delete_user_404_for_unknown(self, client) -> None:
        c, _ = client
        assert c.delete("/api/users/nobody").status_code == 404

    def test_delete_user_204_when_found(self, client) -> None:
        c, reports_dir = client
        from jarvis.auth.core import create_user
        create_user(reports_dir / "jarvis.db", "test_user", "pass", "user")
        assert c.delete("/api/users/test_user").status_code == 204

    def test_patch_role_404_for_unknown(self, client) -> None:
        c, _ = client
        assert c.patch("/api/users/nobody/role", json={"role": "admin"}).status_code == 404

    def test_patch_role_422_for_invalid_role(self, client) -> None:
        c, reports_dir = client
        from jarvis.auth.core import create_user
        create_user(reports_dir / "jarvis.db", "user2", "pass", "user")
        assert c.patch("/api/users/user2/role", json={"role": "superuser"}).status_code == 422

    def test_patch_role_updates_user(self, client) -> None:
        c, reports_dir = client
        from jarvis.auth.core import create_user, get_user
        db = reports_dir / "jarvis.db"
        create_user(db, "user3", "pass", "user")
        resp = c.patch("/api/users/user3/role", json={"role": "readonly"})
        assert resp.status_code == 200
        assert resp.json()["role"] == "readonly"


class TestGetScheduleByIdEndpoint:
    def test_returns_200(self, client) -> None:
        c, _ = client
        mock_sched = MagicMock()
        mock_job = MagicMock()
        mock_job.id = "my-job"
        mock_job.func.__name__ = "_research_job"
        mock_job.kwargs = {"topic": "RLHF"}
        mock_job.next_run_time = None
        mock_job.trigger = MagicMock(__str__=lambda _: "cron")
        mock_sched.get_job.return_value = mock_job
        with patch("jarvis.scheduler.core.get_scheduler", return_value=mock_sched):
            resp = c.get("/api/schedules/my-job")
        assert resp.status_code == 200

    def test_returns_404_for_unknown_job(self, client) -> None:
        c, _ = client
        mock_sched = MagicMock()
        mock_sched.get_job.return_value = None
        with patch("jarvis.scheduler.core.get_scheduler", return_value=mock_sched):
            resp = c.get("/api/schedules/no-such")
        assert resp.status_code == 404

    def test_returns_503_when_scheduler_down(self, client) -> None:
        c, _ = client
        with patch("jarvis.scheduler.core.get_scheduler", return_value=None):
            resp = c.get("/api/schedules/any")
        assert resp.status_code == 503


class TestTrainingRunEndpoints:
    def _seed_run(self, db_path: Path) -> int:
        from jarvis.training.tracking import complete_run, start_run
        run_id = start_run(db_path, "crawl")
        complete_run(db_path, run_id, docs_crawled=3)
        return run_id

    def test_get_run_by_id_returns_200(self, client) -> None:
        c, reports_dir = client
        run_id = self._seed_run(reports_dir / "jarvis.db")
        assert c.get(f"/api/training/runs/{run_id}").status_code == 200

    def test_get_run_by_id_returns_run(self, client) -> None:
        c, reports_dir = client
        run_id = self._seed_run(reports_dir / "jarvis.db")
        data = c.get(f"/api/training/runs/{run_id}").json()
        assert data["id"] == run_id
        assert data["docs_crawled"] == 3

    def test_get_run_by_id_404_for_unknown(self, client) -> None:
        c, _ = client
        assert c.get("/api/training/runs/9999").status_code == 404

    def test_delete_run_returns_204(self, client) -> None:
        c, reports_dir = client
        run_id = self._seed_run(reports_dir / "jarvis.db")
        assert c.delete(f"/api/training/runs/{run_id}").status_code == 204

    def test_delete_run_404_for_unknown(self, client) -> None:
        c, _ = client
        assert c.delete("/api/training/runs/9999").status_code == 404


class TestKnowledgeGraphDeleteEndpoints:
    def _add_entity(self, db, name: str) -> None:
        from jarvis.memory.graph import handle_update_knowledge_graph
        handle_update_knowledge_graph({"entities": [{"name": name}]}, db)

    def test_delete_entity_204(self, client) -> None:
        c, reports_dir = client
        self._add_entity(reports_dir / "jarvis.db", "RLHF")
        assert c.delete("/api/knowledge-graph/entities/RLHF").status_code == 204

    def test_delete_entity_404_for_unknown(self, client) -> None:
        c, _ = client
        assert c.delete("/api/knowledge-graph/entities/NoSuchEntity").status_code == 404

    def test_delete_relationship_204(self, client) -> None:
        c, reports_dir = client
        from jarvis.memory.graph import handle_update_knowledge_graph
        db = reports_dir / "jarvis.db"
        handle_update_knowledge_graph(
            {"entities": [{"name": "RLHF"}, {"name": "PPO"}],
             "relationships": [{"from": "RLHF", "relation": "uses", "to": "PPO"}]}, db
        )
        resp = c.request("DELETE", "/api/knowledge-graph/relationships",
                         json={"from": "RLHF", "relation": "uses", "to": "PPO"})
        assert resp.status_code == 204

    def test_delete_relationship_404_for_unknown(self, client) -> None:
        c, _ = client
        resp = c.request("DELETE", "/api/knowledge-graph/relationships",
                         json={"from": "A", "relation": "r", "to": "B"})
        assert resp.status_code == 404

    def test_delete_relationship_422_without_fields(self, client) -> None:
        c, _ = client
        resp = c.request("DELETE", "/api/knowledge-graph/relationships",
                         json={"from": "A"})
        assert resp.status_code == 422


class TestFeedbackListEndpoint:
    def test_returns_200(self, client) -> None:
        c, _ = client
        assert c.get("/api/feedback").status_code == 200

    def test_returns_list(self, client) -> None:
        c, _ = client
        assert isinstance(c.get("/api/feedback").json(), list)

    def test_returns_inserted_feedback(self, client) -> None:
        c, reports_dir = client
        from jarvis.memory.feedback import log_feedback
        log_feedback(reports_dir / "jarvis.db", "s1", "resp", 5, user_id="alice")
        data = c.get("/api/feedback").json()
        assert len(data) >= 1
        assert data[0]["rating"] == 5

    def test_filter_by_session_id(self, client) -> None:
        c, reports_dir = client
        from jarvis.memory.feedback import log_feedback
        db = reports_dir / "jarvis.db"
        log_feedback(db, "s-alpha", "r", 4, user_id="u")
        log_feedback(db, "s-beta", "r", 2, user_id="u")
        data = c.get("/api/feedback?session_id=s-alpha").json()
        assert all(r["session_id"] == "s-alpha" for r in data)


class TestAuditStatsEndpoint:
    def test_returns_200(self, client) -> None:
        c, _ = client
        assert c.get("/api/audit/stats").status_code == 200

    def test_returns_total_calls(self, client) -> None:
        c, _ = client
        data = c.get("/api/audit/stats").json()
        assert "total_calls" in data

    def test_returns_top_tools_list(self, client) -> None:
        c, _ = client
        data = c.get("/api/audit/stats").json()
        assert isinstance(data.get("top_tools"), list)

    def test_returns_risk_breakdown_dict(self, client) -> None:
        c, _ = client
        data = c.get("/api/audit/stats").json()
        assert isinstance(data.get("risk_breakdown"), dict)
