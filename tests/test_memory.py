"""Unit tests for episodic memory, knowledge graph, and feedback. No API keys needed."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from jarvis.memory.episodic import log_episode, handle_search_episodic_memory, prune_old_episodes
from jarvis.memory.graph import handle_update_knowledge_graph, handle_query_knowledge_graph, QUERY_SCHEMA
from jarvis.memory.feedback import log_feedback, get_feedback_stats, handle_record_feedback, prune_old_feedback
from jarvis.memory.failures import prune_old_failures
from jarvis.memory.preferences import (
    upsert_preference, get_preferences, prune_old_preferences,
    get_preference_context, _get_conn as pref_conn,
)


# ── Episodic memory ───────────────────────────────────────────────────────────

class TestEpisodicMemory:
    def test_log_and_search(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        log_episode(db, "sess1", "user", "What is RLHF?")
        log_episode(db, "sess1", "assistant", "RLHF stands for Reinforcement Learning from Human Feedback.")
        result = handle_search_episodic_memory({"query": "RLHF"}, db)
        assert "RLHF" in result
        assert "ERROR" not in result

    def test_search_returns_no_match(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        log_episode(db, "sess1", "user", "Tell me about transformers.")
        result = handle_search_episodic_memory({"query": "quantum physics"}, db)
        assert "No episodes found" in result

    def test_user_id_isolation(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        log_episode(db, "sess1", "user", "Alice secret note", user_id="alice")
        log_episode(db, "sess2", "user", "Bob public note", user_id="bob")
        result = handle_search_episodic_memory({"query": "note"}, db, user_id="alice")
        assert "Alice" in result
        assert "Bob" not in result

    def test_search_empty_db(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        result = handle_search_episodic_memory({"query": "anything"}, db)
        assert "No episodes found" in result

    def test_log_never_raises(self, tmp_path: Path) -> None:
        db = tmp_path / "no_dir" / "jarvis.db"
        log_episode(db, "s", "user", "x")  # must not raise even if path creates dirs

    def test_limit_respected(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        for i in range(20):
            log_episode(db, "sess", "user", f"message about transformers {i}")
        result = handle_search_episodic_memory({"query": "transformers", "limit": 3}, db)
        # Should return at most 3 entries — count occurrence of role prefix
        assert result.count("[user]") <= 3

    def test_like_fallback_when_fts5_missing(self, tmp_path: Path) -> None:
        """Verify LIKE fallback is used when FTS5 virtual table is unavailable."""
        import sqlite3
        from jarvis.memory.episodic import _search

        db = tmp_path / "jarvis.db"
        log_episode(db, "sess1", "user", "discussion about neural networks")

        # Drop the FTS5 virtual table to simulate environment without FTS5
        conn = sqlite3.connect(str(db))
        try:
            conn.execute("DROP TABLE IF EXISTS episodes_fts")
            conn.execute("DROP TRIGGER IF EXISTS episodes_ai")
        except Exception:
            pass
        conn.commit()
        conn.close()

        rows = _search(db, "neural", limit=5)
        assert len(rows) >= 1
        assert any("neural" in row["content"] for row in rows)


# ── Knowledge graph ───────────────────────────────────────────────────────────

class TestKnowledgeGraph:
    def test_add_and_query_entity(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        result = handle_update_knowledge_graph(
            {"entities": [{"name": "RLHF", "type": "technique", "description": "RL from human feedback"}]},
            db,
        )
        assert "ERROR" not in result
        query = handle_query_knowledge_graph({"entity": "RLHF"}, db)
        assert "RLHF" in query
        assert "technique" in query

    def test_add_relationship(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        handle_update_knowledge_graph(
            {"relationships": [{"from": "RLHF", "relation": "uses", "to": "PPO"}]},
            db,
        )
        result = handle_query_knowledge_graph({"entity": "RLHF"}, db)
        assert "PPO" in result
        assert "uses" in result

    def test_upsert_entity(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        handle_update_knowledge_graph(
            {"entities": [{"name": "GPT", "type": "model", "description": "old desc"}]}, db
        )
        handle_update_knowledge_graph(
            {"entities": [{"name": "GPT", "type": "model", "description": "new desc"}]}, db
        )
        result = handle_query_knowledge_graph({"entity": "GPT"}, db)
        assert "new desc" in result

    def test_user_id_namespace(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        handle_update_knowledge_graph(
            {"entities": [{"name": "MyTech", "type": "concept"}], "user_id": "alice"}, db
        )
        handle_update_knowledge_graph(
            {"entities": [{"name": "MyTech", "type": "concept"}], "user_id": "bob"}, db
        )
        # Both should succeed (different namespaces, no unique conflict)
        result = handle_query_knowledge_graph({"entity": "MyTech"}, db)
        assert "MyTech" in result

    def test_empty_input_returns_error(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        result = handle_update_knowledge_graph({}, db)
        assert result.startswith("ERROR")

    def test_query_unknown_entity(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        result = handle_query_knowledge_graph({"entity": "NonExistent"}, db)
        assert "No knowledge found" in result


# ── Feedback ──────────────────────────────────────────────────────────────────

class TestFeedback:
    def test_log_and_stats(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        log_feedback(db, "sess1", "great response", rating=5, comment="perfect")
        log_feedback(db, "sess1", "bad response", rating=1, comment="wrong")
        stats = get_feedback_stats(db, session_id="sess1")
        assert stats["total"] == 2
        assert stats["avg_rating"] == 3.0

    def test_global_stats(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        log_feedback(db, "s1", "r1", rating=4)
        log_feedback(db, "s2", "r2", rating=2)
        stats = get_feedback_stats(db)
        assert stats["total"] == 2

    def test_handle_record_feedback_tool(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        result = handle_record_feedback(
            {"session_id": "sess1", "response_snippet": "some text", "rating": 5, "comment": "nice"},
            db,
        )
        assert "recorded" in result.lower()
        assert "ERROR" not in result

    def test_invalid_rating_returns_error(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        result = handle_record_feedback({"session_id": "s", "rating": 99}, db)
        assert result.startswith("ERROR")

    def test_thumbs_up_down(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        log_feedback(db, "s", "r", rating=1)   # thumbs up
        log_feedback(db, "s", "r", rating=-1)  # thumbs down
        stats = get_feedback_stats(db, "s")
        assert stats["total"] == 2

    def test_empty_db_stats(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        stats = get_feedback_stats(db)
        assert stats["total"] == 0
        assert stats["avg_rating"] == 0.0


# ── Memory pruning ────────────────────────────────────────────────────────────

class TestMemoryPruning:
    def test_prune_old_episodes(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        log_episode(db, "s", "user", "old message")
        deleted = prune_old_episodes(db, retention_days=0)
        assert deleted >= 1

    def test_prune_keeps_recent_episodes(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        log_episode(db, "s", "user", "recent message")
        deleted = prune_old_episodes(db, retention_days=90)
        assert deleted == 0

    def test_prune_old_feedback(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        log_feedback(db, "s", "r", rating=5)
        deleted = prune_old_feedback(db, retention_days=0)
        assert deleted >= 1

    def test_prune_old_failures(self, tmp_path: Path) -> None:
        from jarvis.memory.failures import log_failure
        db = tmp_path / "jarvis.db"
        log_failure(db, "some_tool", {}, "ERROR: test")
        deleted = prune_old_failures(db, retention_days=0)
        assert deleted >= 1

    def test_prune_old_preferences(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        upsert_preference(db, "alice", "communication_style", "verbosity", "concise")
        deleted = prune_old_preferences(db, retention_days=0)
        assert deleted >= 1
        prefs = get_preferences(db, "alice")
        assert prefs == {}

    def test_prune_keeps_recent_preferences(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        upsert_preference(db, "bob", "technical_depth", "level", "expert")
        deleted = prune_old_preferences(db, retention_days=90)
        assert deleted == 0
        prefs = get_preferences(db, "bob")
        assert "technical_depth" in prefs


# ── Knowledge graph: multi-hop traversal ─────────────────────────────────────

class TestKnowledgeGraphMultiHop:
    def _build(self, db, rels):
        handle_update_knowledge_graph({"relationships": rels}, db)

    def test_multi_hop_depth2_follows_transitive_edges(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        self._build(db, [
            {"from": "RLHF", "relation": "uses", "to": "PPO"},
            {"from": "PPO", "relation": "is_variant_of", "to": "Policy Gradient"},
        ])
        result = handle_query_knowledge_graph({"entity": "RLHF", "depth": 2}, db)
        assert "Policy Gradient" in result
        assert "depth=2" in result

    def test_multi_hop_circular_graph_no_infinite_loop(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        self._build(db, [
            {"from": "A", "relation": "links", "to": "B"},
            {"from": "B", "relation": "links", "to": "A"},
        ])
        result = handle_query_knowledge_graph({"entity": "A", "depth": 3}, db)
        assert "A" in result
        assert "B" in result
        assert "ERROR" not in result

    def test_relation_filter_limits_edge_types(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        self._build(db, [
            {"from": "RLHF", "relation": "uses", "to": "PPO"},
            {"from": "RLHF", "relation": "developed_by", "to": "OpenAI"},
        ])
        result = handle_query_knowledge_graph(
            {"entity": "RLHF", "depth": 1, "relation_filter": "uses"}, db
        )
        assert "PPO" in result
        assert "OpenAI" not in result

    def test_depth_and_filter_in_query_schema(self) -> None:
        props = QUERY_SCHEMA["input_schema"]["properties"]
        assert "depth" in props
        assert "relation_filter" in props
        assert props["depth"]["maximum"] == 3


# ── Preference temporal decay ─────────────────────────────────────────────────

class TestPreferenceDecay:
    def test_fresh_preference_appears_in_context(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        upsert_preference(db, "u", "tool_prefs", "language", "Python", confidence=0.5)
        ctx = get_preference_context(db, "u", decay=True)
        assert "Python" in ctx

    def test_ancient_preference_filtered_by_decay(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        conn = pref_conn(db)
        conn.execute(
            "INSERT INTO user_preferences VALUES (?,?,?,?,?,?,?)",
            ("u", "tool_prefs", "language", "COBOL", 0.5, "inferred", time.time() - 500 * 86400),
        )
        conn.commit()
        conn.close()
        ctx = get_preference_context(db, "u", decay=True)
        assert "COBOL" not in ctx

    def test_decay_false_shows_stale_preference(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        conn = pref_conn(db)
        conn.execute(
            "INSERT INTO user_preferences VALUES (?,?,?,?,?,?,?)",
            ("u", "tool_prefs", "language", "COBOL", 0.5, "inferred", time.time() - 500 * 86400),
        )
        conn.commit()
        conn.close()
        ctx = get_preference_context(db, "u", decay=False)
        assert "COBOL" in ctx

    def test_recent_pref_outranks_old_high_confidence(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        conn = pref_conn(db)
        # Old preference (100 days): eff = 0.9 * exp(-0.01*100) ≈ 0.33 → stays in context
        conn.execute(
            "INSERT INTO user_preferences VALUES (?,?,?,?,?,?,?)",
            ("u", "domain_interest", "area", "Databases", 0.9, "explicit", time.time() - 100 * 86400),
        )
        # Fresh preference: eff = 0.5 * 1.0 = 0.5 → outranks old one
        conn.execute(
            "INSERT INTO user_preferences VALUES (?,?,?,?,?,?,?)",
            ("u", "domain_interest", "topic", "LLMs", 0.5, "inferred", time.time()),
        )
        conn.commit()
        conn.close()
        ctx = get_preference_context(db, "u", decay=True)
        assert "LLMs" in ctx
        assert "Databases" in ctx
        # Fresh LLMs should appear before stale Databases in the output
        assert ctx.index("LLMs") < ctx.index("Databases")


# ── Consolidator: session clustering ─────────────────────────────────────────

class TestConsolidatorClustering:
    def test_consolidation_calls_llm_per_session(self, tmp_path: Path) -> None:
        from unittest.mock import MagicMock, patch
        from jarvis.memory.episodic import log_episode
        from jarvis.memory.consolidator import consolidate_user_memory

        db = tmp_path / "jarvis.db"
        log_episode(db, "sess-A", "user", "I prefer Python", user_id="alice")
        log_episode(db, "sess-A", "assistant", "Got it", user_id="alice")
        log_episode(db, "sess-B", "user", "I like concise answers", user_id="alice")

        mock_resp = MagicMock()
        mock_resp.message.content = ""

        with patch("ollama.chat", return_value=mock_resp) as mock_chat:
            consolidate_user_memory(db, "alice", "test-model", lookback_hours=9999)

        # 2 sessions → at least 2 extraction calls (plus up to 2 summary calls)
        assert mock_chat.call_count >= 2

    def test_consolidation_merges_duplicates_by_highest_confidence(self, tmp_path: Path) -> None:
        from unittest.mock import MagicMock, patch
        from jarvis.memory.episodic import log_episode
        from jarvis.memory.consolidator import consolidate_user_memory
        from jarvis.memory.preferences import get_preferences

        db = tmp_path / "jarvis.db"
        log_episode(db, "sess-A", "user", "I use Python", user_id="u")
        log_episode(db, "sess-B", "user", "I use Python too", user_id="u")

        responses = [
            "PREFERENCE|tool_prefs|language|Python|0.7|explicit",  # sess-A extraction
            "",                                                      # sess-A summary
            "PREFERENCE|tool_prefs|language|Python|0.4|inferred",  # sess-B extraction
            "",                                                      # sess-B summary
        ]
        idx = 0

        def side_effect(**kwargs):
            nonlocal idx
            resp = MagicMock()
            resp.message.content = responses[idx] if idx < len(responses) else ""
            idx += 1
            return resp

        with patch("ollama.chat", side_effect=side_effect):
            count = consolidate_user_memory(db, "u", "test-model", lookback_hours=9999)

        assert count == 1
        prefs = get_preferences(db, "u")
        assert prefs["tool_prefs"]["language"] == "Python"

    def test_consolidation_logs_conflict_on_disagreement(self, tmp_path: Path) -> None:
        from unittest.mock import MagicMock, patch
        from jarvis.memory.episodic import log_episode
        from jarvis.memory.consolidator import consolidate_user_memory

        db = tmp_path / "jarvis.db"
        log_episode(db, "sess-A", "user", "I use Python", user_id="u")
        log_episode(db, "sess-B", "user", "I use Rust now", user_id="u")

        responses = [
            "PREFERENCE|tool_prefs|language|Python|0.9|explicit",
            "",
            "PREFERENCE|tool_prefs|language|Rust|0.8|explicit",
            "",
        ]
        idx = 0

        def side_effect(**kwargs):
            nonlocal idx
            resp = MagicMock()
            resp.message.content = responses[idx] if idx < len(responses) else ""
            idx += 1
            return resp

        with patch("ollama.chat", side_effect=side_effect):
            with patch("jarvis.memory.consolidator.log") as mock_log:
                consolidate_user_memory(db, "u", "test-model", lookback_hours=9999)
                warning_calls = mock_log.warning.call_args_list
                assert any("preference_conflict" in str(c) for c in warning_calls)
