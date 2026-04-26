"""Unit tests for episodic memory, knowledge graph, and feedback. No API keys needed."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from jarvis.memory.episodic import log_episode, handle_search_episodic_memory
from jarvis.memory.graph import handle_update_knowledge_graph, handle_query_knowledge_graph
from jarvis.memory.feedback import log_feedback, get_feedback_stats, handle_record_feedback


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
