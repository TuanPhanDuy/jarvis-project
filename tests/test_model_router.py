"""Unit tests for ModelRouter smart routing logic. No API keys needed."""
from __future__ import annotations

from jarvis.models.router import ModelRouter


class TestModelRouter:
    def test_always_primary_uses_primary_for_empty_messages(self) -> None:
        r = ModelRouter("big", "small", "always_primary")
        assert r.select([]) == "big"

    def test_always_primary_uses_primary_even_with_tool_results(self) -> None:
        r = ModelRouter("big", "small", "always_primary")
        msgs = [{"role": "tool", "content": "some result"}]
        assert r.select(msgs) == "big"

    def test_smart_empty_messages_uses_primary(self) -> None:
        r = ModelRouter("big", "small", "smart")
        assert r.select([]) == "big"

    def test_smart_user_only_uses_primary(self) -> None:
        r = ModelRouter("big", "small", "smart")
        msgs = [{"role": "user", "content": "What is RLHF?"}]
        assert r.select(msgs) == "big"

    def test_smart_tool_result_in_messages_uses_fast(self) -> None:
        r = ModelRouter("big", "small", "smart")
        msgs = [{"role": "tool", "content": "search results here"}]
        assert r.select(msgs) == "small"

    def test_smart_mixed_messages_with_tool_result_uses_fast(self) -> None:
        r = ModelRouter("big", "small", "smart")
        msgs = [
            {"role": "user", "content": "Research RLHF"},
            {"role": "assistant", "content": "searching..."},
            {"role": "tool", "content": "found: reward modeling"},
        ]
        assert r.select(msgs) == "small"

    def test_unknown_strategy_falls_back_to_primary(self) -> None:
        r = ModelRouter("big", "small", "unknown_strategy")
        msgs = [{"role": "tool", "content": "result"}]
        assert r.select(msgs) == "big"

    def test_system_messages_do_not_trigger_fast_model(self) -> None:
        r = ModelRouter("big", "small", "smart")
        msgs = [{"role": "system", "content": "you are an assistant"}]
        assert r.select(msgs) == "big"

    def test_primary_same_as_fast_always_returns_same(self) -> None:
        r = ModelRouter("same-model", "same-model", "smart")
        assert r.select([]) == "same-model"
        assert r.select([{"role": "tool", "content": "x"}]) == "same-model"

    def test_agent_model_map_overrides_primary(self) -> None:
        r = ModelRouter("big", "small", "always_primary", {"coder": "codellama:7b"})
        assert r.select([], agent_type="coder") == "codellama:7b"

    def test_agent_model_map_overrides_smart_routing(self) -> None:
        r = ModelRouter("big", "small", "smart", {"researcher": "qwen2.5:14b"})
        msgs = [{"role": "tool", "content": "results"}]
        # Normally smart would use "small"; override wins
        assert r.select(msgs, agent_type="researcher") == "qwen2.5:14b"

    def test_agent_type_not_in_map_falls_through_to_strategy(self) -> None:
        r = ModelRouter("big", "small", "smart", {"coder": "codellama:7b"})
        msgs = [{"role": "tool", "content": "results"}]
        # "researcher" not in map → smart routing → fast
        assert r.select(msgs, agent_type="researcher") == "small"

    def test_empty_agent_type_uses_strategy(self) -> None:
        r = ModelRouter("big", "small", "always_primary", {"coder": "codellama:7b"})
        assert r.select([], agent_type="") == "big"

    def test_no_agent_model_map_behaves_as_before(self) -> None:
        r = ModelRouter("big", "small", "smart")
        msgs = [{"role": "tool", "content": "result"}]
        assert r.select(msgs) == "small"


class TestHybridSearch:
    def test_bm25_score_matching_term(self) -> None:
        from jarvis.tools.memory import _bm25_score
        score = _bm25_score("RLHF reward", "RLHF uses reward modeling for language models")
        assert score > 0

    def test_bm25_score_no_match(self) -> None:
        from jarvis.tools.memory import _bm25_score
        score = _bm25_score("quantum physics", "RLHF uses reward modeling")
        assert score == 0.0

    def test_bm25_score_more_terms_higher(self) -> None:
        from jarvis.tools.memory import _bm25_score
        score_both = _bm25_score("RLHF reward", "RLHF uses reward modeling")
        score_one = _bm25_score("RLHF reward", "RLHF overview")
        assert score_both > score_one

    def test_weights_sum_to_one(self) -> None:
        from jarvis.tools.memory import _SEMANTIC_WEIGHT, _BM25_WEIGHT
        assert abs(_SEMANTIC_WEIGHT + _BM25_WEIGHT - 1.0) < 0.001
