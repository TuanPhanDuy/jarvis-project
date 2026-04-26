"""Unit tests for ModelRouter smart routing logic. No API keys needed."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from jarvis.models.router import ModelRouter, _MessagesAPI


def _make_router(strategy="smart", primary_model="claude-sonnet-4-6", fast_model="claude-haiku-4-5-20251001"):
    primary = MagicMock()
    return ModelRouter(primary=primary, primary_model=primary_model, fast_model=fast_model, strategy=strategy)


class TestModelRouterInit:
    def test_attributes_set(self) -> None:
        router = _make_router()
        assert router.primary_model == "claude-sonnet-4-6"
        assert router.fast_model == "claude-haiku-4-5-20251001"
        assert router.strategy == "smart"

    def test_fast_model_defaults_to_primary(self) -> None:
        primary = MagicMock()
        router = ModelRouter(primary=primary, primary_model="claude-sonnet-4-6")
        assert router.fast_model == "claude-sonnet-4-6"

    def test_messages_api_attached(self) -> None:
        router = _make_router()
        assert isinstance(router.messages, _MessagesAPI)


class TestSmartRouting:
    def test_plain_string_user_message_uses_primary(self) -> None:
        router = _make_router(strategy="smart")
        api = router.messages
        messages = [{"role": "user", "content": "What is RLHF?"}]
        model = api._select_model({"messages": messages})
        assert model == "claude-sonnet-4-6"

    def test_tool_result_content_uses_fast(self) -> None:
        router = _make_router(strategy="smart")
        api = router.messages
        messages = [
            {"role": "user", "content": "Research RLHF"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "1", "name": "web_search", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1", "content": "results"}]},
        ]
        model = api._select_model({"messages": messages})
        assert model == "claude-haiku-4-5-20251001"

    def test_empty_messages_uses_primary(self) -> None:
        router = _make_router(strategy="smart")
        model = router.messages._select_model({"messages": []})
        assert model == "claude-sonnet-4-6"

    def test_always_primary_strategy_ignores_content_type(self) -> None:
        router = _make_router(strategy="always_primary")
        # _select_model is only called when strategy == "smart"
        # When always_primary, create() does NOT call _select_model
        primary = router.primary
        router.messages.create(model="claude-sonnet-4-6", messages=[
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "y"}]}
        ], max_tokens=100)
        # model param passed through unchanged
        call_kwargs = primary.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-6"

    def test_smart_strategy_overrides_model_param(self) -> None:
        router = _make_router(strategy="smart")
        primary = router.primary
        tool_result_messages = [
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1", "content": "data"}]}
        ]
        router.messages.create(model="claude-sonnet-4-6", messages=tool_result_messages, max_tokens=100)
        call_kwargs = primary.messages.create.call_args[1]
        # Should have been overridden to fast model
        assert call_kwargs["model"] == "claude-haiku-4-5-20251001"

    def test_stream_always_uses_primary_client(self) -> None:
        router = _make_router(strategy="smart")
        primary = router.primary
        router.messages.stream(model="x", messages=[], max_tokens=10)
        primary.messages.stream.assert_called_once()


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
