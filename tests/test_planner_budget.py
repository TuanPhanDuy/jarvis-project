"""Tests for PlannerAgent system-prompt token-budget guard."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from jarvis.agents.planner import _MAX_EXTRAS_CHARS, _trim_extras, PlannerAgent


class TestTrimExtras:
    def test_empty_list_unchanged(self) -> None:
        assert _trim_extras([], 100) == []

    def test_under_budget_unchanged(self) -> None:
        extras = ["short", "also short"]
        result = _trim_extras(extras, 1000)
        assert result == ["short", "also short"]

    def test_exactly_at_budget_unchanged(self) -> None:
        extras = ["abc", "def"]  # 6 chars total
        result = _trim_extras(extras, 6)
        assert result == ["abc", "def"]

    def test_over_budget_drops_last_item(self) -> None:
        extras = ["keep_me", "x" * 500]
        result = _trim_extras(extras, 100)
        assert result == ["keep_me"]

    def test_drops_multiple_items_until_fits(self) -> None:
        extras = ["a" * 50, "b" * 50, "c" * 50]  # 150 total
        result = _trim_extras(extras, 60)
        assert result == ["a" * 50]  # only first fits

    def test_all_dropped_if_first_alone_exceeds_budget(self) -> None:
        extras = ["x" * 200, "y" * 200]
        result = _trim_extras(extras, 100)
        assert result == []

    def test_priority_order_preserved(self) -> None:
        # personality (high) first, entities (low) last — entities should be dropped
        personality = "personality context"
        prefs = "user preferences"
        entities = "e" * 4000  # very long, pushes over budget
        extras = [personality, prefs, entities]
        result = _trim_extras(extras, 50)
        assert personality in result
        assert entities not in result


class TestPlannerSystemPromptBudget:
    def _make_agent(self, user_id="test-user"):
        with patch("jarvis.prompts.loader.load_prompt", return_value="base prompt"):
            agent = PlannerAgent(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                tool_schemas=[],
                tool_registry={},
                user_id=user_id,
            )
        return agent

    def test_anonymous_user_returns_base_prompt(self) -> None:
        agent = self._make_agent(user_id="anonymous")
        with patch("jarvis.agents.planner.load_prompt", return_value="base"):
            prompt = agent.get_system_prompt()
        assert prompt == "base"

    def test_extras_capped_under_max(self) -> None:
        agent = self._make_agent()

        long_personality = "P" * 2000
        long_prefs = "Q" * 2000
        long_summaries = "S" * 2000
        long_entities = "E" * 2000

        with (
            patch("jarvis.agents.planner.load_prompt", return_value="base"),
            patch("jarvis.config.get_settings") as mock_settings,
            patch("jarvis.memory.preferences.get_preferences", return_value={}),
            patch("jarvis.memory.preferences.get_preference_context", return_value=long_prefs),
            patch("jarvis.memory.personality.get_personality_context", return_value=long_personality),
            patch("jarvis.memory.preferences.get_recent_session_summaries", return_value=[long_summaries]),
            patch("jarvis.memory.graph.get_recent_entities", return_value=["e1", "e2"]),
        ):
            mock_settings.return_value.reports_dir = MagicMock()
            mock_settings.return_value.reports_dir.__truediv__ = lambda s, x: MagicMock()
            prompt = agent.get_system_prompt()

        extras_portion = prompt[len("base"):]
        assert len(extras_portion) <= _MAX_EXTRAS_CHARS + 10  # +10 for separator "\n\n"

    def test_extras_within_budget_all_included(self) -> None:
        agent = self._make_agent()

        with (
            patch("jarvis.agents.planner.load_prompt", return_value="base"),
            patch("jarvis.config.get_settings") as mock_settings,
            patch("jarvis.memory.preferences.get_preferences", return_value={}),
            patch("jarvis.memory.preferences.get_preference_context", return_value="short prefs"),
            patch("jarvis.memory.personality.get_personality_context", return_value="short personality"),
            patch("jarvis.memory.preferences.get_recent_session_summaries", return_value=["summary1"]),
            patch("jarvis.memory.graph.get_recent_entities", return_value=["GPT-4", "RLHF"]),
        ):
            mock_settings.return_value.reports_dir = MagicMock()
            mock_settings.return_value.reports_dir.__truediv__ = lambda s, x: MagicMock()
            prompt = agent.get_system_prompt()

        assert "short personality" in prompt
        assert "short prefs" in prompt
        assert "summary1" in prompt
        assert "GPT-4" in prompt
