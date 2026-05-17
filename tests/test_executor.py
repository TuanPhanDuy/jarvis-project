"""Tests for ExecutorAgent — plan execution, topo sort, critic retry, failure handling."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from jarvis.agents.critic import CritiqueResult
from jarvis.agents.executor import ExecutorAgent, PlanStep, _topo_levels


def _make_executor(tmp_path):
    return ExecutorAgent(
        model="llama3.2",
        max_tokens=512,
        sub_tool_schemas=[],
        sub_tool_registry={},
        db_path=tmp_path / "plans.db",
    )


def _good_critique():
    return CritiqueResult(score=8, issues=[], should_retry=False)


def _bad_critique():
    return CritiqueResult(score=3, issues=["too short"], should_retry=True, revised_task="try harder")


class TestTopoLevels:
    def test_dependent_step_in_later_level(self):
        a = PlanStep(id="A", description="first", agent_type="researcher")
        b = PlanStep(id="B", description="second", agent_type="coder", depends_on=["A"])
        levels = _topo_levels([b, a])
        # A must appear in an earlier level than B
        level_of = {s.id: i for i, level in enumerate(levels) for s in level}
        assert level_of["A"] < level_of["B"]

    def test_independent_steps_in_same_level(self):
        steps = [
            PlanStep(id="X", description="x", agent_type="coder"),
            PlanStep(id="Y", description="y", agent_type="coder"),
        ]
        levels = _topo_levels(steps)
        # No deps → both should land in the first level
        assert len(levels) == 1
        ids = {s.id for s in levels[0]}
        assert ids == {"X", "Y"}


class TestExecutorAgent:
    def test_execute_plan_injects_context(self, tmp_path, mock_ollama):
        """Step B receives step A's result as injected context."""
        captured_tasks: list[tuple[str, str]] = []

        def fake_run_step(self_inner, agent_type, task):
            captured_tasks.append((agent_type, task))
            return "result_of_A"

        with patch.object(ExecutorAgent, "_run_step", fake_run_step):
            with patch("jarvis.agents.critic.CriticAgent.critique", return_value=_good_critique()):
                executor = _make_executor(tmp_path)
                steps = [
                    PlanStep(id="A", description="research it", agent_type="researcher"),
                    PlanStep(id="B", description="code it", agent_type="coder", depends_on=["A"]),
                ]
                executor.execute_plan("goal", steps)

        assert len(captured_tasks) == 2
        _, task_b = captured_tasks[1]
        assert "result_of_A" in task_b

    def test_critic_retry_on_low_score(self, tmp_path, mock_ollama):
        """A step with score < 5 triggers one retry (_run_step called twice for that step)."""
        call_count = {"n": 0}

        def fake_run_step(self_inner, agent_type, task):
            call_count["n"] += 1
            return "short result"

        critiques = [_bad_critique(), _good_critique()]
        critique_iter = iter(critiques)

        with patch.object(ExecutorAgent, "_run_step", fake_run_step):
            with patch("jarvis.agents.critic.CriticAgent.critique", side_effect=critique_iter):
                executor = _make_executor(tmp_path)
                steps = [PlanStep(id="S", description="do something", agent_type="coder")]
                executor.execute_plan("goal", steps)

        assert call_count["n"] == 2

    def test_failed_step_marks_plan_partial(self, tmp_path, mock_ollama):
        """A step returning ERROR:... sets plan status to partial_failure."""

        def fake_run_step(self_inner, agent_type, task):
            return "ERROR: something broke"

        with patch.object(ExecutorAgent, "_run_step", fake_run_step):
            with patch("jarvis.agents.critic.CriticAgent.critique", return_value=_good_critique()):
                executor = _make_executor(tmp_path)
                steps = [PlanStep(id="F", description="failing step", agent_type="coder")]
                result = executor.execute_plan("goal", steps)

        assert "ERROR" in result or "failing step" in result

    def test_unknown_agent_type_returns_error(self, tmp_path, mock_ollama):
        """_run_step with an unrecognised agent_type returns an ERROR string."""
        executor = _make_executor(tmp_path)
        result = executor._run_step("wizard", "do magic")
        assert result.startswith("ERROR")
        assert "wizard" in result


class TestGoalVerification:
    def _chat_returning(self, content: str):
        resp = MagicMock()
        resp.message.content = content
        return MagicMock(return_value=resp)

    def test_achieved_verdict_returns_empty(self, tmp_path):
        executor = _make_executor(tmp_path)
        steps = [PlanStep(id="A", description="research", agent_type="researcher",
                          status="done", result="detailed research result")]
        with patch("ollama.chat", self._chat_returning("ACHIEVED")), \
             patch("jarvis.config.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(goal_verification_enabled=True)
            verdict = executor._verify_goal("Research RLHF", steps)
        assert verdict == ""

    def test_gaps_verdict_returned_as_string(self, tmp_path):
        executor = _make_executor(tmp_path)
        steps = [PlanStep(id="A", description="research", agent_type="researcher",
                          status="done", result="brief result")]
        with patch("ollama.chat", self._chat_returning("GAPS: Missing PPO details")), \
             patch("jarvis.config.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(goal_verification_enabled=True)
            verdict = executor._verify_goal("Research RLHF deeply", steps)
        assert "GAPS" in verdict

    def test_verification_disabled_returns_empty(self, tmp_path):
        executor = _make_executor(tmp_path)
        steps = [PlanStep(id="A", description="x", agent_type="researcher", status="done", result="r")]
        mock_chat = MagicMock()
        with patch("ollama.chat", mock_chat), \
             patch("jarvis.config.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(goal_verification_enabled=False)
            verdict = executor._verify_goal("goal", steps)
        assert verdict == ""
        mock_chat.assert_not_called()

    def test_ollama_failure_returns_empty(self, tmp_path):
        executor = _make_executor(tmp_path)
        steps = [PlanStep(id="A", description="x", agent_type="researcher", status="done", result="r")]
        with patch("ollama.chat", side_effect=RuntimeError("down")), \
             patch("jarvis.config.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(goal_verification_enabled=True)
            verdict = executor._verify_goal("goal", steps)
        assert verdict == ""
