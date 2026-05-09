"""Tests for ExecutorAgent — plan execution, topo sort, critic retry, failure handling."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from jarvis.agents.critic import CritiqueResult
from jarvis.agents.executor import ExecutorAgent, PlanStep, _topo_sort


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


class TestTopoSort:
    def test_respects_dependency_order(self):
        a = PlanStep(id="A", description="first", agent_type="researcher")
        b = PlanStep(id="B", description="second", agent_type="coder", depends_on=["A"])
        order = _topo_sort([b, a])
        ids = [s.id for s in order]
        assert ids.index("A") < ids.index("B")

    def test_no_deps_preserves_insertion_order(self):
        steps = [
            PlanStep(id="X", description="x", agent_type="coder"),
            PlanStep(id="Y", description="y", agent_type="coder"),
        ]
        assert [s.id for s in _topo_sort(steps)] == ["X", "Y"]


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
