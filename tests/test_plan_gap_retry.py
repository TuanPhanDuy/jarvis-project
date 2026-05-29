"""Tests for plan gap auto-retry (ExecutorAgent._fill_gaps)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from jarvis.agents.executor import ExecutorAgent, PlanStep


def _make_executor(tmp_path):
    return ExecutorAgent(
        model="llama3.2",
        max_tokens=512,
        sub_tool_schemas=[],
        sub_tool_registry={},
        db_path=tmp_path / "plans.db",
    )


class TestFillGaps:
    def test_ollama_failure_returns_empty(self, tmp_path):
        executor = _make_executor(tmp_path)
        steps = [PlanStep(id="A", description="research", agent_type="researcher",
                          status="done", result="some result")]
        with patch("ollama.chat", side_effect=RuntimeError("down")):
            result = executor._fill_gaps("goal", "GAPS: missing details", steps)
        assert result == ""

    def test_empty_task_in_response_returns_empty(self, tmp_path):
        executor = _make_executor(tmp_path)
        steps = [PlanStep(id="A", description="research", agent_type="researcher",
                          status="done", result="result")]
        mock_resp = MagicMock()
        mock_resp.message.content = "AGENT: researcher\nTASK:"  # empty task
        with patch("ollama.chat", return_value=mock_resp):
            result = executor._fill_gaps("goal", "GAPS: missing X", steps)
        assert result == ""

    def test_valid_response_triggers_run_step(self, tmp_path):
        executor = _make_executor(tmp_path)
        steps = [PlanStep(id="A", description="research", agent_type="researcher",
                          status="done", result="partial result")]

        mock_resp = MagicMock()
        mock_resp.message.content = "AGENT: coder\nTASK: Implement the missing feature X."

        with patch("ollama.chat", return_value=mock_resp), \
             patch.object(ExecutorAgent, "_run_step", return_value="gap filled result") as mock_run:
            result = executor._fill_gaps("Build feature X", "GAPS: missing implementation", steps)

        mock_run.assert_called_once()
        call_agent_type, call_task = mock_run.call_args[0]
        assert call_agent_type == "coder"
        assert "Implement the missing feature X" in call_task

    def test_result_includes_agent_type_prefix(self, tmp_path):
        executor = _make_executor(tmp_path)
        steps = [PlanStep(id="A", description="research", agent_type="researcher",
                          status="done", result="result")]

        mock_resp = MagicMock()
        mock_resp.message.content = "AGENT: analyst\nTASK: Analyze the data gap."

        with patch("ollama.chat", return_value=mock_resp), \
             patch.object(ExecutorAgent, "_run_step", return_value="analysis complete"):
            result = executor._fill_gaps("goal", "GAPS: no analysis", steps)

        assert "[ANALYST]" in result

    def test_invalid_agent_type_defaults_to_researcher(self, tmp_path):
        executor = _make_executor(tmp_path)
        steps = [PlanStep(id="A", description="step", agent_type="researcher",
                          status="done", result="r")]

        mock_resp = MagicMock()
        mock_resp.message.content = "AGENT: wizard\nTASK: Do something magical."

        with patch("ollama.chat", return_value=mock_resp), \
             patch.object(ExecutorAgent, "_run_step", return_value="ok") as mock_run:
            executor._fill_gaps("goal", "GAPS: magic missing", steps)

        agent_type_used = mock_run.call_args[0][0]
        assert agent_type_used == "researcher"

    def test_gap_context_injected_into_task(self, tmp_path):
        executor = _make_executor(tmp_path)
        steps = [PlanStep(id="A", description="research", agent_type="researcher",
                          status="done", result="partial")]
        gap_text = "GAPS: PPO details missing"

        mock_resp = MagicMock()
        mock_resp.message.content = "AGENT: researcher\nTASK: Research PPO in detail."

        with patch("ollama.chat", return_value=mock_resp), \
             patch.object(ExecutorAgent, "_run_step", return_value="done") as mock_run:
            executor._fill_gaps("goal", gap_text, steps)

        _, task_passed = mock_run.call_args[0]
        assert "GAPS: PPO details missing" in task_passed


class TestExecutePlanWithGaps:
    def test_gap_fill_appended_when_gaps_found(self, tmp_path, mock_ollama):
        """When _verify_goal returns a gap, _fill_gaps is called and its result is in the output."""

        def fake_run_step(self_inner, agent_type, task, **kwargs):
            return "step result"

        with patch.object(ExecutorAgent, "_run_step", fake_run_step), \
             patch("jarvis.agents.critic.CriticAgent.critique",
                   return_value=MagicMock(score=8, issues=[], should_retry=False)), \
             patch.object(ExecutorAgent, "_verify_goal", return_value="GAPS: something missing"), \
             patch.object(ExecutorAgent, "_fill_gaps", return_value="[RESEARCHER] gap filled") as mock_fill:
            executor = ExecutorAgent(
                model="llama3.2", max_tokens=512,
                sub_tool_schemas=[], sub_tool_registry={},
                db_path=tmp_path / "plans.db",
            )
            result = executor.execute_plan(
                "goal",
                [PlanStep(id="A", description="step", agent_type="researcher")],
            )

        mock_fill.assert_called_once()
        assert "gap filled" in result

    def test_no_gap_fill_when_verified(self, tmp_path, mock_ollama):
        """When _verify_goal returns empty (ACHIEVED), _fill_gaps is not called."""

        def fake_run_step(self_inner, agent_type, task, **kwargs):
            return "step result"

        with patch.object(ExecutorAgent, "_run_step", fake_run_step), \
             patch("jarvis.agents.critic.CriticAgent.critique",
                   return_value=MagicMock(score=8, issues=[], should_retry=False)), \
             patch.object(ExecutorAgent, "_verify_goal", return_value=""), \
             patch.object(ExecutorAgent, "_fill_gaps") as mock_fill:
            executor = ExecutorAgent(
                model="llama3.2", max_tokens=512,
                sub_tool_schemas=[], sub_tool_registry={},
                db_path=tmp_path / "plans.db",
            )
            executor.execute_plan(
                "goal",
                [PlanStep(id="A", description="step", agent_type="researcher")],
            )

        mock_fill.assert_not_called()
