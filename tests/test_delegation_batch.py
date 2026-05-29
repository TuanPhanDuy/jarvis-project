"""Tests for delegate_batch — parallel multi-agent task dispatch."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from jarvis.tools.delegation_batch import build_batch_delegation_handler, SCHEMA


def _make_handler():
    return build_batch_delegation_handler(
        model="llama3.2",
        max_tokens=512,
        sub_tool_schemas=[],
        sub_tool_registry={},
    )


class TestDelegateBatchSchema:
    def test_schema_name(self):
        assert SCHEMA["name"] == "delegate_batch"

    def test_schema_requires_tasks(self):
        assert "tasks" in SCHEMA["input_schema"]["required"]

    def test_task_item_requires_id_agent_task(self):
        item_schema = SCHEMA["input_schema"]["properties"]["tasks"]["items"]
        assert set(item_schema["required"]) == {"id", "agent_type", "task"}

    def test_agent_type_enum_has_all_types(self):
        item_schema = SCHEMA["input_schema"]["properties"]["tasks"]["items"]
        enum = item_schema["properties"]["agent_type"]["enum"]
        assert {"researcher", "coder", "qa", "analyst", "devops", "consensus"}.issubset(set(enum))

    def test_schema_supports_timeout_seconds(self):
        assert "timeout_seconds" in SCHEMA["input_schema"]["properties"]

    def test_schema_max_items_is_ten(self):
        assert SCHEMA["input_schema"]["properties"]["tasks"]["maxItems"] == 10


class TestDelegateBatchHandler:
    def test_empty_tasks_returns_error(self):
        handler = _make_handler()
        result = handler({"tasks": []})
        assert result.startswith("ERROR")

    def test_missing_tasks_key_returns_error(self):
        handler = _make_handler()
        result = handler({})
        assert result.startswith("ERROR")

    def test_unknown_agent_type_returns_error(self, mock_ollama):
        handler = _make_handler()
        result = handler({
            "tasks": [
                {"id": "t1", "agent_type": "wizard", "task": "do magic"},
                {"id": "t2", "agent_type": "researcher", "task": "research AI"},
            ]
        })
        assert "ERROR" in result
        assert "wizard" in result

    def test_empty_task_text_returns_error(self, mock_ollama):
        handler = _make_handler()
        result = handler({
            "tasks": [
                {"id": "t1", "agent_type": "researcher", "task": ""},
                {"id": "t2", "agent_type": "coder", "task": "write a script"},
            ]
        })
        assert "ERROR" in result

    def test_results_returned_in_input_order(self, mock_ollama):
        call_log: list[str] = []

        def fake_run_turn(self_inner, messages, on_chunk=None):
            task_text = messages[0]["content"]
            call_log.append(task_text)
            return f"result_for_{task_text[:10]}", messages

        with patch("jarvis.agents.researcher.ResearcherAgent.run_turn", fake_run_turn):
            handler = _make_handler()
            result = handler({
                "tasks": [
                    {"id": "alpha", "agent_type": "researcher", "task": "first task"},
                    {"id": "beta",  "agent_type": "researcher", "task": "second task"},
                    {"id": "gamma", "agent_type": "researcher", "task": "third task"},
                ]
            })

        # Output sections appear in the order alpha → beta → gamma
        alpha_pos = result.index("ALPHA")
        beta_pos  = result.index("BETA")
        gamma_pos = result.index("GAMMA")
        assert alpha_pos < beta_pos < gamma_pos

    def test_all_agents_types_dispatched(self, mock_ollama):
        """Verify each agent type can be invoked without error."""
        agent_calls: dict[str, int] = {}

        def fake_run_turn(self_inner, messages, on_chunk=None):
            cls_name = type(self_inner).__name__
            agent_calls[cls_name] = agent_calls.get(cls_name, 0) + 1
            return "ok", messages

        with patch("jarvis.agents.researcher.ResearcherAgent.run_turn", fake_run_turn), \
             patch("jarvis.agents.coder.CoderAgent.run_turn", fake_run_turn), \
             patch("jarvis.agents.qa.QAAgent.run_turn", fake_run_turn):
            handler = _make_handler()
            result = handler({
                "tasks": [
                    {"id": "r", "agent_type": "researcher", "task": "research something"},
                    {"id": "c", "agent_type": "coder",      "task": "write some code"},
                    {"id": "q", "agent_type": "qa",         "task": "review the code"},
                ]
            })

        assert "ERROR" not in result

    def test_single_failure_does_not_block_other_tasks(self, mock_ollama):
        """A failed task returns ERROR for that slot but others still complete."""
        call_count = {"n": 0}

        def fake_run_turn(self_inner, messages, on_chunk=None):
            call_count["n"] += 1
            content = messages[0]["content"]
            if "fail_me" in content:
                raise RuntimeError("agent exploded")
            return "success", messages

        with patch("jarvis.agents.researcher.ResearcherAgent.run_turn", fake_run_turn):
            handler = _make_handler()
            result = handler({
                "tasks": [
                    {"id": "good", "agent_type": "researcher", "task": "normal task"},
                    {"id": "bad",  "agent_type": "researcher", "task": "fail_me please"},
                ]
            })

        assert "success" in result
        assert "ERROR" in result

    def test_output_includes_task_ids_as_headers(self, mock_ollama):
        with patch("jarvis.agents.researcher.ResearcherAgent.run_turn",
                   lambda self, msgs, on_chunk=None: ("done", msgs)):
            handler = _make_handler()
            result = handler({
                "tasks": [
                    {"id": "my_task_1", "agent_type": "researcher", "task": "research it"},
                    {"id": "my_task_2", "agent_type": "researcher", "task": "research more"},
                ]
            })

        assert "MY_TASK_1" in result
        assert "MY_TASK_2" in result

    def test_parallel_execution_runs_all_tasks(self, mock_ollama):
        """All tasks must be executed — verify via call count."""
        call_count = {"n": 0}

        def fake_run_turn(self_inner, messages, on_chunk=None):
            call_count["n"] += 1
            return "result", messages

        with patch("jarvis.agents.researcher.ResearcherAgent.run_turn", fake_run_turn):
            handler = _make_handler()
            handler({
                "tasks": [
                    {"id": f"t{i}", "agent_type": "researcher", "task": f"task {i}"}
                    for i in range(4)
                ]
            })

        assert call_count["n"] == 4


class TestDelegateBatchConsensus:
    def test_consensus_agent_type_dispatches_consensus_agent(self, mock_ollama):
        from jarvis.agents.consensus import ConsensusAgent

        consensus_calls: list[str] = []

        def fake_consensus_run(self_inner, prompt):
            consensus_calls.append(prompt)
            return "consensus result"

        with patch.object(ConsensusAgent, "run", fake_consensus_run):
            handler = _make_handler()
            result = handler({
                "tasks": [
                    {"id": "c1", "agent_type": "consensus", "task": "What is RLHF?"},
                    {"id": "r1", "agent_type": "researcher", "task": "Describe transformers"},
                ]
            })

        assert len(consensus_calls) == 1
        assert "consensus result" in result

    def test_unknown_agent_type_includes_consensus_in_valid_list(self, mock_ollama):
        handler = _make_handler()
        result = handler({
            "tasks": [
                {"id": "bad", "agent_type": "wizard", "task": "do magic"},
                {"id": "ok",  "agent_type": "researcher", "task": "research AI"},
            ]
        })
        assert "ERROR" in result
        assert "consensus" in result


class TestDelegateBatchTimeout:
    def test_timeout_populates_error_for_unfinished_tasks(self, mock_ollama):
        import time

        def slow_run_turn(self_inner, messages, on_chunk=None):
            time.sleep(5)
            return "slow result", messages

        with patch("jarvis.agents.researcher.ResearcherAgent.run_turn", slow_run_turn):
            handler = _make_handler()
            result = handler({
                "tasks": [
                    {"id": "slow", "agent_type": "researcher", "task": "slow task"},
                    {"id": "slow2", "agent_type": "researcher", "task": "slow task 2"},
                ],
                "timeout_seconds": 0.05,
            })

        assert "ERROR" in result


class TestRegistryIntegration:
    def test_delegate_batch_in_planner_registry(self, tmp_path):
        """build_planner_registry should include delegate_batch schema and handler."""
        from jarvis.tools.registry import build_planner_registry

        planner_schemas, planner_registry = build_planner_registry(
            base_schemas=[],
            base_registry={},
            model="llama3.2",
            max_tokens=512,
            db_path=tmp_path / "plans.db",
        )
        schema_names = {s["name"] for s in planner_schemas}
        assert "delegate_batch" in schema_names
        assert "delegate_batch" in planner_registry
