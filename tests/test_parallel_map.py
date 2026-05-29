"""Tests for parallel_map — scatter-gather multi-agent parallel research."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from jarvis.tools.parallel_map import build_parallel_map_handler, SCHEMA


def _make_handler():
    return build_parallel_map_handler(
        model="llama3.2",
        max_tokens=512,
        sub_tool_schemas=[],
        sub_tool_registry={},
    )


def _fake_run_turn(self_inner, messages, on_chunk=None):
    topic = messages[0]["content"]
    return f"result_for_{topic[:20]}", messages


class TestParallelMapSchema:
    def test_schema_name(self):
        assert SCHEMA["name"] == "parallel_map"

    def test_requires_task_template_and_topics(self):
        required = SCHEMA["input_schema"]["required"]
        assert "task_template" in required
        assert "topics" in required

    def test_agent_type_enum_includes_all_five(self):
        enum = SCHEMA["input_schema"]["properties"]["agent_type"]["enum"]
        assert set(enum) >= {"researcher", "coder", "qa", "analyst", "devops"}

    def test_topics_min_max(self):
        topics_schema = SCHEMA["input_schema"]["properties"]["topics"]
        assert topics_schema["minItems"] == 2
        assert topics_schema["maxItems"] == 10


class TestParallelMapValidation:
    def test_missing_task_template_returns_error(self):
        handler = _make_handler()
        result = handler({"topics": ["RLHF", "transformers"]})
        assert result.startswith("ERROR")

    def test_missing_topics_returns_error(self):
        handler = _make_handler()
        result = handler({"task_template": "Research {topic}"})
        assert result.startswith("ERROR")

    def test_missing_placeholder_returns_error(self):
        handler = _make_handler()
        result = handler({
            "task_template": "Research AI without the placeholder",
            "topics": ["RLHF", "transformers"],
        })
        assert result.startswith("ERROR")
        assert "{topic}" in result

    def test_too_many_topics_returns_error(self):
        handler = _make_handler()
        result = handler({
            "task_template": "Research {topic}",
            "topics": [f"topic_{i}" for i in range(11)],
        })
        assert result.startswith("ERROR")

    def test_unknown_agent_type_returns_error(self, mock_ollama):
        handler = _make_handler()
        result = handler({
            "task_template": "Research {topic}",
            "topics": ["RLHF", "transformers"],
            "agent_type": "wizard",
        })
        assert result.startswith("ERROR")
        assert "wizard" in result


class TestParallelMapExecution:
    def test_results_include_all_topics(self, mock_ollama):
        with patch("jarvis.agents.researcher.ResearcherAgent.run_turn", _fake_run_turn):
            handler = _make_handler()
            result = handler({
                "task_template": "Research {topic} deeply.",
                "topics": ["RLHF", "transformers", "LoRA"],
                "synthesize": False,
            })

        assert "RLHF" in result
        assert "transformers" in result
        assert "LoRA" in result

    def test_topics_appear_in_original_order(self, mock_ollama):
        with patch("jarvis.agents.researcher.ResearcherAgent.run_turn", _fake_run_turn):
            handler = _make_handler()
            result = handler({
                "task_template": "Explain {topic}.",
                "topics": ["alpha", "beta", "gamma"],
                "synthesize": False,
            })

        assert result.index("alpha") < result.index("beta") < result.index("gamma")

    def test_synthesize_true_adds_synthesis_section(self, mock_ollama):
        call_count = {"n": 0}

        def counting_run_turn(self_inner, messages, on_chunk=None):
            call_count["n"] += 1
            return "detailed result", messages

        with patch("jarvis.agents.researcher.ResearcherAgent.run_turn", counting_run_turn):
            handler = _make_handler()
            result = handler({
                "task_template": "Research {topic}.",
                "topics": ["RLHF", "RLHF-v2"],
                "synthesize": True,
            })

        # 2 topic agents + 1 synthesis agent = 3 calls
        assert call_count["n"] == 3
        assert "Synthesis" in result

    def test_synthesize_false_skips_synthesis(self, mock_ollama):
        call_count = {"n": 0}

        def counting_run_turn(self_inner, messages, on_chunk=None):
            call_count["n"] += 1
            return "result", messages

        with patch("jarvis.agents.researcher.ResearcherAgent.run_turn", counting_run_turn):
            handler = _make_handler()
            handler({
                "task_template": "Research {topic}.",
                "topics": ["RLHF", "transformers"],
                "synthesize": False,
            })

        assert call_count["n"] == 2

    def test_single_topic_skips_synthesis_even_if_enabled(self, mock_ollama):
        call_count = {"n": 0}

        def counting_run_turn(self_inner, messages, on_chunk=None):
            call_count["n"] += 1
            return "result", messages

        # single topic: synthesize should be skipped
        with patch("jarvis.agents.researcher.ResearcherAgent.run_turn", counting_run_turn):
            handler = _make_handler()
            # schema enforces minItems=2 but handler shouldn't crash on 1
            handler._handler = None  # bypass validation at schema level
            result = build_parallel_map_handler(
                model="llama3.2", max_tokens=512,
                sub_tool_schemas=[], sub_tool_registry={},
            )({"task_template": "Research {topic}.", "topics": ["RLHF"], "synthesize": True})

        assert call_count["n"] == 1
        assert "Synthesis" not in result

    def test_failed_agent_returns_error_for_that_topic(self, mock_ollama):
        def exploding_run_turn(self_inner, messages, on_chunk=None):
            if "fail" in messages[0]["content"]:
                raise RuntimeError("agent exploded")
            return "ok result", messages

        with patch("jarvis.agents.researcher.ResearcherAgent.run_turn", exploding_run_turn):
            handler = _make_handler()
            result = handler({
                "task_template": "Research {topic}.",
                "topics": ["safe_topic", "fail_topic"],
                "synthesize": False,
            })

        assert "ok result" in result
        assert "ERROR" in result

    def test_placeholder_substituted_into_task(self, mock_ollama):
        captured_tasks: list[str] = []

        def capturing_run_turn(self_inner, messages, on_chunk=None):
            captured_tasks.append(messages[0]["content"])
            return "result", messages

        with patch("jarvis.agents.researcher.ResearcherAgent.run_turn", capturing_run_turn):
            handler = _make_handler()
            handler({
                "task_template": "Deep dive into {topic} and its applications.",
                "topics": ["RLHF", "constitutional AI"],
                "synthesize": False,
            })

        assert any("RLHF" in t for t in captured_tasks)
        assert any("constitutional AI" in t for t in captured_tasks)
        assert all("{topic}" not in t for t in captured_tasks)


class TestRegistryIntegration:
    def test_parallel_map_in_planner_registry(self, tmp_path):
        from jarvis.tools.registry import build_planner_registry

        planner_schemas, planner_registry = build_planner_registry(
            base_schemas=[],
            base_registry={},
            model="llama3.2",
            max_tokens=512,
            db_path=tmp_path / "plans.db",
        )
        schema_names = {s["name"] for s in planner_schemas}
        assert "parallel_map" in schema_names
        assert "parallel_map" in planner_registry
