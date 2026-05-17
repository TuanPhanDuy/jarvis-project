"""Tests for plan template registry and parameter substitution."""
from __future__ import annotations

import pytest

from jarvis.agents.plan_templates import get_template, list_templates, _TEMPLATES
from jarvis.agents.executor import PlanStep


class TestListTemplates:
    def test_returns_all_template_names(self):
        names = {t["name"] for t in list_templates()}
        assert "research-report" in names
        assert "code-review" in names
        assert "data-analysis" in names
        assert "implement-and-test" in names

    def test_each_entry_has_required_keys(self):
        for tpl in list_templates():
            assert "name" in tpl
            assert "description" in tpl
            assert "params" in tpl
            assert "step_count" in tpl

    def test_step_counts_are_positive(self):
        for tpl in list_templates():
            assert tpl["step_count"] >= 5


class TestGetTemplate:
    def test_unknown_template_returns_none(self):
        assert get_template("nonexistent") is None

    def test_returns_goal_and_steps(self):
        result = get_template("research-report", topic="RLHF")
        assert result is not None
        goal, steps = result
        assert isinstance(goal, str)
        assert isinstance(steps, list)
        assert all(isinstance(s, PlanStep) for s in steps)

    def test_param_substituted_in_goal(self):
        goal, _ = get_template("research-report", topic="Constitutional AI")
        assert "Constitutional AI" in goal
        assert "{topic}" not in goal

    def test_param_substituted_in_step_descriptions(self):
        _, steps = get_template("research-report", topic="PPO algorithm")
        for step in steps:
            assert "{topic}" not in step.description
            assert "PPO algorithm" in step.description

    def test_steps_have_correct_agent_types(self):
        _, steps = get_template("code-review", repo_path="/home/user/myrepo")
        agent_types = {s.agent_type for s in steps}
        valid = {"researcher", "coder", "qa", "analyst", "devops"}
        assert agent_types.issubset(valid)

    def test_steps_have_unique_ids(self):
        _, steps = get_template("implement-and-test", feature="auth module", language="Python")
        ids = [s.id for s in steps]
        assert len(ids) == len(set(ids))

    def test_depends_on_references_valid_step_ids(self):
        _, steps = get_template("data-analysis", data_source="sales.csv", question="What drives revenue?")
        step_ids = {s.id for s in steps}
        for step in steps:
            for dep in step.depends_on:
                assert dep in step_ids, f"Step {step.id} depends on unknown step '{dep}'"

    def test_parallel_steps_exist(self):
        """At least 2 steps should have no depends_on (can run in parallel)."""
        _, steps = get_template("research-report", topic="transformers")
        independent = [s for s in steps if not s.depends_on]
        assert len(independent) >= 2

    def test_template_does_not_mutate_original(self):
        """Calling get_template twice should return independent step lists."""
        _, steps1 = get_template("research-report", topic="A")
        _, steps2 = get_template("research-report", topic="B")
        assert steps1[0].description != steps2[0].description


class TestPlanTemplateTool:
    def test_schema_lists_all_templates(self):
        from jarvis.tools.plan_template_tool import SCHEMA
        enum = SCHEMA["input_schema"]["properties"]["template"]["enum"]
        assert set(enum) == {"research-report", "code-review", "data-analysis", "implement-and-test"}

    def test_template_in_planner_registry(self, tmp_path):
        from jarvis.tools.registry import build_planner_registry
        _, registry = build_planner_registry(
            base_schemas=[], base_registry={},
            model="llama3.2", max_tokens=512,
            db_path=tmp_path / "plans.db",
        )
        assert "plan_from_template" in registry

    def test_handler_unknown_template_returns_error(self, tmp_path):
        from jarvis.tools.plan_template_tool import build_plan_template_handler
        handler = build_plan_template_handler(
            model="llama3.2", max_tokens=512,
            sub_tool_schemas=[], sub_tool_registry={},
            db_path=tmp_path / "plans.db",
        )
        result = handler({"template": "nonexistent-template"})
        assert result.startswith("ERROR")
        assert "nonexistent-template" in result
