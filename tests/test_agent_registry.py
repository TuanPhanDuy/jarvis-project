"""Tests for the agent capability registry."""
from __future__ import annotations

import pytest

from jarvis.agents.registry import get_agent_info, list_agents


class TestListAgents:
    def test_returns_nonempty_list(self):
        agents = list_agents()
        assert len(agents) > 0

    def test_each_entry_has_required_fields(self):
        for agent in list_agents():
            assert "name" in agent
            assert "description" in agent
            assert "allowed_tools" in agent
            assert "tool_count" in agent
            assert "prompt_file" in agent
            assert "prompt_source" in agent

    def test_known_agents_present(self):
        names = {a["name"] for a in list_agents()}
        for expected in ("planner", "researcher", "coder", "qa", "data_analyst", "devops", "critic"):
            assert expected in names

    def test_researcher_has_web_search(self):
        researcher = next(a for a in list_agents() if a["name"] == "researcher")
        assert "web_search" in researcher["allowed_tools"]

    def test_coder_has_execute_python(self):
        coder = next(a for a in list_agents() if a["name"] == "coder")
        assert "execute_python" in coder["allowed_tools"]

    def test_qa_does_not_have_web_search(self):
        qa = next(a for a in list_agents() if a["name"] == "qa")
        assert "web_search" not in qa["allowed_tools"]

    def test_critic_has_no_tools(self):
        critic = next(a for a in list_agents() if a["name"] == "critic")
        assert critic["allowed_tools"] == []
        assert critic["tool_count"] == 0

    def test_planner_has_null_tool_list(self):
        planner = next(a for a in list_agents() if a["name"] == "planner")
        assert planner["allowed_tools"] is None
        assert planner["tool_count"] == "all"

    def test_descriptions_nonempty(self):
        for agent in list_agents():
            assert len(agent["description"]) > 5

    def test_prompt_source_is_file_or_override(self):
        from jarvis.prompts.overrides import clear_all_overrides
        clear_all_overrides()
        for agent in list_agents():
            assert agent["prompt_source"] in ("file", "override")

    def test_prompt_source_override_detected(self):
        from jarvis.prompts.overrides import set_override, clear_all_overrides
        clear_all_overrides()
        set_override("researcher", "custom prompt")
        researcher = next(a for a in list_agents() if a["name"] == "researcher")
        assert researcher["prompt_source"] == "override"
        clear_all_overrides()


class TestGetAgentInfo:
    def test_returns_none_for_unknown(self):
        assert get_agent_info("nonexistent_agent") is None

    def test_returns_dict_for_known(self):
        info = get_agent_info("researcher")
        assert info is not None
        assert info["name"] == "researcher"

    def test_case_insensitive(self):
        assert get_agent_info("Researcher") is not None
        assert get_agent_info("CODER") is not None

    def test_coder_info(self):
        info = get_agent_info("coder")
        assert "execute_python" in info["allowed_tools"]
        assert info["prompt_file"] == "coder"
