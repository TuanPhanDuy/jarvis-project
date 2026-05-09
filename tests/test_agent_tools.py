"""Tests verifying each specialist agent filters its tool registry correctly."""
from __future__ import annotations

import pytest

ALL_TOOL_NAMES = [
    "web_search", "read_url", "browse",
    "search_memory", "search_episodic_memory", "query_knowledge_graph",
    "analyze_text", "summarize_youtube",
    "save_report", "update_report",
    "filesystem_search",
    "execute_python", "run_command",
    "git_context",
    "database_query", "query_database",
    "ingest_document",
    "system_info",
    "delegate_task", "create_plan",
    "get_weather", "generate_tool", "ask_local_model",
    "read_calendar", "analyze_image",
]

_MOCK_SCHEMAS = [{"name": n, "description": n, "input_schema": {}} for n in ALL_TOOL_NAMES]
_MOCK_REGISTRY = {n: lambda _: "ok" for n in ALL_TOOL_NAMES}


def _names(agent) -> set[str]:
    return {s["name"] for s in agent._tool_schemas}


def _reg_names(agent) -> set[str]:
    return set(agent._tool_registry.keys())


class TestResearcherToolFilter:
    def test_allows_web_search(self):
        from jarvis.agents.researcher import ResearcherAgent
        a = ResearcherAgent("llama3.2", 512, _MOCK_SCHEMAS, _MOCK_REGISTRY)
        assert "web_search" in _names(a)

    def test_blocks_run_command(self):
        from jarvis.agents.researcher import ResearcherAgent
        a = ResearcherAgent("llama3.2", 512, _MOCK_SCHEMAS, _MOCK_REGISTRY)
        assert "run_command" not in _names(a)

    def test_blocks_delegate_task(self):
        from jarvis.agents.researcher import ResearcherAgent
        a = ResearcherAgent("llama3.2", 512, _MOCK_SCHEMAS, _MOCK_REGISTRY)
        assert "delegate_task" not in _names(a)


class TestCoderToolFilter:
    def test_allows_execute_python(self):
        from jarvis.agents.coder import CoderAgent
        a = CoderAgent("llama3.2", 512, _MOCK_SCHEMAS, _MOCK_REGISTRY)
        assert "execute_python" in _names(a)

    def test_blocks_web_search(self):
        from jarvis.agents.coder import CoderAgent
        a = CoderAgent("llama3.2", 512, _MOCK_SCHEMAS, _MOCK_REGISTRY)
        assert "web_search" not in _names(a)

    def test_blocks_delegate_task(self):
        from jarvis.agents.coder import CoderAgent
        a = CoderAgent("llama3.2", 512, _MOCK_SCHEMAS, _MOCK_REGISTRY)
        assert "delegate_task" not in _names(a)


class TestQAToolFilter:
    def test_allows_git_context(self):
        from jarvis.agents.qa import QAAgent
        a = QAAgent("llama3.2", 512, _MOCK_SCHEMAS, _MOCK_REGISTRY)
        assert "git_context" in _names(a)

    def test_blocks_web_search(self):
        from jarvis.agents.qa import QAAgent
        a = QAAgent("llama3.2", 512, _MOCK_SCHEMAS, _MOCK_REGISTRY)
        assert "web_search" not in _names(a)

    def test_blocks_delegate_task(self):
        from jarvis.agents.qa import QAAgent
        a = QAAgent("llama3.2", 512, _MOCK_SCHEMAS, _MOCK_REGISTRY)
        assert "delegate_task" not in _names(a)


class TestAnalystToolFilter:
    def test_allows_query_database(self):
        from jarvis.agents.data_analyst import DataAnalystAgent
        a = DataAnalystAgent("llama3.2", 512, _MOCK_SCHEMAS, _MOCK_REGISTRY)
        assert "query_database" in _names(a)

    def test_blocks_run_command(self):
        from jarvis.agents.data_analyst import DataAnalystAgent
        a = DataAnalystAgent("llama3.2", 512, _MOCK_SCHEMAS, _MOCK_REGISTRY)
        assert "run_command" not in _names(a)

    def test_blocks_delegate_task(self):
        from jarvis.agents.data_analyst import DataAnalystAgent
        a = DataAnalystAgent("llama3.2", 512, _MOCK_SCHEMAS, _MOCK_REGISTRY)
        assert "delegate_task" not in _names(a)


class TestDevOpsToolFilter:
    def test_allows_system_info(self):
        from jarvis.agents.devops import DevOpsAgent
        a = DevOpsAgent("llama3.2", 512, _MOCK_SCHEMAS, _MOCK_REGISTRY)
        assert "system_info" in _names(a)

    def test_blocks_web_search(self):
        from jarvis.agents.devops import DevOpsAgent
        a = DevOpsAgent("llama3.2", 512, _MOCK_SCHEMAS, _MOCK_REGISTRY)
        assert "web_search" not in _names(a)

    def test_blocks_delegate_task(self):
        from jarvis.agents.devops import DevOpsAgent
        a = DevOpsAgent("llama3.2", 512, _MOCK_SCHEMAS, _MOCK_REGISTRY)
        assert "delegate_task" not in _names(a)


class TestRegistryFilterConsistency:
    """schemas and registry must always be in sync after filtering."""

    @pytest.mark.parametrize("AgentClass,module", [
        ("ResearcherAgent", "jarvis.agents.researcher"),
        ("CoderAgent", "jarvis.agents.coder"),
        ("QAAgent", "jarvis.agents.qa"),
        ("DataAnalystAgent", "jarvis.agents.data_analyst"),
        ("DevOpsAgent", "jarvis.agents.devops"),
    ])
    def test_schema_registry_in_sync(self, AgentClass, module):
        import importlib
        mod = importlib.import_module(module)
        cls = getattr(mod, AgentClass)
        agent = cls("llama3.2", 512, _MOCK_SCHEMAS, _MOCK_REGISTRY)
        assert _names(agent) == _reg_names(agent)
