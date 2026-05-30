"""Tests for POST /api/pipeline multi-agent chaining endpoint."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def reset_auth():
    import jarvis.api.server as _s
    _s._require_auth = None
    yield
    _s._require_auth = None


@pytest.fixture()
def client():
    from jarvis.api.server import app
    return TestClient(app, raise_server_exceptions=False)


def _mock_agent(output: str = "step output") -> MagicMock:
    agent = MagicMock()
    agent.run_turn.return_value = (output, [{"role": "assistant", "content": output}])
    agent.get_usage_summary.return_value = {
        "input_tokens": 5, "output_tokens": 3,
        "cache_write_tokens": 0, "cache_read_tokens": 0, "estimated_cost_usd": 0.0,
    }
    agent._approval_gate = None
    return agent


def _post(client, prompt: str, steps: list, **kwargs):
    payload = {"prompt": prompt, "steps": steps, **kwargs}
    return client.post("/api/pipeline", json=payload)


class TestPipelineValidation:
    def test_empty_prompt_422(self, client):
        resp = _post(client, "", [{"agent_type": "researcher"}])
        assert resp.status_code == 422

    def test_empty_steps_422(self, client):
        resp = _post(client, "hello", [])
        assert resp.status_code == 422

    def test_unknown_agent_type_422(self, client):
        resp = _post(client, "hi", [{"agent_type": "wizard"}])
        assert resp.status_code == 422

    def test_valid_agent_types_accepted(self, client):
        for agent_type in ("planner", "researcher", "coder", "qa", "analyst", "devops"):
            with patch("jarvis.api.server._build_agent_for_session",
                       return_value=_mock_agent(f"{agent_type} done")):
                resp = _post(client, "test", [{"agent_type": agent_type}])
            assert resp.status_code == 200, f"{agent_type} should be valid"


class TestPipelineExecution:
    def test_single_step_returns_output(self, client):
        with patch("jarvis.api.server._build_agent_for_session",
                   return_value=_mock_agent("research done")):
            resp = _post(client, "research topic X", [{"agent_type": "researcher"}])
        assert resp.status_code == 200
        body = resp.json()
        assert body["final_output"] == "research done"
        assert len(body["steps"]) == 1

    def test_multi_step_chains_output(self, client):
        outputs = ["first step", "second step", "final"]
        call_count = [0]

        def make_agent(*args, **kwargs):
            idx = call_count[0]
            call_count[0] += 1
            return _mock_agent(outputs[idx % len(outputs)])

        with patch("jarvis.api.server._build_agent_for_session", side_effect=make_agent):
            resp = _post(client, "start", [
                {"agent_type": "researcher"},
                {"agent_type": "analyst"},
                {"agent_type": "planner"},
            ])
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["steps"]) == 3
        assert body["final_output"] == outputs[2]

    def test_response_includes_steps_with_usage(self, client):
        with patch("jarvis.api.server._build_agent_for_session",
                   return_value=_mock_agent("done")):
            resp = _post(client, "prompt", [{"agent_type": "planner"}])
        body = resp.json()
        step = body["steps"][0]
        assert "agent_type" in step
        assert "output" in step
        assert "usage" in step

    def test_total_usage_aggregated(self, client):
        agents = [_mock_agent("out1"), _mock_agent("out2")]
        idx = [0]

        def side_effect(*a, **kw):
            a = agents[idx[0] % len(agents)]
            idx[0] += 1
            return a

        with patch("jarvis.api.server._build_agent_for_session", side_effect=side_effect):
            resp = _post(client, "go", [{"agent_type": "researcher"}, {"agent_type": "coder"}])
        total = resp.json()["total_usage"]
        assert total["input_tokens"] == 10   # 5 + 5
        assert total["output_tokens"] == 6   # 3 + 3

    def test_step_with_instructions(self, client):
        agent = _mock_agent("analyzed")
        with patch("jarvis.api.server._build_agent_for_session", return_value=agent):
            resp = _post(client, "raw data", [
                {"agent_type": "analyst", "instructions": "Summarize key findings"}
            ])
        assert resp.status_code == 200
        # Verify instructions were part of the message content
        call_args = agent.run_turn.call_args
        msgs = call_args[0][0]
        assert any("Summarize key findings" in str(m.get("content", "")) for m in msgs)
