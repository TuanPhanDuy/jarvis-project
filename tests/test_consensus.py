"""Tests for ConsensusAgent — parallel multi-agent synthesis."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from jarvis.agents.consensus import ConsensusAgent
from jarvis.agents.critic import CritiqueResult


def _make_consensus(n=3):
    return ConsensusAgent(
        model="test-model",
        max_tokens=512,
        tool_schemas=[],
        tool_registry={},
        n_agents=n,
    )


class TestConsensusAgent:
    def test_returns_best_scored_response(self):
        responses = ["weak answer", "strong detailed answer about RLHF", "mediocre response"]
        scores = [2, 5, 3]

        def fake_run_turn(messages, on_chunk=None):
            # Return responses in order
            idx = fake_run_turn.call_count
            fake_run_turn.call_count += 1
            return responses[idx % len(responses)], messages
        fake_run_turn.call_count = 0

        critiques = [CritiqueResult(score=s, issues=[], should_retry=False) for s in scores]

        with patch("jarvis.agents.researcher.ResearcherAgent.run_turn", side_effect=lambda m, on_chunk=None: (responses.pop(0) if responses else ("", m), m)), \
             patch("jarvis.agents.critic.CriticAgent.critique", side_effect=lambda task, result: critiques.pop(0)):
            agent = _make_consensus(n=3)
            # Patch run_turn directly on instances via ResearcherAgent
            call_count = {"n": 0}
            original_responses = ["weak answer", "strong detailed answer about RLHF", "mediocre response"]
            original_critiques = [CritiqueResult(2, [], False), CritiqueResult(5, [], False), CritiqueResult(3, [], False)]

            def patched_run(self_inner, messages, on_chunk=None):
                r = original_responses[call_count["n"] % len(original_responses)]
                call_count["n"] += 1
                return r, messages

            def patched_critique(self_inner, task, result):
                idx = original_responses.index(result) if result in original_responses else 0
                return original_critiques[idx]

            with patch("jarvis.agents.researcher.ResearcherAgent.run_turn", patched_run), \
                 patch("jarvis.agents.critic.CriticAgent.critique", patched_critique):
                best = agent.run("Explain RLHF")

        assert "strong" in best

    def test_filters_empty_responses(self):
        def patched_run(self_inner, messages, on_chunk=None):
            return "", messages

        with patch("jarvis.agents.researcher.ResearcherAgent.run_turn", patched_run):
            agent = _make_consensus(n=2)
            result = agent.run("test prompt")
        assert result == ""

    def test_minimum_n_agents_is_2(self):
        agent = ConsensusAgent("m", 512, [], {}, n_agents=1)
        assert agent._n_agents == 2

    def test_single_valid_response_returned_directly(self):
        call_count = {"n": 0}

        def patched_run(self_inner, messages, on_chunk=None):
            if call_count["n"] == 0:
                call_count["n"] += 1
                return "only valid response", messages
            call_count["n"] += 1
            return "", messages

        with patch("jarvis.agents.researcher.ResearcherAgent.run_turn", patched_run):
            agent = _make_consensus(n=2)
            result = agent.run("test")
        assert result == "only valid response"
