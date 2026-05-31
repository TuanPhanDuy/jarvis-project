"""Tests for the agent debate pipeline."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from jarvis.agents.debate import _confidence, _verdict, run_debate


class TestConfidence:
    def test_score_1_is_zero(self):
        assert _confidence(1) == 0.0

    def test_score_5_is_one(self):
        assert _confidence(5) == 1.0

    def test_score_3_is_midpoint(self):
        assert _confidence(3) == 0.5

    def test_clamps_below_range(self):
        assert _confidence(0) == 0.0

    def test_clamps_above_range(self):
        assert _confidence(10) == 1.0


class TestVerdict:
    def test_score_4_is_well_supported(self):
        assert _verdict(4, []) == "well_supported"

    def test_score_5_is_well_supported(self):
        assert _verdict(5, []) == "well_supported"

    def test_score_3_is_partially_supported(self):
        assert _verdict(3, []) == "partially_supported"

    def test_score_2_is_poorly_supported(self):
        assert _verdict(2, ["lacks citations"]) == "poorly_supported"

    def test_score_1_is_poorly_supported(self):
        assert _verdict(1, ["wrong", "incomplete"]) == "poorly_supported"


class TestRunDebate:
    def _make_researcher(self, summary: str) -> MagicMock:
        researcher = MagicMock()
        researcher.run_turn.return_value = (summary, [])
        return researcher

    def _make_critic(self, score: int, issues: list, retry: bool = False, revised=None) -> MagicMock:
        from jarvis.agents.critic import CritiqueResult
        critic = MagicMock()
        critic.critique.return_value = CritiqueResult(
            score=score, issues=issues, should_retry=retry, revised_task=revised
        )
        return critic

    def test_returns_required_fields(self):
        researcher = self._make_researcher("Transformers use self-attention.")
        critic = self._make_critic(4, [])
        result = run_debate("What is a transformer?", researcher, critic)
        assert "question" in result
        assert "research_summary" in result
        assert "critique_issues" in result
        assert "verdict" in result
        assert "confidence_score" in result
        assert "critic_score" in result
        assert "retry_recommended" in result
        assert "revised_question" in result

    def test_question_passed_to_researcher(self):
        researcher = self._make_researcher("summary")
        critic = self._make_critic(3, [])
        run_debate("What is RLHF?", researcher, critic)
        call_messages = researcher.run_turn.call_args[0][0]
        assert any("What is RLHF?" in str(m) for m in call_messages)

    def test_research_summary_passed_to_critic(self):
        summary = "RLHF stands for Reinforcement Learning from Human Feedback."
        researcher = self._make_researcher(summary)
        critic = self._make_critic(4, [])
        run_debate("What is RLHF?", researcher, critic)
        call_args = critic.critique.call_args
        assert summary in call_args[1].get("result", call_args[0][1] if call_args[0] else "")

    def test_high_score_gives_well_supported_verdict(self):
        researcher = self._make_researcher("Detailed analysis...")
        critic = self._make_critic(5, [])
        result = run_debate("question", researcher, critic)
        assert result["verdict"] == "well_supported"
        assert result["confidence_score"] == 1.0

    def test_low_score_gives_poorly_supported_verdict(self):
        researcher = self._make_researcher("Vague answer")
        critic = self._make_critic(1, ["too vague", "no sources"], retry=True)
        result = run_debate("question", researcher, critic)
        assert result["verdict"] == "poorly_supported"
        assert result["retry_recommended"] is True
        assert "too vague" in result["critique_issues"]

    def test_revised_question_propagated(self):
        researcher = self._make_researcher("answer")
        critic = self._make_critic(2, ["needs scope"], retry=True, revised="Narrower question?")
        result = run_debate("question", researcher, critic)
        assert result["revised_question"] == "Narrower question?"

    def test_confidence_is_float_between_0_and_1(self):
        researcher = self._make_researcher("answer")
        for score in range(1, 6):
            critic = self._make_critic(score, [])
            result = run_debate("q", researcher, critic)
            assert 0.0 <= result["confidence_score"] <= 1.0
