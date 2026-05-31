"""Agent debate pipeline — researcher + critic multi-pass workflow.

Flow:
  1. ResearcherAgent answers the question (one turn, no conversation history).
  2. CriticAgent evaluates the research and produces a structured critique.
  3. A verdict and confidence score are derived from the critique score.

The result is a self-contained dict suitable for returning from an API endpoint.
"""
from __future__ import annotations

from dataclasses import asdict


def _confidence(score: int) -> float:
    """Map critic score 1-5 → confidence 0.0-1.0."""
    return round(max(0.0, min(1.0, (score - 1) / 4.0)), 2)


def _verdict(score: int, issues: list[str]) -> str:
    if score >= 4:
        return "well_supported"
    if score == 3:
        return "partially_supported"
    return "poorly_supported"


def run_debate(
    question: str,
    researcher,
    critic,
) -> dict:
    """Run a researcher→critic debate and return a structured result.

    Args:
        question:   The question or claim to investigate.
        researcher: An instantiated ResearcherAgent (or BaseAgent subclass).
        critic:     An instantiated CriticAgent.

    Returns:
        {question, research_summary, critique_issues, verdict, confidence_score,
         retry_recommended, revised_question}
    """
    messages = [{"role": "user", "content": question}]
    research_summary, _ = researcher.run_turn(messages)

    critique_result = critic.critique(task=question, result=research_summary)

    return {
        "question": question,
        "research_summary": research_summary,
        "critique_issues": critique_result.issues,
        "verdict": _verdict(critique_result.score, critique_result.issues),
        "confidence_score": _confidence(critique_result.score),
        "critic_score": critique_result.score,
        "retry_recommended": critique_result.should_retry,
        "revised_question": critique_result.revised_task,
    }
