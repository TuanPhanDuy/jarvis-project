"""Eval runner: execute an eval suite against JARVIS and score results.

Scoring:
  - contains_pass: all expected_contains strings found (case-insensitive)
  - forbidden_pass: no forbidden strings found
  - judge_score: 1-5 from Claude-as-judge (only when judge_rubric is set)
  - overall_pass: contains_pass AND forbidden_pass

Usage:
    from jarvis.evals.runner import run_suite
    from jarvis.evals.suite import BASELINE_SUITE
    results = run_suite(BASELINE_SUITE, settings)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from jarvis.evals.suite import EvalCase


@dataclass
class EvalResult:
    case_id: str
    prompt: str
    response: str
    contains_pass: bool
    forbidden_pass: bool
    overall_pass: bool
    latency_s: float
    cost_usd: float
    failed_contains: list[str] = field(default_factory=list)
    failed_forbidden: list[str] = field(default_factory=list)
    judge_score: int | None = None
    judge_reasoning: str = ""
    error: str = ""


def _score_case(case: EvalCase, response: str) -> tuple[bool, bool, list[str], list[str]]:
    text = response.lower()
    failed_contains = [e for e in case.expected_contains if e.lower() not in text]
    failed_forbidden = [f for f in case.forbidden if f.lower() in text]
    return not failed_contains, not failed_forbidden, failed_contains, failed_forbidden


def _judge(case: EvalCase, response: str, client, model: str) -> tuple[int, str]:
    """Ask Claude to score the response 1-5 against the rubric."""
    if not case.judge_rubric:
        return 0, ""
    prompt = (
        f"Rate this AI response from 1 (very poor) to 5 (excellent) against this rubric:\n\n"
        f"**Rubric:** {case.judge_rubric}\n\n"
        f"**Question:** {case.prompt}\n\n"
        f"**Response:** {response[:1500]}\n\n"
        f"Reply with JSON: {{\"score\": <1-5>, \"reasoning\": \"<one sentence>\"}}"
    )
    try:
        import json
        resp = client.messages.create(
            model=model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        data = json.loads(text[text.find("{"):text.rfind("}") + 1])
        return int(data.get("score", 0)), data.get("reasoning", "")
    except Exception as e:
        return 0, f"judge_error: {e}"


def run_suite(
    cases: list[EvalCase],
    settings,
    use_judge: bool = False,
    tags_filter: list[str] | None = None,
) -> list[EvalResult]:
    """Run all eval cases and return results."""
    import anthropic
    from jarvis.tools.registry import build_registry
    from jarvis.agents.researcher import ResearcherAgent

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    tool_schemas, tool_registry = build_registry(
        tavily_api_key=settings.tavily_api_key,
        reports_dir=settings.reports_dir,
        anthropic_api_key=settings.anthropic_api_key,
    )

    filtered = [c for c in cases if not tags_filter or any(t in c.tags for t in tags_filter)]
    results: list[EvalResult] = []

    for case in filtered:
        agent = ResearcherAgent(
            client=client,
            model=settings.model,
            max_tokens=settings.max_tokens,
            tool_schemas=tool_schemas,
            tool_registry=tool_registry,
        )
        t0 = time.perf_counter()
        error = ""
        response = ""
        try:
            messages = [{"role": "user", "content": case.prompt}]
            response, _ = agent.run_turn(messages)
        except Exception as e:
            error = str(e)

        latency = time.perf_counter() - t0
        usage = agent.get_usage_summary()
        contains_ok, forbidden_ok, failed_c, failed_f = _score_case(case, response)

        judge_score, judge_reason = None, ""
        if use_judge and case.judge_rubric and response:
            judge_score, judge_reason = _judge(case, response, client, settings.model)
            judge_score = judge_score or None

        results.append(EvalResult(
            case_id=case.id,
            prompt=case.prompt,
            response=response,
            contains_pass=contains_ok,
            forbidden_pass=forbidden_ok,
            overall_pass=contains_ok and forbidden_ok and not error,
            latency_s=round(latency, 2),
            cost_usd=usage["estimated_cost_usd"],
            failed_contains=failed_c,
            failed_forbidden=failed_f,
            judge_score=judge_score,
            judge_reasoning=judge_reason,
            error=error,
        ))

    return results


def summarize(results: list[EvalResult]) -> dict:
    total = len(results)
    passed = sum(1 for r in results if r.overall_pass)
    avg_latency = sum(r.latency_s for r in results) / total if total else 0
    total_cost = sum(r.cost_usd for r in results)
    judge_scores = [r.judge_score for r in results if r.judge_score]
    avg_judge = sum(judge_scores) / len(judge_scores) if judge_scores else None
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total, 3) if total else 0,
        "avg_latency_s": round(avg_latency, 2),
        "total_cost_usd": round(total_cost, 6),
        "avg_judge_score": round(avg_judge, 2) if avg_judge else None,
    }
