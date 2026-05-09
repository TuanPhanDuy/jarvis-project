"""Eval runner: execute an eval suite against JARVIS and score results."""
from __future__ import annotations

import dataclasses
import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from datetime import datetime, timezone
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


def _judge(case: EvalCase, response: str, model: str) -> tuple[int, str]:
    """Score the response 1-5 against the rubric using the local Ollama model."""
    if not case.judge_rubric:
        return 0, ""
    import ollama
    prompt = (
        f"Rate this AI response from 1 (very poor) to 5 (excellent) against this rubric:\n\n"
        f"**Rubric:** {case.judge_rubric}\n\n"
        f"**Question:** {case.prompt}\n\n"
        f"**Response:** {response[:1500]}\n\n"
        f"Reply with JSON only: {{\"score\": <1-5>, \"reasoning\": \"<one sentence>\"}}"
    )
    try:
        resp = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.message.content.strip()
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
    from jarvis.tools.registry import build_registry
    from jarvis.agents.researcher import ResearcherAgent

    tool_schemas, tool_registry = build_registry(
        reports_dir=settings.reports_dir,
        vision_model=settings.vision_model,
    )

    filtered = [c for c in cases if not tags_filter or any(t in c.tags for t in tags_filter)]
    results: list[EvalResult] = []
    _pool = ThreadPoolExecutor(max_workers=1)

    for case in filtered:
        agent = ResearcherAgent(
            model=settings.model,
            max_tokens=settings.max_tokens,
            tool_schemas=tool_schemas,
            tool_registry=tool_registry,
        )
        t0 = time.perf_counter()
        error = ""
        response = ""
        try:
            future = _pool.submit(agent.run_turn, [{"role": "user", "content": case.prompt}])
            response, _ = future.result(timeout=case.timeout_seconds)
        except FuturesTimeoutError:
            error = "timed_out"
        except Exception as e:
            error = str(e)

        latency = time.perf_counter() - t0
        usage = agent.get_usage_summary()
        contains_ok, forbidden_ok, failed_c, failed_f = _score_case(case, response)

        judge_score, judge_reason = None, ""
        if use_judge and case.judge_rubric and response:
            score, reason = _judge(case, response, settings.model)
            judge_score = score or None
            judge_reason = reason

        results.append(EvalResult(
            case_id=case.id,
            prompt=case.prompt,
            response=response,
            contains_pass=contains_ok,
            forbidden_pass=forbidden_ok,
            overall_pass=contains_ok and forbidden_ok and not error,
            latency_s=round(latency, 2),
            cost_usd=usage.get("estimated_cost_usd", 0.0),
            failed_contains=failed_c,
            failed_forbidden=failed_f,
            judge_score=judge_score,
            judge_reasoning=judge_reason,
            error=error,
        ))

    return results


def persist_results(results: list[EvalResult], summary: dict, output_dir: Path) -> None:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        history_path = output_dir / "eval_history.jsonl"
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": summary,
            "results": [dataclasses.asdict(r) for r in results],
        }
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass


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
