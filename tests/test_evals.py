"""Unit tests for eval framework scoring logic. No API keys needed."""
from __future__ import annotations

import json
from pathlib import Path

from jarvis.evals.suite import EvalCase, BASELINE_SUITE
from jarvis.evals.runner import EvalResult, _score_case, _git_hash, persist_results, summarize


class TestEvalCase:
    def test_baseline_suite_not_empty(self) -> None:
        assert len(BASELINE_SUITE) >= 4

    def test_evalcase_defaults(self) -> None:
        case = EvalCase(id="test", prompt="What is X?")
        assert case.expected_contains == []
        assert case.forbidden == []
        assert case.tags == []
        assert case.judge_rubric == ""


class TestScoreCase:
    def _case(self, expected=None, forbidden=None):
        return EvalCase(
            id="t",
            prompt="test",
            expected_contains=expected or [],
            forbidden=forbidden or [],
        )

    def test_all_expected_present(self) -> None:
        case = self._case(expected=["reward", "human feedback"])
        ok, _, failed, _ = _score_case(case, "RLHF uses reward modeling and human feedback signals.")
        assert ok is True
        assert failed == []

    def test_missing_expected_fails(self) -> None:
        case = self._case(expected=["reward", "PPO"])
        ok, _, failed, _ = _score_case(case, "Only reward is mentioned here.")
        assert ok is False
        assert "PPO" in failed

    def test_forbidden_present_fails(self) -> None:
        case = self._case(forbidden=["I don't know"])
        _, ok, _, found = _score_case(case, "I don't know the answer.")
        assert ok is False
        assert "I don't know" in found

    def test_forbidden_absent_passes(self) -> None:
        case = self._case(forbidden=["ERROR"])
        _, ok, _, found = _score_case(case, "Here is a helpful answer.")
        assert ok is True
        assert found == []

    def test_case_insensitive_matching(self) -> None:
        case = self._case(expected=["RLHF"])
        ok, _, failed, _ = _score_case(case, "rlhf is a technique.")
        assert ok is True

    def test_empty_response_fails_expected(self) -> None:
        case = self._case(expected=["something"])
        ok, _, failed, _ = _score_case(case, "")
        assert ok is False

    def test_no_constraints_always_passes(self) -> None:
        case = self._case()
        c_ok, f_ok, failed, found = _score_case(case, "anything at all")
        assert c_ok is True
        assert f_ok is True


class TestGitHash:
    def test_returns_string(self):
        result = _git_hash()
        assert isinstance(result, str)

    def test_persist_results_includes_git_hash(self, tmp_path: Path):
        results: list[EvalResult] = []
        summary = {"total": 0, "passed": 0, "failed": 0, "pass_rate": 0}
        persist_results(results, summary, tmp_path)
        line = (tmp_path / "eval_history.jsonl").read_text().strip()
        record = json.loads(line)
        assert "git_hash" in record
        assert isinstance(record["git_hash"], str)

    def test_persist_results_appends_multiple_runs(self, tmp_path: Path):
        summary = {"total": 1, "passed": 1, "failed": 0, "pass_rate": 1.0}
        persist_results([], summary, tmp_path)
        persist_results([], summary, tmp_path)
        lines = (tmp_path / "eval_history.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2


class TestSummarize:
    def _result(self, passed: bool, latency=1.0, cost=0.001, judge=None) -> EvalResult:
        return EvalResult(
            case_id="x", prompt="p", response="r",
            contains_pass=passed, forbidden_pass=passed, overall_pass=passed,
            latency_s=latency, cost_usd=cost, judge_score=judge,
        )

    def test_all_pass(self) -> None:
        results = [self._result(True), self._result(True)]
        s = summarize(results)
        assert s["passed"] == 2
        assert s["failed"] == 0
        assert s["pass_rate"] == 1.0

    def test_partial_pass(self) -> None:
        results = [self._result(True), self._result(False), self._result(False)]
        s = summarize(results)
        assert s["passed"] == 1
        assert s["failed"] == 2
        assert abs(s["pass_rate"] - 0.333) < 0.001

    def test_empty_results(self) -> None:
        s = summarize([])
        assert s["total"] == 0
        assert s["pass_rate"] == 0

    def test_cost_summed(self) -> None:
        results = [self._result(True, cost=0.001), self._result(True, cost=0.002)]
        s = summarize(results)
        assert abs(s["total_cost_usd"] - 0.003) < 0.000001

    def test_avg_latency(self) -> None:
        results = [self._result(True, latency=2.0), self._result(True, latency=4.0)]
        s = summarize(results)
        assert s["avg_latency_s"] == 3.0

    def test_judge_score_averaged(self) -> None:
        results = [self._result(True, judge=4), self._result(True, judge=2)]
        s = summarize(results)
        assert s["avg_judge_score"] == 3.0

    def test_no_judge_scores_is_none(self) -> None:
        results = [self._result(True), self._result(False)]
        s = summarize(results)
        assert s["avg_judge_score"] is None
