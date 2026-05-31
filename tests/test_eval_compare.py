"""Tests for eval run comparison."""
from __future__ import annotations

import pytest

from jarvis.evals.trend import compare_runs, record_run


def _make_results(case_pass_map: dict[str, bool]) -> list[dict]:
    return [
        {"case_id": cid, "overall_pass": passed, "prompt": "", "response": "",
         "latency_s": 0.1, "cost_usd": 0.0}
        for cid, passed in case_pass_map.items()
    ]


class TestCompareRuns:
    def test_returns_none_when_run_a_missing(self, tmp_path):
        db = tmp_path / "db"
        record_run(db, "run-b", 1, 1, 0, 1.0, results=_make_results({"c1": True}))
        assert compare_runs(db, "missing", "run-b") is None

    def test_returns_none_when_run_b_missing(self, tmp_path):
        db = tmp_path / "db"
        record_run(db, "run-a", 1, 1, 0, 1.0, results=_make_results({"c1": True}))
        assert compare_runs(db, "run-a", "missing") is None

    def test_both_missing_returns_none(self, tmp_path):
        db = tmp_path / "db"
        assert compare_runs(db, "x", "y") is None

    def test_identical_runs_no_regressions(self, tmp_path):
        db = tmp_path / "db"
        cases = {"c1": True, "c2": False}
        record_run(db, "a", 2, 1, 1, 0.5, results=_make_results(cases))
        record_run(db, "b", 2, 1, 1, 0.5, results=_make_results(cases))
        result = compare_runs(db, "a", "b")
        assert result["improved"] == []
        assert result["regressed"] == []
        assert result["delta_pass_rate"] == 0.0

    def test_improvement_detected(self, tmp_path):
        db = tmp_path / "db"
        record_run(db, "a", 2, 0, 2, 0.0, results=_make_results({"c1": False, "c2": False}))
        record_run(db, "b", 2, 2, 0, 1.0, results=_make_results({"c1": True, "c2": True}))
        result = compare_runs(db, "a", "b")
        assert set(result["improved"]) == {"c1", "c2"}
        assert result["regressed"] == []
        assert result["delta_pass_rate"] == 1.0

    def test_regression_detected(self, tmp_path):
        db = tmp_path / "db"
        record_run(db, "a", 2, 2, 0, 1.0, results=_make_results({"c1": True, "c2": True}))
        record_run(db, "b", 2, 0, 2, 0.0, results=_make_results({"c1": False, "c2": False}))
        result = compare_runs(db, "a", "b")
        assert result["improved"] == []
        assert set(result["regressed"]) == {"c1", "c2"}
        assert result["delta_pass_rate"] == -1.0

    def test_mixed_improved_and_regressed(self, tmp_path):
        db = tmp_path / "db"
        record_run(db, "a", 3, 2, 1, 2/3,
                   results=_make_results({"c1": True, "c2": True, "c3": False}))
        record_run(db, "b", 3, 2, 1, 2/3,
                   results=_make_results({"c1": True, "c2": False, "c3": True}))
        result = compare_runs(db, "a", "b")
        assert result["improved"] == ["c3"]
        assert result["regressed"] == ["c2"]
        assert result["unchanged_pass"] == 1   # c1 stayed passing
        assert result["unchanged_fail"] == 0

    def test_unchanged_counts_correct(self, tmp_path):
        db = tmp_path / "db"
        record_run(db, "a", 4, 2, 2, 0.5,
                   results=_make_results({"c1": True, "c2": True, "c3": False, "c4": False}))
        record_run(db, "b", 4, 2, 2, 0.5,
                   results=_make_results({"c1": True, "c2": True, "c3": False, "c4": False}))
        result = compare_runs(db, "a", "b")
        assert result["unchanged_pass"] == 2
        assert result["unchanged_fail"] == 2

    def test_result_excludes_raw_results_from_summary(self, tmp_path):
        db = tmp_path / "db"
        record_run(db, "a", 1, 1, 0, 1.0, results=_make_results({"c1": True}))
        record_run(db, "b", 1, 1, 0, 1.0, results=_make_results({"c1": True}))
        result = compare_runs(db, "a", "b")
        # run_a and run_b summaries should not include full results list
        assert "results" not in result["run_a"]
        assert "results" not in result["run_b"]

    def test_required_keys_present(self, tmp_path):
        db = tmp_path / "db"
        record_run(db, "a", 1, 1, 0, 1.0, results=_make_results({"c1": True}))
        record_run(db, "b", 1, 1, 0, 1.0, results=_make_results({"c1": True}))
        result = compare_runs(db, "a", "b")
        for key in ("run_a", "run_b", "delta_pass_rate", "improved", "regressed",
                    "unchanged_pass", "unchanged_fail"):
            assert key in result
