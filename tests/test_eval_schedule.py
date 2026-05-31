"""Tests for scheduled eval runs."""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jarvis.scheduler.core import _eval_run_job
from jarvis.evals.trend import get_run


class TestEvalRunJob:
    def _make_settings(self, tmp_path):
        s = MagicMock()
        s.model = "test-model"
        s.max_tokens = 512
        s.reports_dir = tmp_path
        s.eval_pass_rate_threshold = 0.8
        s.fast_model = "test-fast"
        s.routing_strategy = "always_primary"
        return s

    def _mock_run_suite(self):
        from jarvis.evals.runner import EvalResult
        return [
            EvalResult(
                case_id="c1", prompt="q", response="a",
                contains_pass=True, forbidden_pass=True, overall_pass=True,
                latency_s=0.1, cost_usd=0.0,
            )
        ]

    def test_persists_results_to_trend_db(self, tmp_path):
        db_path = tmp_path / "jarvis.db"
        with patch("jarvis.scheduler.core.get_settings", return_value=self._make_settings(tmp_path)), \
             patch("jarvis.evals.runner.run_suite", return_value=self._mock_run_suite()), \
             patch("jarvis.evals.runner.summarize", return_value={
                 "total": 1, "passed": 1, "failed": 0, "pass_rate": 1.0, "avg_latency_s": 0.1
             }):
            _eval_run_job(
                db_path_str=str(db_path),
                reports_dir_str=str(tmp_path),
                tags_json="[]",
                use_judge=False,
            )
        from jarvis.evals.trend import get_trend
        runs = get_trend(db_path)
        assert len(runs) >= 1
        sched_runs = [r for r in runs if r["run_id"].startswith("sched-")]
        assert len(sched_runs) == 1

    def test_filters_by_tags(self, tmp_path):
        db_path = tmp_path / "jarvis.db"
        captured_cases = []

        def fake_run_suite(cases, **kwargs):
            captured_cases.extend(cases)
            return self._mock_run_suite()

        with patch("jarvis.scheduler.core.get_settings", return_value=self._make_settings(tmp_path)), \
             patch("jarvis.evals.runner.run_suite", side_effect=fake_run_suite), \
             patch("jarvis.evals.runner.summarize", return_value={
                 "total": 1, "passed": 1, "failed": 0, "pass_rate": 1.0, "avg_latency_s": 0.1
             }):
            _eval_run_job(
                db_path_str=str(db_path),
                reports_dir_str=str(tmp_path),
                tags_json=json.dumps(["ml"]),
                use_judge=False,
            )
        for case in captured_cases:
            assert "ml" in case.tags

    def test_never_raises_on_error(self, tmp_path):
        # Should swallow exceptions gracefully
        _eval_run_job(
            db_path_str=str(tmp_path / "db"),
            reports_dir_str=str(tmp_path),
            tags_json="[]",
            use_judge=False,
        )

    def test_job_registered_in_job_functions(self):
        from jarvis.scheduler.core import JOB_FUNCTIONS
        assert "eval_run" in JOB_FUNCTIONS


class TestEvalRunJobUnit:
    def test_eval_run_callable(self):
        from jarvis.scheduler.core import JOB_FUNCTIONS
        assert callable(JOB_FUNCTIONS["eval_run"])

    def test_eval_run_accepts_correct_kwargs(self, tmp_path):
        """Verify the function signature accepts the expected keyword arguments."""
        import inspect
        from jarvis.scheduler.core import _eval_run_job
        sig = inspect.signature(_eval_run_job)
        params = set(sig.parameters.keys())
        assert "db_path_str" in params
        assert "reports_dir_str" in params
        assert "tags_json" in params
        assert "use_judge" in params
