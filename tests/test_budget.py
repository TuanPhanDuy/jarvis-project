"""Unit tests for per-user token budget enforcement. No API keys needed."""
from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.api.budget import (
    BudgetExceededError,
    check_budget,
    get_budget_status,
    record_spend,
    set_budget,
    _current_period,
)


class TestBudget:
    def test_no_budget_set_allows_all_spend(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        check_budget(db, "user1")  # should not raise

    def test_unlimited_budget_zero(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        set_budget(db, "user1", 0.0)  # 0 = unlimited
        record_spend(db, "user1", 999.0)
        check_budget(db, "user1")  # should not raise

    def test_under_budget_allows(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        set_budget(db, "user1", 10.0)
        record_spend(db, "user1", 5.0)
        check_budget(db, "user1")  # should not raise

    def test_over_budget_raises(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        set_budget(db, "user1", 1.0)
        record_spend(db, "user1", 1.5)
        with pytest.raises(BudgetExceededError) as exc_info:
            check_budget(db, "user1")
        assert exc_info.value.user_id == "user1"
        assert exc_info.value.budget == 1.0
        assert exc_info.value.spent >= 1.5

    def test_spend_accumulates(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        set_budget(db, "user1", 5.0)
        record_spend(db, "user1", 2.0)
        record_spend(db, "user1", 2.0)
        status = get_budget_status(db, "user1")
        assert abs(status["spent_usd"] - 4.0) < 0.0001

    def test_get_budget_status_unknown_user(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        status = get_budget_status(db, "unknown")
        assert status["monthly_budget_usd"] == 0
        assert status["spent_usd"] == 0.0
        assert status["remaining_usd"] is None

    def test_remaining_calculated_correctly(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        set_budget(db, "user1", 10.0)
        record_spend(db, "user1", 3.0)
        status = get_budget_status(db, "user1")
        assert abs(status["remaining_usd"] - 7.0) < 0.0001

    def test_current_period_format(self) -> None:
        period = _current_period()
        assert len(period) == 7          # "YYYY-MM"
        assert period[4] == "-"
        assert period[:4].isdigit()
        assert period[5:].isdigit()

    def test_period_reset_allows_spend(self, tmp_path: Path) -> None:
        """Simulate a new month by writing an old period directly."""
        import sqlite3
        db = tmp_path / "jarvis.db"
        set_budget(db, "user1", 1.0)
        # Manually set old period and maxed-out spend
        conn = sqlite3.connect(str(db))
        conn.execute("UPDATE usage_budgets SET spent_usd=999, period='2000-01' WHERE user_id='user1'")
        conn.commit()
        conn.close()
        # New period — should not raise even though spent > budget
        check_budget(db, "user1")

    def test_budget_exceeded_error_message(self, tmp_path: Path) -> None:
        db = tmp_path / "jarvis.db"
        set_budget(db, "alice", 0.50)
        record_spend(db, "alice", 0.75)
        with pytest.raises(BudgetExceededError) as exc_info:
            check_budget(db, "alice")
        assert "alice" in str(exc_info.value)
        assert "0.50" in str(exc_info.value) or "0.5" in str(exc_info.value)
