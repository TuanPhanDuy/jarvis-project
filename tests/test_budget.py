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


class TestGetAllBudgetStatuses:
    def test_empty_db_returns_empty_list(self, tmp_path: Path) -> None:
        from jarvis.api.budget import get_all_budget_statuses
        db = tmp_path / "jarvis.db"
        # Create the DB without any rows
        set_budget(db, "__probe__", 0)
        # Remove the probe user row isn't easy, but we can test by creating it fresh
        db2 = tmp_path / "empty.db"
        set_budget(db2, "probe", 0)
        from jarvis.api.budget import _get_conn
        conn = _get_conn(db2)
        conn.execute("DELETE FROM usage_budgets")
        conn.commit()
        conn.close()
        result = get_all_budget_statuses(db2)
        assert result == []

    def test_returns_all_users(self, tmp_path: Path) -> None:
        from jarvis.api.budget import get_all_budget_statuses
        db = tmp_path / "jarvis.db"
        set_budget(db, "alice", 10.0)
        set_budget(db, "bob", 20.0)
        result = get_all_budget_statuses(db)
        user_ids = {r["user_id"] for r in result}
        assert "alice" in user_ids
        assert "bob" in user_ids

    def test_entry_has_required_keys(self, tmp_path: Path) -> None:
        from jarvis.api.budget import get_all_budget_statuses
        db = tmp_path / "jarvis.db"
        set_budget(db, "alice", 5.0)
        result = get_all_budget_statuses(db)
        assert len(result) == 1
        for key in ("user_id", "monthly_budget_usd", "spent_usd", "remaining_usd", "period"):
            assert key in result[0]

    def test_spent_reflected_in_all_view(self, tmp_path: Path) -> None:
        from jarvis.api.budget import get_all_budget_statuses
        db = tmp_path / "jarvis.db"
        set_budget(db, "carol", 10.0)
        record_spend(db, "carol", 3.0)
        result = get_all_budget_statuses(db)
        carol = next(r for r in result if r["user_id"] == "carol")
        assert carol["spent_usd"] == pytest.approx(3.0, abs=1e-4)
        assert carol["remaining_usd"] == pytest.approx(7.0, abs=1e-4)
