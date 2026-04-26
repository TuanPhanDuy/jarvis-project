"""Token budget enforcement: per-user monthly USD spending limits.

DB: reports_dir/jarvis.db — usage_budgets table.

Flow:
  1. After each agent turn, call record_spend(db_path, user_id, cost_usd).
  2. Before each turn (optional), call check_budget(db_path, user_id) — raises
     BudgetExceededError if the user has exhausted their monthly budget.
  3. Budgets reset automatically on the first call after a new calendar month.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path


class BudgetExceededError(Exception):
    def __init__(self, user_id: str, budget: float, spent: float) -> None:
        self.user_id = user_id
        self.budget = budget
        self.spent = spent
        super().__init__(
            f"User '{user_id}' has exceeded monthly budget "
            f"(${spent:.4f} spent of ${budget:.4f} limit)."
        )


def _get_conn(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS usage_budgets (
            user_id             TEXT    PRIMARY KEY,
            monthly_budget_usd  REAL    NOT NULL DEFAULT 0,   -- 0 = unlimited
            spent_usd           REAL    NOT NULL DEFAULT 0,
            period              TEXT    NOT NULL DEFAULT ''    -- 'YYYY-MM'
        )
    """)
    conn.commit()
    return conn


def _current_period() -> str:
    import datetime
    d = datetime.datetime.utcnow()
    return f"{d.year}-{d.month:02d}"


def set_budget(db_path: Path, user_id: str, monthly_budget_usd: float) -> None:
    """Set or update the monthly USD budget for a user."""
    conn = _get_conn(db_path)
    conn.execute(
        """
        INSERT INTO usage_budgets (user_id, monthly_budget_usd, spent_usd, period)
        VALUES (?, ?, 0, ?)
        ON CONFLICT(user_id) DO UPDATE SET monthly_budget_usd = excluded.monthly_budget_usd
        """,
        (user_id, monthly_budget_usd, _current_period()),
    )
    conn.commit()
    conn.close()


def record_spend(db_path: Path, user_id: str, cost_usd: float) -> None:
    """Accumulate spending for a user. Resets counter if a new month started."""
    conn = _get_conn(db_path)
    period = _current_period()
    row = conn.execute(
        "SELECT monthly_budget_usd, spent_usd, period FROM usage_budgets WHERE user_id = ?",
        (user_id,),
    ).fetchone()

    if row is None:
        conn.execute(
            "INSERT INTO usage_budgets (user_id, monthly_budget_usd, spent_usd, period) VALUES (?,0,?,?)",
            (user_id, cost_usd, period),
        )
    else:
        spent = 0.0 if row["period"] != period else row["spent_usd"]
        conn.execute(
            "UPDATE usage_budgets SET spent_usd = ?, period = ? WHERE user_id = ?",
            (spent + cost_usd, period, user_id),
        )
    conn.commit()
    conn.close()


def check_budget(db_path: Path, user_id: str) -> None:
    """Raise BudgetExceededError if user is over their monthly limit (0 = unlimited)."""
    conn = _get_conn(db_path)
    row = conn.execute(
        "SELECT monthly_budget_usd, spent_usd, period FROM usage_budgets WHERE user_id = ?",
        (user_id,),
    ).fetchone()
    conn.close()

    if row is None or row["monthly_budget_usd"] == 0:
        return  # no budget set = unlimited
    if row["period"] != _current_period():
        return  # new month, counter hasn't been reset yet — allow
    if row["spent_usd"] >= row["monthly_budget_usd"]:
        raise BudgetExceededError(user_id, row["monthly_budget_usd"], row["spent_usd"])


def get_budget_status(db_path: Path, user_id: str) -> dict:
    conn = _get_conn(db_path)
    row = conn.execute(
        "SELECT monthly_budget_usd, spent_usd, period FROM usage_budgets WHERE user_id = ?",
        (user_id,),
    ).fetchone()
    conn.close()
    if row is None:
        return {"user_id": user_id, "monthly_budget_usd": 0, "spent_usd": 0.0,
                "remaining_usd": None, "period": _current_period()}
    period = _current_period()
    spent = row["spent_usd"] if row["period"] == period else 0.0
    budget = row["monthly_budget_usd"]
    remaining = None if budget == 0 else max(0.0, budget - spent)
    return {
        "user_id": user_id,
        "monthly_budget_usd": budget,
        "spent_usd": round(spent, 6),
        "remaining_usd": round(remaining, 6) if remaining is not None else None,
        "period": period,
    }
