"""ExecutorAgent — runs a multi-step plan with dependency tracking and critique.

Each step is run by the appropriate specialist sub-agent. Results from completed
steps are injected as context for dependent downstream steps. CriticAgent
scores each output and triggers one retry for low-quality results.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from collections.abc import Callable
from pathlib import Path

import structlog

log = structlog.get_logger()


@dataclass
class PlanStep:
    id: str
    description: str
    agent_type: str            # "researcher" | "coder" | "qa"
    depends_on: list[str] = field(default_factory=list)
    status: str = "pending"    # pending | running | done | failed
    result: str | None = None


def _get_conn(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS plans (
            id           TEXT PRIMARY KEY,
            goal         TEXT NOT NULL,
            steps_json   TEXT NOT NULL,
            status       TEXT NOT NULL DEFAULT 'running',
            created_at   REAL NOT NULL,
            completed_at REAL,
            session_id   TEXT,
            user_id      TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_plan_session ON plans(session_id);
        CREATE INDEX IF NOT EXISTS idx_plan_user    ON plans(user_id);
    """)
    conn.commit()
    return conn


def _topo_sort(steps: list[PlanStep]) -> list[PlanStep]:
    by_id = {s.id: s for s in steps}
    visited: set[str] = set()
    order: list[PlanStep] = []

    def visit(step: PlanStep) -> None:
        if step.id in visited:
            return
        for dep_id in step.depends_on:
            if dep_id in by_id:
                visit(by_id[dep_id])
        visited.add(step.id)
        order.append(step)

    for s in steps:
        visit(s)
    return order


class ExecutorAgent:
    """Executes a list of PlanSteps using specialist sub-agents."""

    def __init__(
        self,
        model: str,
        max_tokens: int,
        sub_tool_schemas: list[dict],
        sub_tool_registry: dict[str, Callable[[dict], str]],
        db_path: Path | None = None,
        session_id: str = "",
        user_id: str | None = None,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._sub_tool_schemas = sub_tool_schemas
        self._sub_tool_registry = sub_tool_registry
        self._db_path = db_path
        self._session_id = session_id
        self._user_id = user_id

    def execute_plan(self, goal: str, steps: list[PlanStep]) -> str:
        plan_id = str(uuid.uuid4())
        self._store_plan(plan_id, goal, steps)

        from jarvis.agents.critic import build_critic
        critic = build_critic(self._model, self._max_tokens)

        context: dict[str, str] = {}
        sorted_steps = _topo_sort(steps)

        for step in sorted_steps:
            step.status = "running"
            self._update_plan(plan_id, steps)

            dep_context = ""
            for dep_id in step.depends_on:
                if dep_id in context:
                    dep_context += f"\n\n[Context from step '{dep_id}']:\n{context[dep_id][:1000]}"

            task = step.description
            if dep_context:
                task = f"{task}\n\nContext from previous steps:{dep_context}"

            result = self._run_step(step.agent_type, task)

            critique = critic.critique(step.description, result)
            log.info("plan_step_critiqued", plan_id=plan_id, step_id=step.id,
                     score=critique.score, retry=critique.should_retry)

            if critique.should_retry and critique.revised_task:
                revised = critique.revised_task
                if dep_context:
                    revised = f"{revised}\n\nContext from previous steps:{dep_context}"
                result = self._run_step(step.agent_type, revised)

            step.status = "failed" if result.startswith("ERROR:") else "done"
            step.result = result
            context[step.id] = result
            self._update_plan(plan_id, steps)

        any_failed = any(s.status == "failed" for s in sorted_steps)
        self._complete_plan(plan_id, steps, failed=any_failed)

        parts = [f"**Plan: {goal}**\n"]
        for step in sorted_steps:
            parts.append(f"### Step: {step.description}")
            parts.append(step.result or "(no result)")
            parts.append("")
        return "\n".join(parts)

    def _run_step(self, agent_type: str, task: str) -> str:
        from jarvis.agents.coder import CoderAgent
        from jarvis.agents.qa import QAAgent
        from jarvis.agents.researcher import ResearcherAgent

        agent_classes = {
            "researcher": ResearcherAgent,
            "coder": CoderAgent,
            "qa": QAAgent,
        }
        AgentClass = agent_classes.get(agent_type)
        if AgentClass is None:
            return f"ERROR: unknown agent_type '{agent_type}'"
        try:
            agent = AgentClass(
                model=self._model,
                max_tokens=self._max_tokens,
                tool_schemas=self._sub_tool_schemas,
                tool_registry=self._sub_tool_registry,
                session_id=self._session_id,
                user_id=self._user_id,
            )
            result, _ = agent.run_turn([{"role": "user", "content": task}])
            return result
        except Exception as exc:
            return f"ERROR: step failed — {exc}"

    def _store_plan(self, plan_id: str, goal: str, steps: list[PlanStep]) -> None:
        if not self._db_path:
            return
        try:
            conn = _get_conn(self._db_path)
            conn.execute(
                "INSERT INTO plans (id, goal, steps_json, status, created_at, session_id, user_id) VALUES (?,?,?,?,?,?,?)",
                (plan_id, goal, json.dumps([s.__dict__ for s in steps]), "running",
                 time.time(), self._session_id, self._user_id),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

    def _update_plan(self, plan_id: str, steps: list[PlanStep]) -> None:
        if not self._db_path:
            return
        try:
            conn = _get_conn(self._db_path)
            conn.execute("UPDATE plans SET steps_json = ? WHERE id = ?",
                         (json.dumps([s.__dict__ for s in steps]), plan_id))
            conn.commit()
            conn.close()
        except Exception:
            pass

    def _complete_plan(self, plan_id: str, steps: list[PlanStep], failed: bool = False) -> None:
        if not self._db_path:
            return
        try:
            status = "partial_failure" if failed else "done"
            conn = _get_conn(self._db_path)
            conn.execute(
                "UPDATE plans SET status = ?, completed_at = ?, steps_json = ? WHERE id = ?",
                (status, time.time(), json.dumps([s.__dict__ for s in steps]), plan_id),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass
