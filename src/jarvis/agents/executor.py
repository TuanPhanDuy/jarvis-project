"""ExecutorAgent — runs a multi-step plan with dependency tracking and critique.

Each step is run by the appropriate specialist sub-agent. Results from completed
steps are injected as context for dependent downstream steps. CriticAgent
scores each output and triggers one retry for low-quality results.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import structlog

from jarvis.agents.step_summarizer import summarize_if_large

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


def _topo_levels(steps: list[PlanStep]) -> list[list[PlanStep]]:
    """Group steps into dependency levels; all steps in a level can run in parallel."""
    by_id = {s.id: s for s in steps}
    levels: list[list[PlanStep]] = []
    assigned: set[str] = set()

    while len(assigned) < len(steps):
        level = [
            s for s in steps
            if s.id not in assigned
            and all(dep in assigned for dep in s.depends_on if dep in by_id)
        ]
        if not level:
            level = [s for s in steps if s.id not in assigned]
        for s in level:
            assigned.add(s.id)
        levels.append(level)
    return levels


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
        context_lock = threading.Lock()
        levels = _topo_levels(steps)
        all_steps_ordered: list[PlanStep] = []

        for level in levels:
            for step in level:
                step.status = "running"
            self._update_plan(plan_id, steps)

            def _run_level_step(step: PlanStep) -> None:
                with context_lock:
                    dep_context = "".join(
                        f"\n\n[Context from step '{dep_id}']:\n"
                        f"{summarize_if_large(context[dep_id], dep_id, self._model)}"
                        for dep_id in step.depends_on
                        if dep_id in context
                    )

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
                with context_lock:
                    context[step.id] = result

            if len(level) == 1:
                _run_level_step(level[0])
            else:
                with ThreadPoolExecutor(max_workers=len(level)) as pool:
                    futures = {pool.submit(_run_level_step, step): step for step in level}
                    for future in as_completed(futures):
                        future.result()

            self._update_plan(plan_id, steps)
            all_steps_ordered.extend(level)

        any_failed = any(s.status == "failed" for s in steps)
        self._complete_plan(plan_id, steps, failed=any_failed)

        parts = [f"**Plan: {goal}**\n"]
        for step in all_steps_ordered:
            parts.append(f"### Step: {step.description}")
            parts.append(step.result or "(no result)")
            parts.append("")

        verification = self._verify_goal(goal, all_steps_ordered)
        if verification:
            parts.append(f"## Goal Verification\n{verification}")
            gap_result = self._fill_gaps(goal, verification, all_steps_ordered)
            if gap_result:
                parts.append(f"## Gap Remediation\n{gap_result}")

        return "\n".join(parts)

    def _run_step(self, agent_type: str, task: str) -> str:
        from jarvis.agents.coder import CoderAgent
        from jarvis.agents.data_analyst import DataAnalystAgent
        from jarvis.agents.devops import DevOpsAgent
        from jarvis.agents.qa import QAAgent
        from jarvis.agents.researcher import ResearcherAgent

        agent_classes = {
            "researcher": ResearcherAgent,
            "coder": CoderAgent,
            "qa": QAAgent,
            "analyst": DataAnalystAgent,
            "devops": DevOpsAgent,
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

    def _verify_goal(self, goal: str, steps: list[PlanStep]) -> str:
        """Check whether completed steps fully achieved the goal; return gap analysis or empty string."""
        try:
            from jarvis.config import get_settings
            if not get_settings().goal_verification_enabled:
                return ""
        except Exception:
            pass

        results_summary = "\n".join(
            f"Step '{s.description}' [{s.status}]: {(s.result or '')[:300]}"
            for s in steps
        )
        prompt = (
            f"Original goal: {goal}\n\n"
            f"Completed steps and results:\n{results_summary}\n\n"
            "Did the completed steps fully achieve the original goal?\n"
            "If YES, reply: ACHIEVED\n"
            "If NO, reply: GAPS: <one-sentence description of what remains>"
        )
        try:
            import ollama
            resp = ollama.chat(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1, "num_predict": 128},
            )
            verdict = (resp.message.content or "").strip()
            if verdict.upper().startswith("ACHIEVED"):
                log.info("goal_achieved", goal=goal[:80])
                return ""
            log.info("goal_gaps_found", goal=goal[:80], verdict=verdict[:120])
            return verdict
        except Exception as exc:
            log.debug("goal_verification_skipped", error=str(exc))
            return ""

    def _fill_gaps(self, goal: str, gaps: str, completed: list[PlanStep]) -> str:
        """Generate and execute one remediation step to address identified gaps.

        Asks the LLM to propose a single targeted task, then executes it.
        Returns empty string if gap-filling is not possible.
        """
        try:
            context_summary = "\n".join(
                f"- {s.description}: {(s.result or '')[:200]}"
                for s in completed
                if s.status == "done"
            )
            prompt = (
                f"Goal: {goal}\n\n"
                f"Gap identified after execution: {gaps}\n\n"
                f"Already completed steps:\n{context_summary}\n\n"
                "Propose ONE concise remediation task (1-2 sentences) to address the gap.\n"
                "Format your response as:\n"
                "AGENT: <researcher|coder|qa|analyst|devops>\n"
                "TASK: <task description>"
            )
            import ollama
            resp = ollama.chat(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.2, "num_predict": 150},
            )
            text = (resp.message.content or "").strip()

            agent_type = "researcher"
            task_desc = ""
            for line in text.splitlines():
                upper = line.upper()
                if upper.startswith("AGENT:"):
                    candidate = line.split(":", 1)[1].strip().lower()
                    if candidate in {"researcher", "coder", "qa", "analyst", "devops"}:
                        agent_type = candidate
                elif upper.startswith("TASK:"):
                    task_desc = line.split(":", 1)[1].strip()

            if not task_desc:
                return ""

            full_task = f"{task_desc}\n\nContext — gap to address: {gaps}"
            result = self._run_step(agent_type, full_task)
            log.info("gap_remediation_done",
                     agent_type=agent_type, task=task_desc[:60], result_len=len(result))
            return f"[{agent_type.upper()}] {result}"
        except Exception as exc:
            log.debug("gap_fill_skipped", error=str(exc))
            return ""

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
        except Exception as exc:
            log.error("plan_store_failed", plan_id=plan_id, error=str(exc))

    def _update_plan(self, plan_id: str, steps: list[PlanStep]) -> None:
        if not self._db_path:
            return
        try:
            conn = _get_conn(self._db_path)
            conn.execute("UPDATE plans SET steps_json = ? WHERE id = ?",
                         (json.dumps([s.__dict__ for s in steps]), plan_id))
            conn.commit()
            conn.close()
        except Exception as exc:
            log.error("plan_update_failed", plan_id=plan_id, error=str(exc))

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
        except Exception as exc:
            log.error("plan_complete_failed", plan_id=plan_id, error=str(exc))
