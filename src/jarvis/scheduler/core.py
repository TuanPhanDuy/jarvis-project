"""Proactive agent scheduler — runs research and monitor jobs on a cron schedule.

Uses APScheduler 3.x with a SQLite-backed job store so schedules survive restarts.
Jobs publish tasks to RabbitMQ which the worker processes asynchronously.

DB location: reports_dir/scheduler.db
"""
from __future__ import annotations

from pathlib import Path

import structlog

log = structlog.get_logger()

_scheduler = None


# ── Job functions ─────────────────────────────────────────────────────────────
# Must be module-level so APScheduler can pickle/restore them across restarts.

def _research_job(topic: str, session_id: str, rabbitmq_url: str, queue_name: str) -> None:
    """Scheduled job: research a topic and save a report."""
    try:
        from jarvis.queue.producer import publish_task
        publish_task(
            message=f"Research this topic thoroughly and save a comprehensive report: {topic}",
            session_id=session_id,
            researcher_mode=True,
            rabbitmq_url=rabbitmq_url,
            queue_name=queue_name,
        )
        log.info("scheduled_research_published", topic=topic)
    except Exception as exc:
        log.error("scheduled_research_failed", topic=topic, error=str(exc))


def _monitor_job(query: str, session_id: str, rabbitmq_url: str, queue_name: str) -> None:
    """Scheduled job: search for updates matching a query and summarize."""
    try:
        from jarvis.queue.producer import publish_task
        publish_task(
            message=(
                f"Search the web for the latest news and developments about: {query}. "
                "Summarize any significant new findings and save a brief report."
            ),
            session_id=session_id,
            researcher_mode=True,
            rabbitmq_url=rabbitmq_url,
            queue_name=queue_name,
        )
        log.info("scheduled_monitor_published", query=query)
    except Exception as exc:
        log.error("scheduled_monitor_failed", query=query, error=str(exc))


def _graph_dedup_job(db_path_str: str) -> None:
    """Scheduled job: merge near-duplicate entities in the knowledge graph (all users)."""
    try:
        from pathlib import Path
        from jarvis.memory.graph_dedup import deduplicate_entities
        import sqlite3

        db_path = Path(db_path_str)
        if not db_path.exists():
            return
        conn = sqlite3.connect(str(db_path))
        user_ids = [row[0] for row in conn.execute(
            "SELECT DISTINCT user_id FROM entities"
        ).fetchall()]
        conn.close()
        total = 0
        for uid in user_ids:
            merged = deduplicate_entities(db_path, uid)
            if merged:
                log.info("graph_dedup_done", user_id=uid, merged_pairs=merged)
                total += merged
        if total:
            log.info("graph_dedup_total", merged_pairs=total)
    except Exception as exc:
        log.error("graph_dedup_failed", error=str(exc))


def _memory_consolidation_job(db_path_str: str) -> None:
    """Scheduled job: consolidate recent episodes into user preferences (all users)."""
    try:
        from pathlib import Path
        from jarvis.config import get_settings
        from jarvis.memory.consolidator import consolidate_user_memory, get_all_user_ids

        settings = get_settings()
        db_path = Path(db_path_str)
        user_ids = get_all_user_ids(db_path)
        for uid in user_ids:
            count = consolidate_user_memory(db_path, uid, settings.model)
            log.info("memory_consolidated", user_id=uid, preferences=count)
    except Exception as exc:
        log.error("memory_consolidation_failed", error=str(exc))


def _feedback_analyze_job(db_path_str: str, reports_dir_str: str) -> None:
    """Scheduled job: analyze feedback and tool failures, save improvement report."""
    try:
        from jarvis.config import get_settings
        from jarvis.evals.feedback_analyzer import run_analysis

        settings = get_settings()
        result = run_analysis(
            db_path=Path(db_path_str),
            reports_dir=Path(reports_dir_str),
            model=settings.model,
        )
        log.info("feedback_analyze_job_done", result=result)
    except Exception as exc:
        log.error("feedback_analyze_job_failed", error=str(exc))


def _system_snapshot_job(db_path_str: str) -> None:
    """Scheduled job: take a system topology snapshot into the knowledge graph."""
    try:
        from jarvis.twin.main import take_snapshot
        take_snapshot(db_path=Path(db_path_str))
        log.info("system_snapshot_done")
    except Exception as exc:
        log.error("system_snapshot_failed", error=str(exc))


def _digest_job(session_id: str, rabbitmq_url: str, queue_name: str) -> None:
    """Scheduled job: generate a morning briefing / daily digest."""
    try:
        from jarvis.queue.producer import publish_task
        publish_task(
            message=(
                "Generate a morning briefing for the user: summarize any new research reports saved, "
                "list upcoming scheduled tasks, and highlight anything notable from recent episodic memory."
            ),
            session_id=session_id,
            researcher_mode=False,
            rabbitmq_url=rabbitmq_url,
            queue_name=queue_name,
        )
        log.info("digest_published", session_id=session_id)
    except Exception as exc:
        log.error("digest_failed", error=str(exc))


def _analyze_job(topic: str, session_id: str, rabbitmq_url: str, queue_name: str) -> None:
    """Scheduled job: run a deep analysis on a topic using analyst + researcher agents in parallel."""
    try:
        from jarvis.queue.producer import publish_task
        publish_task(
            message=(
                f"Run a comprehensive analysis on: {topic}\n"
                "Use delegate_batch to parallelize: (1) researcher gathers latest data and literature, "
                "(2) analyst derives statistics and trends, (3) researcher finds competing perspectives. "
                "Synthesize all findings into a structured analysis report and save it."
            ),
            session_id=session_id,
            researcher_mode=False,
            rabbitmq_url=rabbitmq_url,
            queue_name=queue_name,
        )
        log.info("scheduled_analyze_published", topic=topic)
    except Exception as exc:
        log.error("scheduled_analyze_failed", topic=topic, error=str(exc))


def _code_review_job(repo_path: str, session_id: str, rabbitmq_url: str, queue_name: str) -> None:
    """Scheduled job: full code review using qa + devops + analyst in parallel."""
    try:
        from jarvis.queue.producer import publish_task
        publish_task(
            message=(
                f"Run a full code review of the repository at: {repo_path}\n"
                "Use delegate_batch to parallelize: (1) devops runs git log and checks recent changes, "
                "(2) qa reviews code for bugs and quality issues, (3) analyst checks test coverage metrics. "
                "Combine findings into a review report and save it."
            ),
            session_id=session_id,
            researcher_mode=False,
            rabbitmq_url=rabbitmq_url,
            queue_name=queue_name,
        )
        log.info("scheduled_code_review_published", repo_path=repo_path)
    except Exception as exc:
        log.error("scheduled_code_review_failed", repo_path=repo_path, error=str(exc))


def _pipeline_job(
    goal: str,
    session_id: str,
    rabbitmq_url: str,
    queue_name: str,
    n_research_steps: int = 3,
) -> None:
    """Scheduled job: full research-to-implementation pipeline using create_plan with 6+ steps."""
    try:
        from jarvis.queue.producer import publish_task
        publish_task(
            message=(
                f"Execute a full research-to-implementation pipeline for: {goal}\n"
                f"Use create_plan with at least {n_research_steps + 3} steps. "
                f"Start with {n_research_steps} parallel research steps covering different aspects, "
                "then implementation steps that depend on the research, "
                "then a qa review step, then a devops deployment/verification step. "
                "Each step must have a self-contained, detailed description."
            ),
            session_id=session_id,
            researcher_mode=False,
            rabbitmq_url=rabbitmq_url,
            queue_name=queue_name,
        )
        log.info("scheduled_pipeline_published", goal=goal[:80])
    except Exception as exc:
        log.error("scheduled_pipeline_failed", goal=goal[:80], error=str(exc))


# ── Scheduler lifecycle ───────────────────────────────────────────────────────

def _add_builtin_jobs(scheduler, db_path: Path, reports_dir: Path) -> None:
    """Register built-in recurring jobs on first startup (idempotent via job-id guard)."""
    from apscheduler.triggers.cron import CronTrigger

    _builtin = [
        (
            "builtin_memory_consolidate",
            _memory_consolidation_job,
            CronTrigger(hour=2, minute=0, timezone="UTC"),
            [str(db_path)],
        ),
        (
            "builtin_feedback_analyze",
            _feedback_analyze_job,
            CronTrigger(day_of_week="mon", hour=3, minute=0, timezone="UTC"),
            [str(db_path), str(reports_dir)],
        ),
        (
            "builtin_system_snapshot",
            _system_snapshot_job,
            CronTrigger(hour="*/6", minute=0, timezone="UTC"),
            [str(db_path)],
        ),
        (
            "builtin_graph_dedup",
            _graph_dedup_job,
            CronTrigger(hour=1, minute=30, timezone="UTC"),
            [str(db_path)],
        ),
    ]
    for job_id, func, trigger, args in _builtin:
        if not scheduler.get_job(job_id):
            scheduler.add_job(func, trigger, args=args, id=job_id, misfire_grace_time=3600)
            log.info("builtin_job_registered", job_id=job_id)


def start_scheduler(db_path: Path):
    """Start the background scheduler with a persistent SQLite job store."""
    global _scheduler
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore

    db_path.parent.mkdir(parents=True, exist_ok=True)
    reports_dir = db_path.parent
    jobstores = {"default": SQLAlchemyJobStore(url=f"sqlite:///{db_path}")}
    _scheduler = BackgroundScheduler(jobstores=jobstores, timezone="UTC")
    _scheduler.start()
    _add_builtin_jobs(_scheduler, db_path, reports_dir)
    job_count = len(_scheduler.get_jobs())
    log.info("scheduler_started", db=str(db_path), existing_jobs=job_count)
    return _scheduler


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        log.info("scheduler_stopped")


def get_scheduler():
    return _scheduler


# ── Job helpers ───────────────────────────────────────────────────────────────

JOB_FUNCTIONS = {
    "research": _research_job,
    "monitor": _monitor_job,
    "memory_consolidate": _memory_consolidation_job,
    "digest": _digest_job,
    "feedback_analyze": _feedback_analyze_job,
    "system_snapshot": _system_snapshot_job,
    "analyze": _analyze_job,
    "code_review": _code_review_job,
    "pipeline": _pipeline_job,
    "graph_dedup": _graph_dedup_job,
}
