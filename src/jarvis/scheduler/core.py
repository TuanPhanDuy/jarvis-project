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


# ── Scheduler lifecycle ───────────────────────────────────────────────────────

def start_scheduler(db_path: Path):
    """Start the background scheduler with a persistent SQLite job store."""
    global _scheduler
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore

    db_path.parent.mkdir(parents=True, exist_ok=True)
    jobstores = {"default": SQLAlchemyJobStore(url=f"sqlite:///{db_path}")}
    _scheduler = BackgroundScheduler(jobstores=jobstores, timezone="UTC")
    _scheduler.start()
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
}
