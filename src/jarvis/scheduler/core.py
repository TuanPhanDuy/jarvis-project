"""Proactive agent scheduler — runs research and monitor jobs on a cron schedule.

Uses APScheduler 3.x with a SQLite-backed job store so schedules survive restarts.
Jobs publish tasks to RabbitMQ which the worker processes asynchronously.

DB location: reports_dir/scheduler.db
"""
from __future__ import annotations

from pathlib import Path

import structlog
from jarvis.config import get_settings
from jarvis.training.crawler import ResearchCrawler
from jarvis.training.tracking import (
    complete_run,
    count_new_docs_since,
    get_last_run,
    start_run,
)

log = structlog.get_logger()

_scheduler = None


def _fire(
    event: str,
    payload: dict,
    db_path: Path,
    title: str,
    severity: str = "info",
) -> None:
    """Best-effort: push a notification and fire matching webhooks. Never raises."""
    try:
        from jarvis.events.notifications import push_notification
        push_notification(db_path, event=event, title=title, severity=severity,
                          body=str(payload))
    except Exception:
        pass
    try:
        from jarvis.events.webhooks import fire_event
        fire_event(db_path, event=event, payload=payload)
    except Exception:
        pass


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


def _auto_crawl_job(db_path_str: str, reports_dir_str: str) -> None:
    """Scheduled job: crawl AI research from the internet and index into ChromaDB."""
    run_id: int = -1
    try:
        settings = get_settings()
        db_path = Path(db_path_str)
        reports_dir = Path(reports_dir_str)
        run_id = start_run(db_path, "crawl")

        crawler = ResearchCrawler(reports_dir)
        topics = [t.strip() for t in settings.auto_training_topics.split(",")]
        max_n = settings.training_max_papers_per_source
        total = 0

        for topic in topics:
            for fetch_fn, kwargs in [
                (crawler.crawl_arxiv, {"topic": topic, "max_results": max_n}),
                (crawler.crawl_hf_blog, {"max_posts": max_n}),
                (crawler.crawl_anthropic, {"max_posts": max_n}),
                (crawler.crawl_papers_with_code, {"topic": topic, "max_results": max_n}),
            ]:
                docs = fetch_fn(**kwargs)
                names = crawler.ingest_all(docs)
                total += len([n for n in names if not n.startswith("ERROR")])

        complete_run(db_path, run_id, docs_crawled=total,
                     notes=f"topics: {settings.auto_training_topics}")
        log.info("auto_crawl_complete", docs=total)
        _fire("training.complete", {"type": "crawl", "docs": total}, db_path,
              f"Research crawl complete — {total} documents indexed")
    except Exception as exc:
        log.error("auto_crawl_failed", error=str(exc))
        _fire("tool.error", {"job": "auto_crawl", "error": str(exc)}, Path(db_path_str),
              f"Research crawl failed: {exc}", severity="error")
        try:
            if run_id >= 0:
                complete_run(Path(db_path_str), run_id, status="failed", notes=str(exc))
        except Exception:
            pass


def _auto_finetune_job(db_path_str: str, reports_dir_str: str) -> None:
    """Scheduled job: generate training data and fine-tune the model if enough new docs exist."""
    run_id: int = -1
    try:
        from jarvis.training.data_generator import TrainingDataGenerator
        from jarvis.training.dataset_manager import DatasetManager
        from jarvis.training.finetune import Finetuner
        from jarvis.training.modelfile import register_model

        settings = get_settings()
        db_path = Path(db_path_str)
        reports_dir = Path(reports_dir_str)
        data_dir = Path(settings.training_data_dir)

        last_ft = get_last_run(db_path, "finetune")
        since_ts = last_ft.completed_at if last_ft else 0.0
        new_docs = count_new_docs_since(db_path, since_ts, reports_dir)

        if new_docs < settings.auto_training_min_new_docs:
            log.info("auto_finetune_skipped", new_docs=new_docs,
                     min_required=settings.auto_training_min_new_docs)
            return

        run_id = start_run(db_path, "finetune")

        gen = TrainingDataGenerator(reports_dir)
        dataset_path = data_dir / "dataset.jsonl"
        pairs = gen.run(dataset_path, target_pairs=settings.training_target_pairs)

        dm = DatasetManager()
        dm.deduplicate(dataset_path)
        train_path, val_path = dm.split(dataset_path)

        adapter_dir = data_dir / "adapters"
        ft = Finetuner(base_model=settings.training_base_model_mlx, adapter_dir=adapter_dir)
        ft.train(data_dir, epochs=settings.training_lora_epochs,
                 lora_rank=settings.training_lora_rank)

        model_name = settings.auto_training_model_name
        gguf_path = data_dir / f"{model_name}.gguf"
        ft.export_gguf(gguf_path)
        registered = register_model(gguf_path, model_name)

        complete_run(
            db_path, run_id,
            docs_crawled=new_docs,
            pairs_generated=pairs,
            model_name=model_name if registered else "",
            notes="auto-finetune" + ("" if registered else " (registration failed)"),
        )
        log.info("auto_finetune_complete", model=model_name, pairs=pairs)
        _fire("training.complete", {"type": "finetune", "model": model_name, "pairs": pairs},
              db_path, f"Fine-tune complete — model '{model_name}' trained on {pairs} pairs")
    except Exception as exc:
        log.error("auto_finetune_failed", error=str(exc))
        _fire("tool.error", {"job": "auto_finetune", "error": str(exc)}, Path(db_path_str),
              f"Fine-tune failed: {exc}", severity="error")
        try:
            if run_id >= 0:
                complete_run(Path(db_path_str), run_id, status="failed", notes=str(exc))
        except Exception:
            pass


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
        try:
            user_ids = [row[0] for row in conn.execute(
                "SELECT DISTINCT user_id FROM entities"
            ).fetchall()]
        finally:
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


def _prune_memory_job(db_path_str: str) -> None:
    """Scheduled job: delete old episodes, feedback, preferences, and failures."""
    try:
        settings = get_settings()
        db_path = Path(db_path_str)
        retention = settings.memory_retention_days

        from jarvis.memory.episodic import prune_old_episodes
        from jarvis.memory.feedback import prune_old_feedback
        from jarvis.memory.failures import prune_old_failures
        from jarvis.memory.preferences import prune_old_preferences
        from jarvis.memory.turns import prune_old_turns
        from jarvis.security.audit import prune_old_audit

        from jarvis.memory.episodic import apply_importance_decay
        apply_importance_decay(db_path)

        ep_deleted = prune_old_episodes(db_path, retention)
        fb_deleted = prune_old_feedback(db_path, retention)
        fail_deleted = prune_old_failures(db_path, retention)
        pref_deleted = prune_old_preferences(db_path, retention)
        turns_deleted = prune_old_turns(db_path, retention)
        audit_deleted = prune_old_audit(db_path, retention)
        log.info(
            "prune_memory_done",
            episodes_deleted=ep_deleted,
            feedback_deleted=fb_deleted,
            failures_deleted=fail_deleted,
            preferences_deleted=pref_deleted,
            turns_deleted=turns_deleted,
            audit_deleted=audit_deleted,
            retention_days=retention,
        )
    except Exception as exc:
        log.error("prune_memory_failed", error=str(exc))


def _eval_check_job(db_path_str: str, reports_dir_str: str) -> None:
    """Scheduled job: run eval suite; trigger fine-tuning if pass_rate < threshold."""
    try:
        settings = get_settings()
        db_path = Path(db_path_str)
        reports_dir = Path(reports_dir_str)

        from jarvis.evals.suite import BASELINE_SUITE
        from jarvis.evals.runner import run_suite, summarize, persist_results

        results = run_suite(cases=BASELINE_SUITE, settings=settings, use_judge=False)
        summary = summarize(results)
        persist_results(results, summary, reports_dir)

        pass_rate = summary.get("pass_rate", 1.0)
        threshold = settings.eval_pass_rate_threshold
        log.info("eval_check_done", pass_rate=pass_rate, threshold=threshold)

        if pass_rate < threshold:
            log.warning("eval_pass_rate_below_threshold",
                        pass_rate=pass_rate, threshold=threshold,
                        action="triggering_finetune")
            _auto_finetune_job(db_path_str, reports_dir_str)
    except Exception as exc:
        log.error("eval_check_job_failed", error=str(exc))


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
        (
            "builtin_prune_memory",
            _prune_memory_job,
            CronTrigger(hour=3, minute=30, timezone="UTC"),
            [str(db_path)],
        ),
    ]
    for job_id, func, trigger, args in _builtin:
        if not scheduler.get_job(job_id):
            scheduler.add_job(func, trigger, args=args, id=job_id, misfire_grace_time=3600)
            log.info("builtin_job_registered", job_id=job_id)

    _register_auto_training_jobs(scheduler, db_path, reports_dir)


def _parse_cron(expr: str) -> "CronTrigger":
    """Parse a 5-field cron expression (min hour dom month dow) into a CronTrigger."""
    from apscheduler.triggers.cron import CronTrigger
    parts = expr.strip().split()
    if len(parts) != 5:
        raise ValueError(f"Invalid cron expression (expected 5 fields): {expr!r}")
    minute, hour, day, month, day_of_week = parts
    return CronTrigger(
        minute=minute, hour=hour, day=day, month=month,
        day_of_week=day_of_week, timezone="UTC",
    )


def _register_auto_training_jobs(scheduler, db_path: Path, reports_dir: Path) -> None:
    """Register auto-crawl and auto-finetune jobs if JARVIS_AUTO_TRAINING=true."""
    try:
        settings = get_settings()
        if not settings.auto_training_enabled:
            return

        crawl_trigger = _parse_cron(settings.auto_crawl_cron)
        if not scheduler.get_job("builtin_auto_crawl"):
            scheduler.add_job(
                _auto_crawl_job,
                crawl_trigger,
                args=[str(db_path), str(reports_dir)],
                id="builtin_auto_crawl",
                misfire_grace_time=3600,
            )
            log.info("auto_crawl_job_registered", cron=settings.auto_crawl_cron)

        ft_trigger = _parse_cron(settings.auto_finetune_cron)
        if not scheduler.get_job("builtin_auto_finetune"):
            scheduler.add_job(
                _auto_finetune_job,
                ft_trigger,
                args=[str(db_path), str(reports_dir)],
                id="builtin_auto_finetune",
                misfire_grace_time=7200,
            )
            log.info("auto_finetune_job_registered", cron=settings.auto_finetune_cron)

        if settings.auto_eval_enabled:
            eval_trigger = _parse_cron(settings.eval_check_cron)
            if not scheduler.get_job("builtin_eval_check"):
                scheduler.add_job(
                    _eval_check_job,
                    eval_trigger,
                    args=[str(db_path), str(reports_dir)],
                    id="builtin_eval_check",
                    misfire_grace_time=3600,
                )
                log.info("eval_check_job_registered", cron=settings.eval_check_cron)
    except Exception as exc:
        log.error("auto_training_job_registration_failed", error=str(exc))


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

def _eval_run_job(
    db_path_str: str,
    reports_dir_str: str,
    tags_json: str = "[]",
    use_judge: bool = False,
) -> None:
    """Scheduled job: run eval suite, persist results, and fire a notification."""
    import json as _json
    import time as _time
    try:
        settings = get_settings()
        db_path = Path(db_path_str)
        reports_dir = Path(reports_dir_str)
        tags = _json.loads(tags_json)

        from jarvis.evals.suite import BASELINE_SUITE
        from jarvis.evals.runner import run_suite, summarize
        from jarvis.evals.trend import record_run

        cases = [c for c in BASELINE_SUITE if (not tags or any(t in c.tags for t in tags))]
        results = run_suite(cases=cases, settings=settings, use_judge=use_judge)
        summary = summarize(results)
        run_id = f"sched-{int(_time.time())}"
        record_run(
            db_path,
            run_id=run_id,
            total=summary["total"],
            passed=summary["passed"],
            failed=summary["failed"],
            pass_rate=summary["pass_rate"],
            tags=tags,
            results=[vars(r) for r in results],
        )
        log.info("scheduled_eval_done", run_id=run_id, pass_rate=summary["pass_rate"])
    except Exception as exc:
        log.error("scheduled_eval_failed", error=str(exc))


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
    "auto_crawl": _auto_crawl_job,
    "auto_finetune": _auto_finetune_job,
    "eval_check": _eval_check_job,
    "eval_run": _eval_run_job,
    "prune_memory": _prune_memory_job,
}
