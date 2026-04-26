"""RabbitMQ task consumer — core processing logic.

Processes a single QueueTask: builds an agent, runs the turn, returns QueueResult.
Used by the worker process.
"""
from __future__ import annotations

import structlog

import anthropic

from jarvis.api.metrics import QUEUE_TASKS_PROCESSED, TOOL_CALLS_TOTAL, record_usage
from jarvis.api.models import QueueResult, QueueTask, UsageSummary
from jarvis.agents.planner import PlannerAgent
from jarvis.agents.researcher import ResearcherAgent
from jarvis.config import get_settings
from jarvis.tools.registry import build_planner_registry, build_registry

log = structlog.get_logger()


def process_task(task: QueueTask) -> QueueResult:
    """Process a single task synchronously and return the result.

    This is called from within a worker thread/process.
    """
    log.info("task_started", task_id=task.task_id, session_id=task.session_id)

    try:
        settings = get_settings()
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        base_schemas, base_registry = build_registry(
            tavily_api_key=settings.tavily_api_key,
            reports_dir=settings.reports_dir,
            allowed_commands=settings.allowed_commands,
            anthropic_api_key=settings.anthropic_api_key,
        )

        if task.researcher_mode:
            agent: PlannerAgent | ResearcherAgent = ResearcherAgent(
                client=client,
                model=settings.model,
                max_tokens=settings.max_tokens,
                tool_schemas=base_schemas,
                tool_registry=base_registry,
                max_search_calls=settings.max_search_calls,
            )
        else:
            planner_schemas, planner_registry = build_planner_registry(
                base_schemas=base_schemas,
                base_registry=base_registry,
                client=client,
                model=settings.model,
                max_tokens=settings.max_tokens,
            )
            agent = PlannerAgent(
                client=client,
                model=settings.model,
                max_tokens=settings.max_tokens,
                tool_schemas=planner_schemas,
                tool_registry=planner_registry,
            )

        # Instrument tool calls for metrics
        original_dispatch = agent._before_dispatch

        def instrumented(name: str, tool_input: dict) -> None:
            TOOL_CALLS_TOTAL.labels(tool_name=name).inc()
            original_dispatch(name, tool_input)

        agent._before_dispatch = instrumented  # type: ignore[method-assign]

        messages = [{"role": "user", "content": task.message}]
        response_text, _ = agent.run_turn(messages)

        usage = agent.get_usage_summary()
        record_usage(usage)
        QUEUE_TASKS_PROCESSED.labels(status="success").inc()

        log.info("task_complete", task_id=task.task_id)
        return QueueResult(
            task_id=task.task_id,
            session_id=task.session_id,
            response=response_text,
            usage=UsageSummary(**usage),
        )

    except Exception as exc:
        QUEUE_TASKS_PROCESSED.labels(status="error").inc()
        log.error("task_failed", task_id=task.task_id, error=str(exc))
        return QueueResult(
            task_id=task.task_id,
            session_id=task.session_id,
            response="",
            usage=UsageSummary(
                input_tokens=0, output_tokens=0,
                cache_write_tokens=0, cache_read_tokens=0,
                estimated_cost_usd=0.0,
            ),
            error=str(exc),
        )
