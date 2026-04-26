"""Prometheus metrics for JARVIS API.

All counters/gauges are module-level singletons. Import and increment them
from anywhere in the codebase.

Exposed at GET /metrics in Prometheus text format.
"""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# ── Requests ────────────────────────────────────────────────────────────────
REQUESTS_TOTAL = Counter(
    "jarvis_requests_total",
    "Total number of requests processed",
    ["mode"],  # labels: "http", "websocket", "queue"
)

REQUEST_DURATION = Histogram(
    "jarvis_request_duration_seconds",
    "Time from user message to final response",
    ["mode"],
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
)

# ── Connections ──────────────────────────────────────────────────────────────
ACTIVE_WS_CONNECTIONS = Gauge(
    "jarvis_active_websocket_connections",
    "Number of currently open WebSocket connections",
)

# ── Tokens & Cost ────────────────────────────────────────────────────────────
TOKENS_TOTAL = Counter(
    "jarvis_tokens_total",
    "Total tokens consumed",
    ["token_type"],  # labels: "input", "output", "cache_read", "cache_write"
)

COST_USD_TOTAL = Counter(
    "jarvis_cost_usd_total",
    "Estimated total API cost in USD",
)

# ── Tools ────────────────────────────────────────────────────────────────────
TOOL_CALLS_TOTAL = Counter(
    "jarvis_tool_calls_total",
    "Total number of tool invocations",
    ["tool_name"],
)

# ── Queue ────────────────────────────────────────────────────────────────────
QUEUE_TASKS_PUBLISHED = Counter(
    "jarvis_queue_tasks_published_total",
    "Tasks published to RabbitMQ",
)

QUEUE_TASKS_PROCESSED = Counter(
    "jarvis_queue_tasks_processed_total",
    "Tasks successfully processed by the worker",
    ["status"],  # "success", "error"
)


def record_usage(usage: dict) -> None:
    """Record token usage from agent.get_usage_summary() into Prometheus counters."""
    TOKENS_TOTAL.labels(token_type="input").inc(usage.get("input_tokens", 0))
    TOKENS_TOTAL.labels(token_type="output").inc(usage.get("output_tokens", 0))
    TOKENS_TOTAL.labels(token_type="cache_read").inc(usage.get("cache_read_tokens", 0))
    TOKENS_TOTAL.labels(token_type="cache_write").inc(usage.get("cache_write_tokens", 0))
    COST_USD_TOTAL.inc(usage.get("estimated_cost_usd", 0.0))
