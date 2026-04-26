"""OpenTelemetry tracing — configure once at startup, use everywhere.

Exports spans to any OTLP-compatible backend (Grafana Tempo, Jaeger, etc.)
via gRPC. No-ops gracefully when OTel packages are not installed or
JARVIS_OTEL_ENABLED=false.

Spans emitted:
  jarvis.agent.turn        — full agent turn (may recurse for tool use)
  jarvis.tool.dispatch     — individual tool call
  jarvis.api.http          — HTTP /api/chat (via FastAPI auto-instrumentation)
  jarvis.api.websocket     — WebSocket turn

Usage:
    from jarvis.telemetry.tracing import get_tracer, setup_tracing
    setup_tracing(endpoint="http://localhost:4317")

    with get_tracer().start_as_current_span("my.span") as span:
        span.set_attribute("key", "value")
        ...
"""
from __future__ import annotations

_tracer = None


# ── No-op fallbacks ───────────────────────────────────────────────────────────

class _NoopSpan:
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def set_attribute(self, k, v): pass
    def record_exception(self, e): pass


class _NoopTracer:
    def start_as_current_span(self, name, **kw): return _NoopSpan()
    def start_span(self, name, **kw): return _NoopSpan()


# ── Setup ─────────────────────────────────────────────────────────────────────

def setup_tracing(endpoint: str) -> None:
    """Configure OTel SDK and exporter. Call once at server startup.

    Args:
        endpoint: OTLP gRPC endpoint, e.g. "http://localhost:4317".
                  If empty string, tracing is disabled (no-op).
    """
    global _tracer
    if not endpoint:
        _tracer = _NoopTracer()
        return
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        resource = Resource(attributes={"service.name": "jarvis", "service.version": "0.1.0"})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("jarvis")
    except ImportError:
        _tracer = _NoopTracer()


def instrument_fastapi(app) -> None:
    """Auto-instrument a FastAPI app. No-op if package not installed."""
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)
    except ImportError:
        pass


def get_tracer():
    """Return the active tracer (no-op if not configured)."""
    global _tracer
    if _tracer is None:
        _tracer = _NoopTracer()
    return _tracer
