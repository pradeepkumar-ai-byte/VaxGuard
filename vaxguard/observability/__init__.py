"""
VaxGuard Observability Package.

Exposes OpenTelemetry semantic tracing, Prometheus metrics export,
and structured JSON audit logging.
"""

from __future__ import annotations

from vaxguard.core.telemetry_otel import (
    PrometheusRegistry,
    Span,
    SpanKind,
    SpanStatus,
    StatusCode,
    Tracer,
    get_meter,
    get_tracer,
    render_prometheus_metrics,
    trace_guardrail_eval,
)

__all__ = [
    "PrometheusRegistry",
    "Span",
    "SpanKind",
    "SpanStatus",
    "StatusCode",
    "Tracer",
    "get_meter",
    "get_tracer",
    "render_prometheus_metrics",
    "trace_guardrail_eval",
]
