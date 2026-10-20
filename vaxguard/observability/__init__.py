"""
VaxGuard Observability Package.

Exposes OpenTelemetry semantic tracing, Prometheus metrics export,
and structured JSON audit logging.
"""

from __future__ import annotations

from vaxguard.core.telemetry_otel import (
    MetricType,
    PrometheusCounter,
    PrometheusGauge,
    PrometheusHistogram,
    Span,
    SpanEvent,
    SpanStatus,
    StatusCode,
    VaxGuardTelemetry,
    get_telemetry,
    reset_telemetry,
)

__all__ = [
    "MetricType",
    "PrometheusCounter",
    "PrometheusGauge",
    "PrometheusHistogram",
    "Span",
    "SpanEvent",
    "SpanStatus",
    "StatusCode",
    "VaxGuardTelemetry",
    "get_telemetry",
    "reset_telemetry",
]
