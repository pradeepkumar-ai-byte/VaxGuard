"""
VaxGuard Core Modules.
"""

from vaxguard.core.telemetry_otel import (
    VaxGuardTelemetry,
    get_telemetry,
    reset_telemetry,
    Span,
    StatusCode,
)

__all__ = [
    "VaxGuardTelemetry",
    "get_telemetry",
    "reset_telemetry",
    "Span",
    "StatusCode",
]
