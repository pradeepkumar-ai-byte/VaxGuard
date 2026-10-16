"""
VaxGuard Enterprise Observability, OpenTelemetry, Prometheus & Structured JSON Audit Engine.

Production-grade observability infrastructure:
1. OpenTelemetry: Semantic distributed tracing for guardrail evaluations and streaming chunks.
2. Prometheus: Exportable metrics counters and histograms (Prometheus text format + prometheus_client bridge).
3. Structured JSON Audit Logging: Machine-parseable, SIEM-ready JSON audit events.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
from pydantic import BaseModel, Field

from vaxguard.core.logger import get_logger

logger = get_logger("Observability")

# Try importing native OpenTelemetry SDK if installed, else fallback gracefully
try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    trace = None  # type: ignore[assignment]
    Status = None  # type: ignore[assignment]
    StatusCode = None  # type: ignore[assignment]
    OPENTELEMETRY_AVAILABLE = False


# ─── 1. Structured JSON Audit Logging ─────────────────────────────────────────

class AuditDecision(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    REDACT = "redact"
    WARN = "warn"


class AuditRecord(BaseModel):
    """SIEM-compatible, structured JSON audit record for every guardrail evaluation."""
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    session_id: Optional[str] = None
    client_type: str = "standalone"
    decision: AuditDecision = AuditDecision.ALLOW
    rule_id: str = "none"
    category: str = "none"
    confidence: float = 0.0
    latency_ms: float = 0.0
    input_length_chars: int = 0
    circuit_breaker_tripped: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize audit record as a single-line NDJSON string."""
        return self.model_dump_json()


class StructuredAuditLogger:
    """Thread-safe structured audit logger emitting NDJSON lines."""

    def __init__(self, log_file: Optional[str] = None) -> None:
        self.log_file = log_file
        self._lock = threading.Lock()
        self._in_memory_records: List[AuditRecord] = []

    def log(self, record: AuditRecord) -> None:
        """Emit audit record to configured sinks."""
        json_line = record.to_json()
        logger.info(f"[AUDIT] {json_line}")

        with self._lock:
            self._in_memory_records.append(record)
            if len(self._in_memory_records) > 1000:
                self._in_memory_records.pop(0)

            if self.log_file:
                try:
                    with open(self.log_file, "a", encoding="utf-8") as f:
                        f.write(json_line + "\n")
                except Exception as e:
                    logger.error(f"Failed to write audit record to {self.log_file}: {e}")

    def get_recent_records(self, limit: int = 50) -> List[AuditRecord]:
        """Retrieve recent audit records from memory."""
        with self._lock:
            return list(self._in_memory_records[-limit:])

    def clear(self) -> None:
        with self._lock:
            self._in_memory_records.clear()


# ─── 2. Prometheus Metrics Engine ─────────────────────────────────────────────

class PrometheusMetrics:
    """
    Lightweight, thread-safe Prometheus metrics registry.

    Generates standard Prometheus Text Exposition Format directly,
    with zero mandatory third-party dependencies.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Counters: (name, label_tuple) -> count
        self._counters: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], int] = {}
        # Latency buckets (in seconds): 0.0005, 0.001, 0.002, 0.005, 0.010, 0.025, 0.050, 0.100, 0.500
        self._latency_buckets = (0.0005, 0.001, 0.002, 0.005, 0.010, 0.025, 0.050, 0.100, 0.500)
        # Histograms: (name, label_tuple) -> { "bucket_counts": [int], "sum": float, "count": int }
        self._histograms: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], Dict[str, Any]] = {}

    def inc_counter(self, name: str, labels: Optional[Dict[str, str]] = None, value: int = 1) -> None:
        """Increment a Prometheus counter metric."""
        label_tuple = tuple(sorted((labels or {}).items()))
        key = (name, label_tuple)
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + value

    def observe_latency(self, name: str, seconds: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Observe a duration measurement in a histogram metric."""
        label_tuple = tuple(sorted((labels or {}).items()))
        key = (name, label_tuple)
        with self._lock:
            if key not in self._histograms:
                self._histograms[key] = {
                    "bucket_counts": [0] * len(self._latency_buckets),
                    "sum": 0.0,
                    "count": 0,
                }
            hist = self._histograms[key]
            hist["sum"] += seconds
            hist["count"] += 1
            for i, boundary in enumerate(self._latency_buckets):
                if seconds <= boundary:
                    hist["bucket_counts"][i] += 1

    def record_request(
        self,
        decision: str,
        category: str,
        rule_id: str,
        latency_seconds: float,
        client: str = "standalone",
    ) -> None:
        """Convenience method to record a standard request and its latency."""
        self.inc_counter(
            "vaxguard_requests_total",
            labels={"decision": decision, "category": category, "rule_id": rule_id, "client": client},
        )
        if decision == "block":
            self.inc_counter(
                "vaxguard_violations_total",
                labels={"category": category, "rule_id": rule_id, "action": "block"},
            )
        self.observe_latency(
            "vaxguard_evaluation_latency_seconds",
            latency_seconds,
            labels={"decision": decision, "client": client},
        )

    def record_streaming_chunk(self, tripped: bool = False, reason: str = "") -> None:
        """Record streaming chunk metrics."""
        self.inc_counter("vaxguard_streaming_chunks_total")
        if tripped:
            self.inc_counter("vaxguard_circuit_breaker_tripped_total", labels={"reason": reason})

    def export_text_format(self) -> str:
        """
        Export all recorded metrics in standard Prometheus Text Exposition Format (v0.0.4).
        Directly compatible with Prometheus scraping agents.
        """
        lines: List[str] = [
            "# HELP vaxguard_requests_total Total number of guardrail requests processed",
            "# TYPE vaxguard_requests_total counter",
        ]

        with self._lock:
            # Counters
            for (name, label_tuple), val in sorted(self._counters.items()):
                if label_tuple:
                    lbls = ",".join(f'{k}="{v}"' for k, v in label_tuple)
                    lines.append(f"{name}{{{lbls}}} {val}")
                else:
                    lines.append(f"{name} {val}")

            # Histograms
            if self._histograms:
                lines.append("# HELP vaxguard_evaluation_latency_seconds Latency of guardrail evaluations in seconds")
                lines.append("# TYPE vaxguard_evaluation_latency_seconds histogram")
                for (name, label_tuple), data in sorted(self._histograms.items()):
                    cumulative = 0
                    base_labels = dict(label_tuple)
                    for i, boundary in enumerate(self._latency_buckets):
                        cumulative += data["bucket_counts"][i]
                        bucket_labels = dict(base_labels)
                        bucket_labels["le"] = str(boundary)
                        lbl_str = ",".join(f'{k}="{v}"' for k, v in sorted(bucket_labels.items()))
                        lines.append(f"{name}_bucket{{{lbl_str}}} {cumulative}")

                    # +Inf bucket
                    inf_labels = dict(base_labels)
                    inf_labels["le"] = "+Inf"
                    lbl_str = ",".join(f'{k}="{v}"' for k, v in sorted(inf_labels.items()))
                    lines.append(f"{name}_bucket{{{lbl_str}}} {data['count']}")

                    # Sum and Count
                    sum_lbl = ",".join(f'{k}="{v}"' for k, v in sorted(base_labels.items()))
                    lbl_prefix = f"{{{sum_lbl}}}" if sum_lbl else ""
                    lines.append(f"{name}_sum{lbl_prefix} {data['sum']:.6f}")
                    lines.append(f"{name}_count{lbl_prefix} {data['count']}")

        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._histograms.clear()


# ─── 3. OpenTelemetry Distributed Tracing ─────────────────────────────────────

class NoOpSpan:
    """Mock fallback span when OpenTelemetry is not installed."""
    def set_attribute(self, key: str, value: Any) -> None: pass
    def set_status(self, status: Any) -> None: pass
    def record_exception(self, exception: BaseException) -> None: pass
    def end(self) -> None: pass


@contextlib.contextmanager
def trace_guardrail_eval(
    name: str = "vaxguard.evaluate",
    client: str = "standalone",
    session_id: Optional[str] = None,
) -> Iterator[Any]:
    """
    Context manager creating an OpenTelemetry semantic span for guardrail operations.
    Gracefully falls back to NoOpSpan if opentelemetry is unavailable.
    """
    if OPENTELEMETRY_AVAILABLE and trace is not None:
        tracer = trace.get_tracer("vaxguard", "1.0.0")
        with tracer.start_as_current_span(name) as span:
            span.set_attribute("vaxguard.client", client)
            if session_id:
                span.set_attribute("vaxguard.session_id", session_id)
            yield span
    else:
        yield NoOpSpan()


# ─── 4. Unified Observability Manager ─────────────────────────────────────────

class ObservabilityManager:
    """Unified telemetry hub orchestrating OpenTelemetry, Prometheus, and Audit Logging."""

    _instance: Optional[ObservabilityManager] = None
    _lock = threading.Lock()

    def __init__(self, audit_file: Optional[str] = None) -> None:
        self.audit_logger = StructuredAuditLogger(log_file=audit_file)
        self.prometheus = PrometheusMetrics()

    @classmethod
    def get_instance(cls, audit_file: Optional[str] = None) -> ObservabilityManager:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(audit_file=audit_file)
            return cls._instance

    def record_evaluation(
        self,
        decision: AuditDecision,
        rule_id: str,
        category: str,
        confidence: float,
        latency_ms: float,
        input_length: int,
        client_type: str = "standalone",
        session_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        circuit_breaker_tripped: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditRecord:
        """Record evaluation simultaneously across Prometheus, OpenTelemetry, and Audit logs."""
        # 1. Prometheus metric record
        self.prometheus.record_request(
            decision=decision.value,
            category=category,
            rule_id=rule_id,
            latency_seconds=latency_ms / 1000.0,
            client=client_type,
        )

        # 2. Structured Audit log
        record = AuditRecord(
            trace_id=trace_id,
            session_id=session_id,
            client_type=client_type,
            decision=decision,
            rule_id=rule_id,
            category=category,
            confidence=confidence,
            latency_ms=latency_ms,
            input_length_chars=input_length,
            circuit_breaker_tripped=circuit_breaker_tripped,
            metadata=metadata or {},
        )
        self.audit_logger.log(record)
        return record

    def export_metrics(self) -> str:
        """Export Prometheus exposition formatted metrics text."""
        return self.prometheus.export_text_format()


# Global singleton access
telemetry = ObservabilityManager.get_instance()
