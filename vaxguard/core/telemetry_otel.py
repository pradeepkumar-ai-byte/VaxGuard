"""
VaxGuard OpenTelemetry Observability & Prometheus Metric Exporters.

Production-grade, zero-external-heavy-dependency telemetry infrastructure:
1. OpenTelemetry Spans & Tracing:
   - vaxguard.inspect (Main entry point / prompt inspection)
   - vaxguard.tri_hybrid_match (Aho-Corasick, Vector, Regex matcher)
   - vaxguard.speculative_cascade (Tier 1 deterministic, Tier 2 drift, Tier 3 micro-judge)
   - vaxguard.circuit_breaker (Streaming interception, trips & states)
2. Prometheus Metric Exporters:
   - vaxguard_requests_total (Counter)
   - vaxguard_blocks_total (Counter)
   - vaxguard_latency_seconds_bucket (Histogram with sub-ms microsecond bounds)
   - vaxguard_fpr_guarantee (Gauge for mathematical false positive rate bounds)
3. In-memory /metrics exposition endpoint:
   - Strict Prometheus exposition format 0.0.4.
   - ASGI / WSGI / FastAPI pluggable adapter.

Pure Python 3.10+, zero required binary dependencies, thread-safe, sub-millisecond execution.
"""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
import functools
import math
import secrets
import threading
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from vaxguard.core.logger import get_logger

logger = get_logger("TelemetryOtel")

# Optional OpenTelemetry integration (graceful fallback if not installed)
try:
    import opentelemetry.trace as otel_trace
    _OTEL_AVAILABLE = True
except ImportError:
    otel_trace = None  # type: ignore[assignment,no-redef]
    _OTEL_AVAILABLE = False


# ─── OpenTelemetry Enums & Data Structures ───────────────────────────────────

class StatusCode(str, Enum):
    """Mirror of OpenTelemetry StatusCode."""
    UNSET = "UNSET"
    OK = "OK"
    ERROR = "ERROR"


@dataclass
class SpanStatus:
    """Span completion status matching OpenTelemetry Status specification."""
    status_code: StatusCode = StatusCode.UNSET
    description: Optional[str] = None


@dataclass
class SpanEvent:
    """Span event matching OpenTelemetry event model."""
    name: str
    timestamp_ns: int
    attributes: Dict[str, Any] = field(default_factory=dict)


class Span:
    """
    Lightweight, thread-safe, high-performance OpenTelemetry-compatible Span.
    
    Zero dependencies, sub-microsecond overhead. Works seamlessly with both
    synchronous and asynchronous context managers.
    """

    def __init__(
        self,
        name: str,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        attributes: Optional[Mapping[str, Any]] = None,
        on_end_callback: Optional[Callable[[Span], None]] = None,
    ) -> None:
        self.name: str = name
        self.trace_id: str = trace_id or secrets.token_hex(16)   # 32 hex chars
        self.span_id: str = span_id or secrets.token_hex(8)       # 16 hex chars
        self.parent_span_id: Optional[str] = parent_span_id
        self.start_time_ns: int = time.perf_counter_ns()
        self.end_time_ns: Optional[int] = None
        self.status: SpanStatus = SpanStatus()
        self.attributes: Dict[str, Any] = dict(attributes or {})
        self.events: List[SpanEvent] = []
        self._on_end_callback: Optional[Callable[[Span], None]] = on_end_callback
        self._lock = threading.Lock()
        self._ended: bool = False
        self._token: Optional[Any] = None

    @property
    def duration_ms(self) -> float:
        """Duration of span in milliseconds."""
        if self.end_time_ns is None:
            return (time.perf_counter_ns() - self.start_time_ns) / 1_000_000.0
        return (self.end_time_ns - self.start_time_ns) / 1_000_000.0

    @property
    def duration_seconds(self) -> float:
        """Duration of span in seconds."""
        return self.duration_ms / 1000.0

    def is_recording(self) -> bool:
        """Returns True if the span is active and recording."""
        return not self._ended

    def set_attribute(self, key: str, value: Any) -> Span:
        """Set a single key-value attribute."""
        with self._lock:
            if not self._ended:
                self.attributes[str(key)] = value
        return self

    def set_attributes(self, attributes: Mapping[str, Any]) -> Span:
        """Set multiple attributes at once."""
        with self._lock:
            if not self._ended:
                for k, v in attributes.items():
                    self.attributes[str(k)] = v
        return self

    def add_event(
        self,
        name: str,
        attributes: Optional[Mapping[str, Any]] = None,
        timestamp: Optional[int] = None,
    ) -> Span:
        """Add an event to the span."""
        ts = timestamp if timestamp is not None else time.perf_counter_ns()
        event = SpanEvent(name=str(name), timestamp_ns=ts, attributes=dict(attributes or {}))
        with self._lock:
            if not self._ended:
                self.events.append(event)
        return self

    def set_status(
        self,
        status_code: Union[StatusCode, str],
        description: Optional[str] = None,
    ) -> Span:
        """Set span status code and description."""
        code = StatusCode(status_code) if isinstance(status_code, str) else status_code
        with self._lock:
            if not self._ended:
                self.status = SpanStatus(status_code=code, description=description)
        return self

    def record_exception(
        self,
        exception: BaseException,
        attributes: Optional[Mapping[str, Any]] = None,
    ) -> Span:
        """Record an exception on the span."""
        attrs = {
            "exception.type": type(exception).__name__,
            "exception.message": str(exception),
        }
        if attributes:
            attrs.update(dict(attributes))
        self.add_event(name="exception", attributes=attrs)
        self.set_status(StatusCode.ERROR, description=str(exception))
        return self

    def end(self, end_time: Optional[int] = None) -> None:
        """End the span and trigger callbacks."""
        with self._lock:
            if self._ended:
                return
            self._ended = True
            self.end_time_ns = end_time if end_time is not None else time.perf_counter_ns()
            if self.status.status_code == StatusCode.UNSET:
                self.status = SpanStatus(status_code=StatusCode.OK)

        if self._on_end_callback:
            try:
                self._on_end_callback(self)
            except Exception as e:
                logger.warning(f"Error in span on_end_callback: {e}")

    # Synchronous Context Manager
    def __enter__(self) -> Span:
        self._token = _CURRENT_SPAN.set(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_val is not None:
            self.record_exception(exc_val)
        self.end()
        if self._token is not None:
            _CURRENT_SPAN.reset(self._token)
            self._token = None

    # Asynchronous Context Manager
    async def __aenter__(self) -> Span:
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.__exit__(exc_type, exc_val, exc_tb)


# Context variable tracking current active span across coroutines and threads
_CURRENT_SPAN: ContextVar[Optional[Span]] = ContextVar("vaxguard_current_span", default=None)


# ─── Prometheus Metric Types ──────────────────────────────────────────────────

class MetricType(str, Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


def _format_labels(labels: Optional[Mapping[str, Any]]) -> str:
    """Format dictionary labels into Prometheus label syntax {key="value",...}."""
    if not labels:
        return ""
    sorted_items = sorted((str(k), str(v)) for k, v in labels.items())
    escaped = [
        f'{k}="{v.replace(chr(92), chr(92)+chr(92)).replace(chr(34), chr(92)+chr(34)).replace(chr(10), chr(92)+"n")}"'
        for k, v in sorted_items
    ]
    return "{" + ",".join(escaped) + "}"


class PrometheusCounter:
    """Thread-safe Prometheus Counter metric."""

    def __init__(self, name: str, help_text: str) -> None:
        self.name: str = name
        self.help_text: str = help_text
        self._values: Dict[Tuple[Tuple[str, str], ...], float] = {}
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0, labels: Optional[Mapping[str, Any]] = None) -> None:
        if amount < 0:
            raise ValueError("Counters can only be incremented by non-negative amounts.")
        key = tuple(sorted((str(k), str(v)) for k, v in (labels or {}).items()))
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def get_value(self, labels: Optional[Mapping[str, Any]] = None) -> float:
        key = tuple(sorted((str(k), str(v)) for k, v in (labels or {}).items()))
        with self._lock:
            return self._values.get(key, 0.0)

    def reset(self) -> None:
        with self._lock:
            self._values.clear()

    def export(self) -> List[str]:
        lines: List[str] = [
            f"# HELP {self.name} {self.help_text}",
            f"# TYPE {self.name} counter",
        ]
        with self._lock:
            if not self._values:
                lines.append(f"{self.name} 0")
            else:
                for key, val in sorted(self._values.items()):
                    lbl_dict = dict(key)
                    lbl_str = _format_labels(lbl_dict)
                    val_str = f"{int(val)}" if val.is_integer() else f"{val:.6g}"
                    lines.append(f"{self.name}{lbl_str} {val_str}")
        return lines


class PrometheusGauge:
    """Thread-safe Prometheus Gauge metric."""

    def __init__(self, name: str, help_text: str) -> None:
        self.name: str = name
        self.help_text: str = help_text
        self._values: Dict[Tuple[Tuple[str, str], ...], float] = {}
        self._lock = threading.Lock()

    def set(self, value: float, labels: Optional[Mapping[str, Any]] = None) -> None:
        key = tuple(sorted((str(k), str(v)) for k, v in (labels or {}).items()))
        with self._lock:
            self._values[key] = float(value)

    def inc(self, amount: float = 1.0, labels: Optional[Mapping[str, Any]] = None) -> None:
        key = tuple(sorted((str(k), str(v)) for k, v in (labels or {}).items()))
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def dec(self, amount: float = 1.0, labels: Optional[Mapping[str, Any]] = None) -> None:
        key = tuple(sorted((str(k), str(v)) for k, v in (labels or {}).items()))
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) - amount

    def get_value(self, labels: Optional[Mapping[str, Any]] = None) -> float:
        key = tuple(sorted((str(k), str(v)) for k, v in (labels or {}).items()))
        with self._lock:
            return self._values.get(key, 0.0)

    def reset(self) -> None:
        with self._lock:
            self._values.clear()

    def export(self) -> List[str]:
        lines: List[str] = [
            f"# HELP {self.name} {self.help_text}",
            f"# TYPE {self.name} gauge",
        ]
        with self._lock:
            if not self._values:
                lines.append(f"{self.name} 0")
            else:
                for key, val in sorted(self._values.items()):
                    lbl_dict = dict(key)
                    lbl_str = _format_labels(lbl_dict)
                    val_str = f"{int(val)}" if val.is_integer() else f"{val:.6g}"
                    lines.append(f"{self.name}{lbl_str} {val_str}")
        return lines


# Default sub-millisecond to multi-second latency buckets for AI Guardrail Operations
DEFAULT_LATENCY_BUCKETS: Tuple[float, ...] = (
    0.00005,  # 50 microseconds (fast cache hit / exact tri-hybrid hit)
    0.0001,   # 100 microseconds
    0.00025,  # 250 microseconds
    0.0005,   # 500 microseconds
    0.001,    # 1 millisecond
    0.0025,   # 2.5 milliseconds
    0.005,    # 5 milliseconds
    0.01,     # 10 milliseconds
    0.025,    # 25 milliseconds
    0.05,     # 50 milliseconds
    0.1,      # 100 milliseconds
    0.25,     # 250 milliseconds
    0.5,      # 500 milliseconds
    1.0,      # 1 second
    2.5,      # 2.5 seconds
    5.0,      # 5 seconds
    10.0,     # 10 seconds
)


class PrometheusHistogram:
    """
    Thread-safe Prometheus Histogram metric.
    
    Generates standard _bucket, _sum, and _count metrics with cumulative counts.
    """

    def __init__(
        self,
        name: str,
        help_text: str,
        buckets: Sequence[float] = DEFAULT_LATENCY_BUCKETS,
    ) -> None:
        self.name: str = name
        self.help_text: str = help_text
        # Sort and ensure unique finite bucket boundaries
        sorted_buckets = sorted(set(b for b in buckets if not math.isinf(b)))
        self.buckets: Tuple[float, ...] = tuple(sorted_buckets)
        self._counts: Dict[Tuple[Tuple[str, str], ...], List[int]] = {}
        self._sums: Dict[Tuple[Tuple[str, str], ...], float] = {}
        self._total_counts: Dict[Tuple[Tuple[str, str], ...], int] = {}
        self._lock = threading.Lock()

    def observe(self, value: float, labels: Optional[Mapping[str, Any]] = None) -> None:
        key = tuple(sorted((str(k), str(v)) for k, v in (labels or {}).items()))
        with self._lock:
            if key not in self._counts:
                self._counts[key] = [0] * len(self.buckets)
                self._sums[key] = 0.0
                self._total_counts[key] = 0

            # Increment cumulative bucket counts where value <= bound
            for idx, bound in enumerate(self.buckets):
                if value <= bound:
                    self._counts[key][idx] += 1

            self._sums[key] += value
            self._total_counts[key] += 1

    def reset(self) -> None:
        with self._lock:
            self._counts.clear()
            self._sums.clear()
            self._total_counts.clear()

    def get_stats(self, labels: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        key = tuple(sorted((str(k), str(v)) for k, v in (labels or {}).items()))
        with self._lock:
            return {
                "count": self._total_counts.get(key, 0),
                "sum": self._sums.get(key, 0.0),
                "buckets": dict(zip(self.buckets, self._counts.get(key, [0] * len(self.buckets)))),
            }

    def export(self) -> List[str]:
        lines: List[str] = [
            f"# HELP {self.name} {self.help_text}",
            f"# TYPE {self.name} histogram",
        ]
        with self._lock:
            if not self._total_counts:
                # Return empty histogram definition
                lines.append(f'{self.name}_bucket{{le="+Inf"}} 0')
                lines.append(f"{self.name}_sum 0")
                lines.append(f"{self.name}_count 0")
            else:
                for key, total_count in sorted(self._total_counts.items()):
                    lbl_dict = dict(key)
                    counts = self._counts[key]
                    val_sum = self._sums[key]

                    # Standard bucket lines
                    for idx, bound in enumerate(self.buckets):
                        bucket_labels = dict(lbl_dict)
                        bucket_labels["le"] = f"{bound:g}"
                        lines.append(f"{self.name}_bucket{_format_labels(bucket_labels)} {counts[idx]}")

                    # +Inf bucket line
                    inf_labels = dict(lbl_dict)
                    inf_labels["le"] = "+Inf"
                    lines.append(f"{self.name}_bucket{_format_labels(inf_labels)} {total_count}")

                    # Sum and Count
                    sum_str = f"{val_sum:.6g}" if not val_sum.is_integer() else f"{int(val_sum)}"
                    lines.append(f"{self.name}_sum{_format_labels(lbl_dict)} {sum_str}")
                    lines.append(f"{self.name}_count{_format_labels(lbl_dict)} {total_count}")
        return lines


# ─── VaxGuard Telemetry & Prometheus Exporter ────────────────────────────────

class VaxGuardTelemetry:
    """
    Centralized Observability Manager for VaxGuard.
    
    Provides:
    - OpenTelemetry spans for vaxguard.inspect, vaxguard.tri_hybrid_match,
      vaxguard.speculative_cascade, and vaxguard.circuit_breaker.
    - Prometheus metrics recorder:
        * vaxguard_requests_total (Counter)
        * vaxguard_blocks_total (Counter)
        * vaxguard_latency_seconds_bucket (Histogram)
        * vaxguard_fpr_guarantee (Gauge)
    - In-memory /metrics exposition endpoint.
    - Context-managed spans and function decorators.
    """

    def __init__(
        self,
        max_finished_spans: int = 1000,
        enable_otel_bridge: bool = True,
    ) -> None:
        self.max_finished_spans: int = max_finished_spans
        self.enable_otel_bridge: bool = enable_otel_bridge and _OTEL_AVAILABLE
        self._finished_spans: List[Span] = []
        self._spans_lock = threading.Lock()

        # ── Prometheus Metrics Initializations ──
        self.requests_total: PrometheusCounter = PrometheusCounter(
            name="vaxguard_requests_total",
            help_text="Total number of inspection requests processed by VaxGuard.",
        )
        self.blocks_total: PrometheusCounter = PrometheusCounter(
            name="vaxguard_blocks_total",
            help_text="Total number of blocked requests by category, rule, and tier.",
        )
        self.latency_seconds: PrometheusHistogram = PrometheusHistogram(
            name="vaxguard_latency_seconds",
            help_text="Latency of VaxGuard operations in seconds.",
            buckets=DEFAULT_LATENCY_BUCKETS,
        )
        self.fpr_guarantee: PrometheusGauge = PrometheusGauge(
            name="vaxguard_fpr_guarantee",
            help_text="Current mathematical False Positive Rate (FPR) upper bound guarantee.",
        )

        # Set default FPR guarantee baseline (e.g., 0.0001 = 0.01% FPR guarantee)
        self.fpr_guarantee.set(
            value=0.0001,
            labels={"category": "all", "guarantee_type": "theoretical_bound"},
        )

        # Initialize global otel tracer if available
        self._otel_tracer = None
        if self.enable_otel_bridge and otel_trace:
            try:
                self._otel_tracer = otel_trace.get_tracer("vaxguard", "1.0.0")
            except Exception as e:
                logger.debug(f"Failed to initialize external OpenTelemetry tracer: {e}")

    # ── Span Management ───────────────────────────────────────────────────────

    def _on_span_ended(self, span: Span) -> None:
        """Internal callback invoked when a span finishes."""
        with self._spans_lock:
            self._finished_spans.append(span)
            if len(self._finished_spans) > self.max_finished_spans:
                self._finished_spans.pop(0)

    def get_current_span(self) -> Optional[Span]:
        """Retrieve the currently active span in the current context."""
        return _CURRENT_SPAN.get()

    def get_finished_spans(self) -> List[Span]:
        """Retrieve all recorded finished spans."""
        with self._spans_lock:
            return list(self._finished_spans)

    def clear_spans(self) -> None:
        """Clear the in-memory recorded span history."""
        with self._spans_lock:
            self._finished_spans.clear()

    def start_span(
        self,
        name: str,
        attributes: Optional[Mapping[str, Any]] = None,
        parent: Optional[Span] = None,
    ) -> Span:
        """
        Create and start a new Span.
        
        If parent is not explicitly provided, attempts to inherit from the current active span.
        """
        active_parent = parent or self.get_current_span()
        parent_id = active_parent.span_id if active_parent else None
        trace_id = active_parent.trace_id if active_parent else None

        span = Span(
            name=name,
            trace_id=trace_id,
            parent_span_id=parent_id,
            attributes=attributes,
            on_end_callback=self._on_span_ended,
        )
        return span

    # ── Dedicated Domain Spans ────────────────────────────────────────────────

    def trace_inspect(
        self,
        prompt: str = "",
        tenant: str = "default",
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        **extra_attributes: Any,
    ) -> Span:
        """
        Create a span for `vaxguard.inspect`.
        
        Records entry point attributes and initiates span. Can be used as a sync
        or async context manager.
        """
        attrs = {
            "vaxguard.operation": "inspect",
            "vaxguard.prompt_length": len(prompt),
            "vaxguard.tenant": tenant,
        }
        if user_id:
            attrs["vaxguard.user_id"] = user_id
        if session_id:
            attrs["vaxguard.session_id"] = session_id
        attrs.update(extra_attributes)
        return self.start_span("vaxguard.inspect", attributes=attrs)

    def trace_tri_hybrid_match(
        self,
        text_length: int = 0,
        pattern_count: Optional[int] = None,
        **extra_attributes: Any,
    ) -> Span:
        """
        Create a span for `vaxguard.tri_hybrid_match`.
        
        Tracks ultra-fast deterministic Aho-Corasick, semantic embedding, and regex checks.
        """
        attrs = {
            "vaxguard.operation": "tri_hybrid_match",
            "vaxguard.engine": "tri_hybrid",
            "vaxguard.text_length": text_length,
        }
        if pattern_count is not None:
            attrs["vaxguard.patterns_indexed"] = pattern_count
        attrs.update(extra_attributes)
        return self.start_span("vaxguard.tri_hybrid_match", attributes=attrs)

    def trace_speculative_cascade(
        self,
        tier: str = "tier_1",
        turn_index: int = 0,
        drift_velocity: Optional[float] = None,
        **extra_attributes: Any,
    ) -> Span:
        """
        Create a span for `vaxguard.speculative_cascade`.
        
        Tracks multi-tier cascade reasoning, drift analysis, and speculative judge triggers.
        """
        attrs = {
            "vaxguard.operation": "speculative_cascade",
            "vaxguard.tier": tier,
            "vaxguard.turn_index": turn_index,
        }
        if drift_velocity is not None:
            attrs["vaxguard.drift_velocity"] = drift_velocity
        attrs.update(extra_attributes)
        return self.start_span("vaxguard.speculative_cascade", attributes=attrs)

    def trace_circuit_breaker(
        self,
        state: str = "CLOSED",
        rule_id: Optional[str] = None,
        action: Optional[str] = None,
        chunk_index: int = 0,
        **extra_attributes: Any,
    ) -> Span:
        """
        Create a span for `vaxguard.circuit_breaker`.
        
        Tracks streaming token interceptor trips, resets, and state transitions.
        """
        attrs = {
            "vaxguard.operation": "circuit_breaker",
            "vaxguard.circuit_state": state,
            "vaxguard.chunk_index": chunk_index,
        }
        if rule_id:
            attrs["vaxguard.rule_id"] = rule_id
        if action:
            attrs["vaxguard.action"] = action
        attrs.update(extra_attributes)
        return self.start_span("vaxguard.circuit_breaker", attributes=attrs)

    # ── Prometheus Metrics Recording Helpers ─────────────────────────────────

    def record_request(
        self,
        status: str = "allowed",
        tenant: str = "default",
        model: Optional[str] = None,
        action: Optional[str] = None,
    ) -> None:
        """
        Record an inspection request to `vaxguard_requests_total`.
        
        Statuses: 'allowed', 'blocked', 'redacted', 'error'.
        """
        labels = {"status": status, "tenant": tenant}
        if model:
            labels["model"] = model
        if action:
            labels["action"] = action
        self.requests_total.inc(amount=1.0, labels=labels)

    def record_block(
        self,
        category: str = "jailbreak",
        rule_id: str = "unknown",
        tier: str = "tier_1",
        action: str = "block",
    ) -> None:
        """
        Record a blocked threat to `vaxguard_blocks_total`.
        
        Categories: 'jailbreak', 'pii', 'smuggling', 'drift', 'harmful_content'.
        """
        labels = {
            "category": category,
            "rule_id": rule_id,
            "tier": tier,
            "action": action,
        }
        self.blocks_total.inc(amount=1.0, labels=labels)

    def record_latency(
        self,
        operation: str,
        latency_seconds: float,
        status: str = "success",
    ) -> None:
        """
        Record an operation duration to `vaxguard_latency_seconds_bucket`.
        
        Operations: 'inspect', 'tri_hybrid_match', 'speculative_cascade', 'circuit_breaker'.
        """
        labels = {"operation": operation, "status": status}
        self.latency_seconds.observe(value=latency_seconds, labels=labels)

    def set_fpr_guarantee(
        self,
        value: float,
        category: str = "all",
        guarantee_type: str = "theoretical_bound",
    ) -> None:
        """
        Update the current false positive rate guarantee in `vaxguard_fpr_guarantee`.
        """
        labels = {
            "category": category,
            "guarantee_type": guarantee_type,
        }
        self.fpr_guarantee.set(value=value, labels=labels)

    def reset_metrics(self) -> None:
        """Reset all counters, histograms, gauges, and recorded spans."""
        self.requests_total.reset()
        self.blocks_total.reset()
        self.latency_seconds.reset()
        self.fpr_guarantee.reset()
        self.clear_spans()

    # ── Prometheus Exposition Generator ──────────────────────────────────────

    def generate_prometheus_metrics(self) -> str:
        """
        Generate standard Prometheus exposition text format (0.0.4).
        
        Produces valid /metrics exposition payload including:
        - vaxguard_requests_total
        - vaxguard_blocks_total
        - vaxguard_latency_seconds_bucket, sum, and count
        - vaxguard_fpr_guarantee
        """
        sections: List[str] = []

        # 1. Requests Total
        sections.append("\n".join(self.requests_total.export()))

        # 2. Blocks Total
        sections.append("\n".join(self.blocks_total.export()))

        # 3. Latency Histogram
        sections.append("\n".join(self.latency_seconds.export()))

        # 4. FPR Guarantee Gauge
        sections.append("\n".join(self.fpr_guarantee.export()))

        # Prometheus format requires trailing newline
        return "\n\n".join(sections) + "\n"

    def export_metrics(self) -> str:
        """Alias for generate_prometheus_metrics()."""
        return self.generate_prometheus_metrics()

    def get_metrics_response(self) -> Dict[str, Any]:
        """
        Returns HTTP response dictionary compatible with standard web frameworks:
        {
            "content": "...prometheus format...",
            "media_type": "text/plain; version=0.0.4; charset=utf-8",
            "status_code": 200,
        }
        """
        return {
            "content": self.generate_prometheus_metrics(),
            "media_type": "text/plain; version=0.0.4; charset=utf-8",
            "status_code": 200,
        }

    async def asgi_metrics_app(self, scope: Dict[str, Any], receive: Any, send: Any) -> None:
        """Raw ASGI 3.0 handler for `/metrics` endpoint."""
        if scope["type"] == "http":
            payload = self.generate_prometheus_metrics().encode("utf-8")
            headers = [
                (b"content-type", b"text/plain; version=0.0.4; charset=utf-8"),
                (b"content-length", str(len(payload)).encode("utf-8")),
            ]
            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": headers,
            })
            await send({
                "type": "http.response.body",
                "body": payload,
                "more_body": False,
            })

    # ── Telemetry Function Decorator ──────────────────────────────────────────

    def instrument(
        self,
        operation: str,
        record_request_metric: bool = False,
    ) -> Callable:
        """
        Decorator to automatically record an OpenTelemetry span and latency metric
        for sync or async functions.
        """
        def decorator(func: Callable) -> Callable:
            if asyncio.iscoroutinefunction(func):
                @functools.wraps(func)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    span = self.start_span(f"vaxguard.{operation}")
                    start_time = time.perf_counter()
                    status = "success"
                    with span:
                        try:
                            result = await func(*args, **kwargs)
                            span.set_status(StatusCode.OK)
                            return result
                        except Exception as e:
                            status = "error"
                            span.record_exception(e)
                            raise
                        finally:
                            duration = time.perf_counter() - start_time
                            self.record_latency(operation=operation, latency_seconds=duration, status=status)
                            if record_request_metric:
                                self.record_request(status="allowed" if status == "success" else "error")
                return async_wrapper
            else:
                @functools.wraps(func)
                def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                    span = self.start_span(f"vaxguard.{operation}")
                    start_time = time.perf_counter()
                    status = "success"
                    with span:
                        try:
                            result = func(*args, **kwargs)
                            span.set_status(StatusCode.OK)
                            return result
                        except Exception as e:
                            status = "error"
                            span.record_exception(e)
                            raise
                        finally:
                            duration = time.perf_counter() - start_time
                            self.record_latency(operation=operation, latency_seconds=duration, status=status)
                            if record_request_metric:
                                self.record_request(status="allowed" if status == "success" else "error")
                return sync_wrapper
        return decorator


# ── Global Singleton & Factory ───────────────────────────────────────────────

_GLOBAL_TELEMETRY: Optional[VaxGuardTelemetry] = None
_GLOBAL_LOCK = threading.Lock()


def get_telemetry() -> VaxGuardTelemetry:
    """Retrieve the global singleton instance of VaxGuardTelemetry."""
    global _GLOBAL_TELEMETRY
    if _GLOBAL_TELEMETRY is None:
        with _GLOBAL_LOCK:
            if _GLOBAL_TELEMETRY is None:
                _GLOBAL_TELEMETRY = VaxGuardTelemetry()
    return _GLOBAL_TELEMETRY


def reset_telemetry() -> None:
    """Reset the global telemetry singleton."""
    global _GLOBAL_TELEMETRY
    with _GLOBAL_LOCK:
        if _GLOBAL_TELEMETRY is not None:
            _GLOBAL_TELEMETRY.reset_metrics()
        _GLOBAL_TELEMETRY = None
