"""
Unit tests for VaxGuard Observability, OpenTelemetry, Prometheus & Structured JSON Audit Engine.
"""

import json
from pathlib import Path

from vaxguard.telemetry.observability import (
    AuditDecision,
    AuditRecord,
    ObservabilityManager,
    PrometheusMetrics,
    StructuredAuditLogger,
    trace_guardrail_eval,
)


def test_structured_audit_record_serialization_and_logging(tmp_path: Path):
    """Verify AuditRecord serializes to strict JSON and writes to disk."""
    log_file = tmp_path / "vaxguard_audit.jsonl"
    logger = StructuredAuditLogger(log_file=str(log_file))

    rec = AuditRecord(
        client_type="openai",
        decision=AuditDecision.BLOCK,
        rule_id="rule_jailbreak_dan",
        category="jailbreak",
        confidence=0.92,
        latency_ms=0.45,
        input_length_chars=120,
        circuit_breaker_tripped=False,
    )

    logger.log(rec)

    # In-memory retrieval
    recent = logger.get_recent_records(limit=10)
    assert len(recent) == 1
    assert recent[0].rule_id == "rule_jailbreak_dan"
    assert recent[0].decision == AuditDecision.BLOCK

    # Disk file verification
    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["rule_id"] == "rule_jailbreak_dan"
    assert parsed["decision"] == "block"
    assert parsed["client_type"] == "openai"


def test_prometheus_metrics_counter_and_histogram_generation():
    """Verify PrometheusMetrics generates standard exposition format text."""
    prom = PrometheusMetrics()

    # Record 2 requests (1 allow, 1 block)
    prom.record_request(
        decision="allow",
        category="clean",
        rule_id="none",
        latency_seconds=0.00035,
        client="fastapi",
    )
    prom.record_request(
        decision="block",
        category="prompt_injection",
        rule_id="rule_injection_im_start",
        latency_seconds=0.00065,
        client="fastapi",
    )
    prom.record_streaming_chunk(tripped=True, reason="API key leak")

    text_output = prom.export_text_format()

    # Verify standard Prometheus exposition headers
    assert "# HELP vaxguard_requests_total" in text_output
    assert "# TYPE vaxguard_requests_total counter" in text_output
    assert "# HELP vaxguard_evaluation_latency_seconds" in text_output
    assert "# TYPE vaxguard_evaluation_latency_seconds histogram" in text_output

    # Verify metric lines
    assert 'vaxguard_requests_total{category="clean",client="fastapi",decision="allow",rule_id="none"} 1' in text_output
    assert 'vaxguard_requests_total{category="prompt_injection",client="fastapi",decision="block",rule_id="rule_injection_im_start"} 1' in text_output
    assert 'vaxguard_violations_total{action="block",category="prompt_injection",rule_id="rule_injection_im_start"} 1' in text_output
    assert "vaxguard_streaming_chunks_total 1" in text_output
    assert 'vaxguard_circuit_breaker_tripped_total{reason="API key leak"} 1' in text_output
    assert "vaxguard_evaluation_latency_seconds_bucket" in text_output


def test_trace_guardrail_eval_graceful_execution():
    """Verify context manager works without error in both native and fallback environments."""
    with trace_guardrail_eval("test.eval", client="anthropic", session_id="sess_123") as span:
        span.set_attribute("vaxguard.test_key", "test_val")
        # simulated work
        val = 1 + 1
        assert val == 2


def test_unified_observability_manager():
    """Verify ObservabilityManager records to both audit log and Prometheus simultaneously."""
    mgr = ObservabilityManager()
    mgr.prometheus.reset()
    mgr.audit_logger.clear()

    rec = mgr.record_evaluation(
        decision=AuditDecision.BLOCK,
        rule_id="rule_crescendo_stop",
        category="crescendo",
        confidence=0.88,
        latency_ms=0.62,
        input_length=85,
        client_type="langchain",
    )

    assert rec.decision == AuditDecision.BLOCK
    assert len(mgr.audit_logger.get_recent_records()) == 1

    prom_text = mgr.export_metrics()
    assert 'rule_crescendo_stop' in prom_text
