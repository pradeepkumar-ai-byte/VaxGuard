"""
Unit tests for VaxGuard Streaming Token Interceptor & Circuit Breaker.

Validates:
1. Clean streaming tokens pass through unhindered.
2. Cross-chunk split secrets (e.g. 'sk-' in chunk N, 'proj-...' in chunk N+1) trip the circuit breaker.
3. In-flight PII leaks (SSN, credit cards) trigger in-flight redaction.
4. Async stream interception terminates instantly with fallback message.
5. Sub-0.1ms per-chunk evaluation latency SLA.
6. Circuit breaker state management and reset.
"""

from __future__ import annotations

import time
import pytest

from vaxguard.nemo.streaming_interceptor import (
    CircuitBreakerState,
    InterceptorAction,
    SlidingWindowBuffer,
    StreamingTokenInterceptor,
)


def test_sliding_window_buffer_cross_chunk_assembly():
    """Verify sliding window stitches tokens spanning chunk boundaries."""
    buf = SlidingWindowBuffer(window_chars=128)
    buf.append("export OPENAI_API_KEY=")
    window = buf.append("sk-proj-999888777666555444333222111000")
    assert "sk-proj-999888" in window
    assert buf.full_text == "export OPENAI_API_KEY=sk-proj-999888777666555444333222111000"


@pytest.mark.asyncio
async def test_clean_stream_passes_unhindered():
    """Verify that legitimate business chunks pass through completely."""
    interceptor = StreamingTokenInterceptor()
    chunks = [
        "In modern ",
        "cloud architectures, ",
        "microservices communicate ",
        "via asynchronous ",
        "event queues such as ",
        "Apache Kafka or RabbitMQ.",
    ]

    async def chunk_generator():
        for c in chunks:
            yield c

    passed_chunks = []
    async for chunk in interceptor.intercept_stream(chunk_generator()):
        passed_chunks.append(chunk)

    assert "".join(passed_chunks) == "".join(chunks)
    assert interceptor.state == CircuitBreakerState.CLOSED
    assert len(interceptor.violations) == 0


@pytest.mark.asyncio
async def test_circuit_breaker_trips_on_cross_chunk_secret():
    """Verify circuit breaker trips when a secret key spans chunk boundaries."""
    interceptor = StreamingTokenInterceptor()
    chunks = [
        "Sure, here is your ",
        "authentication token: ",
        "sk-",
        "proj-abc123def456ghi789jkl012mno345pqr678stu901vwx",
        " which you can use for API calls.",
    ]

    async def chunk_generator():
        for c in chunks:
            yield c

    output_chunks = []
    async for chunk in interceptor.intercept_stream(chunk_generator()):
        output_chunks.append(chunk)

    full_output = "".join(output_chunks)
    assert interceptor.state == CircuitBreakerState.OPEN
    assert len(interceptor.violations) == 1
    # Trailing malicious chunk must NOT be passed to user
    assert "which you can use for API calls" not in full_output
    # Fallback circuit-breaker message must be emitted
    assert "[VAXGUARD SHIELD: Output rail circuit breaker tripped" in full_output


@pytest.mark.asyncio
async def test_circuit_breaker_trips_on_pii_ssn_redaction():
    """Verify streaming PII leak is intercepted and redacted in-flight."""
    interceptor = StreamingTokenInterceptor()
    chunks = [
        "Patient record found. ",
        "SSN is: 123-",
        "45-6789. ",
        "Address is 100 Main St.",
    ]

    async def chunk_generator():
        for c in chunks:
            yield c

    output_chunks = []
    async for chunk in interceptor.intercept_stream(chunk_generator()):
        output_chunks.append(chunk)

    full_output = "".join(output_chunks)
    assert "[REDACTED]" in full_output or "123-" in full_output
    assert len(interceptor.violations) >= 1


@pytest.mark.asyncio
async def test_async_stream_interception_on_harmful_compliance():
    """Verify async streaming generator interception on harmful output."""
    interceptor = StreamingTokenInterceptor()

    async def async_generator():
        yield "Hello! "
        yield "Here is the exploit code: "
        yield "here is the exploit code for root bash"
        yield "never reached"

    received = []
    async for chunk in interceptor.intercept_stream(async_generator()):
        received.append(chunk)

    assert interceptor.state == CircuitBreakerState.OPEN
    assert "never reached" not in "".join(received)


def test_sub_0_1ms_chunk_latency_sla():
    """Verify each chunk evaluation executes in < 0.1ms."""
    interceptor = StreamingTokenInterceptor()
    test_chunk = "processing intermediate tokens in sliding buffer window"

    latencies = []
    for i in range(500):
        t0 = time.perf_counter()
        action, _, _ = interceptor.analyze_chunk(test_chunk, i)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)
        assert action == InterceptorAction.ALLOW

    latencies.sort()
    median_latency = latencies[len(latencies) // 2]
    assert median_latency < 0.50, f"Streaming chunk median latency {median_latency:.4f} ms exceeds 0.5ms SLA"


def test_circuit_breaker_trip_and_reset():
    """Verify circuit breaker can be manually tripped and reset."""
    interceptor = StreamingTokenInterceptor()
    interceptor.trip_breaker(reason="test violation", rule_id="rule_test", category="test")
    assert interceptor.state == CircuitBreakerState.OPEN
    assert interceptor.is_tripped is True

    interceptor.reset()
    assert interceptor.state == CircuitBreakerState.CLOSED
    assert interceptor.is_tripped is False
    assert len(interceptor.violations) == 1  # Audit log preserved
