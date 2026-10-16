# ADR-002: Sliding-Window Mid-Flight Streaming Interceptor and Circuit Breaker

## Status
Accepted

## Context
Production language model integrations deliver generated completions via token-by-token streaming (Server-Sent Events / chunked HTTP transfers) to provide immediate responsiveness. Traditional safety filters operate post-hoc on completed strings; applying this pattern to streaming workflows forces a damaging choice:
1. **Unprotected Streaming**: Stream tokens directly to the user as generated, letting malicious instructions, toxic completions, or sensitive PII/PHI leak onto end-user screens before post-generation filters can react.
2. **Buffering Delay**: Buffer the entire response before evaluation, negating the UX advantages of streaming and inflating time-to-first-token (TTFT) by multiple seconds.

## Decision
VaxGuard implements a **Sliding-Window Circular Interceptor** (`StreamingTokenInterceptor`):
- **Circular Span Buffer**: As tokens arrive in chunks of arbitrary size, the interceptor appends them to a bounded token window (typically 16 to 32 tokens, with configurable overlap) to capture multi-token patterns, split delimiters, and fragmented sensitive entities (e.g., credit card numbers or private keys spread across consecutive chunks).
- **Sub-0.05 ms Chunk Check**: The sliding window evaluates the active character/token window against high-priority pattern sets within microseconds, imposing negligible latency overhead on stream delivery.
- **Circuit Breaker Mechanism**: When a violation or policy breach is detected mid-flight:
  1. The stream is halted immediately;
  2. Upstream LLM generation is cancelled;
  3. A fallback refusal or redaction marker (e.g., `[Response Intercepted by Safety Policy]`) is emitted;
  4. A structured JSON audit record and OpenTelemetry span are dispatched with the exact violation offset.

## Consequences
### Positive
- **Instant Leak Prevention**: Sensitive tokens are intercepted before escaping to the end-user display.
- **Preserved TTFT**: Clean tokens stream with imperceptible overhead (<0.05 ms per chunk).
- **Fault-Tolerant Async Interception**: Supports both synchronous generators and asynchronous token iterators (`AsyncIterator[str]`).

### Negative / Trade-offs
- Very large entities spanning beyond the configured window buffer require either an expanded window size or coordinated multi-chunk aggregation.
