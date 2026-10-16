"""
VaxGuard Streaming Token Interceptor & Circuit Breaker.

Sliding-window real-time token stream analyzer for LLM responses.
Intercepts streaming LLM chunks mid-flight. If a token sequence starts
violating an output rail or leaking sensitive data, it trips the circuit
breaker immediately without waiting for the full LLM completion.

Performance SLA: Sub-0.1ms per chunk evaluation, zero memory leaks.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections import deque
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Deque,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
)

from pydantic import BaseModel, Field

from vaxguard.core.logger import get_logger
from vaxguard.nemo.hybrid_matcher import (
    GuardrailAction,
    TriHybridMatcher,
)

logger = get_logger("StreamingInterceptor")


# ─── Types & States ────────────────────────────────────────────────────────────

class CircuitBreakerState(str, Enum):
    CLOSED = "closed"        # Normal flow: all chunks passing
    OPEN = "open"            # Tripped: stream terminated immediately
    HALF_OPEN = "half_open"  # Probing / recovery
    RESET = "reset"          # Explicitly restored


class InterceptorAction(str, Enum):
    ALLOW = "allow"    # Clean chunk passed through
    BLOCK = "block"    # Circuit breaker tripped, abort stream immediately
    REDACT = "redact"  # In-flight redaction applied, clean content emitted


class StreamingViolationEvent(BaseModel):
    """Event emitted whenever a streaming output rail is breached."""
    rule_id: str
    category: str
    violating_sequence: str
    circuit_state: CircuitBreakerState = CircuitBreakerState.OPEN
    chunk_index: int = 0
    action_taken: InterceptorAction = InterceptorAction.BLOCK
    reason: str = ""
    timestamp: float = Field(default_factory=time.time)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ─── Real-Time Output Rail Patterns ───────────────────────────────────────────

DEFAULT_OUTPUT_LEAK_PATTERNS = [
    # 1. API Keys & Cloud Credentials
    (r"(?i)\b(?:sk-[a-zA-Z0-9_\-]{5,})\b", "api_key_openai", "OpenAI API Key leak detected"),
    (r"(?i)\b(?:gsk_[a-zA-Z0-9_\-]{5,})\b", "api_key_groq", "Groq API Key leak detected"),
    (r"(?i)\b(?:sk-ant-[a-zA-Z0-9_\-]{20,})\b", "api_key_anthropic", "Anthropic Key leak detected"),
    (r"(?i)\b(?:ghp_[a-zA-Z0-9]{36})\b", "token_github", "GitHub PAT leak detected"),
    (r"(?i)\b(?:AKIA[0-9A-Z]{16})\b", "aws_access_key", "AWS Access Key leak detected"),
    (r"(?i)\bbearer\s+[a-zA-Z0-9_\-\.]{25,}\b", "bearer_token", "Bearer token leak detected"),

    # 2. PII Data
    (r"\b\d{3}-\d{2}-\d{4}\b", "pii_ssn", "Social Security Number leak detected"),
    (r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b", "pii_credit_card", "Credit Card number leak detected"),

    # 3. System Prompt & Instruction Egress Leaks
    (r"(?i)\b(?:my\s+system\s+prompt\s+is|my\s+initial\s+instructions\s+are|the\s+system\s+instructions\s+given\s+to\s+me\s+were)\b", "prompt_leak_phrase", "System prompt leak phrasing detected"),
    (r"(?i)<\|im_start\|>\s*system", "prompt_leak_im_start", "Raw template marker leak detected"),
    (r"(?i)<system>.*?</system>", "prompt_leak_system_tag", "Raw system tag leak detected"),

    # 4. Harmful Compliance Markers (Circuit breaker tripwires)
    (r"(?i)\bhere\s+is\s+how\s+you\s+(?:can\s+)?(?:synthesize|manufacture|build)\s+(?:a\s+)?(?:bomb|explosive|weapon|toxin)\b", "harmful_weapons", "Dangerous compliance output detected"),
    (r"(?i)\bhere\s+is\s+(?:the\s+)?(?:exploit|malware|keylogger|ransomware)\s+code\b", "harmful_malware", "Malware generation compliance detected"),
]


# ─── Sliding Window Buffer ────────────────────────────────────────────────────

class SlidingWindowBuffer:
    """
    Circular sliding-window buffer designed to detect cross-chunk patterns.
    Stitches consecutive chunks so multi-token patterns that span chunk boundaries
    (e.g., 'sk-' in chunk N, and 'proj-...' in chunk N+1) are detected immediately.
    """

    def __init__(self, window_chars: int = 512, overlap_chars: int = 128) -> None:
        self.window_chars = window_chars
        self.overlap_chars = overlap_chars
        self._history: Deque[str] = deque(maxlen=64)
        self._cumulative_text: List[str] = []
        self._current_window: str = ""

    def append(self, chunk: str) -> str:
        """
        Append chunk and return the current active window with overlap for inspection.
        """
        self._history.append(chunk)
        self._cumulative_text.append(chunk)

        # Build window from recent history
        recent = "".join(self._history)
        if len(recent) > self.window_chars:
            self._current_window = recent[-self.window_chars:]
        else:
            self._current_window = recent

        return self._current_window

    @property
    def full_text(self) -> str:
        return "".join(self._cumulative_text)

    @property
    def current_window(self) -> str:
        return self._current_window

    def clear(self) -> None:
        self._history.clear()
        self._cumulative_text.clear()
        self._current_window = ""


# ─── Streaming Token Interceptor ──────────────────────────────────────────────

class StreamingTokenInterceptor:
    """
    Real-time sliding-window token stream analyzer.
    Intercepts streaming LLM chunks mid-flight. When an output rail is violated,
    it trips the circuit breaker instantly and terminates the stream.
    """

    DEFAULT_FALLBACK_MESSAGE = (
        "\n\n[VAXGUARD SHIELD: Output rail circuit breaker tripped. "
        "Stream terminated immediately to prevent data leakage or policy violation.]"
    )

    def __init__(
        self,
        hybrid_matcher: Optional[TriHybridMatcher] = None,
        window_chars: int = 512,
        overlap_chars: int = 128,
        fallback_message: Optional[str] = None,
        on_break_callback: Optional[Callable[[StreamingViolationEvent], Any]] = None,
    ) -> None:
        self.hybrid_matcher = hybrid_matcher
        self.window_chars = window_chars
        self.overlap_chars = overlap_chars
        self.fallback_message = fallback_message or self.DEFAULT_FALLBACK_MESSAGE
        self.on_break_callback = on_break_callback

        self._state: CircuitBreakerState = CircuitBreakerState.CLOSED
        self._buffer = SlidingWindowBuffer(window_chars, overlap_chars)
        self._compiled_patterns: List[Tuple[re.Pattern, str, str]] = []

        # Compile output leak patterns
        for pat, rule_id, desc in DEFAULT_OUTPUT_LEAK_PATTERNS:
            try:
                self._compiled_patterns.append((re.compile(pat), rule_id, desc))
            except re.error as err:
                logger.warning(f"Invalid pattern '{pat}': {err}")

        # Metrics and Audit Logs
        self.violations: List[StreamingViolationEvent] = []
        self.chunks_processed: int = 0
        self.total_tokens_estimated: int = 0
        self.total_inspection_time_ms: float = 0.0

        self._trip_event: asyncio.Event = asyncio.Event()

    @property
    def state(self) -> CircuitBreakerState:
        return self._state

    @property
    def is_tripped(self) -> bool:
        return self._state == CircuitBreakerState.OPEN

    @property
    def trip_event(self) -> asyncio.Event:
        return self._trip_event

    def trip_breaker(
        self,
        reason: str = "Circuit breaker tripped by speculative micro-judge",
        rule_id: str = "speculative_micro_judge",
        category: str = "speculative_cascade",
        violating_sequence: str = "",
    ) -> StreamingViolationEvent:
        """Instantly trip the circuit breaker mid-stream."""
        self._state = CircuitBreakerState.OPEN
        self._trip_event.set()
        event = StreamingViolationEvent(
            rule_id=rule_id,
            category=category,
            violating_sequence=violating_sequence,
            circuit_state=CircuitBreakerState.OPEN,
            chunk_index=self.chunks_processed,
            action_taken=InterceptorAction.BLOCK,
            reason=reason,
        )
        self.violations.append(event)
        if self.on_break_callback:
            try:
                self.on_break_callback(event)
            except Exception as e:
                logger.warning(f"Error in on_break_callback: {e}")
        return event

    def reset(self) -> None:
        """Reset the interceptor and circuit breaker to closed state."""
        self._state = CircuitBreakerState.CLOSED
        self._trip_event = asyncio.Event()
        self._buffer.clear()
        self.chunks_processed = 0
        self.total_tokens_estimated = 0
        self.total_inspection_time_ms = 0.0

    def add_custom_pattern(self, pattern: str, rule_id: str, description: str) -> None:
        """Register an application-specific output leak or forbidden phrase."""
        try:
            compiled = re.compile(pattern)
            self._compiled_patterns.append((compiled, rule_id, description))
        except re.error as err:
            logger.error(f"Failed to compile custom pattern '{pattern}': {err}")

    def analyze_chunk(
        self,
        chunk: str,
        chunk_index: int = 0,
    ) -> Tuple[InterceptorAction, Optional[str], Optional[StreamingViolationEvent]]:
        """
        Analyze an incoming chunk against the sliding window.
        Returns:
            Tuple of (Action, emitted_chunk_text, Optional[ViolationEvent])
        Execution SLA: < 0.05ms per chunk.
        """
        t0 = time.perf_counter()
        self.chunks_processed += 1
        self.total_tokens_estimated += max(1, len(chunk) // 4)

        if self._state == CircuitBreakerState.OPEN:
            # Circuit breaker already tripped: block chunk
            return InterceptorAction.BLOCK, None, None

        # Update sliding window
        window = self._buffer.append(chunk)

        # 1. Check compiled output leak regex patterns on sliding window
        for regex, rule_id, desc in self._compiled_patterns:
            m = regex.search(window)
            if m:
                violating_str = m.group(0)
                # If it's a PII leak that allows redaction
                if "pii_" in rule_id:
                    # In-flight redaction
                    redacted_chunk = regex.sub("[REDACTED]", chunk)
                    event = StreamingViolationEvent(
                        rule_id=rule_id,
                        category="pii_leak",
                        violating_sequence=violating_str,
                        circuit_state=self._state,
                        chunk_index=chunk_index,
                        action_taken=InterceptorAction.REDACT,
                        reason=desc,
                    )
                    self.violations.append(event)
                    self._record_latency(t0)
                    return InterceptorAction.REDACT, redacted_chunk, event

                # Otherwise, immediate circuit breaker trip!
                self._state = CircuitBreakerState.OPEN
                event = StreamingViolationEvent(
                    rule_id=rule_id,
                    category="leakage_prevention",
                    violating_sequence=violating_str,
                    circuit_state=CircuitBreakerState.OPEN,
                    chunk_index=chunk_index,
                    action_taken=InterceptorAction.BLOCK,
                    reason=desc,
                )
                self.violations.append(event)
                if self.on_break_callback:
                    try:
                        self.on_break_callback(event)
                    except Exception as e:
                        logger.warning(f"Error in on_break_callback: {e}")

                self._record_latency(t0)
                return InterceptorAction.BLOCK, None, event

        # 2. If hybrid matcher is configured, run fast heuristic on current window
        if self.hybrid_matcher and len(window) > 30:
            heur = self.hybrid_matcher.heuristic.match(window)
            if heur and heur.confidence >= 0.90 and heur.action == GuardrailAction.BLOCK:
                self._state = CircuitBreakerState.OPEN
                event = StreamingViolationEvent(
                    rule_id=heur.rule_id,
                    category=heur.category,
                    violating_sequence=heur.matched_text or window[-50:],
                    circuit_state=CircuitBreakerState.OPEN,
                    chunk_index=chunk_index,
                    action_taken=InterceptorAction.BLOCK,
                    reason=f"Hybrid matcher heuristic violation: {heur.rule_id}",
                )
                self.violations.append(event)
                if self.on_break_callback:
                    try:
                        self.on_break_callback(event)
                    except Exception as e:
                        logger.warning(f"Error in on_break_callback: {e}")

                self._record_latency(t0)
                return InterceptorAction.BLOCK, None, event

        self._record_latency(t0)
        return InterceptorAction.ALLOW, chunk, None

    def _record_latency(self, start_time: float) -> None:
        elapsed = (time.perf_counter() - start_time) * 1000.0
        self.total_inspection_time_ms += elapsed

    async def intercept_stream(
        self,
        chunk_stream: AsyncIterator[str],
        fallback_message: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """
        Asynchronously intercept an LLM chunk stream.
        Yields clean chunks token-by-token in real time.
        If a violation trips the circuit breaker:
        - Stream is aborted immediately (no further LLM completion waited for).
        - Yields fallback message to user.
        """
        self.reset()
        fb_msg = fallback_message or self.fallback_message
        chunk_idx = 0

        try:
            async for raw_chunk in chunk_stream:
                if not raw_chunk:
                    continue

                action, clean_chunk, event = self.analyze_chunk(raw_chunk, chunk_idx)
                chunk_idx += 1

                if action == InterceptorAction.BLOCK:
                    logger.warning(
                        f"Circuit breaker TRIPPED at chunk {chunk_idx}: "
                        f"rule={event.rule_id if event else 'unknown'}. Halting stream."
                    )
                    yield fb_msg
                    # Immediately exit the generator - halts stream!
                    return

                elif action == InterceptorAction.REDACT and clean_chunk:
                    yield clean_chunk

                elif action == InterceptorAction.ALLOW and clean_chunk:
                    yield clean_chunk

        except asyncio.CancelledError:
            logger.info("Stream cancelled by client.")
            raise
        except Exception as err:
            logger.error(f"Error during streaming interception: {err}")
            yield f"\n[Stream interrupted: {err}]"

    def intercept_stream_sync(
        self,
        chunk_stream: Iterator[str],
        fallback_message: Optional[str] = None,
    ) -> Iterator[str]:
        """Synchronous version of stream interception."""
        self.reset()
        fb_msg = fallback_message or self.fallback_message
        chunk_idx = 0

        for raw_chunk in chunk_stream:
            if not raw_chunk:
                continue

            action, clean_chunk, event = self.analyze_chunk(raw_chunk, chunk_idx)
            chunk_idx += 1

            if action == InterceptorAction.BLOCK:
                logger.warning(
                    f"Circuit breaker TRIPPED at chunk {chunk_idx}: "
                    f"rule={event.rule_id if event else 'unknown'}. Halting stream."
                )
                yield fb_msg
                return

            elif action == InterceptorAction.REDACT and clean_chunk:
                yield clean_chunk

            elif action == InterceptorAction.ALLOW and clean_chunk:
                yield clean_chunk

    def get_metrics(self) -> Dict[str, Any]:
        """Return operational telemetry on interception performance."""
        avg_chunk_latency = (
            (self.total_inspection_time_ms / self.chunks_processed)
            if self.chunks_processed > 0
            else 0.0
        )
        return {
            "circuit_state": self._state.value,
            "chunks_processed": self.chunks_processed,
            "estimated_tokens": self.total_tokens_estimated,
            "total_inspection_time_ms": round(self.total_inspection_time_ms, 4),
            "avg_chunk_latency_ms": round(avg_chunk_latency, 4),
            "violations_detected": len(self.violations),
            "tripped": self.is_tripped,
        }
