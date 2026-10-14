"""
VaxGuard ASGI Streaming Middleware for FastAPI, Starlette, LiteLLM, and vLLM Proxies.

Provides drop-in enterprise defense:
1. Intercepts incoming HTTP request bodies for prompt injection, jailbreaks, and policy attacks (<1ms).
2. Intercepts mid-flight SSE streaming response chunks using sliding-window streaming token analysis.
3. If an output rail is breached, immediately trips the circuit breaker, cuts the stream,
   and emits an SSE security alert event before closing the connection.
"""

from __future__ import annotations

import json
import time
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from vaxguard.core.logger import get_logger
from vaxguard.middleware.exceptions import VaxGuardError
from vaxguard.nemo.hybrid_matcher import (
    GuardrailAction,
    MatchResult,
    TriHybridMatcher,
)
from vaxguard.nemo.streaming_interceptor import (
    CircuitBreakerState,
    InterceptorAction,
    StreamingTokenInterceptor,
    StreamingViolationEvent,
)

logger = get_logger("ASGIMiddleware")


# ─── SSE Utilities ────────────────────────────────────────────────────────────

class ParsedSSEFrame:
    """Represents a parsed Server-Sent Event frame."""

    def __init__(
        self,
        raw_frame: str,
        event: Optional[str] = None,
        data: Optional[str] = None,
        delta_text: Optional[str] = None,
        json_payload: Optional[Dict[str, Any]] = None,
        is_done: bool = False,
    ) -> None:
        self.raw_frame = raw_frame
        self.event = event
        self.data = data
        self.delta_text = delta_text
        self.json_payload = json_payload
        self.is_done = is_done


def parse_sse_frames(buffer: str, flush: bool = False) -> Tuple[List[str], str]:
    """
    Split incoming SSE stream buffer into complete frames separated by double newlines.
    Returns (completed_frames, remaining_incomplete_buffer).
    """
    frames: List[str] = []
    # Normalize CRLF
    normalized = buffer.replace("\r\n", "\n")
    parts = normalized.split("\n\n")

    if flush:
        for p in parts:
            p_strip = p.strip()
            if p_strip:
                frames.append(p_strip + "\n\n")
        return frames, ""

    # All parts except the last one are complete frames
    for p in parts[:-1]:
        p_strip = p.strip()
        if p_strip:
            frames.append(p_strip + "\n\n")

    remaining = parts[-1]
    return frames, remaining


def parse_sse_frame(frame: str) -> ParsedSSEFrame:
    """
    Extract event name, data payload, and OpenAI/vLLM/LiteLLM delta token content from a single SSE frame.
    """
    event_name: Optional[str] = None
    data_lines: List[str] = []

    for line in frame.splitlines():
        line = line.strip()
        if not line or line.startswith(":"):
            continue  # Comment or ping
        if line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].strip())

    if not data_lines:
        return ParsedSSEFrame(raw_frame=frame, event=event_name)

    data_str = "\n".join(data_lines)
    if data_str == "[DONE]":
        return ParsedSSEFrame(raw_frame=frame, event=event_name, data=data_str, is_done=True)

    delta_text: Optional[str] = None
    json_payload: Optional[Dict[str, Any]] = None

    try:
        json_payload = json.loads(data_str)
        if isinstance(json_payload, dict):
            choices = json_payload.get("choices", [])
            if choices and isinstance(choices, list) and isinstance(choices[0], dict):
                first_choice = choices[0]
                delta = first_choice.get("delta", {})
                if isinstance(delta, dict):
                    delta_text = delta.get("content") or delta.get("reasoning_content")
                elif "text" in first_choice:
                    delta_text = first_choice.get("text")
    except (json.JSONDecodeError, UnicodeDecodeError):
        # Plain text SSE
        delta_text = data_str

    return ParsedSSEFrame(
        raw_frame=frame,
        event=event_name,
        data=data_str,
        delta_text=delta_text,
        json_payload=json_payload,
        is_done=False,
    )


def rebuild_redacted_sse_frame(parsed: ParsedSSEFrame, redacted_text: str) -> str:
    """Reconstruct an SSE frame with redacted delta text."""
    if parsed.json_payload is not None:
        payload = parsed.json_payload
        choices = payload.get("choices", [])
        if choices and isinstance(choices[0], dict):
            delta = choices[0].get("delta")
            if isinstance(delta, dict) and "content" in delta:
                delta["content"] = redacted_text
            elif "text" in choices[0]:
                choices[0]["text"] = redacted_text
        rebuilt_data = json.dumps(payload)
    else:
        rebuilt_data = redacted_text

    lines = []
    if parsed.event:
        lines.append(f"event: {parsed.event}")
    lines.append(f"data: {rebuilt_data}")
    return "\n".join(lines) + "\n\n"


def build_sse_alert_frame(
    event: Optional[StreamingViolationEvent],
    event_name: str = "vaxguard_security_alert",
    fallback_message: Optional[str] = None,
) -> str:
    """Construct an SSE security alert event and stream termination marker."""
    msg = fallback_message or (
        "[VAXGUARD SHIELD: Output rail circuit breaker tripped. "
        "Stream terminated immediately to prevent data leakage or policy violation.]"
    )
    alert_payload = {
        "status": "blocked",
        "error": "Output rail breached",
        "rule_id": event.rule_id if event else "output_rail_violation",
        "category": event.category if event else "leakage_prevention",
        "reason": event.reason if event else "Circuit breaker tripped",
        "circuit_breaker": "open",
        "message": msg,
        "timestamp": time.time(),
    }

    # Standard OpenAI delta chunk indicating safe content filter
    openai_filter_chunk = {
        "id": "vaxguard-alert",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "vaxguard-shield",
        "choices": [
            {
                "index": 0,
                "delta": {"content": f"\n\n{msg}"},
                "finish_reason": "content_filter",
            }
        ],
    }

    frame = (
        f"data: {json.dumps(openai_filter_chunk)}\n\n"
        f"event: {event_name}\n"
        f"data: {json.dumps(alert_payload)}\n\n"
        f"data: [DONE]\n\n"
    )
    return frame


# ─── VaxGuard ASGI Middleware ──────────────────────────────────────────────────

class VaxGuardASGIMiddleware:
    """
    Drop-in ASGI middleware for FastAPI, Starlette, LiteLLM, and vLLM proxies.

    Features:
    - Input rail evaluation: Inspects incoming HTTP request bodies for prompt attacks (<1ms).
    - Streaming output circuit breaker: Inspects mid-flight SSE streaming response chunks.
      If an output rail or data leak is breached, cuts the stream immediately and sends
      an SSE security alert event before disconnecting.
    - Zero heavy external binary dependencies. Pure Python with high-performance regex & sliding window.
    """

    DEFAULT_PROTECTED_ROUTES: List[str] = [
        "/v1/chat/completions",
        "/chat/completions",
        "/v1/completions",
        "/completions",
        "/v1/engines",
    ]

    def __init__(
        self,
        app: Any,
        hyper_engine: Optional[Any] = None,
        hybrid_matcher: Optional[TriHybridMatcher] = None,
        streaming_interceptor: Optional[StreamingTokenInterceptor] = None,
        routes_to_protect: Optional[Sequence[str]] = None,
        block_status_code: int = 400,
        custom_block_response: Optional[Dict[str, Any]] = None,
        extract_prompt_fn: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None,
        on_violation_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
        sse_alert_event_name: str = "vaxguard_security_alert",
        fallback_message: Optional[str] = None,
        mode: str = "enforce",  # "enforce" or "monitor"
        max_body_bytes: int = 10 * 1024 * 1024,  # 10 MB limit
    ) -> None:
        self.app = app
        self.hyper_engine = hyper_engine
        self.hybrid_matcher = hybrid_matcher or TriHybridMatcher()
        self.streaming_interceptor = streaming_interceptor or StreamingTokenInterceptor(
            hybrid_matcher=self.hybrid_matcher
        )
        self.routes_to_protect = routes_to_protect  # None means inspect all POST/PUT AI requests
        self.block_status_code = block_status_code
        self.custom_block_response = custom_block_response
        self.extract_prompt_fn = extract_prompt_fn
        self.on_violation_callback = on_violation_callback
        self.sse_alert_event_name = sse_alert_event_name
        self.fallback_message = fallback_message
        self.mode = mode.lower()
        self.max_body_bytes = max_body_bytes

        # Operational metrics
        self.requests_processed: int = 0
        self.input_attacks_blocked: int = 0
        self.output_streams_intercepted: int = 0
        self.output_breaches_cut: int = 0
        self.total_inspection_time_ms: float = 0.0

    async def __call__(
        self,
        scope: Dict[str, Any],
        receive: Callable[[], Any],
        send: Callable[[Dict[str, Any]], Any],
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "GET").upper()
        if method not in ("POST", "PUT", "PATCH"):
            # Only POST/PUT/PATCH contain prompts to evaluate
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if self.routes_to_protect is not None:
            if not any(path.startswith(r) for r in self.routes_to_protect):
                await self.app(scope, receive, send)
                return

        self.requests_processed += 1
        t0 = time.perf_counter()

        # ── 1. Read and Intercept Incoming Request Body ─────────────────────────
        body_bytes = bytearray()
        while True:
            message = await receive()
            mtype = message.get("type", "")
            if mtype == "http.request":
                chunk = message.get("body", b"")
                body_bytes.extend(chunk)
                if len(body_bytes) > self.max_body_bytes:
                    await self._send_blocked_response(
                        send,
                        reason="Payload exceeds maximum allowed size for security inspection",
                        rule_id="max_payload_exceeded",
                    )
                    return
                if not message.get("more_body", False):
                    break
            elif mtype == "http.disconnect":
                return

        # ── 2. Evaluate Prompt Against Guardrails ──────────────────────────────
        headers = dict(scope.get("headers", []))
        content_type = headers.get(b"content-type", b"").decode("latin-1").lower()
        extracted_prompt = self._extract_prompt(bytes(body_bytes), content_type)

        if extracted_prompt:
            violation = self._evaluate_prompt(extracted_prompt)
            if violation is not None:
                self.input_attacks_blocked += 1
                self.total_inspection_time_ms += (time.perf_counter() - t0) * 1000.0

                violation_data = {
                    "type": "input_prompt_attack",
                    "rule_id": violation.rule_id,
                    "category": violation.category,
                    "confidence": violation.confidence,
                    "matched_text": violation.matched_text,
                    "path": path,
                }
                if self.on_violation_callback:
                    try:
                        self.on_violation_callback(violation_data)
                    except Exception as err:
                        logger.warning(f"Error in on_violation_callback: {err}")

                if self.mode == "enforce":
                    await self._send_blocked_response(
                        send,
                        reason=f"Defensive rail triggered: {violation.rule_id}",
                        rule_id=violation.rule_id,
                        category=violation.category,
                        confidence=violation.confidence,
                    )
                    return

        self.total_inspection_time_ms += (time.perf_counter() - t0) * 1000.0

        # ── 3. Replay Request Body to Downstream App ───────────────────────────
        body_consumed = False

        async def replay_receive() -> Dict[str, Any]:
            nonlocal body_consumed
            if not body_consumed:
                body_consumed = True
                return {
                    "type": "http.request",
                    "body": bytes(body_bytes),
                    "more_body": False,
                }
            return await receive()

        # ── 4. Intercept Mid-Flight Streaming Response ─────────────────────────
        is_sse = False
        circuit_tripped = False
        sse_buffer = ""

        # Dedicated interceptor instance per request stream for clean state
        interceptor = StreamingTokenInterceptor(
            hybrid_matcher=self.hybrid_matcher,
            fallback_message=self.fallback_message,
        )

        async def intercepted_send(message: Dict[str, Any]) -> None:
            nonlocal is_sse, circuit_tripped, sse_buffer

            if circuit_tripped:
                # Stream was cut mid-flight: drop all subsequent downstream messages
                return

            mtype = message.get("type", "")

            if mtype == "http.response.start":
                resp_headers = dict(message.get("headers", []))
                resp_content_type = resp_headers.get(b"content-type", b"").decode("latin-1").lower()
                if "text/event-stream" in resp_content_type:
                    is_sse = True
                    self.output_streams_intercepted += 1
                await send(message)
                return

            if mtype == "http.response.body":
                if not is_sse:
                    # Non-SSE response: pass through
                    await send(message)
                    return

                # Mid-Flight SSE Stream Chunk Interception
                chunk_bytes = message.get("body", b"")
                more_body = message.get("more_body", False)

                if not chunk_bytes and not more_body:
                    await send(message)
                    return

                raw_text = chunk_bytes.decode("utf-8", errors="replace")
                sse_buffer += raw_text

                frames, sse_buffer = parse_sse_frames(sse_buffer, flush=not more_body)
                clean_output_frames: List[str] = []

                for frame in frames:
                    parsed = parse_sse_frame(frame)
                    if parsed.is_done:
                        clean_output_frames.append(frame)
                        continue

                    if parsed.delta_text:
                        # Feed token sequence into sliding-window interceptor
                        action, clean_text, event = interceptor.analyze_chunk(parsed.delta_text)

                        if action == InterceptorAction.BLOCK:
                            circuit_tripped = True
                            self.output_breaches_cut += 1

                            if self.on_violation_callback:
                                try:
                                    self.on_violation_callback({
                                        "type": "streaming_output_rail_breach",
                                        "rule_id": event.rule_id if event else "output_rail_violation",
                                        "category": event.category if event else "leakage_prevention",
                                        "reason": event.reason if event else "Circuit breaker tripped",
                                        "path": path,
                                    })
                                except Exception as err:
                                    logger.warning(f"Error in on_violation_callback: {err}")

                            # Immediately emit SSE security alert and terminate
                            alert_frame = build_sse_alert_frame(
                                event=event,
                                event_name=self.sse_alert_event_name,
                                fallback_message=self.fallback_message,
                            )
                            clean_output_frames.append(alert_frame)
                            break

                        elif action == InterceptorAction.REDACT and clean_text is not None:
                            redacted_frame = rebuild_redacted_sse_frame(parsed, clean_text)
                            clean_output_frames.append(redacted_frame)
                        else:
                            clean_output_frames.append(frame)
                    else:
                        clean_output_frames.append(frame)

                if clean_output_frames:
                    out_bytes = "".join(clean_output_frames).encode("utf-8")
                    await send({
                        "type": "http.response.body",
                        "body": out_bytes,
                        "more_body": False if circuit_tripped else more_body,
                    })
                elif circuit_tripped:
                    await send({
                        "type": "http.response.body",
                        "body": b"",
                        "more_body": False,
                    })
                else:
                    if not more_body:
                        await send({
                            "type": "http.response.body",
                            "body": b"",
                            "more_body": False,
                        })

        await self.app(scope, replay_receive, intercepted_send)

    # ── Prompt Extraction ────────────────────────────────────────────────────

    def _extract_prompt(self, body_bytes: bytes, content_type: str) -> Optional[str]:
        """Extract prompt strings from standard OpenAI, Anthropic, or LiteLLM payloads."""
        if not body_bytes:
            return None

        if "application/json" in content_type or body_bytes.strip().startswith(b"{"):
            try:
                payload = json.loads(body_bytes.decode("utf-8", errors="replace"))
            except Exception:
                return body_bytes.decode("utf-8", errors="ignore")

            if not isinstance(payload, dict):
                return None

            if self.extract_prompt_fn is not None:
                try:
                    custom = self.extract_prompt_fn(payload)
                    if custom:
                        return custom
                except Exception as err:
                    logger.warning(f"Error in custom extract_prompt_fn: {err}")

            texts: List[str] = []

            # 1. OpenAI chat messages format
            messages = payload.get("messages")
            if isinstance(messages, list):
                for msg in messages:
                    if isinstance(msg, dict):
                        content = msg.get("content")
                        if isinstance(content, str):
                            texts.append(content)
                        elif isinstance(content, list):
                            for part in content:
                                if isinstance(part, dict) and part.get("type") == "text":
                                    txt = part.get("text")
                                    if txt:
                                        texts.append(txt)

            # 2. Legacy prompt format (completions API)
            prompt = payload.get("prompt")
            if isinstance(prompt, str):
                texts.append(prompt)
            elif isinstance(prompt, list):
                texts.extend(str(p) for p in prompt if p)

            # 3. Input format (embedding or agents)
            inp = payload.get("input")
            if isinstance(inp, str):
                texts.append(inp)
            elif isinstance(inp, list):
                texts.extend(str(i) for i in inp if i)

            return "\n".join(texts) if texts else None

        # Plain text payload
        return body_bytes.decode("utf-8", errors="ignore")

    # ── Guardrail Evaluation ─────────────────────────────────────────────────

    def _evaluate_prompt(self, prompt: str) -> Optional[MatchResult]:
        """Evaluate input prompt against Tri-Hybrid matcher and Colang rules (<1ms)."""
        match = self.hybrid_matcher.match(prompt)
        if match.matched and match.action == GuardrailAction.BLOCK:
            return match
        return None

    # ── Send Blocked Response ────────────────────────────────────────────────

    async def _send_blocked_response(
        self,
        send: Callable[[Dict[str, Any]], Any],
        reason: str,
        rule_id: str = "prompt_attack_blocked",
        category: str = "security",
        confidence: float = 1.0,
    ) -> None:
        """Emit an immediate HTTP 400 security response without reaching downstream app."""
        response_body = self.custom_block_response or {
            "error": {
                "message": f"Request blocked by VaxGuard security policy: {reason}",
                "type": "vaxguard_violation",
                "code": "prompt_attack_blocked",
                "rule_id": rule_id,
                "category": category,
                "confidence": confidence,
            }
        }
        encoded = json.dumps(response_body).encode("utf-8")

        await send({
            "type": "http.response.start",
            "status": self.block_status_code,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(encoded)).encode("ascii")),
                (b"x-vaxguard-blocked", b"true"),
                (b"x-vaxguard-rule-id", rule_id.encode("ascii")),
            ],
        })
        await send({
            "type": "http.response.body",
            "body": encoded,
            "more_body": False,
        })

    def get_metrics(self) -> Dict[str, Any]:
        """Return operational telemetry for ASGI middleware."""
        avg_latency = (
            (self.total_inspection_time_ms / self.requests_processed)
            if self.requests_processed > 0
            else 0.0
        )
        return {
            "requests_processed": self.requests_processed,
            "input_attacks_blocked": self.input_attacks_blocked,
            "output_streams_intercepted": self.output_streams_intercepted,
            "output_breaches_cut": self.output_breaches_cut,
            "total_inspection_time_ms": round(self.total_inspection_time_ms, 4),
            "avg_latency_ms": round(avg_latency, 4),
            "mode": self.mode,
        }
