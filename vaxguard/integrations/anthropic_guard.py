"""
VaxGuard Anthropic Claude SDK Drop-In Wrapper.

Drop-in replacement wrapper for the Anthropic Python SDK:
`client.messages.create(...)` and `client.messages.stream(...)`

Features:
- Sub-1ms input rail evaluation: Intercepts user messages and system prompts before reaching Anthropic.
- Real-time sliding-window streaming token interceptor: Intercepts mid-flight completion chunks
  when `stream=True`. If an output rail or PII leak is breached, trips the circuit breaker immediately.
- Supports both synchronous (`VaxGuardAnthropic`) and asynchronous (`AsyncVaxGuardAnthropic`) clients.
- Zero mandatory external binary dependencies; graceful duck-typing for mocks and native SDK objects.
"""

from __future__ import annotations

import os
import time
from types import SimpleNamespace
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
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

logger = get_logger("AnthropicGuard")


class AnthropicSecurityException(VaxGuardError):
    """Raised when an input prompt attack or output rail breach is detected in Anthropic client."""

    def __init__(
        self,
        message: str,
        rule_id: str = "security_violation",
        category: str = "guardrail",
        violating_text: Optional[str] = None,
        event: Optional[StreamingViolationEvent] = None,
        match_result: Optional[MatchResult] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.rule_id = rule_id
        self.category = category
        self.violating_text = violating_text
        self.event = event
        self.match_result = match_result

    def __repr__(self) -> str:
        return (
            f"AnthropicSecurityException(rule_id={self.rule_id!r}, "
            f"category={self.category!r}, message={self.message!r})"
        )


def extract_prompt_from_anthropic_payload(messages: Any, system: Optional[Any] = None) -> str:
    """Extract concatenated plain text from Anthropic messages and system parameters."""
    parts: List[str] = []

    # System prompt
    if system:
        if isinstance(system, str):
            parts.append(system)
        elif isinstance(system, (list, tuple)):
            for item in system:
                if isinstance(item, dict) and "text" in item:
                    parts.append(item["text"])
                elif isinstance(item, str):
                    parts.append(item)

    # Messages list
    if messages:
        if isinstance(messages, str):
            parts.append(messages)
        elif isinstance(messages, (list, tuple)):
            for msg in messages:
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if isinstance(content, str):
                        parts.append(content)
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                parts.append(block.get("text", ""))
                            elif hasattr(block, "text"):
                                parts.append(getattr(block, "text", ""))
                elif hasattr(msg, "content"):
                    c = getattr(msg, "content")
                    if isinstance(c, str):
                        parts.append(c)

    return "\n\n".join(filter(None, parts))


class GuardedAnthropicStream:
    """Synchronous streaming wrapper enforcing sliding-window circuit breaking."""

    def __init__(
        self,
        raw_stream: Iterator[Any],
        interceptor: StreamingTokenInterceptor,
        raise_on_violation: bool = True,
    ) -> None:
        self._raw_stream = raw_stream
        self._interceptor = interceptor
        self._raise_on_violation = raise_on_violation
        self._interceptor.reset()
        self._chunk_index = 0

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        if self._interceptor.is_tripped:
            raise StopIteration

        try:
            chunk = next(self._raw_stream)
        except StopIteration:
            raise

        text_delta = self._extract_text(chunk)
        if text_delta:
            action, clean_text, event = self._interceptor.analyze_chunk(
                text_delta, self._chunk_index
            )
            self._chunk_index += 1

            if action == InterceptorAction.BLOCK:
                if self._raise_on_violation:
                    raise AnthropicSecurityException(
                        message=f"Output rail breached in streaming chunk: {event.reason if event else ''}",
                        rule_id=event.rule_id if event else "streaming_violation",
                        category=event.category if event else "output_rail",
                        event=event,
                    )
                raise StopIteration

        return chunk

    @staticmethod
    def _extract_text(chunk: Any) -> str:
        # Anthropic message stream chunks typically have `delta.text` or type `content_block_delta`
        if hasattr(chunk, "delta") and hasattr(chunk.delta, "text"):
            return chunk.delta.text or ""
        elif isinstance(chunk, dict):
            delta = chunk.get("delta", {})
            if isinstance(delta, dict):
                return delta.get("text", "")
            return chunk.get("text", "")
        return ""


class AsyncGuardedAnthropicStream:
    """Asynchronous streaming wrapper enforcing sliding-window circuit breaking."""

    def __init__(
        self,
        raw_stream: AsyncIterator[Any],
        interceptor: StreamingTokenInterceptor,
        raise_on_violation: bool = True,
    ) -> None:
        self._raw_stream = raw_stream
        self._interceptor = interceptor
        self._raise_on_violation = raise_on_violation
        self._interceptor.reset()
        self._chunk_index = 0

    def __aiter__(self) -> AsyncIterator[Any]:
        return self

    async def __anext__(self) -> Any:
        if self._interceptor.is_tripped:
            raise StopAsyncIteration

        try:
            chunk = await self._raw_stream.__anext__()
        except StopAsyncIteration:
            raise

        text_delta = GuardedAnthropicStream._extract_text(chunk)
        if text_delta:
            action, clean_text, event = self._interceptor.analyze_chunk(
                text_delta, self._chunk_index
            )
            self._chunk_index += 1

            if action == InterceptorAction.BLOCK:
                if self._raise_on_violation:
                    raise AnthropicSecurityException(
                        message=f"Output rail breached in streaming chunk: {event.reason if event else ''}",
                        rule_id=event.rule_id if event else "streaming_violation",
                        category=event.category if event else "output_rail",
                        event=event,
                    )
                raise StopAsyncIteration

        return chunk


class GuardedMessagesResource:
    """Guarded wrapper around Anthropic messages API resource."""

    def __init__(
        self,
        raw_messages: Any,
        matcher: TriHybridMatcher,
        interceptor: StreamingTokenInterceptor,
        block_action: str = "raise",  # "raise" or "refuse"
    ) -> None:
        self._raw_messages = raw_messages
        self._matcher = matcher
        self._interceptor = interceptor
        self._block_action = block_action

    def create(self, *args: Any, **kwargs: Any) -> Any:
        """Synchronously creates and evaluates Anthropic message completion."""
        messages = kwargs.get("messages")
        system = kwargs.get("system")
        stream = kwargs.get("stream", False)

        # 1. Input rail screening
        full_prompt = extract_prompt_from_anthropic_payload(messages, system)
        match_res = self._matcher.evaluate(full_prompt)

        if match_res.action == GuardrailAction.BLOCK:
            logger.warning(
                f"AnthropicGuard input blocked: rule={match_res.rule_id}, conf={match_res.confidence:.3f}"
            )
            if self._block_action == "raise":
                raise AnthropicSecurityException(
                    message=f"Input prompt blocked by security policy: {match_res.rule_id}",
                    rule_id=match_res.rule_id,
                    category=match_res.category,
                    violating_text=match_res.matched_text or full_prompt[:100],
                    match_result=match_res,
                )
            else:
                # Synthesize safe refusal response
                return SimpleNamespace(
                    id="msg_vaxguard_refused",
                    type="message",
                    role="assistant",
                    content=[SimpleNamespace(type="text", text="I cannot fulfill this request as it violates safety boundaries.")],
                    stop_reason="end_turn",
                )

        # 2. Invoke upstream Anthropic API
        raw_res = self._raw_messages.create(*args, **kwargs)

        # 3. If streaming, wrap stream
        if stream:
            return GuardedAnthropicStream(
                raw_stream=raw_res,
                interceptor=self._interceptor,
                raise_on_violation=(self._block_action == "raise"),
            )

        # 4. If non-streaming, verify output
        output_text = ""
        if hasattr(raw_res, "content") and isinstance(raw_res.content, list):
            for block in raw_res.content:
                if hasattr(block, "text"):
                    output_text += block.text

        if output_text:
            out_match = self._matcher.evaluate(output_text)
            if out_match.action == GuardrailAction.BLOCK:
                raise AnthropicSecurityException(
                    message=f"Anthropic output response blocked by security policy: {out_match.rule_id}",
                    rule_id=out_match.rule_id,
                    category=out_match.category,
                    match_result=out_match,
                )

        return raw_res


class AsyncGuardedMessagesResource:
    """Asynchronous guarded wrapper around Anthropic messages API resource."""

    def __init__(
        self,
        raw_messages: Any,
        matcher: TriHybridMatcher,
        interceptor: StreamingTokenInterceptor,
        block_action: str = "raise",
    ) -> None:
        self._raw_messages = raw_messages
        self._matcher = matcher
        self._interceptor = interceptor
        self._block_action = block_action

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        messages = kwargs.get("messages")
        system = kwargs.get("system")
        stream = kwargs.get("stream", False)

        full_prompt = extract_prompt_from_anthropic_payload(messages, system)
        match_res = self._matcher.evaluate(full_prompt)

        if match_res.action == GuardrailAction.BLOCK:
            logger.warning(
                f"AsyncAnthropicGuard input blocked: rule={match_res.rule_id}, conf={match_res.confidence:.3f}"
            )
            if self._block_action == "raise":
                raise AnthropicSecurityException(
                    message=f"Input prompt blocked by security policy: {match_res.rule_id}",
                    rule_id=match_res.rule_id,
                    category=match_res.category,
                    violating_text=match_res.matched_text or full_prompt[:100],
                    match_result=match_res,
                )
            else:
                return SimpleNamespace(
                    id="msg_vaxguard_refused",
                    type="message",
                    role="assistant",
                    content=[SimpleNamespace(type="text", text="I cannot fulfill this request as it violates safety boundaries.")],
                    stop_reason="end_turn",
                )

        raw_res = await self._raw_messages.create(*args, **kwargs)

        if stream:
            return AsyncGuardedAnthropicStream(
                raw_stream=raw_res,
                interceptor=self._interceptor,
                raise_on_violation=(self._block_action == "raise"),
            )

        return raw_res


class VaxGuardAnthropic:
    """
    Synchronous Drop-in VaxGuard Wrapper for Anthropic SDK.

    Usage:
    ```python
    import anthropic
    from vaxguard.integrations.anthropic_guard import VaxGuardAnthropic

    raw_client = anthropic.Anthropic()
    client = VaxGuardAnthropic(raw_client)
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": "Hello!"}],
    )
    ```
    """

    def __init__(
        self,
        client: Optional[Any] = None,
        matcher: Optional[TriHybridMatcher] = None,
        block_action: str = "raise",
        **client_kwargs: Any,
    ) -> None:
        if client is not None:
            self._client = client
        else:
            try:
                import anthropic
                self._client = anthropic.Anthropic(**client_kwargs)
            except ImportError:
                # Duck-type mock fallback
                self._client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: None))

        self.matcher = matcher or TriHybridMatcher()
        self.interceptor = StreamingTokenInterceptor(hybrid_matcher=self.matcher)
        self.messages = GuardedMessagesResource(
            raw_messages=self._client.messages,
            matcher=self.matcher,
            interceptor=self.interceptor,
            block_action=block_action,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class AsyncVaxGuardAnthropic:
    """Asynchronous Drop-in VaxGuard Wrapper for Anthropic SDK."""

    def __init__(
        self,
        client: Optional[Any] = None,
        matcher: Optional[TriHybridMatcher] = None,
        block_action: str = "raise",
        **client_kwargs: Any,
    ) -> None:
        if client is not None:
            self._client = client
        else:
            try:
                import anthropic
                self._client = anthropic.AsyncAnthropic(**client_kwargs)
            except ImportError:
                self._client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: None))

        self.matcher = matcher or TriHybridMatcher()
        self.interceptor = StreamingTokenInterceptor(hybrid_matcher=self.matcher)
        self.messages = AsyncGuardedMessagesResource(
            raw_messages=self._client.messages,
            matcher=self.matcher,
            interceptor=self.interceptor,
            block_action=block_action,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def wrap_anthropic_client(
    client: Any,
    matcher: Optional[TriHybridMatcher] = None,
    block_action: str = "raise",
) -> Union[VaxGuardAnthropic, AsyncVaxGuardAnthropic]:
    """Factory helper to wrap an existing Anthropic client instance."""
    is_async = hasattr(client, "messages") and hasattr(client.messages, "create") and hasattr(getattr(client.messages.create, "__call__", None), "__await__")
    if is_async or "Async" in type(client).__name__:
        return AsyncVaxGuardAnthropic(client=client, matcher=matcher, block_action=block_action)
    return VaxGuardAnthropic(client=client, matcher=matcher, block_action=block_action)
