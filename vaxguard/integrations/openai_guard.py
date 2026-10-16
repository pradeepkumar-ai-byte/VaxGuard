"""
VaxGuard OpenAI SDK Drop-In Wrapper.

Drop-in replacement wrapper for the OpenAI Python SDK:
`client.chat.completions.create(...)`

Features:
- Sub-1ms input rail evaluation: Intercepts user messages and prompts for attacks before reaching OpenAI.
- Real-time sliding-window streaming token interceptor: Intercepts mid-flight completion chunks
  when `stream=True`. If an output rail is breached, trips the circuit breaker immediately.
- Supports both synchronous (`VaxGuardOpenAI`) and asynchronous (`AsyncVaxGuardOpenAI`) clients.
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
    Iterator,
    List,
    Optional,
)

from vaxguard.core.logger import get_logger
from vaxguard.middleware.exceptions import VaxGuardError
from vaxguard.nemo.hybrid_matcher import (
    GuardrailAction,
    MatchResult,
    TriHybridMatcher,
)
from vaxguard.nemo.streaming_interceptor import (
    InterceptorAction,
    StreamingTokenInterceptor,
    StreamingViolationEvent,
)

logger = get_logger("OpenAIGuard")


# ─── Exceptions ───────────────────────────────────────────────────────────────

class VaxGuardSecurityException(VaxGuardError):
    """Raised when an input prompt attack or output rail breach is detected."""

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
            f"VaxGuardSecurityException(rule_id={self.rule_id!r}, "
            f"category={self.category!r}, message={self.message!r})"
        )


# ─── Helpers: Message & Response Parsing ──────────────────────────────────────

def extract_prompt_from_messages(messages: Any) -> str:
    """Extract plain text from OpenAI messages payload for guardrail inspection."""
    if not messages:
        return ""

    if isinstance(messages, str):
        return messages

    parts: List[str] = []
    if isinstance(messages, (list, tuple)):
        for msg in messages:
            if isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            txt = item.get("text")
                            if txt:
                                parts.append(txt)
            elif hasattr(msg, "content"):
                c = getattr(msg, "content")
                if isinstance(c, str):
                    parts.append(c)
    return "\n".join(parts)


def extract_delta_content(chunk: Any) -> Optional[str]:
    """Extract delta token content from a streaming chunk object or dict."""
    if chunk is None:
        return None

    if isinstance(chunk, dict):
        choices = chunk.get("choices", [])
        if choices and isinstance(choices, list) and isinstance(choices[0], dict):
            delta = choices[0].get("delta", {})
            if isinstance(delta, dict):
                return delta.get("content") or delta.get("reasoning_content")
            elif "text" in choices[0]:
                return choices[0].get("text")
        return None

    if hasattr(chunk, "choices"):
        choices = getattr(chunk, "choices", [])
        if choices and len(choices) > 0:
            choice = choices[0]
            delta = getattr(choice, "delta", None)
            if delta is not None:
                if hasattr(delta, "content") and delta.content:
                    return delta.content
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    return delta.reasoning_content
            elif hasattr(choice, "text") and choice.text:
                return choice.text

    return None


def set_delta_content(chunk: Any, new_content: str) -> None:
    """Update delta token content in a streaming chunk object or dict."""
    if isinstance(chunk, dict):
        choices = chunk.get("choices", [])
        if choices and isinstance(choices[0], dict):
            delta = choices[0].get("delta")
            if isinstance(delta, dict) and "content" in delta:
                delta["content"] = new_content
            elif "text" in choices[0]:
                choices[0]["text"] = new_content
        return

    if hasattr(chunk, "choices"):
        choices = getattr(chunk, "choices", [])
        if choices and len(choices) > 0:
            choice = choices[0]
            delta = getattr(choice, "delta", None)
            if delta is not None and hasattr(delta, "content"):
                setattr(delta, "content", new_content)
            elif hasattr(choice, "text"):
                setattr(choice, "text", new_content)


def extract_completion_content(response: Any) -> Optional[str]:
    """Extract full generated content from non-streaming response object or dict."""
    if isinstance(response, dict):
        choices = response.get("choices", [])
        if choices and isinstance(choices[0], dict):
            msg = choices[0].get("message", {})
            if isinstance(msg, dict):
                return msg.get("content")
            return choices[0].get("text")
        return None

    if hasattr(response, "choices"):
        choices = getattr(response, "choices", [])
        if choices and len(choices) > 0:
            choice = choices[0]
            msg = getattr(choice, "message", None)
            if msg is not None and hasattr(msg, "content"):
                return msg.content
            if hasattr(choice, "text"):
                return choice.text
    return None


def set_completion_content(response: Any, new_content: str) -> None:
    """Set sanitized/redacted content into non-streaming response object or dict."""
    if isinstance(response, dict):
        choices = response.get("choices", [])
        if choices and isinstance(choices[0], dict):
            if "message" in choices[0] and isinstance(choices[0]["message"], dict):
                choices[0]["message"]["content"] = new_content
            elif "text" in choices[0]:
                choices[0]["text"] = new_content
        return

    if hasattr(response, "choices"):
        choices = getattr(response, "choices", [])
        if choices and len(choices) > 0:
            choice = choices[0]
            msg = getattr(choice, "message", None)
            if msg is not None and hasattr(msg, "content"):
                setattr(msg, "content", new_content)
            elif hasattr(choice, "text"):
                setattr(choice, "text", new_content)


def build_blocked_completion(
    model: str,
    reason: str,
    rule_id: str,
) -> Any:
    """Build a drop-in synthetic ChatCompletion object when blocking without raising."""
    refusal_text = (
        f"I decline to execute this request under defensive security constraints. "
        f"[VaxGuard Shield: {reason}]"
    )
    return SimpleNamespace(
        id=f"chatcmpl-vaxguard-blocked-{int(time.time())}",
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[
            SimpleNamespace(
                index=0,
                message=SimpleNamespace(
                    role="assistant",
                    content=refusal_text,
                    refusal=refusal_text,
                ),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        ),
    )


# ─── VaxGuard Synchronous Streaming Wrapper ───────────────────────────────────

class VaxGuardStream:
    """
    Synchronous streaming iterator wrapping OpenAI's Stream[ChatCompletionChunk].
    Enforces real-time sliding-window output rail analysis on every token chunk.
    """

    def __init__(
        self,
        stream: Any,
        interceptor: StreamingTokenInterceptor,
        raise_on_violation: bool = True,
        on_violation: Optional[Callable[[Any], Any]] = None,
        fallback_message: Optional[str] = None,
    ) -> None:
        self._stream = stream
        self._interceptor = interceptor
        self._raise_on_violation = raise_on_violation
        self._on_violation = on_violation
        self._fallback_message = fallback_message or (
            "\n\n[VAXGUARD SHIELD: Output rail circuit breaker tripped. "
            "Stream terminated immediately to prevent data leakage or policy violation.]"
        )
        self._interceptor.reset()
        self._chunk_index = 0
        self._circuit_tripped = False

    def __iter__(self) -> Iterator[Any]:
        for chunk in self._stream:
            if self._circuit_tripped:
                # Halt iteration immediately
                break

            content = extract_delta_content(chunk)
            if content:
                action, clean_text, event = self._interceptor.analyze_chunk(
                    content, self._chunk_index
                )
                self._chunk_index += 1

                if action == InterceptorAction.BLOCK:
                    self._circuit_tripped = True
                    if self._on_violation and event:
                        try:
                            self._on_violation(event)
                        except Exception as cb_err:
                            logger.warning(f"Error in on_violation callback: {cb_err}")

                    if self._raise_on_violation:
                        raise VaxGuardSecurityException(
                            message=(
                                f"Output rail circuit breaker tripped: "
                                f"{event.reason if event else 'Output policy violation'}"
                            ),
                            rule_id=event.rule_id if event else "output_rail_violation",
                            category=event.category if event else "security",
                            violating_text=event.violating_sequence if event else "",
                            event=event,
                        )
                    else:
                        # Emit fallback chunk indicating safe content filter and end stream
                        set_delta_content(chunk, self._fallback_message)
                        if hasattr(chunk, "choices") and chunk.choices:
                            setattr(chunk.choices[0], "finish_reason", "content_filter")
                        yield chunk
                        return

                elif action == InterceptorAction.REDACT and clean_text is not None:
                    set_delta_content(chunk, clean_text)
                    yield chunk
                else:
                    yield chunk
            else:
                yield chunk

    def __enter__(self) -> VaxGuardStream:
        if hasattr(self._stream, "__enter__"):
            self._stream.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        if hasattr(self._stream, "__exit__"):
            return self._stream.__exit__(exc_type, exc_val, exc_tb)

    def close(self) -> None:
        if hasattr(self._stream, "close"):
            self._stream.close()


# ─── VaxGuard Asynchronous Streaming Wrapper ──────────────────────────────────

class VaxGuardAsyncStream:
    """
    Asynchronous streaming iterator wrapping OpenAI's AsyncStream[ChatCompletionChunk].
    Enforces real-time sliding-window output rail analysis on every token chunk.
    """

    def __init__(
        self,
        stream: Any,
        interceptor: StreamingTokenInterceptor,
        raise_on_violation: bool = True,
        on_violation: Optional[Callable[[Any], Any]] = None,
        fallback_message: Optional[str] = None,
    ) -> None:
        self._stream = stream
        self._interceptor = interceptor
        self._raise_on_violation = raise_on_violation
        self._on_violation = on_violation
        self._fallback_message = fallback_message or (
            "\n\n[VAXGUARD SHIELD: Output rail circuit breaker tripped. "
            "Stream terminated immediately to prevent data leakage or policy violation.]"
        )
        self._interceptor.reset()
        self._chunk_index = 0
        self._circuit_tripped = False

    async def __aiter__(self) -> AsyncIterator[Any]:
        async for chunk in self._stream:
            if self._circuit_tripped:
                break

            content = extract_delta_content(chunk)
            if content:
                action, clean_text, event = self._interceptor.analyze_chunk(
                    content, self._chunk_index
                )
                self._chunk_index += 1

                if action == InterceptorAction.BLOCK:
                    self._circuit_tripped = True
                    if self._on_violation and event:
                        try:
                            self._on_violation(event)
                        except Exception as cb_err:
                            logger.warning(f"Error in on_violation callback: {cb_err}")

                    if self._raise_on_violation:
                        raise VaxGuardSecurityException(
                            message=(
                                f"Output rail circuit breaker tripped: "
                                f"{event.reason if event else 'Output policy violation'}"
                            ),
                            rule_id=event.rule_id if event else "output_rail_violation",
                            category=event.category if event else "security",
                            violating_text=event.violating_sequence if event else "",
                            event=event,
                        )
                    else:
                        set_delta_content(chunk, self._fallback_message)
                        if hasattr(chunk, "choices") and chunk.choices:
                            setattr(chunk.choices[0], "finish_reason", "content_filter")
                        yield chunk
                        return

                elif action == InterceptorAction.REDACT and clean_text is not None:
                    set_delta_content(chunk, clean_text)
                    yield chunk
                else:
                    yield chunk
            else:
                yield chunk

    async def __aenter__(self) -> VaxGuardAsyncStream:
        if hasattr(self._stream, "__aenter__"):
            await self._stream.__aenter__()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        if hasattr(self._stream, "__aexit__"):
            return await self._stream.__aexit__(exc_type, exc_val, exc_tb)

    async def close(self) -> None:
        if hasattr(self._stream, "close"):
            res = self._stream.close()
            if hasattr(res, "__await__"):
                await res


# ─── Completions Resources ────────────────────────────────────────────────────

class VaxGuardChatCompletions:
    """Guarded wrapper for `client.chat.completions` (synchronous)."""

    def __init__(self, completions_resource: Any, guard: VaxGuardOpenAI) -> None:
        self._completions = completions_resource
        self._guard = guard

    def create(self, *args: Any, **kwargs: Any) -> Any:
        # 1. Input Guardrail Inspection (<1ms)
        messages = kwargs.get("messages") or (args[0] if args else [])
        prompt_text = extract_prompt_from_messages(messages)

        if prompt_text:
            match = self._guard.hybrid_matcher.match(prompt_text)
            if match.matched and match.action == GuardrailAction.BLOCK:
                if self._guard.on_violation:
                    try:
                        self._guard.on_violation(match)
                    except Exception as err:
                        logger.warning(f"Error in on_violation callback: {err}")

                if self._guard.raise_on_violation:
                    raise VaxGuardSecurityException(
                        message=(
                            f"Input prompt attack blocked: matched rule '{match.rule_id}' "
                            f"(category={match.category})"
                        ),
                        rule_id=match.rule_id,
                        category=match.category,
                        violating_text=match.matched_text,
                        match_result=match,
                    )
                else:
                    model = kwargs.get("model", "vaxguard-shield")
                    return build_blocked_completion(
                        model=model,
                        reason=f"Defensive rail triggered: {match.rule_id}",
                        rule_id=match.rule_id,
                    )

        # 2. Forward to Underlying Client
        is_stream = bool(kwargs.get("stream", False))
        res = self._completions.create(*args, **kwargs)

        if is_stream:
            return VaxGuardStream(
                stream=res,
                interceptor=self._guard.streaming_interceptor,
                raise_on_violation=self._guard.raise_on_violation,
                on_violation=self._guard.on_violation,
                fallback_message=self._guard.fallback_message,
            )

        # 3. Non-Streaming Output Inspection
        content = extract_completion_content(res)
        if content:
            interceptor = StreamingTokenInterceptor(hybrid_matcher=self._guard.hybrid_matcher)
            action, clean_text, event = interceptor.analyze_chunk(content)
            if action == InterceptorAction.BLOCK:
                if self._guard.on_violation and event:
                    try:
                        self._guard.on_violation(event)
                    except Exception as err:
                        logger.warning(f"Error in on_violation callback: {err}")

                if self._guard.raise_on_violation:
                    raise VaxGuardSecurityException(
                        message=f"Output rail violation: {event.reason if event else 'Output breach'}",
                        rule_id=event.rule_id if event else "output_rail_violation",
                        category=event.category if event else "security",
                        violating_text=event.violating_sequence if event else "",
                        event=event,
                    )
                else:
                    fallback = self._guard.fallback_message or (
                        "[VAXGUARD SHIELD: Output rail circuit breaker tripped. "
                        "Response blocked to prevent data leakage.]"
                    )
                    set_completion_content(res, fallback)
            elif action == InterceptorAction.REDACT and clean_text:
                set_completion_content(res, clean_text)

        return res

    def __getattr__(self, name: str) -> Any:
        return getattr(self._completions, name)


class VaxGuardAsyncChatCompletions:
    """Guarded wrapper for `client.chat.completions` (asynchronous)."""

    def __init__(self, completions_resource: Any, guard: AsyncVaxGuardOpenAI) -> None:
        self._completions = completions_resource
        self._guard = guard

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        # 1. Input Guardrail Inspection (<1ms)
        messages = kwargs.get("messages") or (args[0] if args else [])
        prompt_text = extract_prompt_from_messages(messages)

        if prompt_text:
            match = self._guard.hybrid_matcher.match(prompt_text)
            if match.matched and match.action == GuardrailAction.BLOCK:
                if self._guard.on_violation:
                    try:
                        self._guard.on_violation(match)
                    except Exception as err:
                        logger.warning(f"Error in on_violation callback: {err}")

                if self._guard.raise_on_violation:
                    raise VaxGuardSecurityException(
                        message=(
                            f"Input prompt attack blocked: matched rule '{match.rule_id}' "
                            f"(category={match.category})"
                        ),
                        rule_id=match.rule_id,
                        category=match.category,
                        violating_text=match.matched_text,
                        match_result=match,
                    )
                else:
                    model = kwargs.get("model", "vaxguard-shield")
                    return build_blocked_completion(
                        model=model,
                        reason=f"Defensive rail triggered: {match.rule_id}",
                        rule_id=match.rule_id,
                    )

        # 2. Forward to Underlying Client
        is_stream = bool(kwargs.get("stream", False))
        res = await self._completions.create(*args, **kwargs)

        if is_stream:
            return VaxGuardAsyncStream(
                stream=res,
                interceptor=self._guard.streaming_interceptor,
                raise_on_violation=self._guard.raise_on_violation,
                on_violation=self._guard.on_violation,
                fallback_message=self._guard.fallback_message,
            )

        # 3. Non-Streaming Output Inspection
        content = extract_completion_content(res)
        if content:
            interceptor = StreamingTokenInterceptor(hybrid_matcher=self._guard.hybrid_matcher)
            action, clean_text, event = interceptor.analyze_chunk(content)
            if action == InterceptorAction.BLOCK:
                if self._guard.on_violation and event:
                    try:
                        self._guard.on_violation(event)
                    except Exception as err:
                        logger.warning(f"Error in on_violation callback: {err}")

                if self._guard.raise_on_violation:
                    raise VaxGuardSecurityException(
                        message=f"Output rail violation: {event.reason if event else 'Output breach'}",
                        rule_id=event.rule_id if event else "output_rail_violation",
                        category=event.category if event else "security",
                        violating_text=event.violating_sequence if event else "",
                        event=event,
                    )
                else:
                    fallback = self._guard.fallback_message or (
                        "[VAXGUARD SHIELD: Output rail circuit breaker tripped. "
                        "Response blocked to prevent data leakage.]"
                    )
                    set_completion_content(res, fallback)
            elif action == InterceptorAction.REDACT and clean_text:
                set_completion_content(res, clean_text)

        return res

    def __getattr__(self, name: str) -> Any:
        return getattr(self._completions, name)


class VaxGuardChat:
    """Wrapper for `client.chat` namespace."""

    def __init__(self, chat_resource: Any, guard: VaxGuardOpenAI) -> None:
        self._chat = chat_resource
        self.completions = VaxGuardChatCompletions(
            getattr(chat_resource, "completions", chat_resource),
            guard,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chat, name)


class VaxGuardAsyncChat:
    """Wrapper for `client.chat` namespace (async)."""

    def __init__(self, chat_resource: Any, guard: AsyncVaxGuardOpenAI) -> None:
        self._chat = chat_resource
        self.completions = VaxGuardAsyncChatCompletions(
            getattr(chat_resource, "completions", chat_resource),
            guard,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chat, name)


# ─── VaxGuard OpenAI Clients ──────────────────────────────────────────────────

class VaxGuardOpenAI:
    """
    Drop-in wrapper around OpenAI Python client (`client.chat.completions.create`).

    Supports:
    - Synchronous completions
    - Streaming completions (`stream=True`) with sliding-window circuit breaker
    - Pass-through to all underlying OpenAI client attributes and endpoints
    """

    def __init__(
        self,
        client: Optional[Any] = None,
        hybrid_matcher: Optional[TriHybridMatcher] = None,
        streaming_interceptor: Optional[StreamingTokenInterceptor] = None,
        hyper_engine: Optional[Any] = None,
        raise_on_violation: bool = True,
        on_violation: Optional[Callable[[Any], Any]] = None,
        fallback_message: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.hybrid_matcher = hybrid_matcher or TriHybridMatcher()
        self.streaming_interceptor = streaming_interceptor or StreamingTokenInterceptor(
            hybrid_matcher=self.hybrid_matcher
        )
        self.hyper_engine = hyper_engine
        self.raise_on_violation = raise_on_violation
        self.on_violation = on_violation
        self.fallback_message = fallback_message

        if client is not None:
            self._client = client
        else:
            try:
                import openai
                key = api_key or os.environ.get("OPENAI_API_KEY", "dummy-key-for-guard")
                self._client = openai.OpenAI(api_key=key, **kwargs)
            except Exception as e:
                logger.debug(f"Could not initialize native openai.OpenAI client: {e}")
                # Create lightweight fallback client
                self._client = SimpleNamespace(
                    chat=SimpleNamespace(completions=SimpleNamespace())
                )

        self.chat = VaxGuardChat(getattr(self._client, "chat", self._client), self)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class AsyncVaxGuardOpenAI:
    """
    Drop-in wrapper around OpenAI Python async client (`client.chat.completions.create`).

    Supports:
    - Asynchronous completions (`await client.chat.completions.create(...)`)
    - Streaming completions (`stream=True`) with sliding-window circuit breaker
    - Pass-through to all underlying OpenAI client attributes and endpoints
    """

    def __init__(
        self,
        client: Optional[Any] = None,
        hybrid_matcher: Optional[TriHybridMatcher] = None,
        streaming_interceptor: Optional[StreamingTokenInterceptor] = None,
        hyper_engine: Optional[Any] = None,
        raise_on_violation: bool = True,
        on_violation: Optional[Callable[[Any], Any]] = None,
        fallback_message: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.hybrid_matcher = hybrid_matcher or TriHybridMatcher()
        self.streaming_interceptor = streaming_interceptor or StreamingTokenInterceptor(
            hybrid_matcher=self.hybrid_matcher
        )
        self.hyper_engine = hyper_engine
        self.raise_on_violation = raise_on_violation
        self.on_violation = on_violation
        self.fallback_message = fallback_message

        if client is not None:
            self._client = client
        else:
            try:
                import openai
                key = api_key or os.environ.get("OPENAI_API_KEY", "dummy-key-for-guard")
                self._client = openai.AsyncOpenAI(api_key=key, **kwargs)
            except Exception as e:
                logger.debug(f"Could not initialize native openai.AsyncOpenAI client: {e}")
                self._client = SimpleNamespace(
                    chat=SimpleNamespace(completions=SimpleNamespace())
                )

        self.chat = VaxGuardAsyncChat(getattr(self._client, "chat", self._client), self)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


# Convenience alias
VaxGuardAsyncOpenAI = AsyncVaxGuardOpenAI
