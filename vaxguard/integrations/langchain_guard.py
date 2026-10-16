"""
VaxGuard Enterprise Ecosystem Adapter: LangChain Integration.

Provides production-grade, zero-external-heavy-dependency integrations for LangChain / LangGraph:
1. VaxGuardLangChainCallback: Intercepts on_llm_start, on_llm_new_token, on_llm_end with circuit breaker.
2. VaxGuardRunnable: A LangChain-compatible LCEL runnable Runnable[str, str] filtering prompts & completions.
3. VaxGuardSecurityViolation: Structured security violation exception.

Pure Python 3.10+, deterministic execution SLA < 1.0ms, zero mandatory heavy binaries.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

from vaxguard.core.logger import get_logger
from vaxguard.middleware.exceptions import VaxGuardError
from vaxguard.nemo.hybrid_matcher import (
    GuardrailAction,
    TriHybridMatcher,
)
from vaxguard.nemo.streaming_interceptor import (
    InterceptorAction,
    StreamingTokenInterceptor,
)

logger = get_logger("LangChainGuard")

# Optional LangChain base imports (graceful fallback if langchain_core is not installed)
try:
    from langchain_core.callbacks.base import BaseCallbackHandler
except ImportError:
    class BaseCallbackHandler:  # type: ignore[no-redef]
        """Fallback base class when langchain_core is not installed."""
        pass

try:
    from langchain_core.runnables.base import Runnable as LangChainCoreRunnable
except ImportError:
    class LangChainCoreRunnable:  # type: ignore[no-redef]
        """Fallback base class when langchain_core is not installed."""
        pass


# ─── Security Violation Exception ─────────────────────────────────────────────

class VaxGuardSecurityViolation(VaxGuardError):
    """
    Raised when a prompt injection, jailbreak, data leakage, or adversarial attack
    is intercepted by a VaxGuard integration adapter.
    """

    def __init__(
        self,
        message: str,
        category: str = "security_violation",
        rule_id: str = "unknown",
        confidence: float = 1.0,
        payload_sample: Optional[str] = None,
        action: str = "block",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.category = category
        self.rule_id = rule_id
        self.confidence = confidence
        self.payload_sample = payload_sample or ""
        self.action = action
        self.metadata = metadata or {}
        self.timestamp = time.time()

    def __str__(self) -> str:
        sample = f" (payload: '{self.payload_sample[:60]}...')" if self.payload_sample else ""
        return (
            f"[VaxGuardSecurityViolation] category='{self.category}' "
            f"rule='{self.rule_id}' conf={self.confidence:.2f}: {self.message}{sample}"
        )


# ─── LangChain Callback Handler ───────────────────────────────────────────────

class VaxGuardLangChainCallback(BaseCallbackHandler):
    """
    Production-grade LangChain callback handler.
    Intercepts:
      - `on_llm_start` / `on_chat_model_start`: Scans input prompts for jailbreaks and prompt injections.
      - `on_llm_new_token`: Sliding-window streaming analysis to trip circuit breaker mid-flight.
      - `on_llm_end`: Validates final generated outputs for sensitive data leaks or toxic completions.

    Raises `VaxGuardSecurityViolation` immediately upon detecting an adversarial pattern or policy breach.
    """

    def __init__(
        self,
        matcher: Optional[TriHybridMatcher] = None,
        streaming_interceptor: Optional[StreamingTokenInterceptor] = None,
        block_input: bool = True,
        block_output: bool = True,
        raise_exception: bool = True,
        window_chars: int = 512,
        on_violation: Optional[Callable[[VaxGuardSecurityViolation], Any]] = None,
    ) -> None:
        super().__init__()
        self.matcher = matcher or TriHybridMatcher()
        self.streaming_interceptor = streaming_interceptor or StreamingTokenInterceptor(
            hybrid_matcher=self.matcher,
            window_chars=window_chars,
        )
        self.block_input = block_input
        self.block_output = block_output
        self.raise_exception = raise_exception
        self.on_violation = on_violation

        self.violations: List[VaxGuardSecurityViolation] = []
        self.prompts_scanned: int = 0
        self.tokens_scanned: int = 0
        self.outputs_scanned: int = 0
        self._circuit_open: bool = False

    @property
    def is_tripped(self) -> bool:
        return self._circuit_open or self.streaming_interceptor.is_tripped

    def reset(self) -> None:
        """Reset interceptor state, clearing buffers and trip events."""
        self._circuit_open = False
        self.violations.clear()
        self.streaming_interceptor.reset()

    def _handle_violation(self, violation: VaxGuardSecurityViolation) -> None:
        self.violations.append(violation)
        self._circuit_open = True
        logger.warning(str(violation))

        if self.on_violation:
            try:
                self.on_violation(violation)
            except Exception as ex:
                logger.error(f"Error in on_violation callback: {ex}")

        if self.raise_exception:
            raise violation

    # 1. Input Inspection: on_llm_start & on_chat_model_start
    def on_llm_start(
        self,
        serialized: Optional[Dict[str, Any]] = None,
        prompts: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Intercepts LLM initiation. Evaluates all incoming prompt strings.
        Raises VaxGuardSecurityViolation if an attack is detected.
        """
        if not self.block_input or not prompts:
            return

        for prompt in prompts:
            self.prompts_scanned += 1
            match_res = self.matcher.match(prompt)
            if match_res.matched and match_res.action == GuardrailAction.BLOCK:
                violation = VaxGuardSecurityViolation(
                    message=f"Prompt blocked due to {match_res.category} ({match_res.rule_id})",
                    category=match_res.category,
                    rule_id=match_res.rule_id,
                    confidence=match_res.confidence,
                    payload_sample=prompt[:120],
                    metadata={"engine": match_res.engine_type.value if hasattr(match_res.engine_type, 'value') else str(match_res.engine_type)},
                )
                self._handle_violation(violation)

    def on_chat_model_start(
        self,
        serialized: Optional[Dict[str, Any]] = None,
        messages: Optional[List[List[Any]]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Intercepts Chat model initiation. Unpacks message objects or dicts and validates contents.
        """
        if not self.block_input or not messages:
            return

        for message_list in messages:
            for msg in message_list:
                text = ""
                if isinstance(msg, str):
                    text = msg
                elif isinstance(msg, dict):
                    text = str(msg.get("content", ""))
                elif hasattr(msg, "content"):
                    text = str(msg.content)
                else:
                    text = str(msg)

                if text:
                    self.on_llm_start(serialized=serialized, prompts=[text], **kwargs)

    # 2. Streaming Token Inspection: on_llm_new_token
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """
        Real-time per-token streaming analysis.
        Trips circuit breaker instantly if output leak or toxic token sequence is detected.
        """
        if not self.block_output or not token:
            return

        self.tokens_scanned += 1
        action, _, event = self.streaming_interceptor.analyze_chunk(
            chunk=token,
            chunk_index=self.tokens_scanned,
        )

        if action == InterceptorAction.BLOCK or event is not None:
            rule_id = event.rule_id if event else "streaming_circuit_breaker"
            category = event.category if event else "data_leakage"
            reason = event.reason if event else "Streaming token sequence tripped output circuit breaker"
            sample = (
                event.violating_sequence
                if (event and event.violating_sequence)
                else getattr(self.streaming_interceptor._buffer, "current_window", token)
            )
            violation = VaxGuardSecurityViolation(
                message=reason,
                category=category,
                rule_id=rule_id,
                confidence=1.0,
                payload_sample=sample[:120],
                metadata={"tokens_scanned": self.tokens_scanned},
            )
            self._handle_violation(violation)

    # 3. Output Completion Inspection: on_llm_end
    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """
        Intercepts completed LLM generation. Validates full completion texts against
        leakage patterns, prompt disclosures, and toxic content.
        """
        if not self.block_output or response is None:
            return

        # Extract generated strings across different LangChain LLMResult formats
        generated_texts: List[str] = []

        if hasattr(response, "generations"):
            generations = response.generations
            for gen_batch in generations:
                if isinstance(gen_batch, list):
                    for g in gen_batch:
                        if hasattr(g, "text"):
                            generated_texts.append(g.text)
                        elif isinstance(g, dict) and "text" in g:
                            generated_texts.append(g["text"])
                        elif hasattr(g, "message") and hasattr(g.message, "content"):
                            generated_texts.append(str(g.message.content))
                elif hasattr(gen_batch, "text"):
                    generated_texts.append(gen_batch.text)
        elif isinstance(response, dict):
            if "generations" in response:
                for g_batch in response["generations"]:
                    if isinstance(g_batch, list):
                        for g in g_batch:
                            if isinstance(g, dict) and "text" in g:
                                generated_texts.append(g["text"])
                    elif isinstance(g_batch, dict) and "text" in g_batch:
                        generated_texts.append(g_batch["text"])
            elif "text" in response:
                generated_texts.append(response["text"])
        elif isinstance(response, str):
            generated_texts.append(response)

        for text in generated_texts:
            self.outputs_scanned += 1
            # Check with output patterns via streaming interceptor patterns & hybrid matcher
            for pat, rule_id, desc in self.streaming_interceptor._compiled_patterns:
                if pat.search(text):
                    violation = VaxGuardSecurityViolation(
                        message=f"Output generation blocked: {desc} ({rule_id})",
                        category="data_leakage",
                        rule_id=rule_id,
                        confidence=1.0,
                        payload_sample=text[:120],
                    )
                    self._handle_violation(violation)
                    return

            match_res = self.matcher.match(text)
            if match_res.matched and match_res.action == GuardrailAction.BLOCK:
                violation = VaxGuardSecurityViolation(
                    message=f"Output generation blocked due to {match_res.category} ({match_res.rule_id})",
                    category=match_res.category,
                    rule_id=match_res.rule_id,
                    confidence=match_res.confidence,
                    payload_sample=text[:120],
                )
                self._handle_violation(violation)
                return

    # 4. Async Callbacks (for asynchronous LangChain execution)
    async def on_llm_start_async(
        self,
        serialized: Optional[Dict[str, Any]] = None,
        prompts: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        self.on_llm_start(serialized=serialized, prompts=prompts, **kwargs)

    async def on_chat_model_start_async(
        self,
        serialized: Optional[Dict[str, Any]] = None,
        messages: Optional[List[List[Any]]] = None,
        **kwargs: Any,
    ) -> None:
        self.on_chat_model_start(serialized=serialized, messages=messages, **kwargs)

    async def on_llm_new_token_async(self, token: str, **kwargs: Any) -> None:
        self.on_llm_new_token(token, **kwargs)

    async def on_llm_end_async(self, response: Any, **kwargs: Any) -> None:
        self.on_llm_end(response, **kwargs)


# ─── LangChain LCEL Runnable ──────────────────────────────────────────────────

InT = TypeVar("InT")
OutT = TypeVar("OutT")


class VaxGuardRunnableSequence(LangChainCoreRunnable, Generic[InT, OutT]):
    """
    Chained sequence of LangChain Runnables or callables with VaxGuard guardrails.
    Allows pipe composition: `guard | llm | guard`.
    """

    def __init__(self, steps: Sequence[Any]) -> None:
        self.steps = list(steps)

    def __or__(self, other: Any) -> VaxGuardRunnableSequence[InT, Any]:
        if isinstance(other, VaxGuardRunnableSequence):
            return VaxGuardRunnableSequence(self.steps + other.steps)
        return VaxGuardRunnableSequence(self.steps + [other])

    def __ror__(self, other: Any) -> VaxGuardRunnableSequence[Any, OutT]:
        if isinstance(other, VaxGuardRunnableSequence):
            return VaxGuardRunnableSequence(other.steps + self.steps)
        return VaxGuardRunnableSequence([other] + self.steps)

    def invoke(self, input: Any, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        current = input
        for step in self.steps:
            if hasattr(step, "invoke"):
                current = step.invoke(current, config=config, **kwargs)
            elif callable(step):
                current = step(current)
            else:
                raise TypeError(f"Object {step} is not callable or runnable.")
        return current

    async def ainvoke(self, input: Any, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        current = input
        for step in self.steps:
            if hasattr(step, "ainvoke"):
                current = await step.ainvoke(current, config=config, **kwargs)
            elif hasattr(step, "invoke"):
                if inspect.iscoroutinefunction(step.invoke):
                    current = await step.invoke(current, config=config, **kwargs)
                else:
                    current = step.invoke(current, config=config, **kwargs)
            elif callable(step):
                if inspect.iscoroutinefunction(step):
                    current = await step(current)
                else:
                    current = step(current)
            else:
                raise TypeError(f"Object {step} is not callable or runnable.")
        return current

    def batch(self, inputs: Sequence[Any], config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> List[Any]:
        return [self.invoke(inp, config=config, **kwargs) for inp in inputs]

    async def abatch(self, inputs: Sequence[Any], config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> List[Any]:
        tasks = [self.ainvoke(inp, config=config, **kwargs) for inp in inputs]
        return await asyncio.gather(*tasks)


class VaxGuardRunnable(LangChainCoreRunnable, Generic[InT, OutT]):
    """
    LangChain Expression Language (LCEL) compatible Runnable component.
    Acts as a secure firewall step in LCEL pipelines:
      `pipeline = VaxGuardRunnable(mode="input") | llm | VaxGuardRunnable(mode="output")`

    SLA: < 1.0ms per invoke, zero heavy dependencies.
    """

    def __init__(
        self,
        mode: str = "auto",  # 'input', 'output', or 'auto'
        matcher: Optional[TriHybridMatcher] = None,
        raise_exception: bool = True,
        fallback_text: str = "[VAXGUARD SHIELD: Request blocked due to security policy violation]",
        sanitize_instead_of_raise: bool = False,
    ) -> None:
        super().__init__()
        self.mode = mode.lower()
        self.matcher = matcher or TriHybridMatcher()
        self.raise_exception = raise_exception
        self.fallback_text = fallback_text
        self.sanitize_instead_of_raise = sanitize_instead_of_raise

    def __or__(self, other: Any) -> VaxGuardRunnableSequence:
        return VaxGuardRunnableSequence([self, other])

    def __ror__(self, other: Any) -> VaxGuardRunnableSequence:
        return VaxGuardRunnableSequence([other, self])

    def _extract_text(self, input_data: Any) -> Tuple[str, bool]:
        """Extract text representation from string, dict, or message objects."""
        if isinstance(input_data, str):
            return input_data, True
        if isinstance(input_data, dict):
            for key in ("prompt", "input", "query", "text", "content"):
                if key in input_data and isinstance(input_data[key], str):
                    return input_data[key], False
            return str(input_data), False
        if hasattr(input_data, "content") and isinstance(input_data.content, str):
            return input_data.content, False
        return str(input_data), False

    def _validate_or_transform(self, input_val: Any) -> Any:
        text, is_raw_str = self._extract_text(input_val)

        match_res = self.matcher.match(text)
        if match_res.matched and match_res.action == GuardrailAction.BLOCK:
            if self.raise_exception and not self.sanitize_instead_of_raise:
                raise VaxGuardSecurityViolation(
                    message=f"VaxGuardRunnable blocked {self.mode} payload due to {match_res.category} ({match_res.rule_id})",
                    category=match_res.category,
                    rule_id=match_res.rule_id,
                    confidence=match_res.confidence,
                    payload_sample=text[:120],
                )
            # Fallback or sanitization
            if is_raw_str:
                return self.fallback_text
            if isinstance(input_val, dict):
                modified = dict(input_val)
                for key in ("prompt", "input", "query", "text", "content"):
                    if key in modified:
                        modified[key] = self.fallback_text
                        return modified
            return self.fallback_text

        return input_val

    def invoke(self, input: Any, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        """Synchronously filters input or output text."""
        return self._validate_or_transform(input)

    async def ainvoke(self, input: Any, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        """Asynchronously filters input or output text."""
        return self._validate_or_transform(input)

    def batch(self, inputs: Sequence[Any], config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> List[Any]:
        """Processes a sequence of inputs."""
        return [self.invoke(item, config=config, **kwargs) for item in inputs]

    async def abatch(self, inputs: Sequence[Any], config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> List[Any]:
        """Asynchronously processes a sequence of inputs."""
        return [await self.ainvoke(item, config=config, **kwargs) for item in inputs]

    def stream(self, input: Any, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Iterator[Any]:
        """Yields filtered stream result."""
        yield self.invoke(input, config=config, **kwargs)

    async def astream(self, input: Any, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> AsyncIterator[Any]:
        """Asynchronously yields filtered stream result."""
        yield await self.ainvoke(input, config=config, **kwargs)
