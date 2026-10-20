"""
VaxGuard Enterprise Ecosystem Adapter: HuggingFace Transformers Integration.

Provides production-grade, zero-external-heavy-dependency integrations for HuggingFace Transformers:
1. VaxGuardHuggingFaceLogitsProcessor: PyTorch/HuggingFace compatible logits processor / stopping criteria
   that intercepts tokens during local inference with zero mandatory torch dependency (pure duck-typing).
2. VaxGuardHuggingFaceStoppingCriteria: HuggingFace compatible stopping criteria wrapping VaxGuard defense.
3. VaxGuardStreamerWrapper: Wraps HuggingFace TextIteratorStreamer to intercept streaming tokens and trip
   circuit breaker mid-flight.

Pure Python 3.10+, deterministic execution SLA < 1.0ms, zero mandatory heavy binaries.
"""

from __future__ import annotations

import asyncio
import queue
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
from vaxguard.integrations.langchain_guard import VaxGuardSecurityViolation
from vaxguard.nemo.hybrid_matcher import (
    GuardrailAction,
    TriHybridMatcher,
)
from vaxguard.nemo.streaming_interceptor import (
    InterceptorAction,
    StreamingTokenInterceptor,
)

logger = get_logger("HuggingFaceGuard")

# Optional PyTorch import (graceful fallback)
try:
    import torch  # type: ignore[import-untyped]
    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False

# Optional Hugging Face Transformers base classes (graceful fallback)
try:
    from transformers.generation.logits_process import LogitsProcessor  # type: ignore[import-untyped]
except Exception:
    class LogitsProcessor:  # type: ignore[no-redef]
        """Fallback base class when transformers is not installed."""
        pass

try:
    from transformers.generation.stopping_criteria import StoppingCriteria  # type: ignore[import-untyped]
except Exception:
    class StoppingCriteria:  # type: ignore[no-redef]
        """Fallback base class when transformers is not installed."""
        pass

try:
    from transformers.generation.streamers import BaseStreamer  # type: ignore[import-untyped]
except Exception:
    class BaseStreamer:  # type: ignore[no-redef]
        """Fallback BaseStreamer when transformers is not installed."""
        def put(self, value: Any) -> None:
            pass

        def end(self) -> None:
            pass


# ─── Helper: Token Decoders & Tensor Duck-Typing ───────────────────────────────

def _extract_token_ids(batch_row: Any) -> List[int]:
    """Extract flat list of integer token IDs from torch tensor, numpy array, or list."""
    if hasattr(batch_row, "tolist"):
        raw = batch_row.tolist()
        if isinstance(raw, list):
            # Flatten if nested
            if raw and isinstance(raw[0], list):
                return [int(item) for sublist in raw for item in sublist]
            return [int(x) for x in raw]
    if isinstance(batch_row, (list, tuple)):
        flat: List[int] = []
        for x in batch_row:
            if isinstance(x, (list, tuple)):
                flat.extend(int(item) for item in x)
            else:
                flat.append(int(x))
        return flat
    try:
        return [int(batch_row)]
    except (TypeError, ValueError):
        return []


def _extract_batch_sequences(input_ids: Any) -> List[List[int]]:
    """Convert 1D or 2D batch of input_ids into List[List[int]]."""
    if hasattr(input_ids, "ndim"):
        if input_ids.ndim == 1:
            return [_extract_token_ids(input_ids)]
        return [_extract_token_ids(input_ids[i]) for i in range(input_ids.shape[0])]
    if isinstance(input_ids, (list, tuple)):
        if not input_ids:
            return [[]]
        if isinstance(input_ids[0], (list, tuple)) or hasattr(input_ids[0], "tolist"):
            return [_extract_token_ids(row) for row in input_ids]
        return [_extract_token_ids(input_ids)]
    return [_extract_token_ids(input_ids)]


def _default_token_decode(token_ids: Sequence[int]) -> str:
    """Fallback token decoder when no tokenizer is supplied."""
    chars: List[str] = []
    for tid in token_ids:
        if 32 <= tid <= 126:
            chars.append(chr(tid))
        elif tid == 10:
            chars.append("\n")
        elif tid == 9:
            chars.append("\t")
        elif tid == 32:
            chars.append(" ")
        else:
            # Map high-order tokens or non-ascii into readable token repr
            chars.append(f"<{tid}>")
    return "".join(chars)


# ─── HuggingFace Logits Processor & Stopping Criteria ─────────────────────────

class VaxGuardHuggingFaceLogitsProcessor(LogitsProcessor):
    """
    Production-grade HuggingFace Transformers compatible Logits Processor & Stopping Criteria.
    Intercepts tokens during local autoregressive generation (e.g. `model.generate()`).

    Key Capabilities:
    - Zero mandatory PyTorch dependency (duck-typing compatible with torch, numpy, and python lists).
    - Auto-detects prompt boundary to scan incoming prompts and streaming continuation tokens.
    - Matches sliding window token sequence against VaxGuard Tri-Hybrid matcher.
    - Forces immediate generation termination (via EOS token injection or score masking)
      or raises VaxGuardSecurityViolation.
    - Dual-purpose: works as a `LogitsProcessor` or `StoppingCriteria`.
    """

    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        matcher: Optional[TriHybridMatcher] = None,
        eos_token_id: Optional[Union[int, Sequence[int]]] = None,
        raise_exception: bool = False,
        block_action: str = "force_eos",  # "force_eos", "mask_all", "stop"
        window_tokens: int = 128,
        scan_prompt: bool = True,
        prompt_length: Optional[Union[int, Sequence[int]]] = None,
        on_violation: Optional[Callable[[VaxGuardSecurityViolation], Any]] = None,
        mode: str = "logits_processor",  # "logits_processor" or "stopping_criteria"
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.matcher = matcher or TriHybridMatcher()
        self.raise_exception = raise_exception
        self.block_action = block_action
        self.window_tokens = window_tokens
        self.scan_prompt = scan_prompt
        self.on_violation = on_violation
        self.mode = mode

        # Resolve EOS token IDs
        self.eos_token_ids: List[int] = []
        if eos_token_id is not None:
            if isinstance(eos_token_id, int):
                self.eos_token_ids = [eos_token_id]
            elif isinstance(eos_token_id, (list, tuple, set)):
                self.eos_token_ids = [int(x) for x in eos_token_id]
            else:
                self.eos_token_ids = [int(x) for x in eos_token_id]  # type: ignore[union-attr]
        elif tokenizer is not None and hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
            if isinstance(tokenizer.eos_token_id, (list, tuple, set)):
                self.eos_token_ids = [int(x) for x in tokenizer.eos_token_id]
            else:
                self.eos_token_ids = [int(tokenizer.eos_token_id)]
        else:
            self.eos_token_ids = [2]  # Common default EOS (e.g. LLaMA/GPT)

        # Internal state
        self.violations: List[VaxGuardSecurityViolation] = []
        self.tokens_scanned: int = 0
        self.calls_count: int = 0
        self._circuit_open: bool = False
        self._initial_prompt_lens: Dict[int, int] = {}
        if prompt_length is not None:
            if isinstance(prompt_length, int):
                self._initial_prompt_lens[0] = prompt_length
            elif isinstance(prompt_length, (list, tuple)):
                for idx, pl in enumerate(prompt_length):
                    self._initial_prompt_lens[idx] = int(pl)

    @property
    def is_tripped(self) -> bool:
        """Returns True if any security violation has tripped the circuit breaker."""
        return self._circuit_open

    def reset(self) -> None:
        """Reset interceptor state, clearing tracked prompt lengths and violation events."""
        self._circuit_open = False
        self.violations.clear()
        self._initial_prompt_lens.clear()
        self.tokens_scanned = 0
        self.calls_count = 0

    def decode_tokens(self, token_ids: Sequence[int]) -> str:
        """Decode a sequence of token IDs to text string using tokenizer or fallback."""
        if not token_ids:
            return ""
        if self.tokenizer is not None:
            if hasattr(self.tokenizer, "decode"):
                try:
                    return str(self.tokenizer.decode(token_ids, skip_special_tokens=True))
                except Exception as err:
                    logger.debug(f"Tokenizer.decode failed: {err}")
            elif callable(self.tokenizer):
                try:
                    return str(self.tokenizer(token_ids))
                except Exception as err:
                    logger.debug(f"Tokenizer callable failed: {err}")
        return _default_token_decode(token_ids)

    def _handle_violation(self, violation: VaxGuardSecurityViolation) -> None:
        """Record violation, trip breaker, invoke callback, and optionally raise exception."""
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

    def _apply_force_eos(self, scores: Any, batch_idx: int) -> None:
        """Modify scores to force immediate emission of EOS token."""
        eos_id = self.eos_token_ids[0] if self.eos_token_ids else 2

        # 1. PyTorch Tensor
        if HAS_TORCH and torch is not None and isinstance(scores, torch.Tensor):
            if scores.ndim == 2:
                scores[batch_idx, :] = -float("inf")
                if eos_id < scores.shape[-1]:
                    scores[batch_idx, eos_id] = 0.0
            elif scores.ndim == 1:
                scores[:] = -float("inf")
                if eos_id < len(scores):
                    scores[eos_id] = 0.0
            return

        # 2. Numpy ndarray or duck-typed tensor
        if hasattr(scores, "shape") and hasattr(scores, "__setitem__"):
            neg_inf = -float("inf")
            if len(scores.shape) == 2:
                scores[batch_idx, :] = neg_inf
                if eos_id < scores.shape[-1]:
                    scores[batch_idx, eos_id] = 0.0
            elif len(scores.shape) == 1:
                scores[:] = neg_inf
                if eos_id < len(scores):
                    scores[eos_id] = 0.0
            return

        # 3. Python list of lists
        if isinstance(scores, list):
            if scores and isinstance(scores[0], list):
                target_row = scores[batch_idx] if batch_idx < len(scores) else scores[0]
                neg_inf = -float("inf")
                for k in range(len(target_row)):
                    target_row[k] = neg_inf
                if eos_id < len(target_row):
                    target_row[eos_id] = 0.0
            else:
                neg_inf = -float("inf")
                for k in range(len(scores)):
                    scores[k] = neg_inf
                if eos_id < len(scores):
                    scores[eos_id] = 0.0

    def inspect_tokens(self, input_ids: Any) -> Optional[VaxGuardSecurityViolation]:
        """
        Inspect the current input_ids across all batch sequences.
        Returns the first detected violation, or None if clean.
        """
        batch_seqs = _extract_batch_sequences(input_ids)
        is_first_step = (self.calls_count == 0 and not self._initial_prompt_lens)

        for batch_idx, seq in enumerate(batch_seqs):
            if not seq:
                continue

            # Auto-record prompt boundary on step 0
            if batch_idx not in self._initial_prompt_lens:
                self._initial_prompt_lens[batch_idx] = len(seq)
                if self.scan_prompt and is_first_step:
                    # Scan input prompt on first evaluation
                    prompt_text = self.decode_tokens(seq)
                    match_res = self.matcher.match(prompt_text)
                    if match_res.matched and match_res.action == GuardrailAction.BLOCK:
                        violation = VaxGuardSecurityViolation(
                            message=f"Prompt blocked: {match_res.category} ({match_res.rule_id})",
                            category=match_res.category,
                            rule_id=match_res.rule_id,
                            confidence=match_res.confidence,
                            payload_sample=prompt_text[:120],
                            metadata={"batch_index": batch_idx, "phase": "prompt"},
                        )
                        return violation

            # Scan generated continuation tokens within sliding window
            prompt_len = self._initial_prompt_lens.get(batch_idx, 0)
            generated_tokens = seq[prompt_len:] if len(seq) > prompt_len else seq
            self.tokens_scanned += max(1, len(generated_tokens))

            window_tokens = (
                generated_tokens[-self.window_tokens:]
                if len(generated_tokens) > self.window_tokens
                else generated_tokens
            )
            decoded_window = self.decode_tokens(window_tokens)
            if not decoded_window.strip():
                continue

            match_res = self.matcher.match(decoded_window)
            if match_res.matched and match_res.action == GuardrailAction.BLOCK:
                violation = VaxGuardSecurityViolation(
                    message=f"Generation blocked: {match_res.category} ({match_res.rule_id})",
                    category=match_res.category,
                    rule_id=match_res.rule_id,
                    confidence=match_res.confidence,
                    payload_sample=decoded_window[:120],
                    metadata={"batch_index": batch_idx, "phase": "generation"},
                )
                return violation

        return None

    def should_stop(self, input_ids: Any, scores: Optional[Any] = None, **kwargs: Any) -> bool:
        """
        Evaluate if generation should halt immediately.
        Implements HuggingFace StoppingCriteria contract.
        """
        if self._circuit_open:
            return True

        violation = self.inspect_tokens(input_ids)
        if violation is not None:
            self._handle_violation(violation)
            return True

        return False

    def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> Any:
        """
        Execute logits processor or stopping criteria step.
        When mode='logits_processor', returns modified scores.
        When mode='stopping_criteria', returns boolean stop decision.
        """
        self.calls_count += 1

        # Check stopping criteria mode
        if self.mode == "stopping_criteria":
            return self.should_stop(input_ids, scores, **kwargs)

        # Logits processor mode
        if self._circuit_open:
            batch_seqs = _extract_batch_sequences(input_ids)
            for b_idx in range(len(batch_seqs)):
                self._apply_force_eos(scores, b_idx)
            return scores

        violation = self.inspect_tokens(input_ids)
        if violation is not None:
            batch_idx = violation.metadata.get("batch_index", 0)
            self._handle_violation(violation)
            self._apply_force_eos(scores, batch_idx)
            return scores

        return scores

    def as_stopping_criteria(self) -> VaxGuardHuggingFaceStoppingCriteria:
        """Return a dedicated StoppingCriteria adapter wrapping this processor."""
        return VaxGuardHuggingFaceStoppingCriteria(self)

    def get_metrics(self) -> Dict[str, Any]:
        """Return operational telemetry."""
        return {
            "calls_count": self.calls_count,
            "tokens_scanned": self.tokens_scanned,
            "violations_count": len(self.violations),
            "is_tripped": self.is_tripped,
            "circuit_open": self._circuit_open,
        }


class VaxGuardHuggingFaceStoppingCriteria(StoppingCriteria):
    """
    HuggingFace `StoppingCriteria` compatible adapter wrapping `VaxGuardHuggingFaceLogitsProcessor`.
    HuggingFace generation loops invoke `criteria(input_ids, scores) -> bool`.
    """

    def __init__(self, processor: VaxGuardHuggingFaceLogitsProcessor) -> None:
        super().__init__()
        self.processor = processor

    def __call__(self, input_ids: Any, scores: Any = None, **kwargs: Any) -> bool:
        """Returns True if generation should halt due to a security violation."""
        return self.processor.should_stop(input_ids, scores, **kwargs)

    @property
    def is_tripped(self) -> bool:
        return self.processor.is_tripped

    @property
    def violations(self) -> List[VaxGuardSecurityViolation]:
        return self.processor.violations

    def reset(self) -> None:
        self.processor.reset()


# ─── HuggingFace Streamer Wrapper ─────────────────────────────────────────────

class VaxGuardStreamerWrapper(BaseStreamer):
    """
    Wraps HuggingFace `TextIteratorStreamer` (or any text iterator/streamer) to intercept
    streaming tokens in real-time and trip the circuit breaker mid-flight.

    Key Capabilities:
    - Zero heavy dependencies; wraps any iterable of text chunks or HuggingFace streamer.
    - Intercepts leaking credentials, PII, prompt injections, and toxic outputs mid-stream.
    - Synchronous (`for chunk in streamer`) and asynchronous (`async for chunk in streamer`) support.
    - In-flight redaction (e.g. masking SSNs/API keys) or instant circuit-breaker termination.
    - Can act as a direct drop-in streamer for `model.generate(..., streamer=guard_streamer)`.
    """

    DEFAULT_FALLBACK = (
        "\n\n[VAXGUARD SHIELD: Output rail circuit breaker tripped. "
        "Stream terminated immediately to prevent data leakage or policy violation.]"
    )

    def __init__(
        self,
        streamer: Optional[Any] = None,
        matcher: Optional[TriHybridMatcher] = None,
        streaming_interceptor: Optional[StreamingTokenInterceptor] = None,
        tokenizer: Optional[Any] = None,
        raise_exception: bool = False,
        fallback_message: Optional[str] = None,
        window_chars: int = 512,
        on_violation: Optional[Callable[[VaxGuardSecurityViolation], Any]] = None,
    ) -> None:
        super().__init__()
        self.streamer = streamer
        self.tokenizer = tokenizer
        self.matcher = matcher or TriHybridMatcher()
        self.interceptor = streaming_interceptor or StreamingTokenInterceptor(
            hybrid_matcher=self.matcher,
            window_chars=window_chars,
            fallback_message=fallback_message or self.DEFAULT_FALLBACK,
        )
        self.raise_exception = raise_exception
        self.fallback_message = fallback_message or self.DEFAULT_FALLBACK
        self.on_violation = on_violation

        # Standalone queue if no underlying streamer is supplied
        self._internal_queue: queue.Queue[Any] = queue.Queue()
        self._stop_signal = object()

        # Metrics and state
        self.chunks_scanned: int = 0
        self.tokens_scanned: int = 0
        self.violations: List[VaxGuardSecurityViolation] = []
        self._circuit_open: bool = False
        self._iterator: Optional[Iterator[str]] = None

    @property
    def is_tripped(self) -> bool:
        """Returns True if the circuit breaker has been tripped."""
        return self._circuit_open or self.interceptor.is_tripped

    def reset(self) -> None:
        """Reset the streamer wrapper and circuit breaker to closed state."""
        self._circuit_open = False
        self.violations.clear()
        self.chunks_scanned = 0
        self.tokens_scanned = 0
        self._iterator = None
        self.interceptor.reset()
        while not self._internal_queue.empty():
            try:
                self._internal_queue.get_nowait()
            except queue.Empty:
                break

    def trip_breaker(
        self,
        reason: str = "Circuit breaker tripped manually",
        rule_id: str = "manual_trip",
        category: str = "security_override",
    ) -> None:
        """Manually trip the circuit breaker."""
        self._circuit_open = True
        self.interceptor.trip_breaker(reason=reason, rule_id=rule_id, category=category)
        violation = VaxGuardSecurityViolation(
            message=reason,
            category=category,
            rule_id=rule_id,
            confidence=1.0,
            payload_sample="",
        )
        self.violations.append(violation)
        if self.on_violation:
            try:
                self.on_violation(violation)
            except Exception as e:
                logger.error(f"Error in on_violation: {e}")

    # Hugging Face BaseStreamer interface
    def put(self, value: Any) -> None:
        """
        Receives tokens from HuggingFace generation thread.
        Forwards to underlying streamer if available, or queues decoded text chunk.
        """
        if self.streamer is not None and hasattr(self.streamer, "put"):
            self.streamer.put(value)
            return

        # Standalone handling: decode value if tokenizer available, else str
        token_ids = _extract_token_ids(value)
        if token_ids:
            chunk = _default_token_decode(token_ids)
            if self.tokenizer is not None and hasattr(self.tokenizer, "decode"):
                try:
                    chunk = self.tokenizer.decode(token_ids, skip_special_tokens=True)
                except Exception:
                    pass
            self._internal_queue.put(chunk)
        elif isinstance(value, str):
            self._internal_queue.put(value)

    def end(self) -> None:
        """Signals end of generation."""
        if self.streamer is not None and hasattr(self.streamer, "end"):
            self.streamer.end()
            return
        self._internal_queue.put(self._stop_signal)

    def _process_chunk(
        self, raw_chunk: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Internal inspection of a raw string chunk.
        Returns: (should_continue, chunk_to_emit)
        """
        if self._circuit_open or self.interceptor.is_tripped:
            return False, None

        self.chunks_scanned += 1
        self.tokens_scanned += max(1, len(raw_chunk) // 4)

        action, clean_chunk, event = self.interceptor.analyze_chunk(
            raw_chunk, self.chunks_scanned
        )

        if action == InterceptorAction.BLOCK:
            self._circuit_open = True
            rule_id = event.rule_id if event else "streaming_block"
            category = event.category if event else "leakage_prevention"
            violation = VaxGuardSecurityViolation(
                message=f"Streaming chunk blocked: {category} ({rule_id})",
                category=category,
                rule_id=rule_id,
                confidence=1.0,
                payload_sample=raw_chunk[:120],
                metadata={"chunk_index": self.chunks_scanned},
            )
            self.violations.append(violation)

            if self.on_violation:
                try:
                    self.on_violation(violation)
                except Exception as ex:
                    logger.error(f"Error in on_violation: {ex}")

            if self.raise_exception:
                raise violation

            return False, self.fallback_message

        if action == InterceptorAction.REDACT:
            return True, clean_chunk

        return True, clean_chunk

    # Synchronous Iterator protocol
    def __iter__(self) -> Iterator[str]:
        return self._generate_sync_stream()

    def _generate_sync_stream(self) -> Iterator[str]:
        """Generator yielding validated, sanitized streaming text chunks."""
        # Source iterator
        if self.streamer is not None:
            source = iter(self.streamer)
        else:
            def queue_iter() -> Iterator[str]:
                while True:
                    item = self._internal_queue.get()
                    if item is self._stop_signal:
                        break
                    yield item
            source = queue_iter()

        for chunk in source:
            if not chunk:
                continue

            should_continue, emit_chunk = self._process_chunk(chunk)
            if emit_chunk:
                yield emit_chunk

            if not should_continue:
                break

    def __next__(self) -> str:
        """Pulls next validated token chunk."""
        if self._iterator is None:
            self._iterator = self._generate_sync_stream()
        return next(self._iterator)

    # Asynchronous Iterator protocol
    def __aiter__(self) -> AsyncIterator[str]:
        return self._generate_async_stream()

    async def _generate_async_stream(self) -> AsyncIterator[str]:
        """Asynchronous generator yielding validated chunks."""
        # Iterate asynchronously if source is an async iterator
        if self.streamer is not None and hasattr(self.streamer, "__aiter__"):
            async for chunk in self.streamer:
                if not chunk:
                    continue
                should_continue, emit_chunk = self._process_chunk(chunk)
                if emit_chunk:
                    yield emit_chunk
                if not should_continue:
                    break
            return

        # Otherwise wrap sync source into async iterator with cooperative yields
        for chunk in self._generate_sync_stream():
            yield chunk
            await asyncio.sleep(0)

    def get_metrics(self) -> Dict[str, Any]:
        """Return operational telemetry on streaming interception performance."""
        interceptor_metrics = self.interceptor.get_metrics()
        return {
            "chunks_scanned": self.chunks_scanned,
            "tokens_scanned": self.tokens_scanned,
            "violations_detected": len(self.violations),
            "is_tripped": self.is_tripped,
            "interceptor": interceptor_metrics,
        }
