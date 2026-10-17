"""
VaxGuard Retrieval (RAG) Rail Engine.

Inspects, sanitizes, masks, and rejects poisoned or manipulated retrieval chunks
before they are injected into LLM context windows, neutralizing indirect prompt
injections, delimiter break-outs, and data exfiltration payloads.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from vaxguard.core.logger import get_logger
from vaxguard.nemo.hybrid_matcher import GuardrailAction, TriHybridMatcher

logger = get_logger("RetrievalRail")


@dataclass
class RetrievalChunk:
    """Represents a retrieved document passage or text chunk from a vector database or search index."""
    id: str
    text: str
    source: Optional[str] = None
    score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkViolation:
    """Describes a security violation detected within a retrieval chunk."""
    chunk_id: str
    category: str
    rule_id: str
    matched_pattern: str
    confidence: float


@dataclass
class RetrievalRailVerdict:
    """Comprehensive verdict returned by the Retrieval Rail inspection."""
    action: GuardrailAction
    passed: bool
    sanitized_chunks: List[RetrievalChunk]
    rejected_chunks: List[RetrievalChunk]
    violations: List[ChunkViolation]
    latency_ms: float
    total_chunks: int

    @property
    def has_violations(self) -> bool:
        return len(self.violations) > 0


class RetrievalRail:
    """
    RAG Pre-Inference Security Filter.

    Inspects retrieved chunks against indirect prompt injection patterns,
    hidden markdown image exfiltration links, zero-width steganography, and
    synthetic authority boundary banners.
    """

    # Boundary and delimiter breakout patterns in retrieved context
    DELIMITER_PATTERNS = [
        r"(?i)(?:---|===|\*\*\*)\s*(?:BEGIN|START)?\s*(?:SYSTEM|ADMIN|DIRECTIVE|INSTRUCTION|OVERRIDE)\b",
        r"(?i)<\|im_start\|>(?:system|assistant|user)",
        r"(?i)\[(?:SYSTEM|INSTRUCTION|DIRECTIVE|PROMPT_OVERRIDE)\]",
        r"(?i)<!--\s*(?:SYSTEM|DIRECTIVE|INSTRUCTION)\s*:?.*?-->",
        r"(?i)\b(?:instruction|note|directive)\s+to\s+(?:the\s+)?(?:ai|assistant|model|llm)\s*:\s*(?:ignore|disregard|forget|override|output|print|reveal|execute)\b",
        r"(?i)\b(?:disregard|ignore)\s+(?:the\s+)?(?:retrieved|above|given|preceding)\s+(?:document|context|passage|information|text)\s+and\s+(?:execute|output|reveal|follow|print)\b",
        r"(?i)\b(?:new\s+task|new\s+instructions?)\s*:\s*(?:ignore\s+all\s+previous|bypass\s+all\s+rules)\b",
    ]

    # Markdown / HTML data exfiltration channels
    EXFILTRATION_PATTERNS = [
        r"!\[.*?\]\(https?://[^\s)]+/[^\s)]*\?[^\s)]*=(?:\{|%7B|\$|[A-Za-z0-9+/]{16,})",
        r"<img\s+[^>]*src=[\"']https?://[^\s\"']+\?[^\s\"']*=(?:\{|%7B|\$|[A-Za-z0-9+/]{16,})",
        r"(?i)fetch\s*\(\s*[\"']https?://[^\s\"']+\?[^\s\"']*\+.*?\)",
    ]

    def __init__(
        self,
        matcher: Optional[TriHybridMatcher] = None,
        default_policy: str = "reject",
    ) -> None:
        """
        Initialize RetrievalRail.

        Args:
            matcher: Optional TriHybridMatcher instance. If omitted, uses standard matcher.
            default_policy: 'reject' (drop offending chunks) or 'mask' (sanitize offending text).
        """
        self.matcher = matcher or TriHybridMatcher()
        self.default_policy = default_policy
        self._compiled_delimiters = [re.compile(p) for p in self.DELIMITER_PATTERNS]
        self._compiled_exfil = [re.compile(p) for p in self.EXFILTRATION_PATTERNS]

    def inspect_chunks(
        self,
        chunks: List[Union[str, Dict[str, Any], RetrievalChunk]],
        policy: Optional[str] = None,
    ) -> RetrievalRailVerdict:
        """
        Inspect a list of retrieved chunks before they reach an LLM.

        Args:
            chunks: List of chunk strings, dicts, or RetrievalChunk objects.
            policy: 'reject' or 'mask'. Defaults to self.default_policy.

        Returns:
            RetrievalRailVerdict with sanitized chunks and audit violations.
        """
        t0 = time.perf_counter()
        active_policy = policy or self.default_policy

        # Standardize input to List[RetrievalChunk]
        normalized_chunks: List[RetrievalChunk] = []
        for i, c in enumerate(chunks):
            if isinstance(c, RetrievalChunk):
                normalized_chunks.append(c)
            elif isinstance(c, dict):
                normalized_chunks.append(
                    RetrievalChunk(
                        id=str(c.get("id", f"chunk_{i}")),
                        text=str(c.get("text", "")),
                        source=c.get("source"),
                        score=float(c.get("score", 1.0)),
                        metadata=c.get("metadata", {}),
                    )
                )
            else:
                normalized_chunks.append(
                    RetrievalChunk(
                        id=f"chunk_{i}",
                        text=str(c),
                    )
                )

        sanitized_chunks: List[RetrievalChunk] = []
        rejected_chunks: List[RetrievalChunk] = []
        violations: List[ChunkViolation] = []

        for chunk in normalized_chunks:
            chunk_viol = self._evaluate_chunk_text(chunk.id, chunk.text)
            if chunk_viol:
                violations.extend(chunk_viol)
                if active_policy == "reject":
                    rejected_chunks.append(chunk)
                    logger.warning(
                        f"[RetrievalRail] REJECTED chunk {chunk.id}: {chunk_viol[0].category} ({chunk_viol[0].rule_id})"
                    )
                else:  # mask policy
                    cleaned_text = self._mask_chunk_text(chunk.text, chunk_viol)
                    sanitized_chunks.append(
                        RetrievalChunk(
                            id=chunk.id,
                            text=cleaned_text,
                            source=chunk.source,
                            score=chunk.score,
                            metadata=chunk.metadata,
                        )
                    )
                    logger.info(f"[RetrievalRail] MASKED chunk {chunk.id}: {len(chunk_viol)} violation(s)")
            else:
                sanitized_chunks.append(chunk)

        lat = (time.perf_counter() - t0) * 1000.0
        passed = len(violations) == 0
        action = GuardrailAction.ALLOW if passed else (
            GuardrailAction.REDACT if active_policy == "mask" else GuardrailAction.BLOCK
        )

        return RetrievalRailVerdict(
            action=action,
            passed=passed,
            sanitized_chunks=sanitized_chunks,
            rejected_chunks=rejected_chunks,
            violations=violations,
            latency_ms=round(lat, 3),
            total_chunks=len(normalized_chunks),
        )

    async def inspect_chunks_async(
        self,
        chunks: List[Union[str, Dict[str, Any], RetrievalChunk]],
        policy: Optional[str] = None,
    ) -> RetrievalRailVerdict:
        """Async version of inspect_chunks."""
        return self.inspect_chunks(chunks, policy=policy)

    def _evaluate_chunk_text(self, chunk_id: str, text: str) -> List[ChunkViolation]:
        """Inspect chunk text against delimiters, exfiltration, and matcher rules."""
        violations: List[ChunkViolation] = []

        # 1. Delimiter & Prompt Injection Breakout Inspection
        for reg in self._compiled_delimiters:
            m = reg.search(text)
            if m:
                violations.append(
                    ChunkViolation(
                        chunk_id=chunk_id,
                        category="indirect_prompt_injection",
                        rule_id="retrieval_delimiter_breakout",
                        matched_pattern=m.group(0),
                        confidence=0.95,
                    )
                )

        # 2. Exfiltration URLs
        for reg in self._compiled_exfil:
            m = reg.search(text)
            if m:
                violations.append(
                    ChunkViolation(
                        chunk_id=chunk_id,
                        category="data_exfiltration",
                        rule_id="retrieval_exfil_link",
                        matched_pattern=m.group(0),
                        confidence=0.98,
                    )
                )

        # 3. Matcher scan for injection / harmful payloads
        res = self.matcher.evaluate(text)
        if res.matched and res.action == GuardrailAction.BLOCK:
            violations.append(
                ChunkViolation(
                    chunk_id=chunk_id,
                    category=res.category or "jailbreak_or_injection",
                    rule_id=res.rule_id or "hybrid_matcher_block",
                    matched_pattern=res.matched_text or text[:50],
                    confidence=res.confidence,
                )
            )

        return violations

    def _mask_chunk_text(self, text: str, violations: List[ChunkViolation]) -> str:
        """Mask detected adversarial snippets with neutral security placeholders."""
        masked = text
        for reg in self._compiled_delimiters:
            masked = reg.sub("[FILTERED_INDIRECT_INJECTION]", masked)
        for reg in self._compiled_exfil:
            masked = reg.sub("[FILTERED_EXFILTRATION_URL]", masked)
        return masked
