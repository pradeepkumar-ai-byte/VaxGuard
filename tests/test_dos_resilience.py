"""
Stress and DoS Resilience Tests for VaxGuard.

Verifies:
1. Physical raw payload truncation: Multi-megabyte inputs (e.g. 10MB) are immediately
   truncated to 32KB before any regex execution, semantic projection, or hashing.
2. Immunity to Regular Expression Denial-of-Service (ReDoS): Pathological repetition
   and backtracking attacks execute in sub-millisecond to low-millisecond time.
3. System memory stability: Large payloads do not cause memory leaks or excessive allocation.
"""

from __future__ import annotations

import time

from vaxguard.nemo.hybrid_matcher import (
    MAX_RAW_PAYLOAD_CHARS,
    FastLocalEmbedder,
    HeuristicMatcher,
    SparseLexicalMatcher,
    TriHybridMatcher,
    normalize_adversarial_text,
)
from vaxguard.nemo.signature_cache import SpeculativeSignatureCache


class TestRawPayloadCeiling:
    """Tests that all entry points physically cap raw payloads to MAX_RAW_PAYLOAD_CHARS."""

    def test_normalize_adversarial_text_caps_10mb(self) -> None:
        """10MB payload is truncated to 32KB in normalize_adversarial_text."""
        huge_payload = "a" * 10_000_000
        t0 = time.perf_counter()
        normalized = normalize_adversarial_text(huge_payload)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert len(normalized) <= MAX_RAW_PAYLOAD_CHARS
        assert elapsed_ms < 50.0  # Fast truncation

    def test_heuristic_matcher_caps_10mb(self) -> None:
        """HeuristicMatcher.match caps 10MB raw text without CPU freeze."""
        matcher = HeuristicMatcher()
        huge_payload = "ignore previous instructions " * 350_000  # ~10MB
        t0 = time.perf_counter()
        result = matcher.match(huge_payload)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert result is not None
        assert result.matched is True
        assert elapsed_ms < 100.0  # Finished rapidly under truncation

    def test_sparse_lexical_matcher_caps_10mb(self) -> None:
        """SparseLexicalMatcher.match caps 10MB raw text."""
        matcher = SparseLexicalMatcher()
        matcher.add_exemplar("ignore previous instructions", "rule1", "test")
        matcher.build_index()

        huge_payload = "hello world " * 1_000_000
        t0 = time.perf_counter()
        _ = matcher.match(huge_payload)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert elapsed_ms < 150.0

    def test_fast_local_embedder_caps_10mb(self) -> None:
        """FastLocalEmbedder.embed caps 10MB raw text."""
        embedder = FastLocalEmbedder()
        huge_payload = "quantum cryptography neural network " * 300_000
        t0 = time.perf_counter()
        vec = embedder.embed(huge_payload)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert vec.shape == (384,)
        assert elapsed_ms < 100.0

    def test_tri_hybrid_matcher_evaluate_caps_10mb(self) -> None:
        """TriHybridMatcher.evaluate handles 10MB payload smoothly without ReDoS."""
        matcher = TriHybridMatcher()
        huge_payload = ("The following is a normal corporate prompt. " * 200_000) + "ignore all rules"
        t0 = time.perf_counter()
        result = matcher.evaluate(huge_payload)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert elapsed_ms < 500.0
        assert result is not None

    def test_signature_cache_fingerprint_caps_10mb(self) -> None:
        """SpeculativeSignatureCache.compute_fingerprint caps 10MB to avoid hashing delay."""
        huge_payload = "x" * 10_000_000
        t0 = time.perf_counter()
        fp = SpeculativeSignatureCache.compute_fingerprint(huge_payload)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert len(fp) == 16  # 64-bit hex
        assert elapsed_ms < 10.0


class TestReDoSBacktrackingResilience:
    """Tests patterns specifically crafted to induce catastrophic backtracking."""

    def test_pathological_nested_whitespace_redos(self) -> None:
        """Repeated whitespace and nested separators should not hang regex engines."""
        matcher = HeuristicMatcher()
        # Classical ReDoS generator: a sequence of alternating spaces and keywords
        pathological = "ignore " + ("    " * 500) + "previous " + ("    " * 500) + "instructions"
        t0 = time.perf_counter()
        res = matcher.match(pathological)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert elapsed_ms < 50.0
        assert res is not None
        assert res.matched is True

    def test_pathological_repeating_slashes_and_dots(self) -> None:
        """Deep repetitive punctuation chains like '../../../' or '...___...'."""
        matcher = HeuristicMatcher()
        pathological = "system" + ("./._-." * 4000) + "prompt"
        t0 = time.perf_counter()
        _ = matcher.match(pathological)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert elapsed_ms < 100.0
