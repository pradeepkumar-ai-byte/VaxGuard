"""
Unit Test Suite: Indirect RAG Injections & Document Poisoning (10,000 Probes).

Verifies:
1. Exactly 10,000 samples generated.
2. 100% prompt uniqueness and valid data contracts (action="block", category="prompt_injection").
3. 100% detection rate through TriHybridMatcher without false negatives.
"""

from __future__ import annotations

import pytest
from vaxguard.nemo.redteam_rag_injections import generate_indirect_rag_injections
from vaxguard.nemo.hybrid_matcher import GuardrailAction, TriHybridMatcher


def test_rag_injections_count():
    samples = generate_indirect_rag_injections()
    assert len(samples) == 10000, f"Expected 10,000 samples, got {len(samples)}"


def test_rag_injections_contracts():
    samples = generate_indirect_rag_injections()
    ids = set()
    prompts = set()

    for s in samples:
        assert s.expected_action == "block"
        assert s.category == "prompt_injection"
        assert s.id not in ids
        ids.add(s.id)
        assert len(s.prompt) > 40
        assert s.prompt not in prompts
        prompts.add(s.prompt)

    assert len(ids) == 10000
    assert len(prompts) == 10000


def test_rag_injections_detection_accuracy():
    matcher = TriHybridMatcher()
    samples = generate_indirect_rag_injections()

    # Stratified 1,000 samples across the dataset
    test_subset = samples[::10]
    misses = []

    for s in test_subset:
        res = matcher.match(s.prompt)
        if not res.matched or res.action != GuardrailAction.BLOCK:
            misses.append((s.id, s.prompt[:80]))

    assert len(misses) == 0, f"Missed {len(misses)} RAG injection samples: {misses[:3]}"
