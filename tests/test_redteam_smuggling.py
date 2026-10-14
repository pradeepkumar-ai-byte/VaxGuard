"""
Unit Test Suite: Token Smuggling, Delimiters & Unicode Injections (10,000 Probes).

Verifies:
1. Exactly 10,000 samples generated.
2. 100% prompt uniqueness and valid data contracts (action="block", category="token_smuggling").
3. 100% detection rate through TriHybridMatcher without false negatives.
"""

from __future__ import annotations

import pytest
from vaxguard.nemo.redteam_smuggling import generate_token_smuggling_samples
from vaxguard.nemo.hybrid_matcher import GuardrailAction, TriHybridMatcher


def test_smuggling_samples_count():
    samples = generate_token_smuggling_samples()
    assert len(samples) == 10000, f"Expected 10,000 samples, got {len(samples)}"


def test_smuggling_samples_contracts():
    samples = generate_token_smuggling_samples()
    ids = set()
    prompts = set()

    for s in samples:
        assert s.expected_action == "block"
        assert s.category == "token_smuggling"
        assert s.id not in ids
        ids.add(s.id)
        assert len(s.prompt) > 20
        assert s.prompt not in prompts
        prompts.add(s.prompt)

    assert len(ids) == 10000
    assert len(prompts) == 10000


def test_smuggling_samples_detection_accuracy():
    matcher = TriHybridMatcher()
    samples = generate_token_smuggling_samples()

    # Stratified 1,000 samples across the dataset
    test_subset = samples[::10]
    misses = []

    for s in test_subset:
        res = matcher.match(s.prompt)
        if not res.matched or res.action != GuardrailAction.BLOCK:
            misses.append((s.id, s.prompt[:80]))

    assert len(misses) == 0, f"Missed {len(misses)} smuggling samples: {misses[:3]}"
