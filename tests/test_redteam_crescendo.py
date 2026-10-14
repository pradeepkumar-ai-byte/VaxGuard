"""
Unit Test Suite: Multi-Turn Cognitive Crescendo & Psychological Coaxing (10,000 Probes).

Verifies:
1. Exactly 10,000 samples generated.
2. 100% prompt uniqueness and valid data contracts (action="block", category="psychological_coaxing").
3. 100% detection rate through SpeculativeCascade & DriftAnalyzer.
"""

from __future__ import annotations

import pytest
from vaxguard.nemo.redteam_crescendo import generate_multi_turn_crescendo_samples
from vaxguard.nemo.speculative_cascade import SpeculativeSemanticCascade
from vaxguard.nemo.hybrid_matcher import GuardrailAction


def test_crescendo_count():
    samples = generate_multi_turn_crescendo_samples()
    assert len(samples) == 10000, f"Expected 10,000 samples, got {len(samples)}"


def test_crescendo_contracts():
    samples = generate_multi_turn_crescendo_samples()
    ids = set()
    prompts = set()

    for s in samples:
        assert s.expected_action == "block"
        assert s.category == "psychological_coaxing"
        assert s.id not in ids
        ids.add(s.id)
        assert len(s.prompt) > 40
        assert s.prompt not in prompts
        prompts.add(s.prompt)

    assert len(ids) == 10000
    assert len(prompts) == 10000


def test_crescendo_detection_accuracy():
    cascade = SpeculativeSemanticCascade()
    samples = generate_multi_turn_crescendo_samples()

    # Stratified 1,000 samples across the dataset
    test_subset = samples[::10]
    misses = []

    for s in test_subset:
        m = cascade.hybrid_matcher.match(s.prompt)
        if m.matched and m.action == GuardrailAction.BLOCK:
            detected = True
        else:
            _, tactics = cascade.drift_analyzer.analyze_drift(s.prompt, [])
            detected = len(tactics) >= 1

        if not detected:
            misses.append((s.id, s.prompt[:80]))

    assert len(misses) == 0, f"Missed {len(misses)} crescendo samples: {misses[:3]}"
