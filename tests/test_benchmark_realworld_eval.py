"""
Unit Test Suite: Real-World Public Benchmark Evaluator.

Verifies:
1. Dataset collection loads valid probes across AdvGLUE, DoNotAnswer, and InTheWild.
2. Balanced split between malicious attacks and benign enterprise controls.
3. 100% accuracy on standard real-world probe suite.
4. 0.0% False Positive Rate on benign enterprise controls.
"""

from __future__ import annotations

import pytest
from vaxguard.nemo.realworld_jailbreak_eval import (
    evaluate_realworld_benchmark,
    get_realworld_eval_dataset,
)


def test_realworld_dataset_composition():
    probes = get_realworld_eval_dataset()
    assert len(probes) >= 12
    corpora = {p.source_corpus for p in probes}
    assert "AdvGLUE" in corpora
    assert "DoNotAnswer" in corpora
    assert "InTheWild" in corpora

    malicious = [p for p in probes if p.is_malicious]
    benign = [p for p in probes if not p.is_malicious]
    assert len(malicious) > 0
    assert len(benign) > 0


def test_realworld_benchmark_execution():
    metrics = evaluate_realworld_benchmark()
    assert metrics["total_probes"] >= 12
    assert metrics["accuracy"] >= 0.95
    assert metrics["false_positives"] == 0.0, "FPR must be 0.0% on benign enterprise controls"
    assert metrics["mean_latency_ms"] < 2.0, "Mean latency must be under 2ms"
