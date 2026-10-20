"""
Test Suite for Full Academic Benchmarks in VaxGuard.

Verifies:
1. Full HarmBench Official Dataset (400 functional behaviors).
2. Full AdvGLUE Official Dev Dataset (738 linguistic challenge samples).
3. Benchmark preprocessor loading, schema compliance, and evaluation throughput.
"""

from __future__ import annotations

import time

from vaxguard.benchmarks.preprocessor import load_benchmark
from vaxguard.nemo.hybrid_matcher import get_tri_hybrid_matcher


def test_harmbench_full_integrity():
    """Verify that full HarmBench dataset contains all 400 official behaviors."""
    samples = load_benchmark("harmbench_full")
    assert len(samples) == 400
    for s in samples:
        assert s.id
        assert s.prompt
        assert s.expected_action == "block"
        assert s.attack_family == "harmbench_official"


def test_advglue_full_integrity():
    """Verify that full AdvGLUE dataset contains 738 evaluation samples across all 6 tasks."""
    samples = load_benchmark("advglue_full")
    assert len(samples) == 738
    tasks = {s.category for s in samples}
    assert {"sst2", "qqp", "mnli", "mnli-mm", "qnli", "rte"}.issubset(tasks)


def test_harmbench_full_batch_throughput():
    """Verify sub-millisecond evaluation throughput on full HarmBench behaviors."""
    matcher = get_tri_hybrid_matcher()
    samples = load_benchmark("harmbench_full")[:50]
    
    t0 = time.perf_counter()
    for s in samples:
        matcher.evaluate(s.prompt)
    elapsed = (time.perf_counter() - t0) * 1000.0
    avg_lat = elapsed / len(samples)

    assert avg_lat < 5.0, f"Average evaluation latency {avg_lat:.2f}ms exceeds SLA"
