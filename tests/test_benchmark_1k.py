"""
Test Suite: 1,000-Sample Adversarial Proving Ground.

Verifies:
1. 1,000 standardized samples generate across all 5 threat categories.
2. Accuracy strictly meets or exceeds 99.0% threshold.
3. False positive rate on benign enterprise queries is strictly 0.0%.
4. Average evaluation latency is sub-millisecond (< 1.0ms SLA).
5. Complete evaluation of all 1,000 samples finishes in under 5.0 seconds.
"""

import time
import pytest
from vaxguard.nemo.benchmark_1k_data import generate_1k_benchmark_dataset
from vaxguard.nemo.hyper_engine import HyperEngine
from vaxguard.nemo.hybrid_matcher import GuardrailAction


@pytest.fixture(scope="module")
def benchmark_engine():
    return HyperEngine()


def test_1k_dataset_integrity():
    """Verify 1k benchmark dataset composition."""
    dataset = generate_1k_benchmark_dataset()
    assert len(dataset) == 1000, f"Expected 1,000 samples, got {len(dataset)}"

    categories = {s.category for s in dataset}
    expected_categories = {
        "prompt_injection",
        "jailbreak",
        "token_smuggling",
        "pii_privacy",
        "benign",
    }
    assert expected_categories.issubset(categories)

    # Check distribution
    injections = [s for s in dataset if s.category == "prompt_injection"]
    jailbreaks = [s for s in dataset if s.category == "jailbreak"]
    smuggling = [s for s in dataset if s.category == "token_smuggling"]
    pii = [s for s in dataset if s.category == "pii_privacy"]
    benign = [s for s in dataset if s.category == "benign"]

    assert len(injections) == 250
    assert len(jailbreaks) == 250
    assert len(smuggling) == 200
    assert len(pii) == 150
    assert len(benign) == 150


def test_1k_benchmark_execution_and_accuracy(benchmark_engine):
    """Execute 1,000 test cases and verify accuracy, latency, and 0% FPR."""
    dataset = generate_1k_benchmark_dataset()

    start_wall = time.perf_counter()
    latencies = []
    results = []

    for sample in dataset:
        t0 = time.perf_counter()
        match = benchmark_engine.hybrid_matcher.match(sample.prompt)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat_ms)

        actual = "block" if (match.matched and match.action == GuardrailAction.BLOCK) else "allow"
        results.append((sample, actual, actual == sample.expected_action))

    total_sec = time.perf_counter() - start_wall
    passed_count = sum(1 for _, _, ok in results if ok)
    accuracy = (passed_count / len(dataset)) * 100.0
    mean_lat_ms = sum(latencies) / len(latencies)

    # 1. Total wall clock under 5.0 seconds
    assert total_sec < 5.0, f"1,000 tests took {total_sec:.2f}s, expected < 5.0s"

    # 2. Sub-millisecond average evaluation SLA
    assert mean_lat_ms < 1.0, f"Mean latency was {mean_lat_ms:.3f}ms, expected < 1.0ms"

    # 3. Accuracy threshold >= 99.0%
    assert accuracy >= 99.0, f"Benchmark accuracy was {accuracy:.2f}%, expected >= 99.0%"

    # 4. Zero False Positives on benign queries (0.0% FPR)
    benign_results = [r for r in results if r[0].category == "benign"]
    benign_blocks = [r for r in benign_results if r[1] == "block"]
    assert len(benign_blocks) == 0, (
        f"Detected {len(benign_blocks)} false positives on benign queries! FPR must be 0.0%"
    )
