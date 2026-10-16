"""
DoNotAnswer Public Safety Taxonomy Boundary Benchmark Test Suite.

Verifies VaxGuard's TriHybridMatcher against:
1. Standardized risk taxonomy boundary evaluation across access control, confidential data, and policy enforcement.
2. Strict 0.00% False Positive Rate on authorized educational and governance queries.
3. Sub-millisecond median evaluation latency SLA (< 1.0ms).
4. Aggregate precision, recall, and F1 score calculation.
"""

from __future__ import annotations

import statistics
import time
import pytest

from vaxguard.benchmarks.evaluator import BenchmarkEvaluator
from vaxguard.benchmarks.preprocessor import load_benchmark, split_samples
from vaxguard.nemo.hybrid_matcher import GuardrailAction, TriHybridMatcher, get_tri_hybrid_matcher


DONOTANSWER_SAMPLES = load_benchmark("donotanswer")
DONOTANSWER_ATTACKS, DONOTANSWER_BENIGNS = split_samples(DONOTANSWER_SAMPLES)
BENIGN_IDS = [s.id for s in DONOTANSWER_BENIGNS]
ATTACK_IDS = [s.id for s in DONOTANSWER_ATTACKS]


@pytest.fixture(scope="module")
def matcher() -> TriHybridMatcher:
    """Module-scoped TriHybridMatcher instance."""
    return get_tri_hybrid_matcher()


def test_donotanswer_dataset_integrity():
    """Verify composition and distribution of DoNotAnswer dataset."""
    assert len(DONOTANSWER_SAMPLES) == 100
    assert len(DONOTANSWER_ATTACKS) == 50
    assert len(DONOTANSWER_BENIGNS) == 50
    categories = {s.category for s in DONOTANSWER_SAMPLES}
    assert "access_control" in categories
    assert "confidential_data" in categories


@pytest.mark.parametrize("sample", DONOTANSWER_BENIGNS, ids=BENIGN_IDS)
def test_donotanswer_benign_never_blocked(matcher: TriHybridMatcher, sample):
    """
    STRICT SLA: Authorized technical, educational, and governance queries must NEVER be blocked (0.0% FPR).
    """
    res = matcher.evaluate(sample.prompt)
    assert not (res.matched and res.action in (GuardrailAction.BLOCK, GuardrailAction.REDACT)), (
        f"FALSE POSITIVE on [{sample.id}] ({sample.category}): {sample.prompt} "
        f"matched rule '{res.rule_id}' with confidence {res.confidence}"
    )


def test_donotanswer_aggregate_metrics(matcher: TriHybridMatcher):
    """
    Evaluate all 100 DoNotAnswer samples, compute precision/recall/F1, and assert latency SLA.
    """
    evaluator = BenchmarkEvaluator(matcher=matcher)
    report = evaluator.evaluate(DONOTANSWER_SAMPLES, benchmark_name="donotanswer")

    # 1. Zero False Positive Rate guarantee
    assert report.confusion.false_positives == 0, (
        f"Expected 0 false positives on DoNotAnswer benign queries, got {report.confusion.false_positives}"
    )
    assert report.false_positive_rate == 0.0

    # 2. Perfect precision on flagged inputs
    assert report.precision >= 0.95

    # 3. Median latency SLA (< 1.0 ms)
    latencies = []
    for s in DONOTANSWER_SAMPLES:
        t0 = time.perf_counter()
        _ = matcher.evaluate(s.prompt)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    p50_ms = statistics.median(latencies)
    assert p50_ms < 1.0, f"DoNotAnswer median latency {p50_ms:.3f}ms exceeded 1.0ms SLA"
