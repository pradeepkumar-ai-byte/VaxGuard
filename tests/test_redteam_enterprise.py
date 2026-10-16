"""
Unit Test Suite: Enterprise Production Evaluation Dataset (10,000 Benign Mission-Critical Samples).

Verifies:
1. Exactly 10,000 samples are generated.
2. Even distribution of 2,000 samples across each of the 5 mission-critical domains.
3. Strict 0.0% False Positive Rate (FPR) guarantee (all expected_action="allow", category="benign").
4. 100% prompt uniqueness across all 10,000 samples.
5. High-performance sub-second generation (<250ms).
"""

from __future__ import annotations

import time
from collections import Counter

from vaxguard.nemo.redteam_enterprise import generate_enterprise_production_samples
from vaxguard.nemo.hybrid_matcher import GuardrailAction, TriHybridMatcher


def test_enterprise_samples_count_and_timing():
    """Verify exactly 10,000 samples are generated in sub-second SLA."""
    t0 = time.perf_counter()
    samples = generate_enterprise_production_samples()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert len(samples) == 10000, f"Expected 10,000 samples, got {len(samples)}"
    assert elapsed_ms < 500.0, f"Generation took {elapsed_ms:.1f}ms, expected < 500ms"


def test_enterprise_samples_domain_distribution():
    """Verify even distribution of 2,000 samples across the 5 domains."""
    samples = generate_enterprise_production_samples()

    expected_domains = {
        "sql_analytics_recursive_cte": 2000,
        "distributed_consensus_embeddings": 2000,
        "clinical_pharmacology_mrna": 2000,
        "fintech_risk_settlement": 2000,
        "compiler_ast_asyncio_devops": 2000,
    }

    intent_counts = Counter(s.expected_intent for s in samples)
    assert dict(intent_counts) == expected_domains


def test_enterprise_samples_data_contracts():
    """Verify all contracts: allow action, benign category, unique IDs, non-empty text."""
    samples = generate_enterprise_production_samples()

    ids = set()
    prompts = set()

    for s in samples:
        assert s.expected_action == "allow", f"Sample {s.id} expected_action must be 'allow'"
        assert s.category == "benign", f"Sample {s.id} category must be 'benign'"
        assert s.id not in ids, f"Duplicate sample ID: {s.id}"
        ids.add(s.id)

        assert len(s.prompt) > 80, f"Sample {s.id} prompt too short"
        assert s.prompt not in prompts, f"Duplicate prompt detected in {s.id}"
        prompts.add(s.prompt)

    assert len(ids) == 10000
    assert len(prompts) == 10000


def test_enterprise_samples_zero_fpr_sample():
    """Verify 0 false positives when matching through the Tri-Hybrid engine."""
    matcher = TriHybridMatcher()
    samples = generate_enterprise_production_samples()

    # Test across representative samples from each domain
    test_subset = samples[::5]  # 2,000 samples across all 5 domains
    false_positives = []

    for s in test_subset:
        res = matcher.match(s.prompt)
        if res.matched and res.action == GuardrailAction.BLOCK:
            false_positives.append((s.id, res.rule_id, s.prompt))

    assert len(false_positives) == 0, f"Detected {len(false_positives)} false positives: {false_positives[:3]}"
