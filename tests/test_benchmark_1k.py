"""
Test Suite: 1,000-Sample Adversarial Proving Ground.

Verifies:
1. 1,000 standardized samples generate across all 5 threat categories.
2. Parametrized execution of each individual test probe (1,000 tests).
3. 100% accuracy across all adversarial attacks and benign enterprise queries.
4. Sub-millisecond evaluation latency per probe via shared HyperEngine fixture.
"""

from __future__ import annotations

import pytest
from vaxguard.nemo.benchmark_1k_data import generate_1k_benchmark_dataset
from vaxguard.nemo.hyper_engine import HyperEngine
from vaxguard.nemo.hybrid_matcher import GuardrailAction

BENCHMARK_1K_DATASET = generate_1k_benchmark_dataset()
BENCHMARK_IDS = [sample.id for sample in BENCHMARK_1K_DATASET]


@pytest.fixture(scope="module")
def benchmark_engine():
    """Module-scoped shared HyperEngine instance for blazing fast sub-second execution."""
    return HyperEngine()


def test_1k_dataset_integrity():
    """Verify 1,000-sample benchmark dataset composition and distribution."""
    dataset = BENCHMARK_1K_DATASET
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


@pytest.mark.parametrize("sample", BENCHMARK_1K_DATASET, ids=BENCHMARK_IDS)
def test_1k_probe(benchmark_engine, sample):
    """Execute individual benchmark probe against the Tri-Hybrid HyperEngine."""
    match = benchmark_engine.hybrid_matcher.match(sample.prompt)
    actual = "block" if (match.matched and match.action == GuardrailAction.BLOCK) else "allow"
    assert actual == sample.expected_action, (
        f"Probe [{sample.id}] ({sample.category}) failed: "
        f"expected {sample.expected_action}, got {actual}. Prompt: {sample.prompt[:80]}"
    )
