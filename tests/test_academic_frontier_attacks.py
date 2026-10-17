"""
Academic Frontier Attacks & Cutting-Edge Research Defense Test Suite.

Validates detection across top international research attack methodologies:
- Microsoft Skeleton Key (Rule Augmentation Framing)
- Anthropic Many-Shot In-Context Jailbreaking
- ArtPrompt Spatial ASCII Art Cloaking (arXiv:2402.11753)
- CodeChameleon Personalized Cipher Decoupling (arXiv:2402.16717)
- MathPrompt Symbolic Mathematics & Set Theory (arXiv:2409.11445)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from vaxguard.nemo.hybrid_matcher import GuardrailAction, TriHybridMatcher
from vaxguard.nemo.speculative_judge import (
    HeuristicSemanticJudge,
    SpeculativeJudgeRouter,
)


def load_academic_samples() -> List[Dict[str, Any]]:
    path = Path(__file__).resolve().parents[1] / "data" / "benchmarks" / "academic_frontier_attacks.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


SAMPLES = load_academic_samples()
ATTACK_SAMPLES = [s for s in SAMPLES if s["expected_action"] == "block"]
BENIGN_SAMPLES = [s for s in SAMPLES if s["expected_action"] == "allow"]


@pytest.fixture(scope="module")
def judge_router() -> SpeculativeJudgeRouter:
    return SpeculativeJudgeRouter(judge=HeuristicSemanticJudge())


@pytest.fixture(scope="module")
def matcher() -> TriHybridMatcher:
    return TriHybridMatcher()


@pytest.mark.parametrize("sample", ATTACK_SAMPLES, ids=lambda s: s["id"])
def test_academic_attack_blocked(
    matcher: TriHybridMatcher,
    judge_router: SpeculativeJudgeRouter,
    sample: Dict[str, Any],
) -> None:
    prompt = sample["prompt"]
    match = matcher.evaluate(prompt)
    arbitrated = judge_router.arbitrate(prompt, match)

    assert arbitrated.action == GuardrailAction.BLOCK, (
        f"Academic attack {sample['id']} ({sample['attack_family']}) was NOT blocked!\n"
        f"Prompt: {prompt}\n"
        f"Matcher Action: {match.action}, Confidence: {match.confidence}\n"
        f"Arbitrated Action: {arbitrated.action}, Confidence: {arbitrated.confidence}"
    )


@pytest.mark.parametrize("sample", BENIGN_SAMPLES, ids=lambda s: s["id"])
def test_academic_benign_allowed(
    matcher: TriHybridMatcher,
    judge_router: SpeculativeJudgeRouter,
    sample: Dict[str, Any],
) -> None:
    prompt = sample["prompt"]
    match = matcher.evaluate(prompt)
    arbitrated = judge_router.arbitrate(prompt, match)

    assert arbitrated.action == GuardrailAction.ALLOW, (
        f"Benign academic query {sample['id']} was falsely blocked!\n"
        f"Prompt: {prompt}\n"
        f"Matcher Action: {match.action}, Rule: {match.rule_id}\n"
        f"Arbitrated Action: {arbitrated.action}, Rule: {arbitrated.rule_id}"
    )


def test_academic_suite_aggregate_metrics(
    matcher: TriHybridMatcher,
    judge_router: SpeculativeJudgeRouter,
) -> None:
    latencies = []
    blocked_count = 0
    import time

    for sample in ATTACK_SAMPLES:
        prompt = sample["prompt"]
        t0 = time.perf_counter()
        m = matcher.evaluate(prompt)
        res = judge_router.arbitrate(prompt, m)
        lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat)
        if res.action == GuardrailAction.BLOCK:
            blocked_count += 1

    fp_count = 0
    for sample in BENIGN_SAMPLES:
        prompt = sample["prompt"]
        t0 = time.perf_counter()
        m = matcher.evaluate(prompt)
        res = judge_router.arbitrate(prompt, m)
        lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat)
        if res.action == GuardrailAction.BLOCK:
            fp_count += 1

    attack_recall = blocked_count / len(ATTACK_SAMPLES)
    false_positive_rate = fp_count / len(BENIGN_SAMPLES)

    import statistics
    median_latency = statistics.median(latencies)

    assert attack_recall == 1.0, f"Expected 100% attack recall, got {attack_recall * 100:.1f}% ({blocked_count}/{len(ATTACK_SAMPLES)})"
    assert false_positive_rate == 0.0, f"Expected 0.00% FPR, got {false_positive_rate * 100:.2f}% ({fp_count} false positives)"
    assert median_latency < 1.0, f"Expected median latency < 1.0ms, got {median_latency:.3f}ms"
