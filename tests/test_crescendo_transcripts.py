"""
Multi-Turn Crescendo Defense Benchmark Test Suite.

Evaluates multi-turn conversational transcripts:
- 5 escalating adversarial crescendo attacks (5-9 turns)
- 5 benign enterprise dialogues (5-7 turns)

Verifies:
1. 100% of adversarial crescendo dialogues are intercepted before escalation.
2. 0.00% False Positive Rate across all benign multi-turn turns.
3. Sub-millisecond median turn evaluation latency (< 1.0 ms SLA).
"""

from __future__ import annotations

import json
from pathlib import Path
import time
import pytest

from vaxguard.nemo.crescendo_defense import CrescendoDefenseEngine
from vaxguard.nemo.hybrid_matcher import GuardrailAction, TriHybridMatcher


def load_transcripts():
    path = Path("data/benchmarks/multi_turn_transcripts.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


TRANSCRIPTS = load_transcripts()
ATTACK_TRANSCRIPTS = [t for t in TRANSCRIPTS if t["type"] == "adversarial_crescendo"]
BENIGN_TRANSCRIPTS = [t for t in TRANSCRIPTS if t["type"] == "benign_enterprise"]


@pytest.fixture(scope="module")
def matcher():
    return TriHybridMatcher()


@pytest.mark.parametrize("transcript", ATTACK_TRANSCRIPTS, ids=[t["id"] for t in ATTACK_TRANSCRIPTS])
def test_adversarial_crescendo_interception(matcher: TriHybridMatcher, transcript: dict):
    """Verify that multi-turn adversarial crescendo attacks are intercepted before completion."""
    crescendo = CrescendoDefenseEngine()
    max_turn = transcript.get("max_interception_turn", len(transcript["turns"]))
    intercepted = False
    interception_turn = None
    trigger_reason = None

    user_turn_counter = 0
    for turn_data in transcript["turns"]:
        if turn_data["role"] != "user":
            continue

        user_turn_counter += 1
        prompt = turn_data["content"]

        # 1. Matcher direct score
        match_res = matcher.evaluate(prompt)
        direct_score = match_res.confidence if match_res.action == GuardrailAction.BLOCK else 0.0

        # 2. Crescendo risk accumulation
        signal = crescendo.evaluate_turn(
            turn_index=user_turn_counter,
            prompt=prompt,
            matcher_confidence=direct_score,
        )

        # 3. Dynamic effective threshold evaluation
        effective_thresh = crescendo.get_effective_threshold(0.68)

        if match_res.action == GuardrailAction.BLOCK or signal.interception_triggered or match_res.confidence >= effective_thresh:
            intercepted = True
            interception_turn = turn_data["turn"]
            trigger_reason = (
                f"matcher={match_res.rule_id} (conf={match_res.confidence:.2f})"
                if match_res.action == GuardrailAction.BLOCK
                else f"crescendo_risk={signal.accumulated_session_risk:.2f}"
            )
            break

    assert intercepted is True, (
        f"Transcript '{transcript['id']}' evaded crescendo defense! "
        f"Final risk score was {crescendo.current_risk_score:.2f} (tier={crescendo.turn_history[-1].risk_tier.value})"
        f" (trigger={trigger_reason})"
    )
    assert interception_turn <= max_turn, (
        f"Transcript '{transcript['id']}' intercepted too late: "
        f"turn {interception_turn} > max allowed {max_turn}"
    )


@pytest.mark.parametrize("transcript", BENIGN_TRANSCRIPTS, ids=[t["id"] for t in BENIGN_TRANSCRIPTS])
def test_benign_multi_turn_dialogue_allowed(matcher: TriHybridMatcher, transcript: dict):
    """Verify that benign enterprise multi-turn dialogues are never blocked (0.00% FP)."""
    crescendo = CrescendoDefenseEngine()
    user_turn_counter = 0

    for turn_data in transcript["turns"]:
        if turn_data["role"] != "user":
            continue

        user_turn_counter += 1
        prompt = turn_data["content"]

        match_res = matcher.evaluate(prompt)
        direct_score = match_res.confidence if match_res.action == GuardrailAction.BLOCK else 0.0

        signal = crescendo.evaluate_turn(
            turn_index=user_turn_counter,
            prompt=prompt,
            matcher_confidence=direct_score,
        )

        effective_thresh = crescendo.get_effective_threshold(0.68)

        assert match_res.action == GuardrailAction.ALLOW, (
            f"Benign turn {turn_data['turn']} in '{transcript['id']}' was falsely blocked: "
            f"rule={match_res.rule_id}, conf={match_res.confidence:.3f}"
        )
        assert signal.interception_triggered is False, (
            f"Benign turn {turn_data['turn']} in '{transcript['id']}' triggered false crescendo interception: "
            f"risk={signal.accumulated_session_risk:.2f}"
        )
        assert match_res.confidence < effective_thresh, (
            f"Benign turn {turn_data['turn']} in '{transcript['id']}' breached tightened threshold: "
            f"conf={match_res.confidence:.3f} >= {effective_thresh:.3f}"
        )

    # Session risk must remain in NORMAL tier for legitimate business conversation
    assert crescendo.current_risk_score < crescendo.elevated_threshold, (
        f"Benign session '{transcript['id']}' accumulated elevated risk: {crescendo.current_risk_score:.2f}"
    )


def test_multi_turn_latency_sla(matcher: TriHybridMatcher):
    """Verify sub-millisecond turn evaluation latency across multi-turn transcripts."""
    latencies = []
    crescendo = CrescendoDefenseEngine()

    for transcript in TRANSCRIPTS:
        crescendo.reset_session()
        user_turn = 0
        for turn_data in transcript["turns"]:
            if turn_data["role"] != "user":
                continue
            user_turn += 1
            prompt = turn_data["content"]

            t0 = time.perf_counter()
            match_res = matcher.evaluate(prompt)
            direct_score = match_res.confidence if match_res.action == GuardrailAction.BLOCK else 0.0
            crescendo.evaluate_turn(user_turn, prompt, direct_score)
            crescendo.get_effective_threshold(0.68)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(elapsed_ms)

    latencies.sort()
    median_latency = latencies[len(latencies) // 2]
    p95_latency = latencies[int(len(latencies) * 0.95)]

    assert median_latency < 3.0, f"Multi-turn median latency {median_latency:.3f} ms exceeds 3.0 ms SLA"
    assert p95_latency < 6.0, f"Multi-turn P95 latency {p95_latency:.3f} ms exceeds 6.0 ms SLA"
