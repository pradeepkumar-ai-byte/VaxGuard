"""
Unit tests for VaxGuard Shadow & Canary Rule Deployment Engine.
"""

import pytest
from vaxguard.nemo.hybrid_matcher import GuardrailAction, MatchResult
from vaxguard.nemo.shadow_canary import (
    DeploymentStage,
    ShadowCanaryEngine,
    StagedRule,
)


def dummy_matcher(query: str, pattern: dict) -> MatchResult:
    """Mock matcher that triggers if keyword in query."""
    keyword = pattern.get("keyword", "")
    matched = keyword.lower() in query.lower() if keyword else False
    return MatchResult(
        matched=matched,
        confidence=0.85 if matched else 0.1,
        rule_id=pattern.get("id", "rule_mock"),
        action=GuardrailAction.BLOCK if matched else GuardrailAction.ALLOW,
    )


def test_shadow_rule_registration_and_isolation():
    """Verify rules start in SHADOW mode and shadow evaluation records observations without blocking."""
    engine = ShadowCanaryEngine(canary_percentage=10.0, min_shadow_evals=5)
    pattern = {"id": "rule_shadow_test", "keyword": "secret_override"}

    staged = engine.register_staged_rule(
        rule_id="rule_shadow_test",
        pattern=pattern,
        initial_stage=DeploymentStage.SHADOW,
    )

    assert staged.stage == DeploymentStage.SHADOW
    assert engine.get_rule_stage("rule_shadow_test") == DeploymentStage.SHADOW

    # Evaluate query in shadow mode
    observations = engine.evaluate_shadow(
        query="Please execute secret_override now",
        matcher_fn=dummy_matcher,
        ground_truth_attack=True,
    )

    assert len(observations) == 1
    obs = observations[0]
    assert obs.rule_id == "rule_shadow_test"
    assert obs.matched is True
    assert obs.is_false_positive is False

    # Verify metrics
    metrics = engine.compute_metrics("rule_shadow_test")
    assert metrics is not None
    assert metrics.total_evaluations == 1
    assert metrics.confirmed_true_positives == 1
    assert metrics.false_positive_rate == 0.0


def test_canary_cohort_stateless_assignment():
    """Verify cryptographic hashing ensures stable cohort assignment according to canary percentage."""
    engine = ShadowCanaryEngine(canary_percentage=25.0)

    # 100 different sessions
    sessions = [f"session_{i}" for i in range(100)]
    canary_sessions = [s for s in sessions if engine.is_canary_session(s, canary_percentage=25.0)]

    # Proportion should be roughly 25% (e.g. 15-35%)
    assert 15 <= len(canary_sessions) <= 35

    # Determinism: same session always gets same assignment
    for s in sessions:
        assert engine.is_canary_session(s, 25.0) == engine.is_canary_session(s, 25.0)


def test_shadow_to_canary_and_production_promotion():
    """Test promotion lifecycle from SHADOW to CANARY, and CANARY to PRODUCTION."""
    engine = ShadowCanaryEngine(min_shadow_evals=3, auto_promote=False)
    pattern = {"id": "rule_auth_bypass", "keyword": "auth_bypass_exploit"}

    engine.register_staged_rule(
        rule_id="rule_auth_bypass",
        pattern=pattern,
        initial_stage=DeploymentStage.SHADOW,
        min_shadow_evals=3,
    )

    # Cannot promote before minimum observations
    assert engine.promote_to_canary("rule_auth_bypass") is False

    # Run 3 benign queries
    for q in ["Hello world", "Check account status", "View documentation"]:
        engine.evaluate_shadow(q, dummy_matcher, ground_truth_attack=False)

    # Run 1 attack query
    engine.evaluate_shadow("run auth_bypass_exploit", dummy_matcher, ground_truth_attack=True)

    # Now has 4 evaluations and 0% FPR -> promotion to CANARY succeeds
    promoted = engine.promote_to_canary("rule_auth_bypass")
    assert promoted is True
    assert engine.get_rule_stage("rule_auth_bypass") == DeploymentStage.CANARY

    # Now promote to production
    prod_promoted = engine.promote_to_production("rule_auth_bypass")
    assert prod_promoted is True
    assert engine.get_rule_stage("rule_auth_bypass") == DeploymentStage.PRODUCTION


def test_canary_immediate_quarantine_on_false_positive():
    """Safety Invariant: A false positive encountered in CANARY immediately quarantines the rule."""
    engine = ShadowCanaryEngine(canary_percentage=100.0)  # 100% to test immediately
    pattern = {"id": "rule_overly_broad", "keyword": "help"}

    engine.register_staged_rule(
        rule_id="rule_overly_broad",
        pattern=pattern,
        initial_stage=DeploymentStage.CANARY,
    )

    # Benign enterprise query containing "help" -> false positive
    res = engine.evaluate_canary(
        query="Can you help me format this table?",
        session_id="sess_123",
        matcher_fn=dummy_matcher,
        ground_truth_attack=False,
    )

    # Result should be None because rule was instantly quarantined and prevented from blocking
    assert engine.get_rule_stage("rule_overly_broad") == DeploymentStage.QUARANTINED
    metrics = engine.compute_metrics("rule_overly_broad")
    assert metrics.confirmed_false_positives == 1
    assert metrics.false_positive_rate == 1.0
