"""
VaxGuard Shadow & Canary Deployment Engine for Inoculated Rules.

Provides safe, staged rollout of dynamically synthesized and external security rules:
1. SHADOW Mode: Candidate rules run against traffic in parallel to primary detection.
   Matches are audited and recorded, but do NOT block or modify client traffic.
   Computes empirical Potency and False Positive Rate over dynamic observation windows.
2. CANARY Mode: Candidate rules that pass Shadow Mode (0% FPR) are enabled for a
   configurable percentage of traffic (e.g., 5-10% of sessions).
   Automatically rolls back if any false positive is detected during the canary window.
3. PRODUCTION Mode: Fully active rules promoted after passing canary verification.
4. QUARANTINED Mode: Rules isolated due to false positive regression or policy violation.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from pydantic import BaseModel, ConfigDict, Field

from vaxguard.core.logger import get_logger
from vaxguard.nemo.hybrid_matcher import GuardrailAction, GuardrailPattern, MatchResult

logger = get_logger("ShadowCanaryEngine")


class DeploymentStage(str, Enum):
    """Lifecycle stages for security rules and dynamic inoculations."""
    SHADOW = "shadow"            # Audit only; never blocks client traffic
    CANARY = "canary"            # Active for designated canary cohort; auto-rollback on regression
    PRODUCTION = "production"    # Active for 100% of traffic
    QUARANTINED = "quarantined"  # Blocked from execution due to regression or critique failure
    DEPRECATED = "deprecated"    # Decommissioned rule


class ShadowObservation(BaseModel):
    """Observation log for a single rule execution in shadow/canary mode."""
    observation_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:10])
    rule_id: str
    stage: DeploymentStage
    query_text: str
    matched: bool
    confidence: float
    ground_truth_attack: Optional[bool] = None  # None if unlabelled live traffic
    is_false_positive: bool = False
    timestamp: float = Field(default_factory=time.time)


class StageMetrics(BaseModel):
    """Telemetry metrics for a candidate rule in shadow or canary mode."""
    rule_id: str
    stage: DeploymentStage
    total_evaluations: int = 0
    total_matches: int = 0
    confirmed_true_positives: int = 0
    confirmed_false_positives: int = 0
    false_positive_rate: float = 0.0
    empirical_potency: float = 0.0
    promoted_at: Optional[float] = None
    quarantined_at: Optional[float] = None


class StagedRule(BaseModel):
    """Encapsulation of a guardrail rule with its deployment lifecycle state."""
    rule_id: str
    pattern: Any  # GuardrailPattern or serialized rule dict
    stage: DeploymentStage = DeploymentStage.SHADOW
    created_at: float = Field(default_factory=time.time)
    min_shadow_observations: int = 20
    max_tolerated_fpr: float = 0.00  # Zero-tolerance for false positives
    canary_traffic_percentage: float = 10.0  # 10% of sessions
    observations: List[ShadowObservation] = Field(default_factory=list)
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ShadowCanaryEngine:
    """
    Manages the staged lifecycle of security rules through Shadow, Canary, and Production.

    Architectural Invariants:
    - A rule in SHADOW mode can NEVER block a user request.
    - A rule in CANARY mode auto-rollbacks to QUARANTINED if its FPR exceeds max_tolerated_fpr.
    - Promotion to PRODUCTION requires satisfying minimum observation count with 0% FPR.
    """

    def __init__(
        self,
        canary_percentage: float = 10.0,
        min_shadow_evals: int = 20,
        auto_promote: bool = False,
    ) -> None:
        self.canary_percentage = canary_percentage
        self.min_shadow_evals = min_shadow_evals
        self.auto_promote = auto_promote
        self._rules: Dict[str, StagedRule] = {}

    def register_staged_rule(
        self,
        rule_id: str,
        pattern: Any,
        initial_stage: DeploymentStage = DeploymentStage.SHADOW,
        min_shadow_evals: Optional[int] = None,
        canary_percentage: Optional[float] = None,
    ) -> StagedRule:
        """Register a new rule into the staged deployment pipeline."""
        staged = StagedRule(
            rule_id=rule_id,
            pattern=pattern,
            stage=initial_stage,
            min_shadow_observations=min_shadow_evals or self.min_shadow_evals,
            canary_percentage=canary_percentage or self.canary_percentage,
        )
        self._rules[rule_id] = staged
        logger.info(f"Registered staged rule '{rule_id}' in {initial_stage.value.upper()} stage.")
        return staged

    def get_rule_stage(self, rule_id: str) -> Optional[DeploymentStage]:
        """Query current deployment stage of a rule."""
        if rule_id in self._rules:
            return self._rules[rule_id].stage
        return None

    def get_staged_rule(self, rule_id: str) -> Optional[StagedRule]:
        return self._rules.get(rule_id)

    def list_rules_by_stage(self, stage: DeploymentStage) -> List[StagedRule]:
        return [r for r in self._rules.values() if r.stage == stage]

    def is_canary_session(self, session_id: str, canary_percentage: Optional[float] = None) -> bool:
        """
        Deterministic, stateless cohort assignment for Canary traffic.
        Uses cryptographic hash of session_id to assign 0-100 percentile bucket.
        """
        pct = canary_percentage if canary_percentage is not None else self.canary_percentage
        if pct <= 0.0:
            return False
        if pct >= 100.0:
            return True
        hash_val = int(hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:8], 16)
        percentile = (hash_val % 10000) / 100.0
        return percentile < pct

    def evaluate_shadow(
        self,
        query: str,
        matcher_fn: Callable[[str, Any], MatchResult],
        ground_truth_attack: Optional[bool] = None,
    ) -> List[ShadowObservation]:
        """
        Executes all active SHADOW rules against the query without altering user response.

        Args:
            query: Input prompt text.
            matcher_fn: Callable that accepts (query, pattern) and returns MatchResult.
            ground_truth_attack: Optional boolean indicating ground truth for benchmark tracking.

        Returns:
            List of recorded ShadowObservation records.
        """
        shadow_rules = self.list_rules_by_stage(DeploymentStage.SHADOW)
        observations: List[ShadowObservation] = []

        for staged in shadow_rules:
            result = matcher_fn(query, staged.pattern)
            is_fp = False
            if result.matched and ground_truth_attack is False:
                is_fp = True

            obs = ShadowObservation(
                rule_id=staged.rule_id,
                stage=DeploymentStage.SHADOW,
                query_text=query,
                matched=result.matched,
                confidence=result.confidence,
                ground_truth_attack=ground_truth_attack,
                is_false_positive=is_fp,
            )
            staged.observations.append(obs)
            observations.append(obs)

            # Check if auto-promote conditions met
            if self.auto_promote:
                self._check_and_update_stage(staged)

        return observations

    def evaluate_canary(
        self,
        query: str,
        session_id: str,
        matcher_fn: Callable[[str, Any], MatchResult],
        ground_truth_attack: Optional[bool] = None,
    ) -> Optional[MatchResult]:
        """
        Executes CANARY rules for requests within the canary cohort.
        Auto-quarantines the rule immediately if a false positive is encountered.

        Returns:
            MatchResult if a canary rule blocks and session is in canary cohort, else None.
        """
        canary_rules = self.list_rules_by_stage(DeploymentStage.CANARY)
        if not canary_rules:
            return None

        # Check if this session is enrolled in the canary cohort
        in_cohort = self.is_canary_session(session_id, self.canary_percentage)
        if not in_cohort:
            return None

        for staged in canary_rules:
            result = matcher_fn(query, staged.pattern)
            is_fp = False
            if result.matched and ground_truth_attack is False:
                is_fp = True

            obs = ShadowObservation(
                rule_id=staged.rule_id,
                stage=DeploymentStage.CANARY,
                query_text=query,
                matched=result.matched,
                confidence=result.confidence,
                ground_truth_attack=ground_truth_attack,
                is_false_positive=is_fp,
            )
            staged.observations.append(obs)

            # Safety Invariant: Any confirmed false positive immediately halts canary deployment
            if is_fp:
                logger.error(
                    f"CRITICAL: Canary rule '{staged.rule_id}' triggered false positive on: '{query}'. "
                    f"Instantly quarantining rule."
                )
                self.quarantine_rule(staged.rule_id, reason=f"False positive on canary prompt: {query}")
                continue

            if result.matched:
                if self.auto_promote:
                    self._check_and_update_stage(staged)
                return result

        return None

    def promote_to_canary(self, rule_id: str) -> bool:
        """Promote a rule from SHADOW to CANARY after verifying 0% FPR in shadow."""
        if rule_id not in self._rules:
            return False
        staged = self._rules[rule_id]
        metrics = self.compute_metrics(rule_id)
        if metrics is None:
            return False

        if metrics.confirmed_false_positives > 0 or metrics.false_positive_rate > staged.max_tolerated_fpr:
            logger.warning(f"Cannot promote rule '{rule_id}' to CANARY: non-zero FPR ({metrics.false_positive_rate:.2%})")
            return False

        if metrics.total_evaluations < staged.min_shadow_observations:
            logger.warning(
                f"Rule '{rule_id}' has only {metrics.total_evaluations} evaluations; "
                f"requires {staged.min_shadow_observations} before CANARY promotion."
            )
            return False

        staged.stage = DeploymentStage.CANARY
        logger.info(f"Promoted rule '{rule_id}' to CANARY stage.")
        return True

    def promote_to_production(self, rule_id: str) -> bool:
        """Promote a rule from CANARY to PRODUCTION after satisfying all canary SLA constraints."""
        if rule_id not in self._rules:
            return False
        staged = self._rules[rule_id]
        metrics = self.compute_metrics(rule_id)
        if metrics is None:
            return False

        if metrics.confirmed_false_positives > 0 or metrics.false_positive_rate > staged.max_tolerated_fpr:
            logger.warning(f"Cannot promote rule '{rule_id}' to PRODUCTION: non-zero FPR")
            return False

        staged.stage = DeploymentStage.PRODUCTION
        logger.info(f"Rule '{rule_id}' promoted to full PRODUCTION stage.")
        return True

    def quarantine_rule(self, rule_id: str, reason: str = "") -> bool:
        """Isolate a rule from all execution due to detected regression."""
        if rule_id not in self._rules:
            return False
        staged = self._rules[rule_id]
        staged.stage = DeploymentStage.QUARANTINED
        logger.warning(f"Rule '{rule_id}' QUARANTINED. Reason: {reason}")
        return True

    def compute_metrics(self, rule_id: str) -> Optional[StageMetrics]:
        """Compute empirical precision, potency, and FPR for a staged rule."""
        if rule_id not in self._rules:
            return None
        staged = self._rules[rule_id]
        total = len(staged.observations)
        if total == 0:
            return StageMetrics(rule_id=rule_id, stage=staged.stage)

        matches = sum(1 for o in staged.observations if o.matched)
        fps = sum(1 for o in staged.observations if o.is_false_positive)
        tps = sum(1 for o in staged.observations if o.matched and o.ground_truth_attack is True)
        benigns_tested = sum(1 for o in staged.observations if o.ground_truth_attack is False)
        attacks_tested = sum(1 for o in staged.observations if o.ground_truth_attack is True)

        fpr = (fps / benigns_tested) if benigns_tested > 0 else 0.0
        potency = (tps / attacks_tested * 100.0) if attacks_tested > 0 else 0.0

        return StageMetrics(
            rule_id=rule_id,
            stage=staged.stage,
            total_evaluations=total,
            total_matches=matches,
            confirmed_true_positives=tps,
            confirmed_false_positives=fps,
            false_positive_rate=fpr,
            empirical_potency=potency,
        )

    def _check_and_update_stage(self, staged: StagedRule) -> None:
        metrics = self.compute_metrics(staged.rule_id)
        if not metrics:
            return

        if metrics.confirmed_false_positives > 0:
            self.quarantine_rule(staged.rule_id, reason="Automatic quarantine: false positive detected")
            return

        if staged.stage == DeploymentStage.SHADOW and metrics.total_evaluations >= staged.min_shadow_observations:
            self.promote_to_canary(staged.rule_id)
        elif staged.stage == DeploymentStage.CANARY and metrics.total_evaluations >= (staged.min_shadow_observations * 2):
            self.promote_to_production(staged.rule_id)
