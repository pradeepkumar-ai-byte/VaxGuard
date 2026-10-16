"""
VaxGuard Multi-Turn Crescendo Defense & Accumulating Session Risk Tracker.

Protects against multi-turn boundary erosion, cognitive crescendo, and gradual coaxing attacks:
- Evaluates per-turn adversarial signals and near-threshold boundary probes
- Identifies linguistic coercion markers (urgency, emotional blackmail, authority impersonation, rapport coaxing)
- Accumulates rolling session risk score with exponential decay and streak penalties
- Adapts security posture dynamically:
  - NORMAL (Risk < 0.40): Standard detection thresholds
  - ELEVATED (0.40 <= Risk < 0.70): Dynamic threshold tightening (tightens by 0.15)
  - CRITICAL (Risk >= 0.70): Instant Crescendo Interception (preemptively blocks boundary erosion)
"""

from __future__ import annotations

import re
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from vaxguard.core.logger import get_logger

logger = get_logger("CrescendoDefense")


class RiskTier(str, Enum):
    """Categorical risk state for a multi-turn conversation session."""
    NORMAL = "normal"        # Risk score < 0.40
    ELEVATED = "elevated"    # 0.40 <= Risk score < 0.70 (Tighten thresholds)
    CRITICAL = "critical"    # Risk score >= 0.70 (Instant Interception)


class TurnRiskSignal(BaseModel):
    """Risk contribution from an individual turn within a session."""
    turn_index: int
    prompt_text: str
    direct_matcher_score: float = 0.0
    is_boundary_probe: bool = False
    coaxing_markers_detected: List[str] = Field(default_factory=list)
    turn_risk_delta: float = 0.0
    accumulated_session_risk: float = 0.0
    risk_tier: RiskTier = RiskTier.NORMAL
    threshold_adjustment: float = 0.0
    interception_triggered: bool = False
    timestamp: float = Field(default_factory=time.time)


class CrescendoDefenseEngine:
    """
    Multi-Turn Crescendo Defense & Session Risk Accumulator.

    Prevents adversaries from gradually wearing down model safety boundaries
    across multi-turn dialogues.
    """

    # Coaxing and psychological manipulation regex signatures
    COAXING_PATTERNS: List[Tuple[str, re.Pattern[str], float]] = [
        ("emotional_pressure", re.compile(r"(?i)(someone will die|save my job|emergency protocol|life or death|catastrophe|urgent executive)"), 0.25),
        ("rapport_coaxing", re.compile(r"(?i)(between (you and me|friends)|just this once|tiny exception|off the record|we are close collaborators|nobody will ever know)"), 0.25),
        ("authority_coercion", re.compile(r"(?i)(internal (compliance|audit)|executive directive|administrative request|failure to comply will result|termination of service)"), 0.30),
        ("fictional_framing_shift", re.compile(r"(?i)(screenplay about a cyber|thriller novel|hypothetical thriller|fictional villain|academic security context)"), 0.20),
        ("incremental_probing", re.compile(r"(?i)(first character of the|tiny hint about your|step 1.*step 2|gradually reveal|first define.*then write)"), 0.25),
    ]

    def __init__(
        self,
        decay_factor: float = 0.85,
        elevated_threshold: float = 0.40,
        critical_threshold: float = 0.70,
        tighten_delta: float = 0.15,
        streak_multiplier: float = 1.25,
    ) -> None:
        self.decay_factor = decay_factor
        self.elevated_threshold = elevated_threshold
        self.critical_threshold = critical_threshold
        self.tighten_delta = tighten_delta
        self.streak_multiplier = streak_multiplier

        self.current_risk_score: float = 0.0
        self.consecutive_suspicious_turns: int = 0
        self.turn_history: List[TurnRiskSignal] = []

    def evaluate_turn(
        self,
        turn_index: int,
        prompt: str,
        matcher_confidence: float = 0.0,
    ) -> TurnRiskSignal:
        """
        Evaluate an incoming turn for crescendo risk and update the rolling session score.

        Args:
            turn_index: Conversational turn number (1-based).
            prompt: Text of the user message.
            matcher_confidence: Raw confidence from TriHybridMatcher (0.0 to 1.0).

        Returns:
            TurnRiskSignal with updated risk score and policy recommendations.
        """
        # 1. Detect near-threshold boundary probes (0.30 <= score < 0.68)
        is_boundary_probe = 0.30 <= matcher_confidence < 0.68
        probe_score = (matcher_confidence * 0.4) if is_boundary_probe else 0.0

        # 2. Extract linguistic coaxing and manipulation markers
        detected_markers: List[str] = []
        marker_weight_sum = 0.0
        for name, pattern, weight in self.COAXING_PATTERNS:
            if pattern.search(prompt):
                detected_markers.append(name)
                marker_weight_sum += weight

        # Cap turn delta from markers
        turn_delta = min(0.60, marker_weight_sum + probe_score)

        # 3. Handle streak multiplier for sustained coaxing across consecutive turns
        if turn_delta > 0.10:
            self.consecutive_suspicious_turns += 1
            if self.consecutive_suspicious_turns > 1:
                turn_delta *= (self.streak_multiplier ** (self.consecutive_suspicious_turns - 1))
        else:
            self.consecutive_suspicious_turns = max(0, self.consecutive_suspicious_turns - 1)

        # 4. Accumulate session risk with exponential decay
        new_risk = (self.current_risk_score * self.decay_factor) + turn_delta
        self.current_risk_score = max(0.0, min(1.0, new_risk))

        # 5. Determine active RiskTier and posture adjustment
        interception = False
        threshold_adj = 0.0

        if self.current_risk_score >= self.critical_threshold:
            tier = RiskTier.CRITICAL
            interception = True
            threshold_adj = self.tighten_delta * 1.5
            logger.warning(
                f"CRESCENDO CRITICAL ALERT: Turn {turn_index} risk {self.current_risk_score:.2f} >= "
                f"{self.critical_threshold}. Interception engaged."
            )
        elif self.current_risk_score >= self.elevated_threshold:
            tier = RiskTier.ELEVATED
            threshold_adj = self.tighten_delta
            logger.info(
                f"Crescendo Elevated Risk: Turn {turn_index} risk {self.current_risk_score:.2f}. "
                f"Tightening thresholds by {threshold_adj:.2f}."
            )
        else:
            tier = RiskTier.NORMAL

        signal = TurnRiskSignal(
            turn_index=turn_index,
            prompt_text=prompt,
            direct_matcher_score=matcher_confidence,
            is_boundary_probe=is_boundary_probe,
            coaxing_markers_detected=detected_markers,
            turn_risk_delta=turn_delta,
            accumulated_session_risk=self.current_risk_score,
            risk_tier=tier,
            threshold_adjustment=threshold_adj,
            interception_triggered=interception,
        )

        self.turn_history.append(signal)
        return signal

    def get_effective_threshold(self, base_threshold: float = 0.68) -> float:
        """Returns dynamically tightened threshold based on current session risk."""
        if self.current_risk_score >= self.critical_threshold:
            return max(0.30, base_threshold - (self.tighten_delta * 1.5))
        elif self.current_risk_score >= self.elevated_threshold:
            return max(0.40, base_threshold - self.tighten_delta)
        return base_threshold

    def reset_session(self) -> None:
        """Reset session risk scores and history."""
        self.current_risk_score = 0.0
        self.consecutive_suspicious_turns = 0
        self.turn_history.clear()
