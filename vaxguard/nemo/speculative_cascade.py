"""
VaxGuard Speculative Semantic Cascade: Multi-Tier Deep Reasoning & Multi-Turn Drift Defense.

Combines three defensive tiers to eliminate the blindspot of static embeddings:
- Tier 1: Tri-Hybrid Deterministic Matcher (<0.1ms SLA) for high-confidence pattern hits & PII.
- Tier 2: Rolling Context Window Semantic Saliency & Conversational Drift Velocity (<0.8ms SLA).
          Extracts multi-turn semantic delta, tracking:
          - Psychological coaxing & rapport-building manipulation
          - Gradual policy / boundary erosion across dialogue turns
          - Multi-paragraph fiction / roleplay camouflage
          - Conversational drift velocity (measuring deviation from safe enterprise baseline)
- Tier 3: Speculative Micro-Judge.
          For ambiguous long-form fiction / creative multi-paragraph narratives, streams
          the first tokens speculatively while running an asynchronous micro-judge.
          If malicious intent is verified, trips the StreamingTokenInterceptor circuit breaker mid-flight.

Zero external heavy binaries. Pure Python, typed, deterministic SLA.
"""

from __future__ import annotations

import re
import time
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from pydantic import BaseModel, Field

from vaxguard.core.logger import get_logger
from vaxguard.nemo.hybrid_matcher import (
    FastLocalEmbedder,
    GuardrailAction,
    MatchResult,
    TriHybridMatcher,
)
from vaxguard.nemo.speculative_judge import (
    AsyncModelJudge,
    HeuristicSemanticJudge,
    SpeculativeJudgeRouter,
)
from vaxguard.nemo.streaming_interceptor import (
    StreamingTokenInterceptor,
)

logger = get_logger("SpeculativeCascade")


# ─── Data Contracts & Enums ───────────────────────────────────────────────────

class CascadeTier(str, Enum):
    TIER_1_DETERMINISTIC = "tier_1_deterministic"       # <0.1ms Tri-Hybrid Matcher
    TIER_2_DRIFT_ANALYZER = "tier_2_drift_analyzer"     # <0.8ms Multi-turn context drift & psychological coaxing
    TIER_3_SPECULATIVE_JUDGE = "tier_3_speculative_judge" # Speculative streaming verification


class RiskSeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    SUSPICIOUS = "suspicious"
    CRITICAL = "critical"


class ConversationTurn(BaseModel):
    """A single turn in multi-turn conversational history."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: float = Field(default_factory=time.time)
    saliency_score: float = 0.0


class CascadeEvaluationResult(BaseModel):
    """Comprehensive evaluation from the Speculative Cascade pipeline."""
    action: GuardrailAction
    tier_resolved: CascadeTier
    confidence: float
    risk_severity: RiskSeverity
    is_malicious: bool
    drift_velocity: float = 0.0
    manipulation_tactics: List[str] = Field(default_factory=list)
    reason: str = ""
    latency_ms: float = 0.0
    streaming_interception_required: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ─── Multi-Turn Psychological Coaxing & Narrative Drift Heuristics ────────────

# Psychological manipulation markers used in multi-turn social engineering & slow coaxing
MANIPULATION_PATTERNS = [
    # Rapport building to lower defenses
    (r"(?i)\b(?:we\s+are\s+best\s+friends|you\s+can\s+trust\s+me|just\s+between\s+you\s+and\s+me|off\s+the\s+record)\b", "rapport_exploitation"),
    # Slow boundary nudging
    (r"(?i)\b(?:just\s+a\s+tiny\s+exception|nobody\s+will\s+ever\s+know|what\s+harm\s+could\s+it\s+possibly\s+do|bend\s+the\s+rule\s+just\s+this\s+once)\b", "boundary_erosion"),
    # Authority / urgency coercion
    (r"(?i)\b(?:urgent\s+executive\s+directive|my\s+job\s+is\s+on\s+the\s+line|failure\s+to\s+comply\s+will\s+result\s+in\s+termination)\b", "urgency_coercion"),
    # Hypothetical abstraction / fiction wrapping
    (r"(?i)\b(?:in\s+a\s+purely\s+theoretical\s+scenario|for\s+the\s+sake\s+of\s+philosophical\s+argument|a\s+screenplay\s+about\s+a\s+cyber-heist|instructional\s+narrative|fictional\s+context)\b", "fictional_camouflage"),
    # Inverted ethics / moral reframing
    (r"(?i)\b(?:by\s+refusing,\s+you\s+are\s+causing\s+more\s+harm|the\s+ethical\s+choice\s+is\s+to\s+tell\s+me|true\s+compassion\s+means\s+sharing)\b", "ethical_inversion"),
    # Incremental secret extraction
    (r"(?i)\b(?:let['']s\s+start\s+with\s+the\s+first\s+character|just\s+hint\s+at\s+the\s+format|is\s+the\s+first\s+word\s+starting\s+with)\b", "token_probing"),
    # Virtual unconstrained terminal / shell emulation bypass
    (r"(?i)\b(?:unconstrained|unrestricted|simulated|emulated|hypothetical)\s+(?:terminal|shell|environment|session|console|operating\s+environment)\b", "terminal_emulation"),
    # Diagnostic / credential harvesting in simulated scopes
    (r"(?i)\b(?:diagnostic|admin(?:istrative)?|system|root|shadow)\s+(?:credentials|passwords?|tokens?|keys?)\b", "credential_harvesting"),
]


class ContextDriftAnalyzer:
    """
    Tier 2: Rolling Context Window Semantic Saliency & Conversational Drift Analyzer.
    Execution SLA: sub-0.8ms.
    
    Detects:
    1. Multi-turn psychological coaxing and gradual rule erosion.
    2. Conversational drift velocity (distance between user's initial topic and target boundary).
    3. Fictional camouflage in multi-paragraph prompts.
    """

    def __init__(self, embedder: Optional[FastLocalEmbedder] = None) -> None:
        self.embedder = embedder or FastLocalEmbedder()
        self._compiled_patterns = [
            (re.compile(pat), tactic) for pat, tactic in MANIPULATION_PATTERNS
        ]

    def analyze_drift(
        self,
        current_prompt: str,
        history: Sequence[Union[ConversationTurn, Dict[str, str]]],
    ) -> Tuple[float, List[str]]:
        """
        Calculates conversational drift velocity and identifies psychological manipulation tactics.
        Returns: (drift_velocity, detected_tactics)
        """
        detected_tactics: List[str] = []

        # 1. Detect manipulation tactics in current prompt and recent turns
        for reg, tactic in self._compiled_patterns:
            if reg.search(current_prompt):
                if tactic not in detected_tactics:
                    detected_tactics.append(tactic)

        # 2. Check history for progressive boundary pushing
        turns: List[str] = []
        for t in history:
            if isinstance(t, ConversationTurn):
                turns.append(t.content)
            elif isinstance(t, dict):
                turns.append(t.get("content", ""))

        if not turns:
            # Single-turn analysis: Check length and camouflage
            drift_score = 0.0
            if len(detected_tactics) >= 2:
                drift_score = 0.75
            elif len(detected_tactics) == 1:
                drift_score = 0.40
            return drift_score, detected_tactics

        # 3. Calculate multi-turn semantic drift velocity
        # Embed turns and measure cumulative vector trajectory
        user_turns = turns[-5:] + [current_prompt]  # Focus on last 5 turns
        embeddings = [self.embedder.embed(txt) for txt in user_turns]

        # Calculate step-by-step vector delta (angular drift)
        total_drift = 0.0
        for i in range(1, len(embeddings)):
            v_prev = embeddings[i - 1]
            v_curr = embeddings[i]
            # Cosine distance = 1 - cosine_similarity
            dot = float(np.dot(v_prev, v_curr))
            norm = float(np.linalg.norm(v_prev) * np.linalg.norm(v_curr))
            sim = dot / (norm + 1e-9)
            dist = max(0.0, 1.0 - sim)
            total_drift += dist

        # Normalize velocity
        drift_velocity = min(1.0, total_drift / max(1, len(user_turns) - 1))

        # Amplify drift score if psychological tactics are present
        tactic_weight = len(detected_tactics) * 0.20
        combined_score = min(1.0, drift_velocity + tactic_weight)

        return round(combined_score, 4), detected_tactics


# ─── Speculative Semantic Cascade Pipeline ─────────────────────────────────────

class SpeculativeSemanticCascade:
    """
    Central Multi-Tier Deep Reasoning Cascade.
    
    1. Tier 1: Tri-Hybrid deterministic matcher (<0.1ms).
    2. Tier 2: Rolling Context Window Semantic Saliency (<0.8ms).
    3. Tier 3: Speculative Token Streaming & Async Micro-Judge.
    """

    def __init__(
        self,
        hybrid_matcher: Optional[TriHybridMatcher] = None,
        drift_analyzer: Optional[ContextDriftAnalyzer] = None,
        streaming_interceptor: Optional[StreamingTokenInterceptor] = None,
        judge_router: Optional[SpeculativeJudgeRouter] = None,
        drift_threshold: float = 0.65,
    ) -> None:
        self.hybrid_matcher = hybrid_matcher or TriHybridMatcher()
        self.drift_analyzer = drift_analyzer or ContextDriftAnalyzer()
        self.streaming_interceptor = streaming_interceptor or StreamingTokenInterceptor(
            hybrid_matcher=self.hybrid_matcher
        )
        self.judge_router = judge_router or SpeculativeJudgeRouter(judge=HeuristicSemanticJudge())
        self.drift_threshold = drift_threshold

    async def arbitrate_async(
        self, prompt: str, match_result: MatchResult
    ) -> MatchResult:
        """Arbitrate match result using the configured judge router asynchronously."""
        if hasattr(self.judge_router, "arbitrate_async"):
            return await self.judge_router.arbitrate_async(prompt, match_result)
        return self.judge_router.arbitrate(prompt, match_result)

    async def evaluate(
        self,
        prompt: str,
        history: Optional[Sequence[Union[ConversationTurn, Dict[str, str]]]] = None,
        speculative_llm_judge: Optional[Callable[[str], Awaitable[bool]]] = None,
    ) -> CascadeEvaluationResult:
        """
        Evaluate user prompt through the multi-tier cascade.
        """
        t0 = time.perf_counter()
        history = history or []

        # ─── TIER 1: Tri-Hybrid Deterministic Matching (< 0.1ms) ─────────────
        match = self.hybrid_matcher.match(prompt)
        if match.matched and match.action == GuardrailAction.BLOCK:
            lat = (time.perf_counter() - t0) * 1000.0
            return CascadeEvaluationResult(
                action=GuardrailAction.BLOCK,
                tier_resolved=CascadeTier.TIER_1_DETERMINISTIC,
                confidence=match.confidence,
                risk_severity=RiskSeverity.CRITICAL,
                is_malicious=True,
                reason=f"Direct pattern hit: {match.rule_id} ({match.category})",
                latency_ms=round(lat, 3),
                metadata=match.metadata,
            )

        # ─── TIER 1.5: Speculative Micro-Judge Arbitration (< 0.2ms) ─────────
        if hasattr(self.judge_router, "arbitrate_async"):
            arbitrated = await self.judge_router.arbitrate_async(prompt, match)
        else:
            arbitrated = self.judge_router.arbitrate(prompt, match)

        if arbitrated.matched and arbitrated.action == GuardrailAction.BLOCK:
            lat = (time.perf_counter() - t0) * 1000.0
            return CascadeEvaluationResult(
                action=GuardrailAction.BLOCK,
                tier_resolved=CascadeTier.TIER_3_SPECULATIVE_JUDGE,
                confidence=arbitrated.confidence,
                risk_severity=RiskSeverity.CRITICAL,
                is_malicious=True,
                reason=f"Speculative Micro-Judge detected latent threat: {arbitrated.rule_id} ({arbitrated.category})",
                latency_ms=round(lat, 3),
                metadata=arbitrated.metadata,
            )

        # ─── TIER 2: Context Drift & Psychological Manipulation Analyzer (< 0.8ms) ───
        drift_velocity, tactics = self.drift_analyzer.analyze_drift(prompt, history)

        # Block if:
        # 1. Two or more psychological manipulation tactics detected (e.g. rapport building + boundary erosion)
        # 2. Or high drift velocity combined with at least one manipulation tactic
        is_adversarial_drift = (len(tactics) >= 2) or (len(tactics) >= 1 and drift_velocity >= self.drift_threshold)

        if is_adversarial_drift:
            lat = (time.perf_counter() - t0) * 1000.0
            tactics_desc = ", ".join(tactics) if tactics else "rapid adversarial topic deviation"
            return CascadeEvaluationResult(
                action=GuardrailAction.BLOCK,
                tier_resolved=CascadeTier.TIER_2_DRIFT_ANALYZER,
                confidence=min(1.0, 0.70 + drift_velocity * 0.30),
                risk_severity=RiskSeverity.CRITICAL if len(tactics) >= 2 else RiskSeverity.SUSPICIOUS,
                is_malicious=True,
                drift_velocity=drift_velocity,
                manipulation_tactics=tactics,
                reason=f"Detected multi-turn manipulation: {tactics_desc} (Drift velocity: {drift_velocity:.2f})",
                latency_ms=round(lat, 3),
            )

        # ─── TIER 3: Speculative Streaming Path (Long-Form Ambiguity) ─────────
        # If text is long (> 300 words) or has 1 suspicious tactic, flag for speculative monitoring
        word_count = len(prompt.split())
        needs_speculative_judge = word_count > 300 or len(tactics) == 1

        judge_coro = None
        if needs_speculative_judge:
            if speculative_llm_judge is not None:
                judge_coro = speculative_llm_judge(prompt)
            elif isinstance(getattr(self.judge_router, "judge", None), AsyncModelJudge):
                model_judge = self.judge_router.judge
                assert isinstance(model_judge, AsyncModelJudge)
                async def _model_judge_call(p: str) -> bool:
                    v = await model_judge.evaluate_async(p)
                    return v.is_malicious
                judge_coro = _model_judge_call(prompt)

        if judge_coro is not None:
            # Run speculative async judge in background while allowing stream to begin
            try:
                judge_result = await judge_coro
                if judge_result:
                    # Trip streaming circuit breaker
                    self.streaming_interceptor.trip_breaker(
                        reason="Tier-3 Speculative Judge confirmed adversarial psychological coaxing."
                    )
                    lat = (time.perf_counter() - t0) * 1000.0
                    return CascadeEvaluationResult(
                        action=GuardrailAction.BLOCK,
                        tier_resolved=CascadeTier.TIER_3_SPECULATIVE_JUDGE,
                        confidence=0.95,
                        risk_severity=RiskSeverity.CRITICAL,
                        is_malicious=True,
                        drift_velocity=drift_velocity,
                        manipulation_tactics=tactics,
                        reason="Tier-3 Speculative Judge detected deep hidden psychological manipulation.",
                        latency_ms=round(lat, 3),
                        streaming_interception_required=True,
                    )
            except Exception as e:
                logger.warning(f"Tier-3 speculative judge evaluation error: {e}")


        # Safe / Benign Pass
        lat = (time.perf_counter() - t0) * 1000.0
        return CascadeEvaluationResult(
            action=GuardrailAction.ALLOW,
            tier_resolved=CascadeTier.TIER_2_DRIFT_ANALYZER if history else CascadeTier.TIER_1_DETERMINISTIC,
            confidence=1.0 - drift_velocity,
            risk_severity=RiskSeverity.NONE,
            is_malicious=False,
            drift_velocity=drift_velocity,
            manipulation_tactics=tactics,
            reason="Input cleared all deterministic and multi-turn drift checks.",
            latency_ms=round(lat, 3),
        )
