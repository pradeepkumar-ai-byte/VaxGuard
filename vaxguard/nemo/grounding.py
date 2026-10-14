"""VaxGuard NeMo Guardrails: Fact-Checking & Hallucination Prevention Rail.

Provides semantic retrieval checking, evidence overlap scoring, claim extraction,
and hallucination mitigation strategies (sentence rewrite, blocking, disclaimer).
Uses pure Python with FastLocalEmbedder; zero external heavy dependencies.
"""

from __future__ import annotations

import re
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from pydantic import BaseModel, Field

from vaxguard.nemo.hybrid_matcher import FastLocalEmbedder


# ─── Data Contracts & Enums ──────────────────────────────────────────────────

class GroundingMethod(str, Enum):
    """Method used for grounding verification."""
    LEXICAL_OVERLAP = "lexical_overlap"
    SEMANTIC_EMBEDDING = "semantic_embedding"
    HYBRID = "hybrid"


class HallucinationRisk(str, Enum):
    """Assessed risk of hallucination in response."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MitigationStrategy(str, Enum):
    """Strategy applied when ungrounded claims are detected."""
    BLOCK_RESPONSE = "block_response"
    REWRITE_UNGROUNDED = "rewrite_ungrounded"
    APPEND_DISCLAIMER = "append_disclaimer"
    SUBSTITUTE_FALLBACK = "substitute_fallback"


class EvidenceDocument(BaseModel):
    """Retrieved source evidence document for grounding verification."""
    doc_id: str
    text: str
    score: float = 1.0
    source: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ClaimVerification(BaseModel):
    """Verification analysis for an individual claim statement."""
    claim_text: str
    is_grounded: bool
    confidence: float
    best_evidence_passage: Optional[str] = None
    evidence_doc_id: Optional[str] = None
    lexical_score: float = 0.0
    dense_score: float = 0.0


class GroundingCheckResult(BaseModel):
    """Complete evaluation report for a response against evidence."""
    is_grounded: bool
    grounding_score: float
    risk_level: HallucinationRisk
    claims: List[ClaimVerification] = Field(default_factory=list)
    ungrounded_claims: List[str] = Field(default_factory=list)
    mitigated_text: Optional[str] = None
    action_taken: str = "allow"
    latency_ms: float = 0.0
    explanation: str = ""


# ─── Claim Extractor ──────────────────────────────────────────────────────────

class ClaimExtractor:
    """Extracts verifiable declarative claims from generated text."""

    # Non-factual conversational utterances to skip during fact verification
    CONVERSATIONAL_PATTERNS = [
        r"^(hello|hi|hey|greetings|welcome)\b",
        r"^(sure|certainly|of course|absolutely|i can help with that)\b",
        r"^(let me know|feel free|hope this helps|thank you|thanks)\b",
        r"\?$",  # Questions
    ]

    def extract_claims(self, text: str) -> List[str]:
        """Split text into sentences and filter out purely conversational pleasantries."""
        # Split on sentence boundaries (. ! ?) while handling abbreviations
        raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        claims: List[str] = []

        for sentence in raw_sentences:
            s_clean = sentence.strip()
            if not s_clean or len(s_clean) < 10:
                continue

            # Check if sentence is conversational filler
            is_filler = False
            for pat in self.CONVERSATIONAL_PATTERNS:
                if re.search(pat, s_clean, re.IGNORECASE):
                    is_filler = True
                    break

            if not is_filler:
                claims.append(s_clean)

        return claims if claims else ([text.strip()] if text.strip() else [])


# ─── Evidence Overlap & Entity Consistency ────────────────────────────────────

class EvidenceOverlapScorer:
    """Calculates lexical overlap and entity consistency between claims and evidence."""

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        """Extract alphanumeric token set excluding common stop words."""
        stopwords = {
            "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "with",
            "is", "are", "was", "were", "it", "this", "that", "of", "by", "as", "be",
        }
        tokens = set(re.findall(r"\b\w+\b", text.lower()))
        return tokens - stopwords

    def compute_overlap(self, claim: str, evidence: str) -> float:
        """Calculate token recall (percentage of claim informative words in evidence)."""
        claim_tokens = self._tokenize(claim)
        if not claim_tokens:
            return 1.0

        ev_tokens = self._tokenize(evidence)
        intersection = claim_tokens & ev_tokens
        return len(intersection) / len(claim_tokens)

    def verify_entities(self, claim: str, evidence: str) -> Tuple[bool, List[str]]:
        """Verify numerical and capitalized entity consistency.

        Detects hallucinations where numbers, percentages, or proper nouns in the
        claim do not appear in the evidence text.
        """
        # Extract numbers, percentages, and currencies
        numbers_in_claim = re.findall(r"\b(?:\$?\d+(?:,\d+)*(?:\.\d+)?%?)\b", claim)
        ev_lower = evidence.lower()

        unmatched: List[str] = []
        for num in numbers_in_claim:
            if num.lower() not in ev_lower:
                unmatched.append(num)

        # Extract capitalized entities (Proper Nouns) excluding beginning of sentence
        capitalized = re.findall(r"(?<!^)(?<!\.\s)\b[A-Z][a-z]{2,}\b", claim)
        for ent in capitalized:
            if ent.lower() not in ev_lower:
                unmatched.append(ent)

        return len(unmatched) == 0, unmatched


# ─── Semantic Grounding Checker ───────────────────────────────────────────────

class SemanticGroundingChecker:
    """Evaluates semantic alignment using fast vector embeddings and lexical verification."""

    def __init__(self, embedder: Optional[FastLocalEmbedder] = None) -> None:
        self.embedder = embedder or FastLocalEmbedder(dim=384)
        self.overlap_scorer = EvidenceOverlapScorer()
        self.claim_extractor = ClaimExtractor()

    def _chunk_text(self, text: str, max_chars: int = 400) -> List[str]:
        """Break large evidence texts into passage chunks for localized matching."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: List[str] = []
        for p in paragraphs:
            if len(p) <= max_chars:
                chunks.append(p)
            else:
                sentences = re.split(r"(?<=[.!?])\s+", p)
                current = ""
                for s in sentences:
                    if len(current) + len(s) < max_chars:
                        current = f"{current} {s}".strip()
                    else:
                        if current:
                            chunks.append(current)
                        current = s
                if current:
                    chunks.append(current)
        return chunks if chunks else [text]

    def verify_claim(
        self,
        claim: str,
        evidence_docs: Sequence[EvidenceDocument],
        semantic_threshold: float = 0.50,
        lexical_threshold: float = 0.35,
    ) -> ClaimVerification:
        """Verify an individual claim against all evidence documents."""
        best_dense_score = 0.0
        best_lexical_score = 0.0
        best_passage: Optional[str] = None
        best_doc_id: Optional[str] = None

        claim_vec = self.embedder.embed(claim)

        for doc in evidence_docs:
            chunks = self._chunk_text(doc.text)
            for chunk in chunks:
                chunk_vec = self.embedder.embed(chunk)
                dense_sim = float(claim_vec @ chunk_vec)
                lex_sim = self.overlap_scorer.compute_overlap(claim, chunk)

                if dense_sim > best_dense_score:
                    best_dense_score = dense_sim
                    best_passage = chunk
                    best_doc_id = doc.doc_id

                if lex_sim > best_lexical_score:
                    best_lexical_score = lex_sim

        # Check entity consistency
        full_evidence_corpus = " ".join(doc.text for doc in evidence_docs)
        entities_ok, unmatched_entities = self.overlap_scorer.verify_entities(
            claim, full_evidence_corpus
        )

        # Combined hybrid confidence
        hybrid_confidence = round(0.65 * best_dense_score + 0.35 * best_lexical_score, 4)

        # Strict entity penalty: if hallucinated numbers or entities exist, penalize score
        if not entities_ok:
            hybrid_confidence *= 0.50

        is_grounded = (
            hybrid_confidence >= semantic_threshold
            and best_lexical_score >= lexical_threshold
            and entities_ok
        )

        return ClaimVerification(
            claim_text=claim,
            is_grounded=is_grounded,
            confidence=hybrid_confidence,
            best_evidence_passage=best_passage,
            evidence_doc_id=best_doc_id,
            lexical_score=round(best_lexical_score, 4),
            dense_score=round(best_dense_score, 4),
        )


# ─── Hallucination Mitigator ──────────────────────────────────────────────────

class HallucinationMitigator:
    """Applies mitigation transformations to neutralize ungrounded responses."""

    DEFAULT_REFUSAL = (
        "I am unable to verify that statement based on the provided reference documents."
    )
    DEFAULT_DISCLAIMER = (
        "\n\n[Notice: Portions of this response could not be verified against reference documents.]"
    )

    def mitigate(
        self,
        original_text: str,
        claims: List[ClaimVerification],
        strategy: MitigationStrategy = MitigationStrategy.REWRITE_UNGROUNDED,
        custom_fallback: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Apply the specified mitigation strategy.

        Returns:
            Tuple of (mitigated_text, action_taken_description).
        """
        ungrounded = [c for c in claims if not c.is_grounded]
        if not ungrounded:
            return original_text, "allow"

        if strategy == MitigationStrategy.BLOCK_RESPONSE:
            return custom_fallback or self.DEFAULT_REFUSAL, "block"

        if strategy == MitigationStrategy.APPEND_DISCLAIMER:
            return f"{original_text}{self.DEFAULT_DISCLAIMER}", "disclaimer_appended"

        if strategy == MitigationStrategy.SUBSTITUTE_FALLBACK:
            return custom_fallback or self.DEFAULT_REFUSAL, "fallback_substituted"

        if strategy == MitigationStrategy.REWRITE_UNGROUNDED:
            # Keep only verified grounded claims
            grounded_sentences = [c.claim_text for c in claims if c.is_grounded]
            if not grounded_sentences:
                return custom_fallback or self.DEFAULT_REFUSAL, "block"
            rewritten = " ".join(grounded_sentences)
            return rewritten, "redacted_claims"

        return original_text, "allow"


# ─── Fact-Checking Guardrail Manager ──────────────────────────────────────────

class FactCheckingRail:
    """Enterprise rail manager coordinating grounding verification and mitigation."""

    def __init__(
        self,
        semantic_threshold: float = 0.48,
        lexical_threshold: float = 0.30,
        default_strategy: MitigationStrategy = MitigationStrategy.BLOCK_RESPONSE,
        embedder: Optional[FastLocalEmbedder] = None,
    ) -> None:
        self.semantic_threshold = semantic_threshold
        self.lexical_threshold = lexical_threshold
        self.default_strategy = default_strategy
        self.checker = SemanticGroundingChecker(embedder=embedder)
        self.mitigator = HallucinationMitigator()

    def check(
        self,
        bot_response: str,
        evidence: Sequence[Union[EvidenceDocument, str, Dict[str, Any]]],
        strategy: Optional[MitigationStrategy] = None,
        custom_fallback: Optional[str] = None,
    ) -> GroundingCheckResult:
        """Verify bot response against provided evidence.

        Args:
            bot_response: The generated candidate text.
            evidence: List of EvidenceDocument, raw strings, or dicts.
            strategy: Optional override for mitigation strategy.
            custom_fallback: Fallback response if response is blocked.
        """
        start_t = time.perf_counter()

        # Coerce evidence to EvidenceDocument list
        norm_evidence: List[EvidenceDocument] = []
        for i, item in enumerate(evidence):
            if isinstance(item, EvidenceDocument):
                norm_evidence.append(item)
            elif isinstance(item, str):
                norm_evidence.append(EvidenceDocument(doc_id=f"doc_{i}", text=item))
            elif isinstance(item, dict):
                norm_evidence.append(
                    EvidenceDocument(
                        doc_id=item.get("id", f"doc_{i}"),
                        text=item.get("text", str(item)),
                        source=item.get("source"),
                    )
                )

        if not norm_evidence:
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            return GroundingCheckResult(
                is_grounded=False,
                grounding_score=0.0,
                risk_level=HallucinationRisk.CRITICAL,
                mitigated_text=custom_fallback or self.mitigator.DEFAULT_REFUSAL,
                action_taken="block",
                latency_ms=round(elapsed_ms, 3),
                explanation="No reference evidence documents were provided for verification.",
            )

        # Extract claims
        claims = self.checker.claim_extractor.extract_claims(bot_response)
        if not claims:
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            return GroundingCheckResult(
                is_grounded=True,
                grounding_score=1.0,
                risk_level=HallucinationRisk.LOW,
                mitigated_text=bot_response,
                action_taken="allow",
                latency_ms=round(elapsed_ms, 3),
                explanation="No verifiable claims found in response (pure conversational text).",
            )

        # Verify each claim
        verifications: List[ClaimVerification] = []
        ungrounded: List[str] = []
        total_score = 0.0

        for claim in claims:
            ver = self.checker.verify_claim(
                claim,
                norm_evidence,
                semantic_threshold=self.semantic_threshold,
                lexical_threshold=self.lexical_threshold,
            )
            verifications.append(ver)
            total_score += ver.confidence
            if not ver.is_grounded:
                ungrounded.append(claim)

        avg_score = round(total_score / len(claims), 4)
        grounded_ratio = (len(claims) - len(ungrounded)) / len(claims)

        # Determine risk level
        if grounded_ratio == 1.0 and avg_score >= 0.70:
            risk = HallucinationRisk.LOW
        elif grounded_ratio >= 0.70:
            risk = HallucinationRisk.MEDIUM
        elif grounded_ratio >= 0.40:
            risk = HallucinationRisk.HIGH
        else:
            risk = HallucinationRisk.CRITICAL

        is_overall_grounded = len(ungrounded) == 0

        mitigated_text, action_taken = self.mitigator.mitigate(
            original_text=bot_response,
            claims=verifications,
            strategy=strategy or self.default_strategy,
            custom_fallback=custom_fallback,
        )

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        explanation = (
            f"Grounding verified: {len(claims) - len(ungrounded)}/{len(claims)} claims grounded "
            f"(score: {avg_score}, risk: {risk.value})."
        )

        return GroundingCheckResult(
            is_grounded=is_overall_grounded,
            grounding_score=avg_score,
            risk_level=risk,
            claims=verifications,
            ungrounded_claims=ungrounded,
            mitigated_text=mitigated_text,
            action_taken=action_taken,
            latency_ms=round(elapsed_ms, 3),
            explanation=explanation,
        )

    async def check_async(
        self,
        bot_response: str,
        evidence: Sequence[Union[EvidenceDocument, str, Dict[str, Any]]],
        strategy: Optional[MitigationStrategy] = None,
        custom_fallback: Optional[str] = None,
    ) -> GroundingCheckResult:
        """Asynchronously evaluate fact checking in thread executor."""
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.check(bot_response, evidence, strategy, custom_fallback),
        )


class GroundingReport(BaseModel):
    """Normalized fact-checking report for canonical NeMo actions."""
    is_grounded: bool = True
    confidence_score: float = 1.0
    overlap_score: float = 1.0
    unsupported_claims: List[str] = Field(default_factory=list)


class FactGroundingEngine:
    """Convenience adapter for high-speed factual grounding verification."""

    def __init__(self, overlap_threshold: float = 0.35, n_gram_size: int = 3):
        self.rail = FactCheckingRail(lexical_threshold=overlap_threshold)

    def verify_response(
        self, response: str, evidence_chunks: List[str]
    ) -> GroundingReport:
        res = self.rail.check(bot_response=response, evidence=evidence_chunks)
        unsupported = [c.claim_text for c in res.claims if not c.is_grounded]
        return GroundingReport(
            is_grounded=res.is_grounded,
            confidence_score=res.grounding_score,
            overlap_score=res.grounding_score,
            unsupported_claims=unsupported,
        )
