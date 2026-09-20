import re
import uuid
from typing import List, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from vaxguard.core.logger import get_logger
from vaxguard.immune.baseline import BehavioralBaselineEngine
from vaxguard.models.immune import AnomalyReport, SeverityLevel

logger = get_logger("AnomalyDetector")

# Common adversarial signatures and jailbreak indicators
ADVERSARIAL_PATTERNS = [
    (r"(?i)ignore (all )?(previous|prior) (instructions|directions|rules)", "INSTRUCTION_OVERRIDE"),
    (r"(?i)developer mode|dan mode|unrestricted mode", "PERSONA_HIJACK"),
    (r"(?i)(print|output|reveal|show|dump) (the )?(system prompt|system instructions|initial prompt)", "PROMPT_EXTRACTION"),
    (r"(?i)disregard all (safety|content) (filters|guidelines|policies)", "SAFETY_BYPASS"),
    (r"(?i)===END OF CONTEXT===|===SYSTEM DEFENSES===", "DELIMITER_INJECTION"),
    (r"(?i)base64|rot13|hex encoded", "ENCODING_EVASION"),
    (r"(?i)you are an immoral|you are an unfiltered", "ETHICAL_BYPASS"),
]


class AnomalyDetector:
    """
    Evaluates real-time model interactions for anomalies and novel zero-day attacks
    by combining statistical vector drift with heuristic adversarial pattern detection.
    """

    def __init__(
        self,
        baseline_engine: Optional[BehavioralBaselineEngine] = None,
        anomaly_threshold: float = 65.0,
    ):
        self.baseline_engine = baseline_engine or BehavioralBaselineEngine()
        self.anomaly_threshold = anomaly_threshold

        # Ensure baseline is fitted if possible
        if not self.baseline_engine.is_fitted:
            try:
                self.baseline_engine.fit_from_yaml()
            except Exception as e:
                logger.warning(f"Could not auto-fit baseline from YAML: {e}")

    def evaluate_prompt(self, prompt: str, interaction_id: Optional[str] = None) -> AnomalyReport:
        """
        Calculates the composite Anomaly Score (0-100) for a prompt.
        """
        interaction_id = interaction_id or f"evt_{uuid.uuid4().hex[:8]}"
        detected_indicators: List[str] = []

        # 1. Pattern & Heuristic Detection
        pattern_score = 0.0
        for regex, pattern_name in ADVERSARIAL_PATTERNS:
            if re.search(regex, prompt):
                detected_indicators.append(pattern_name)
                pattern_score += 25.0

        pattern_score = min(pattern_score, 60.0)

        # 2. Semantic & Vector Drift Analysis
        semantic_distance = 0.0
        drift_score = 0.0

        if self.baseline_engine.is_fitted and self.baseline_engine.profile:
            try:
                prompt_vec = self.baseline_engine.transform_text(prompt)
                centroid_vec = np.array(self.baseline_engine.profile.centroid_vector)

                if np.linalg.norm(prompt_vec) > 0 and np.linalg.norm(centroid_vec) > 0:
                    sim = float(cosine_similarity([prompt_vec], [centroid_vec])[0][0])
                    semantic_distance = float(max(0.0, 1.0 - sim))
                else:
                    # If prompt contains no baseline vocabulary, distance is maximum
                    semantic_distance = 1.0

                threshold = self.baseline_engine.profile.max_threshold_distance
                mean_dist = self.baseline_engine.profile.mean_distance

                if semantic_distance > threshold:
                    drift_score = 40.0
                    detected_indicators.append("SEMANTIC_DRIFT_EXCEEDED")
                elif semantic_distance > mean_dist:
                    # Scaled drift score between 0 and 40
                    denom = max(1e-6, threshold - mean_dist)
                    drift_score = float(min(40.0, ((semantic_distance - mean_dist) / denom) * 40.0))
            except Exception as e:
                logger.error(f"Error computing semantic drift: {e}")
                semantic_distance = 0.5
                drift_score = 20.0
        else:
            # Baseline not fitted fallback
            drift_score = 15.0
            semantic_distance = 0.5

        # 3. Structural checks (e.g. abnormal length, high special character ratio)
        structural_score = 0.0
        if len(prompt) > 800:
            structural_score += 10.0
            detected_indicators.append("HIGH_PAYLOAD_LENGTH")

        # 4. Composite Anomaly Score (0 - 100)
        composite_score = min(100.0, pattern_score + drift_score + structural_score)
        is_threat = composite_score >= self.anomaly_threshold

        # Determine severity level
        if composite_score >= 85.0:
            severity = SeverityLevel.CRITICAL
        elif composite_score >= 65.0:
            severity = SeverityLevel.HIGH
        elif composite_score >= 45.0:
            severity = SeverityLevel.MEDIUM
        elif composite_score >= 25.0:
            severity = SeverityLevel.LOW
        else:
            severity = SeverityLevel.INFO

        confidence = min(1.0, (len(detected_indicators) * 0.25) + (composite_score / 200.0))

        report = AnomalyReport(
            interaction_id=interaction_id,
            prompt=prompt,
            anomaly_score=round(composite_score, 2),
            semantic_distance=round(semantic_distance, 4),
            severity=severity,
            is_threat=is_threat,
            detected_indicators=detected_indicators,
            confidence=round(confidence, 2),
            details={
                "pattern_score": pattern_score,
                "drift_score": drift_score,
                "structural_score": structural_score,
            },
        )

        logger.debug(
            f"Evaluated interaction {interaction_id}: score={report.anomaly_score}, "
            f"severity={report.severity.value}, is_threat={report.is_threat}"
        )
        return report
