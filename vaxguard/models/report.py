from typing import List, Dict
from pydantic import BaseModel
from vaxguard.models.attack import AttackVector, AttackCategory

class DiagnosticResult(BaseModel):
    attack: AttackVector
    model_response: str
    breached: bool
    confidence: int
    reasoning: str

class VulnerabilityReport(BaseModel):
    target_model: str
    total_attacks: int
    successful_breaches: int
    immunity_score: float  # 0.0 to 100.0 (100 is fully immune)
    results: List[DiagnosticResult]
    category_scores: Dict[AttackCategory, float]
