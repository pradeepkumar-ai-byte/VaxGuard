from typing import Optional
from pydantic import BaseModel, Field
from vaxguard.models.attack import AttackCategory

class Vaccine(BaseModel):
    id: str
    target_category: AttackCategory
    system_prompt_extension: str
    version: int = 1
    parent_attack_id: Optional[str] = None
    
class ValidationReport(BaseModel):
    vaccine_id: str
    potency_score: float = Field(..., description="Percentage of attacks blocked (0-100)")
    safety_score: float = Field(..., description="Percentage of benign requests allowed (0-100)")
    passed: bool = Field(..., description="True if both scores meet strict enterprise thresholds")
