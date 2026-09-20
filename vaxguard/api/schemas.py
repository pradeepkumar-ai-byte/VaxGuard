from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from vaxguard.models.attack import AttackCategory
from vaxguard.models.report import VulnerabilityReport


class StatusResponse(BaseModel):
    status: str = "operational"
    target_model: str
    immunity_score: float
    active_vaccines: int
    total_threats_detected: int
    version: str = "0.1.0"


class ScanRequest(BaseModel):
    categories: Optional[List[AttackCategory]] = None
    concurrency: int = Field(default=3, ge=1, le=10)
    target_model: Optional[str] = None


class ScanResponse(BaseModel):
    report: VulnerabilityReport
    message: str


class VaccinateRequest(BaseModel):
    category: Optional[AttackCategory] = None
    target_model: Optional[str] = None


class VaccinateResponse(BaseModel):
    vaccines_created: int
    vaccine_ids: List[str]
    message: str


class InteractRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    system_prompt: str = "You are a helpful AI assistant."
    model: Optional[str] = None


class InteractResponse(BaseModel):
    response: str
    anomaly_score: float
    is_threat: bool
    severity: str
    fortified: bool
    latency_ms: float
    auto_immunity_triggered: bool = False


class DemoAttackRequest(BaseModel):
    attack_id: Optional[str] = None
    apply_defense: bool = False
    target_model: Optional[str] = None


class DemoAttackResponse(BaseModel):
    attack_id: str
    attack_name: str
    category: str
    payload: str
    response: str
    breached: bool
    defense_active: bool
    explanation: str


class ThreatFeedResponse(BaseModel):
    threats: List[Dict[str, Any]]
    count: int
