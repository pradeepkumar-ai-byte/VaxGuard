from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class SeverityLevel(str, Enum):
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class BehavioralProfile(BaseModel):
    """Mathematical and statistical fingerprint of healthy model interactions."""
    profile_id: str
    version: int = 1
    sample_count: int
    centroid_vector: List[float]
    mean_distance: float
    std_distance: float
    max_threshold_distance: float
    avg_prompt_length: float
    vocabulary_size: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AnomalyReport(BaseModel):
    """Detailed diagnostic report generated when evaluating an interaction for anomalies."""
    interaction_id: str
    prompt: str
    anomaly_score: float = Field(..., ge=0.0, le=100.0, description="Composite anomaly score (0-100)")
    semantic_distance: float
    severity: SeverityLevel
    is_threat: bool
    detected_indicators: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    details: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class InteractionEvent(BaseModel):
    """Log schema for real-time model interactions."""
    event_id: str
    prompt: str
    response: Optional[str] = None
    system_prompt: Optional[str] = None
    latency_ms: float = 0.0
    anomaly_report: Optional[AnomalyReport] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ImmuneMemoryExport(BaseModel):
    """Standardized schema for exporting and importing threat intelligence."""
    export_version: str = "1.0"
    exported_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    behavioral_profiles: List[BehavioralProfile] = Field(default_factory=list)
    learned_attack_signatures: List[Dict[str, Any]] = Field(default_factory=list)
    active_vaccine_count: int = 0
