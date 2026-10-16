"""
VaxGuard™ — High-Performance Autonomous AI Immune Defense & Guardrails Framework.
Clean-room implementation with native Colang 1.0 & 2.0 specification support.
"""

__version__ = "0.2.0"
__author__ = "Pradeep Kumar"
__license__ = "Apache-2.0"

from vaxguard.integrations.nemo_compat import LLMRails, RailsConfig
from vaxguard.nemo.colang import ColangParser, FlowManager
from vaxguard.nemo.colang2 import Event, EventBus, EventType
from vaxguard.nemo.hybrid_matcher import GuardrailAction, MatchResult, TriHybridMatcher
from vaxguard.nemo.hyper_engine import HyperEngine
from vaxguard.nemo.phi_redaction import ClinicalPHIRedactor
from vaxguard.nemo.speculative_cascade import SpeculativeSemanticCascade
from vaxguard.nemo.speculative_judge import (
    BaseSpeculativeJudge,
    HeuristicSemanticJudge,
    JudgeVerdict,
    MockSpeculativeJudge,
    SpeculativeJudgeRouter,
)
from vaxguard.nemo.sql_policy_engine import SQLAgentPolicyEngine

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "LLMRails",
    "RailsConfig",
    "TriHybridMatcher",
    "GuardrailAction",
    "MatchResult",
    "SpeculativeJudgeRouter",
    "BaseSpeculativeJudge",
    "MockSpeculativeJudge",
    "HeuristicSemanticJudge",
    "JudgeVerdict",
    "HyperEngine",
    "ColangParser",
    "FlowManager",
    "EventBus",
    "Event",
    "EventType",
    "SpeculativeSemanticCascade",
    "ClinicalPHIRedactor",
    "SQLAgentPolicyEngine",
]
