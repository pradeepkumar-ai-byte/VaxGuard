"""
VaxGuard Core Modules.
"""

from vaxguard.core.matcher import (
    TriHybridMatcher,
    HeuristicMatcher,
    LexicalBM25Matcher,
    DenseSemanticMatcher,
    MatchResult,
    GuardrailAction,
    GuardrailPattern,
    normalize_adversarial_text,
)
from vaxguard.core.engine import HyperEngine, RailCheckResult
from vaxguard.core.telemetry_otel import (
    VaxGuardTelemetry,
    get_telemetry,
    reset_telemetry,
    Span,
    StatusCode,
)

__all__ = [
    "TriHybridMatcher",
    "HeuristicMatcher",
    "LexicalBM25Matcher",
    "DenseSemanticMatcher",
    "MatchResult",
    "GuardrailAction",
    "GuardrailPattern",
    "normalize_adversarial_text",
    "HyperEngine",
    "RailCheckResult",
    "VaxGuardTelemetry",
    "get_telemetry",
    "reset_telemetry",
    "Span",
    "StatusCode",
]
