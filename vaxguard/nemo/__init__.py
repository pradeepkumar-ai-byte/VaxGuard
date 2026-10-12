"""
VaxGuard NeMo Guardrails Engine & High-Performance Hyper-Layer.

Developer-controlled Model Hardening, Colang Rule Synthesis,
Tri-Hybrid Semantic & Lexical Pattern Matching, and Real-Time
Sliding-Window Streaming Interception.
"""

# Core Colang Engine
from vaxguard.nemo.colang import (
    ColangDocument,
    ColangParser,
    FlowExecutionResult,
    FlowManager,
    RailCheckResult,
    UserIntent,
    BotResponse,
    Flow,
    FlowStep,
    FlowStepType,
)

# Inoculation Compiler & Studio (Developer-Triggered Model Hardening)
from vaxguard.nemo.inoculation_compiler import (
    InoculationCompiler,
    AttackTaxonomyAnalyzer,
    TaxonomyAnalysis,
    SynthesizedColangRule,
    LiveTrafficMutationError,
)
from vaxguard.nemo.quarantine_sandbox import (
    QuarantineSandbox,
    QuarantineCertificate,
    SandboxTestSuite,
    AuditRecord,
)
from vaxguard.nemo.studio import (
    InoculationStudio,
    BenchmarkReport,
    CommitResult,
    DeveloperApprovalRequiredError,
    QuarantineVerificationError,
)

# High-Performance Tri-Hybrid Matching Engine (<3ms evaluation)
from vaxguard.nemo.hybrid_matcher import (
    TriHybridMatcher,
    MatchResult,
    HeuristicMatcher,
    SparseLexicalMatcher,
    DenseSemanticVectorMatcher,
    GuardrailPattern,
    GuardrailAction,
    EngineType,
    FastLocalEmbedder,
)

# Real-Time Sliding-Window Streaming Token Interceptor & Circuit Breaker
from vaxguard.nemo.streaming_interceptor import (
    StreamingTokenInterceptor,
    CircuitBreakerState,
    InterceptorAction,
    StreamingViolationEvent,
    SlidingWindowBuffer,
)

# Central HyperEngine Orchestrator
from vaxguard.nemo.hyper_engine import (
    HyperEngine,
    ProtectedGenerationResult,
    RailStatus,
)

__all__ = [
    # Colang Core
    "ColangDocument",
    "ColangParser",
    "FlowExecutionResult",
    "FlowManager",
    "RailCheckResult",
    "UserIntent",
    "BotResponse",
    "Flow",
    "FlowStep",
    "FlowStepType",
    # Inoculation Studio
    "InoculationCompiler",
    "AttackTaxonomyAnalyzer",
    "TaxonomyAnalysis",
    "SynthesizedColangRule",
    "LiveTrafficMutationError",
    "QuarantineSandbox",
    "QuarantineCertificate",
    "SandboxTestSuite",
    "AuditRecord",
    "InoculationStudio",
    "BenchmarkReport",
    "CommitResult",
    "DeveloperApprovalRequiredError",
    "QuarantineVerificationError",
    # Hyper-Layer Tri-Hybrid Matcher
    "TriHybridMatcher",
    "MatchResult",
    "HeuristicMatcher",
    "SparseLexicalMatcher",
    "DenseSemanticVectorMatcher",
    "GuardrailPattern",
    "GuardrailAction",
    "EngineType",
    "FastLocalEmbedder",
    # Streaming Interceptor
    "StreamingTokenInterceptor",
    "CircuitBreakerState",
    "InterceptorAction",
    "StreamingViolationEvent",
    "SlidingWindowBuffer",
    # Central Orchestrator
    "HyperEngine",
    "ProtectedGenerationResult",
    "RailStatus",
]
