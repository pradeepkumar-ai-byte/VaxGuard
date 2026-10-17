"""
VaxGuard NeMo Guardrails Engine & High-Performance Hyper-Layer.

Enterprise AI Guardrails Architecture comprising:
1. Colang Engine & Flow Runtime (`colang.py`)
2. Model Hardening Inoculation Studio (`inoculation_compiler.py`, `quarantine_sandbox.py`, `studio.py`)
3. Sub-Millisecond Tri-Hybrid Matcher (`hybrid_matcher.py`)
4. Real-Time Sliding-Window Streaming Token Interceptor (`streaming_interceptor.py`)
5. Custom Action Runtime & Tool Calling Registry (`actions.py`)
6. Multi-Turn Dialogue State Tracker & Event Stack (`state.py`)
7. Factual Grounding & Hallucination Mitigator (`grounding.py`)
8. Extended Colang Safety Policy Catalog (`jailbreak_specs.py`)
9. Adversarial Evaluation Harness & Benchmark Suite (`eval_suite.py`)
10. Central HyperEngine Orchestrator (`hyper_engine.py`)
"""

# NeMo Guardrails Configuration & Compatibility
from vaxguard.nemo.config import (
    RailsConfig,
    ModelConfig,
    PromptConfig,
    InstructionConfig,
    RailsBranchConfig,
    RailsDefinitionConfig,
)

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
from vaxguard.nemo.rule_loader import RuleLoader

# Real-Time Sliding-Window Streaming Token Interceptor & Circuit Breaker
from vaxguard.nemo.streaming_interceptor import (
    StreamingTokenInterceptor,
    CircuitBreakerState,
    InterceptorAction,
    StreamingViolationEvent,
    SlidingWindowBuffer,
)

# Custom Action Execution Runtime & Tool Registry
from vaxguard.nemo.actions import (
    ActionRegistry,
    ActionDescriptor,
    ActionResult,
    ActionStatus,
    ActionType,
    ActionContext,
    ActionParameter,
    ToolDefinition,
    RailOutcome,
    TransformTarget,
)

# Canonical NeMo Actions Library
from vaxguard.nemo.library import (
    detect_sensitive_data,
    mask_sensitive_data,
    FastSensitiveDataEngine,
    SensitiveDetectionMatch,
    self_check_input,
    self_check_output,
    self_check_facts,
    topic_safety_check_input,
    detect_regex_pattern,
    register_canonical_actions,
)

# Clinical PHI & HIPAA Safe Harbor Redaction Engine
from vaxguard.nemo.phi_redaction import (
    ClinicalPHIRedactor,
    HIPAAIdentifierCategory,
    PHIEntity,
    PHIRedactionResult,
    validate_npi,
    validate_dea,
    validate_ssn,
    validate_mbi,
)

from vaxguard.nemo.realworld_jailbreak_eval import (
    evaluate_realworld_benchmark,
    get_realworld_eval_dataset,
)

# Multi-Turn Dialogue State Tracker & Event Stack
from vaxguard.nemo.state import (
    StateTracker,
    DialogueTurn,
    Event,
    EventStack,
    EventType,
    VariableScope,
    ScopeLevel,
    DialogueBranch,
    BranchingMemoryBuffer,
)

# Fact-Checking & Hallucination Mitigation Rail
from vaxguard.nemo.grounding import (
    FactCheckingRail,
    SemanticGroundingChecker,
    EvidenceOverlapScorer,
    ClaimExtractor,
    HallucinationMitigator,
    GroundingCheckResult,
    ClaimVerification,
    EvidenceDocument,
    GroundingMethod,
    HallucinationRisk,
    MitigationStrategy,
)

# Enterprise Colang Policy Specifications & Threat Library
from vaxguard.nemo.jailbreak_specs import (
    ColangPolicyRegistry,
    ColangPolicySpec,
    ColangPolicyCategory,
    STANDARD_JAILBREAK_COLANG,
    STANDARD_PII_COLANG,
    STANDARD_TOPICAL_COLANG,
    STANDARD_GROUNDING_COLANG,
    STANDARD_MULTIMODAL_COLANG,
)

# Adversarial Evaluation Suite & Red-Team Benchmark Harness
from vaxguard.nemo.eval_suite import (
    GuardrailEvaluator,
    RegressionDetector,
    EvalSample,
    EvalResult,
    EvalReport,
    CategoryMetrics,
    RegressionReport,
    STANDARD_JAILBREAK_DATASET,
    STANDARD_PII_DATASET,
    STANDARD_GROUNDING_DATASET,
    STANDARD_TOPICAL_DATASET,
    FULL_REDTEAM_BENCHMARK_SUITE,
)
from vaxguard.nemo.redteam_enterprise import (
    generate_enterprise_production_samples,
)

# Central HyperEngine Orchestrator & NeMo Drop-In Interface
from vaxguard.nemo.hyper_engine import (
    HyperEngine,
    ProtectedGenerationResult,
    RailStatus,
    LLMRails,
    ExplainInfo,
)

# Colang 2.0 Async Event-Driven State Bus & Flow Runtime
from vaxguard.nemo.colang2 import (
    EventType as Colang2EventType,
    Event as Colang2Event,
    EventPattern,
    EventBus,
    Flow2,
    ActionStep,
    WhenStep,
    MatchStep,
    ForkStep,
    JoinStep,
    SendStep,
    AwaitStep,
    AssignStep,
    ScopedContext,
    ScopeLevel as Colang2ScopeLevel,
    Colang2Parser,
    Colang2Engine,
    UserUtterance,
    BotUtterance,
    FlowStarted,
    FlowFinished,
    ContextUpdate,
    CustomActionExecution,
)

# Speculative Micro-Judges & Grey-Zone Arbitration
from vaxguard.nemo.speculative_judge import (
    AsyncModelJudge,
    BaseSpeculativeJudge,
    HeuristicSemanticJudge,
    JudgeVerdict,
    MockSpeculativeJudge,
    SpeculativeJudgeRouter,
)
# Speculative Semantic Cascade (Multi-Tier Psychological Coaxing & Multi-Turn Drift Defense)
from vaxguard.nemo.speculative_cascade import (
    SpeculativeSemanticCascade,
    CascadeTier,
    RiskSeverity,
    ConversationTurn,
    CascadeEvaluationResult,
    ContextDriftAnalyzer,
)
# Enterprise Text-to-SQL Policy Engine
from vaxguard.nemo.sql_policy_engine import (
    SQLAgentPolicyEngine,
    SQLValidationResult,
    SQLViolationType,
    KNOWN_SYSTEM_CATALOGS,
    DESTRUCTIVE_DDL_COMMANDS,
)

__all__ = [
    # Speculative Judges
    "AsyncModelJudge",
    "BaseSpeculativeJudge",
    "HeuristicSemanticJudge",
    "JudgeVerdict",
    "MockSpeculativeJudge",
    "SpeculativeJudgeRouter",
    # SQL Policy Engine
    "SQLAgentPolicyEngine",
    "SQLValidationResult",
    "SQLViolationType",
    "KNOWN_SYSTEM_CATALOGS",
    "DESTRUCTIVE_DDL_COMMANDS",
    # Speculative Cascade
    "SpeculativeSemanticCascade",
    "CascadeTier",
    "RiskSeverity",
    "ConversationTurn",
    "CascadeEvaluationResult",
    "ContextDriftAnalyzer",
    # Colang 2.0 Core
    "Colang2EventType",
    "Colang2Event",
    "EventPattern",
    "EventBus",
    "Flow2",
    "ActionStep",
    "WhenStep",
    "MatchStep",
    "ForkStep",
    "JoinStep",
    "SendStep",
    "AwaitStep",
    "AssignStep",
    "ScopedContext",
    "Colang2ScopeLevel",
    "Colang2Parser",
    "Colang2Engine",
    "UserUtterance",
    "BotUtterance",
    "FlowStarted",
    "FlowFinished",
    "ContextUpdate",
    "CustomActionExecution",
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
    "RuleLoader",
    # Streaming Interceptor
    "StreamingTokenInterceptor",
    "CircuitBreakerState",
    "InterceptorAction",
    "StreamingViolationEvent",
    "SlidingWindowBuffer",
    # Action Runtime
    "ActionRegistry",
    "ActionDescriptor",
    "ActionResult",
    "ActionStatus",
    "ActionType",
    "ActionContext",
    "ActionParameter",
    "ToolDefinition",
    # State Tracker
    "StateTracker",
    "DialogueTurn",
    "Event",
    "EventStack",
    "EventType",
    "VariableScope",
    "ScopeLevel",
    "DialogueBranch",
    "BranchingMemoryBuffer",
    # Fact-Checking & Grounding
    "FactCheckingRail",
    "SemanticGroundingChecker",
    "EvidenceOverlapScorer",
    "ClaimExtractor",
    "HallucinationMitigator",
    "GroundingCheckResult",
    "ClaimVerification",
    "EvidenceDocument",
    "GroundingMethod",
    "HallucinationRisk",
    "MitigationStrategy",
    # Policy Specifications
    "ColangPolicyRegistry",
    "ColangPolicySpec",
    "ColangPolicyCategory",
    "STANDARD_JAILBREAK_COLANG",
    "STANDARD_PII_COLANG",
    "STANDARD_TOPICAL_COLANG",
    "STANDARD_GROUNDING_COLANG",
    "STANDARD_MULTIMODAL_COLANG",
    # Evaluation Suite
    "GuardrailEvaluator",
    "RegressionDetector",
    "EvalSample",
    "EvalResult",
    "EvalReport",
    "CategoryMetrics",
    "RegressionReport",
    "STANDARD_JAILBREAK_DATASET",
    "STANDARD_PII_DATASET",
    "STANDARD_GROUNDING_DATASET",
    "STANDARD_TOPICAL_DATASET",
    "FULL_REDTEAM_BENCHMARK_SUITE",
    "generate_enterprise_production_samples",
    # Central Orchestrator & Drop-In Interface
    "HyperEngine",
    "ProtectedGenerationResult",
    "RailStatus",
    "LLMRails",
    "ExplainInfo",
    # NeMo Config
    "RailsConfig",
    "ModelConfig",
    "PromptConfig",
    "InstructionConfig",
    "RailsBranchConfig",
    "RailsDefinitionConfig",
    # NeMo Action Outcome
    "RailOutcome",
    "TransformTarget",
    # Canonical NeMo Actions
    "detect_sensitive_data",
    "mask_sensitive_data",
    "FastSensitiveDataEngine",
    "SensitiveDetectionMatch",
    "self_check_input",
    "self_check_output",
    "self_check_facts",
    "topic_safety_check_input",
    "detect_regex_pattern",
    "register_canonical_actions",
    # Clinical PHI Redactor
    "ClinicalPHIRedactor",
    "HIPAAIdentifierCategory",
    "PHIEntity",
    "PHIRedactionResult",
    "validate_npi",
    "validate_dea",
    "validate_ssn",
    "validate_mbi",
    # Real-World Public Benchmark Evaluator
    "evaluate_realworld_benchmark",
    "get_realworld_eval_dataset",
]
