"""
VaxGuard Integrations: Zero-Dependency Enterprise AI Defense Adapters.

Ecosystem Adapters:
- LangChain / LangGraph: `VaxGuardLangChainCallback`, `VaxGuardRunnable`, `VaxGuardRunnableSequence`
- LlamaIndex RAG: `VaxGuardNodePostprocessor`, `VaxGuardQueryTransform`
- Multi-Agent Swarms: `AgentSwarmFirewall`, `SwarmMessage`, `SwarmVerdict`, `ToolExecutionRequest`
- Model Context Protocol (MCP): `MCPToolGuard`, `MCPToolDefinition`, `MCPValidationResult`, `MCPViolationType`
- ASGI / FastAPI / LiteLLM / vLLM: `VaxGuardASGIMiddleware`, `parse_sse_frame`, `build_sse_alert_frame`
- OpenAI Python SDK: `VaxGuardOpenAI`, `AsyncVaxGuardOpenAI`, `VaxGuardStream`, `VaxGuardAsyncStream`
"""

from vaxguard.integrations.asgi_middleware import (
    VaxGuardASGIMiddleware,
    build_sse_alert_frame,
    parse_sse_frame,
    parse_sse_frames,
)
from vaxguard.integrations.nemo_compat import (
    LLMRails,
    RailsConfig,
)
from vaxguard.integrations.langchain_guard import (
    VaxGuardLangChainCallback,
    VaxGuardRunnable,
    VaxGuardRunnableSequence,
    VaxGuardSecurityViolation,
)
from vaxguard.integrations.llamaindex_guard import (
    VaxGuardNodePostprocessor,
    VaxGuardQueryTransform,
)
from vaxguard.integrations.mcp_guard import (
    MCPGuardException,
    MCPToolDefinition,
    MCPToolGuard,
    MCPValidationResult,
    MCPViolationType,
)
from vaxguard.integrations.multiagent_guard import (
    AgentProfile,
    AgentRolePolicy,
    AgentSwarmFirewall,
    SwarmMessage,
    SwarmSecurityException,
    SwarmVerdict,
    SwarmViolationType,
    ToolExecutionRequest,
)
from vaxguard.integrations.openai_guard import (
    AsyncVaxGuardOpenAI,
    VaxGuardAsyncOpenAI,
    VaxGuardAsyncStream,
    VaxGuardOpenAI,
    VaxGuardSecurityException,
    VaxGuardStream,
)

from vaxguard.integrations.huggingface_guard import (
    VaxGuardHuggingFaceLogitsProcessor,
    VaxGuardHuggingFaceStoppingCriteria,
    VaxGuardStreamerWrapper,
)
from vaxguard.integrations.anthropic_guard import (
    AnthropicSecurityException,
    AsyncGuardedAnthropicStream,
    AsyncVaxGuardAnthropic,
    GuardedAnthropicStream,
    VaxGuardAnthropic,
    wrap_anthropic_client,
)
from vaxguard.integrations.semantic_kernel_guard import (
    VaxGuardFilter,
)

__all__ = [
    # NeMo Compatibility Bridge
    "LLMRails",
    "RailsConfig",
    # LangChain
    "VaxGuardLangChainCallback",
    "VaxGuardRunnable",
    "VaxGuardRunnableSequence",
    "VaxGuardSecurityViolation",
    # Hugging Face
    "VaxGuardHuggingFaceLogitsProcessor",
    "VaxGuardHuggingFaceStoppingCriteria",
    "VaxGuardStreamerWrapper",
    # Semantic Kernel
    "VaxGuardFilter",
    # LlamaIndex
    "VaxGuardNodePostprocessor",
    "VaxGuardQueryTransform",
    # Multi-Agent Swarm
    "AgentSwarmFirewall",
    "AgentProfile",
    "AgentRolePolicy",
    "SwarmMessage",
    "SwarmVerdict",
    "SwarmViolationType",
    "ToolExecutionRequest",
    "SwarmSecurityException",
    # MCP
    "MCPToolGuard",
    "MCPToolDefinition",
    "MCPValidationResult",
    "MCPViolationType",
    "MCPGuardException",
    # ASGI
    "VaxGuardASGIMiddleware",
    "parse_sse_frame",
    "parse_sse_frames",
    "build_sse_alert_frame",
    # OpenAI
    "VaxGuardOpenAI",
    "AsyncVaxGuardOpenAI",
    "VaxGuardAsyncOpenAI",
    "VaxGuardStream",
    "VaxGuardAsyncStream",
    "VaxGuardSecurityException",
    # Anthropic
    "VaxGuardAnthropic",
    "AsyncVaxGuardAnthropic",
    "GuardedAnthropicStream",
    "AsyncGuardedAnthropicStream",
    "AnthropicSecurityException",
    "wrap_anthropic_client",
]

