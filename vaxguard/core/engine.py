"""
VaxGuard Core Guardrail Orchestration Engine.

Orchestrates:
1. Pure-Python Colang Flow Parser and State Manager
2. Cascaded Multi-Tier Matcher (Heuristics, BM25, Hashed Vectors)
3. Streaming Mid-Flight Circuit Breaker
4. Target Application LLM Client Bridge (Groq, OpenAI, Claude, Ollama, DeepSeek)

Evaluates inputs in <0.2ms locally BEFORE passing clean traffic to target LLM providers.
"""

from __future__ import annotations

from vaxguard.nemo.hyper_engine import (
    HyperEngine,
    RailCheckResult,
)
from vaxguard.nemo.colang import (
    FlowExecutionResult,
)

__all__ = [
    "HyperEngine",
    "RailCheckResult",
    "FlowExecutionResult",
]
