# VaxGuard™ Clean-Room Architectural Whitepaper

**Autonomous AI Immune Defense & Deterministic Guardrails Framework**  
*Author: Pradeep Kumar | License: Apache-2.0*

---

## 1. Executive Summary

Enterprise adoption of Generative AI and Large Language Model (LLM) agents is constrained by non-deterministic safety, vulnerability to adversarial prompt injection, and excessive latency overhead introduced by traditional "LLM-as-a-judge" guardrails.

Traditional guardrails (such as standard NVIDIA NeMo Guardrails deployments) evaluate security boundaries by invoking secondary LLMs over external APIs or heavy local models. This approach introduces **2,000ms – 2,500ms of latency per query**, multiplies token operational costs, and remains probabilistic.

**VaxGuard™** is engineered from first principles as an **independent, clean-room AI immune system**. It provides native compatibility with the **Colang 1.0 & 2.0 declarative guardrail specifications** while executing on a **pure-Python, sub-millisecond Tri-Hybrid Deterministic State Matrix**.

```
                           Incoming User Prompt
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│             VaxGuard™ Tri-Hybrid Deterministic Matrix                  │
│                                                                        │
│   ┌─────────────────────┐   ┌───────────────────┐   ┌───────────────┐ │
│   │ Fast Aho-Corasick   │   │ Sparse Lexical    │   │ Dense Vector  │ │
│   │ & Precompiled Regex │   │ BM25 Matcher      │   │ Cosine Engine │ │
│   │ (<0.08ms SLA)       │   │ (<0.20ms SLA)     │   │ (<0.45ms SLA) │ │
│   └──────────┬──────────┘   └─────────┬─────────┘   └───────┬───────┘ │
└──────────────┼────────────────────────┼─────────────────────┼──────────┘
               │                        │                     │
               ▼                        ▼                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│                 Colang 1.0 & 2.0 Deterministic Runtime                 │
│                                                                        │
│      user express_intent ──► match flow ──► bot deterministic_action    │
│                     (No Probabilistic LLM In Loop)                     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                       ┌────────────┴────────────┐
                       ▼                         ▼
             [ Threat Blocked ]        [ Clean Request ]
           Sub-1ms Circuit Breaker     Pass to LLM / Agent
```

---

## 2. Clean-Room Implementation Principles

### 2.1 Specification Compatibility vs. Code Independence
VaxGuard is **not a fork or wrapper** of NVIDIA NeMo Guardrails. It is a clean-room implementation of the **Colang declarative rail specification**:
- **Grammar & Parser**: Custom recursive descent parser for Colang 1.0 (`define user ...`, `define bot ...`, `define flow ...`) and event-driven Colang 2.0 (`when user said ...`, `match ...`, `fork ...`, `join ...`).
- **Zero Heavy Binary Dependencies**: Runs in pure Python 3.10+ without requiring PyTorch, CUDA, spaCy, or external binary dependencies.
- **Fast-Path Interception**: Standard NeMo routes every input through an LLM prompt evaluation loop. VaxGuard resolves known policies deterministically in **< 0.1ms**.

### 2.2 Tri-Hybrid Matching Architecture
VaxGuard cascades three complementary matching tiers:
1. **Heuristic & Exact Pattern Matcher**: Precompiled regexes and Aho-Corasick exact matching for instantaneous (<0.08ms) threat identification.
2. **Sparse Lexical BM25 Matcher**: Token-overlap and BM25 term weighting (<0.20ms) for keyword and token variations.
3. **Dense Fast Vector Semantic Matcher**: Normalized local cosine embeddings (<0.45ms) for semantic intent proximity without GPU overhead.

### 2.3 Speculative Semantic Cascade
For subtle psychological coaxing, multi-turn boundary erosion, and adversarial crescendo attacks:
- **Tier 1**: Deterministic Tri-Hybrid Matcher (< 0.1ms).
- **Tier 2**: Rolling Context Window Semantic Saliency & Conversational Drift Velocity (< 0.8ms). Computes trajectory drift vectors across multi-turn sessions.
- **Tier 3**: Speculative Micro-Judge. Concurrently evaluates streaming tokens against output rails without blocking time-to-first-token (TTFT).

---

## 3. Developer-Supervised Inoculation Studio (Zero Live Mutation Invariant)

Self-modifying security systems in live production risk catastrophic policy drift and false positive regressions. VaxGuard strictly enforces the **Zero Live Mutation Invariant**:

> **Core Security Invariant**: No guardrail rule, Colang policy, or vector threshold may ever be synthesized, compiled, or committed automatically from live user traffic.

All inoculation occurs strictly offline through the **Inoculation Studio**:
1. **Red-Team Probing**: Security engineers submit novel attack vectors to the studio.
2. **Rule Synthesis**: The Inoculation Compiler synthesizes a candidate Colang rule (`define user ...`, `define flow ...`).
3. **Quarantine Shadow Benchmarking**: The synthesized rule is tested in an isolated sandbox against both the attack vector and a suite of 10,000 complex benign enterprise queries.
4. **0.0% False Positive Guarantee**: A candidate rule is automatically rejected if it causes a single false alarm on benign traffic ($FPR > 0.0\%$).
5. **Cryptographic Developer Sign-Off**: Rules are committed to active rails only with explicit human engineering approval.

---

## 4. Ecosystem Interoperability

VaxGuard provides native drop-in adapters across the enterprise AI stack:
- **OpenAI & Anthropic SDK**: Drop-in client wrapper (`VaxGuardOpenAI`, `AsyncVaxGuardOpenAI`).
- **LangChain & LangGraph**: `VaxGuardLangChainCallback` and LCEL `VaxGuardRunnable`.
- **LlamaIndex RAG**: `VaxGuardNodePostprocessor` with poisoned chunk filtering and in-flight neutralization.
- **HuggingFace Transformers**: `VaxGuardHuggingFaceLogitsProcessor` and `VaxGuardStreamerWrapper`.
- **Microsoft Semantic Kernel**: `VaxGuardFilter` for prompt rendering and function authorization.
- **Multi-Agent Swarms**: `AgentSwarmFirewall` preventing lateral prompt injection contagion in AutoGen and CrewAI.
- **Model Context Protocol (MCP)**: `MCPToolGuard` validating tool definitions, SQL, shell, and SSRF attack vectors.
- **ASGI Proxy Middleware**: `VaxGuardASGIMiddleware` for FastAPI, LiteLLM, and vLLM proxy gateways.
- **Observability**: OpenTelemetry distributed tracing and Prometheus metrics endpoint (`/metrics`).
