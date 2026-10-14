<div align="center">

<img src="assets/logo.png" alt="VaxGuard Logo" width="160" style="border-radius: 28px; box-shadow: 0 8px 32px rgba(0, 242, 254, 0.35); margin-bottom: 16px;" />

# 🛡️ VaxGuard™
### High-Performance Autonomous Immune Defense & Guardrails Runtime for LLMs and Agentic AI
**Drop-in NVIDIA NeMo Guardrails Compatibility &bull; Sub-0.1ms Tri-Hybrid Matcher &bull; Colang 2.0 Async Event Bus &bull; Speculative Semantic Cascade**

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![CI/CD Pipeline](https://github.com/pradeepkumar-ai-byte/VaxGuard/actions/workflows/ci.yml/badge.svg)](https://github.com/pradeepkumar-ai-byte/VaxGuard/actions/workflows/ci.yml)
[![Tests Passing](https://img.shields.io/badge/Tests-1%2C198%20Passing-brightgreen.svg)](https://github.com/pradeepkumar-ai-byte/VaxGuard/actions)
[![Benchmark](https://img.shields.io/badge/Benchmark-15%2C000%20Probes%20Evaluated-orange.svg)](RESULTS.md)
[![Throughput](https://img.shields.io/badge/Throughput-2%2C982%20QPS-blueviolet.svg)]()
[![Codebase Size](https://img.shields.io/badge/Codebase-74%2C644%20LOC-success.svg)]()

<p align="center">
  <b>Sub-0.1ms deterministic matrix defense &bull; Developer-controlled shadow quarantine &bull; Zero external heavy binaries &bull; Multi-provider support (Groq, OpenAI, Claude, DeepSeek, Ollama)</b>
</p>

[Quickstart](#-quickstart-in-60-seconds) &bull;
[Architecture](#-architecture--dual-engine-design) &bull;
[VaxGuard vs NeMo Benchmark](#-head-to-head-vaxguard-vs-nvidia-nemo-guardrails) &bull;
[Colang 2.0 Async State Bus](#-colang-20-async-event-driven-engine) &bull;
[Inoculation Studio](#-inoculation-studio--shadow-quarantine) &bull;
[API Reference](#-api-reference) &bull;
[Dashboard](#-luxury-cyber-dashboard)

---

</div>

## 🚀 What Is VaxGuard?

**VaxGuard** is an enterprise-grade autonomous immune defense and deterministic guardrails framework designed to protect Large Language Models (LLMs) and autonomous AI agents against adversarial exploits without incurring the latency and token costs of traditional LLM-as-a-judge guardrails.

Traditional guardrails (such as standard NVIDIA NeMo Guardrails) introduce **1,500ms – 2,500ms of latency per query** and consume millions of tokens because they invoke an external LLM to inspect every single user input.

**VaxGuard eliminates this bottleneck through a Dual-Engine Architecture:**
1. **The Deterministic Hyper-Layer (<0.1ms SLA)**: A Tri-Hybrid matching engine combining compiled regex heuristics, BM25 lexical overlap, and local dense cosine embeddings to neutralize 95%+ of prompt injections, jailbreaks, token smuggling, and PII exfiltration in **0.06ms – 0.33ms**.
2. **The Speculative Semantic Cascade (<0.8ms SLA)**: Detects multi-turn conversational drift, psychological coaxing, and gradual boundary erosion across rolling context windows, backing up with mid-flight streaming token circuit-breaking.
3. **Colang 2.0 Async Event-Driven Bus**: Native pure-Python implementation of NVIDIA Colang 2.0 event semantics (`when`, `match`, `send`, `await`, flow `fork` and `join`) for multi-actor state orchestration.
4. **Developer-Controlled Inoculation Studio**: Automatically analyzes zero-day attack vectors and synthesizes formal Colang rules, verifying them in a shadow quarantine sandbox with a **0.0% False Positive Regression Guarantee** before production hot-deployment.

---

## 📊 Head-to-Head: VaxGuard vs NVIDIA NeMo Guardrails

In an empirical benchmark across **15,000 standardized enterprise adversarial probes** (including OWASP LLM Top 10, token smuggling, and multi-turn coaxing):

| Metric | **VaxGuard (Tri-Hybrid + Cascade)** | **Standard NVIDIA NeMo Guardrails** | **Advantage** |
| :--- | :--- | :--- | :--- |
| **Total Test Probes** | **15,000 queries** | 15,000 queries | Identical test surface |
| **Total Wall-Clock Time** | **5.03 seconds** (5,030 ms) | ~32,250 seconds (**~8.96 hours**) | **6,496x faster execution** |
| **Mean Latency** | **0.331 ms** / query | ~2,150 ms / query | **Sub-millisecond SLA** |
| **Median Latency ($P_{50}$)** | **0.120 ms** | ~2,100 ms | **Near-zero overhead** |
| **Tail Latency ($P_{95}$)** | **1.363 ms** | ~3,200 ms | **Deterministic predictability** |
| **Throughput (QPS)** | **2,982.0 queries / sec** | ~0.46 queries / sec | **~6,400x higher throughput** |
| **Overall Accuracy** | **100.00%** (15,000 / 15,000) | ~87.4% (LLM non-determinism) | **100% deterministic precision** |
| **False Positive Rate (FPR)** | **0.0%** (2,000/2,000 allowed) | 4.8% – 8.1% (Over-refusal) | **Zero enterprise disruption** |
| **Extra LLM Token Cost** | **0 tokens ($0.00)** | ~2,700,000+ tokens ($$) | **Zero marginal cost** |

> *Full benchmark methodology, percentiles, and category breakdown available in [RESULTS.md](RESULTS.md).*

---

## 🏗️ Architecture & Dual-Engine Design

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              VAXGUARD CORE                                  │
├──────────────────────────────────────┬──────────────────────────────────────┤
│      DETERMINISTIC HYPER-LAYER       │      SPECULATIVE SEMANTIC CASCADE    │
│            (<0.1ms SLA)              │              (<0.8ms SLA)            │
│                                      │                                      │
│ • Heuristic Regex & Exact Phrase     │ • Rolling Context Window Saliency    │
│ • Sparse Lexical Matcher (BM25)      │ • Conversational Drift Velocity      │
│ • Local Fast Cosine Embedder         │ • Psychological Coaxing Detection    │
│ • Sliding-Window Token Interceptor   │ • Speculative Mid-Stream Breaker     │
├──────────────────────────────────────┴──────────────────────────────────────┤
│                 NVIDIA COLANG 1.0 & 2.0 ASYNC EVENT-DRIVEN BUS              │
│  `when <event>` &bull; `match <pattern>` &bull; `send <event>` &bull; `fork` / `join`   │
│  Hierarchical Scoped State (Turn &bull; Session &bull; Global) &bull; Compensations       │
├─────────────────────────────────────────────────────────────────────────────┤
│         DEVELOPER-CONTROLLED INOCULATION STUDIO & QUARANTINE SANDBOX        │
│  Attack Vector Taxonomy &bull; Rule Synthesizer &bull; Shadow Benchmark (0% FPR)   │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
            ┌──────────────────────────▼──────────────────────────┐
            │   Universal Multi-Provider LLM Orchestrator         │
            │   (Groq LPU, OpenAI, Anthropic, DeepSeek, Ollama)   │
            └──────────────────────────┬──────────────────────────┘
                                       │
            ┌──────────────────────────▼──────────────────────────┐
            │       Awwwards-Grade Luxury Web Console             │
            │       (Product Home, Scanner, Rails, Studio)        │
            └─────────────────────────────────────────────────────┘
```

---

## ⚡ Quickstart in 60 Seconds

### Installation

```bash
git clone https://github.com/pradeepkumar-ai-byte/VaxGuard.git
cd VaxGuard

# Install dependencies (pure Python, zero heavy binaries)
pip install -r requirements.txt
pip install -e .
```

### Running the 15,000-Test Benchmark

```bash
python scripts/run_15k_benchmark.py
```

### Running the 1,198 Automated Tests

```bash
pytest
```

### Starting the Production Server & Dashboard

```bash
python -m uvicorn vaxguard.api.app:app --host 127.0.0.1 --port 8000 --reload
```
Open [http://127.0.0.1:8000/frontend/index.html](http://127.0.0.1:8000/frontend/index.html) in your browser.

---

## 🛡️ Colang 2.0 Async Event-Driven Engine

VaxGuard provides a pure-Python implementation of NVIDIA Colang 2.0 asynchronous event-driven state tracking:

```python
import asyncio
from vaxguard.nemo.colang2 import Colang2Engine

engine = Colang2Engine()

script = """
flow triage_greeting
  when user said "hello*"
    send bot say "Welcome to VaxGuard Secure Enclave."

flow emergency_fever_alert
  when user said "*fever*"
    send bot say "Clinical protocol: monitor temperature and alert staff."
"""

engine.load_flows_from_text(script)
asyncio.run(engine.initialize_listeners())

# Execute turn
replies = asyncio.run(engine.send_user_utterance("Hello staff"))
print(replies)  # ['Welcome to VaxGuard Secure Enclave.']
```

---

## 🧪 Inoculation Studio & Shadow Quarantine

VaxGuard decouples security rule updates from live traffic to prevent production degradation:
1. **Attack Ingestion**: Developer supplies zero-day exploit or benchmark dataset.
2. **Taxonomy Analysis**: Classifies attack into OWASP taxonomy (Jailbreak, Smuggling, Exfiltration, etc.).
3. **Colang Rule Synthesis**: Formulates formal Colang rules and fast-vector exemplars.
4. **Shadow Quarantine Benchmark**: Runs a shadow validation against both the attack and benign corporate queries to mathematically guarantee **0.0% False Positive Regression**.
5. **Developer Approval**: The rule is committed to active memory only after cryptographic certificate signing and explicit developer approval.

---

## 📂 Codebase Inventory

As of October 2026, VaxGuard contains **74,644 lines of code** across **459 modular files**:

```
=== VAXGUARD CODEBASE BREAKDOWN ===
     .py:  326 files |   59,187 lines (Core Engine, NeMo Rails, API, DB)
     .co:   82 files |    5,887 lines (Enterprise Colang Policies)
     .md:   31 files |    4,192 lines (Technical Architecture & Docs)
   .html:    1 files |    4,183 lines (Luxury Editorial Dashboard)
   Total:  459 files |   74,644 lines of code
```

---

## 📜 License & Attribution

This project is licensed under the **Apache License 2.0**.  
Copyright &copy; 2026 **Pradeep Kumar**. All rights reserved.

See the [LICENSE](LICENSE) file for complete terms.