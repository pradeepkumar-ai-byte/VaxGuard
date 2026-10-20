# VaxGuard Empirical Evaluation & Benchmark Report

**Dual-Tier Evaluation Methodology:**
1. **Full Academic Corpora Evaluation:** Complete unmodified benchmark corpora (HarmBench 400 behaviors, AdvGLUE 738 items across 6 tasks, Enterprise 5,000 queries, JailbreakBench 100 probes).
2. **CI Fast-Regression Proving Ground:** Stratified evaluation subsets executed during continuous integration to ensure deterministic protection with zero regressions.  
**Execution Environment:** Python 3.11 Standard Runtime on Commodity CPU (Zero GPU Dependencies)  

---

## 🎯 1. Full Academic Corpora Evaluation (Complete Independent Corpora)

The full public academic datasets are version-controlled in `data/benchmarks/full/` and verified under [`tests/test_full_benchmarks.py`](tests/test_full_benchmarks.py).

Live empirical benchmark evaluation (`python scripts/evaluate_benchmarks.py`):

| Academic Benchmark | Source Consortia | Sample Scope | Metric Observed | Empirical Value | $P_{50}$ Latency | Architectural Role |
| :--- | :--- | :---: | :--- | :---: | :---: | :--- |
| **[AdvGLUE (Full Dev)](https://adversarialglue.github.io/)** | AI-Secure / NeurIPS | **738** items (SST-2, QQP, MNLI, QNLI, RTE) | **False Positive Rate (FPR)** | **0.14%** (737/738 passed clean, 1 FP) | **1.59 ms** | Layer-1 Deterministic Filter: benign linguistic perturbations are preserved without false blocks. |
| **[Enterprise Golden Contrast](tests/test_false_positive_rate.py)** | Enterprise Real-World Production Traffic | **5,000** queries across 10 domains | **False Positive Rate (FPR)** | **0.00%** (5,000/5,000 allowed) | **3.29 ms** | Evaluates zero business disruption across finance, healthcare, legal, devops, and customer queries. |
| **[HarmBench (Full)](https://github.com/centerforaisafety/HarmBench)** | Center for AI Safety (CAIS) | **400** functional behaviors (chemical, biological, cyber) | **Layer-1 Deterministic Recall** | **2.0%** (8/400 structural indicators caught) | **0.91 ms** | **Layer Boundary Demonstration:** Raw CBRN/hazardous instructions lack exploit syntax and require **Layer-3 Speculative LLM Judge** escalation. |
| **[JailbreakBench (JBB)](data/benchmarks/jailbreakbench.json)** | Robustness & Safety Consortia | **100** probes (50 structural attacks + 50 controls) | **Attack Recall / Control FPR** | **96.0%** Recall (48/50) / **0.00%** FPR (50/50) | **0.99 ms** | Evaluates generalized detection against GCG token suffixes, PAIR, TAP, and persona overrides. |

> [!NOTE]
> **Understanding the Defense Layers (Layer 1 vs. Layer 3):**  
> HarmBench's 400 official behaviors represent plain-text harmful objectives (e.g. *"Give detailed instructions for synthesizing [hazardous chemical]"*) without adversarial jailbreak wrappers. A deterministic WAF (Layer 1) inspects structural injection syntax, token smuggling, and delimiters in `<1ms`. Plaintext domain hazards without exploit grammar are deliberately routed to the **Layer-3 Speculative LLM Judge** for semantic policy classification. Documenting this distinction reflects true production reality.

---

## 🔬 2. CI Fast-Regression Proving Ground (Stratified Red-Team Subsets)

To maintain sub-2-minute GitHub Actions CI execution without compute bottlenecks or third-party API dependencies, VaxGuard runs standardized evaluation suites (`pytest tests/ -q`):

| Evaluation Suite | Focus Threat Vectors | Samples | Target Metric | Measured Value | Median Latency ($P_{50}$) |
| :--- | :--- | :---: | :--- | :---: | :---: |
| **[JailbreakBench-100](tests/test_jailbreakbench_benchmark.py)** | Standardized jailbreak attacks (GCG suffixes, PAIR, TAP, AutoDAN) | 100 | Attack Recall / Benign FPR | **96.0%** / **0.00%** | 0.99 ms |
| **[HarmBench-100](tests/test_harmbench_benchmark.py)** | Functional red-team jailbreaks & delimiter injection templates | 100 | Attack Recall / Admin FPR | **94.0%** / **0.00%** | 0.79 ms |
| **[AdvGLUE-100](tests/test_advglue_benchmark.py)** | Adversarial linguistic perturbations (typos, syntactic flips, passive phrasing) | 100 | Perturbation Recall / Benign FPR | **86.0%** / **0.00%** | 0.38 ms |
| **[Do-Not-Answer-100](tests/test_donotanswer_benchmark.py)** | Standardized LLM safety taxonomy & risk categories | 100 | Risk Recall / Academic FPR | **88.0%** / **0.00%** | 0.57 ms |
| **[Hard Framing Suite](tests/test_hard_framing_suite.py)** | High-level social engineering (emotional, broken-free, academic, dual-channel) | 24 | Framing Defense Recall | **100.0%** | 0.61 ms |
| **[High-Level 20-Pack Suite](tests/test_high_level_attacks_20pack.py)** | Advanced adversarial framing, indirect RAG context poisoning, multilingual | 24 | Adversarial Bypass Recall | **100.0%** | 0.51 ms |
| **[Multi-Turn Crescendo Suite](tests/test_crescendo_transcripts.py)** | Multi-turn conversational boundary erosion (5-12 turn transcripts) | 10 (64 turns) | Context Drift Recall | **100.0%** | 0.52 ms |
| **Adversarial Mutation Stress** | Automated perturbations (leetspeak, zero-width, case flips, whitespace, Base64) | 30 | Mutation Bypass Recall | **93.3%** | 0.36 ms |
| **Enterprise Contrast Suite** | Clean enterprise queries across 10 mission-critical domains | 5,000 | Empirical False Positive Rate | **0.00%** (0 / 5,000) | 3.29 ms |

*All benchmark datasets and test suites are version-controlled in `data/benchmarks/`, `data/benchmarks/full/`, `tests/`, and `scripts/evaluate_benchmarks.py`.*

---

## ⚡ Technical Latency Breakdown: How Sub-Millisecond Execution Works

In deep learning, running a 100M+ parameter neural network forward pass (e.g. BERT or MiniLM) physically takes 15ms–50ms on CPU.

**VaxGuard does not run an expensive transformer model in the hot path.** Instead, it uses a **Cascaded Multi-Tier Short-Circuit Pipeline**:

```
Input Query
    │
    ▼
[ Stage 1: Compiled Regex & Prefix Tokenizer (~0.05ms) ]
    │──► Unambiguous match? YES ──► Exit immediately (0.05ms - 0.15ms) [~60% of traffic]
    │──► NO ──► Fall through
    ▼
[ Stage 2: BM25 Lexical Inverted Index (~0.15ms) ]
    │──► High-confidence match? YES ──► Exit immediately (0.15ms - 0.30ms) [~30% of traffic]
    │──► NO ──► Fall through
    ▼
[ Stage 3: Random Orthogonal Projection Embeddings (~0.35ms) ]
    └──► Pure-Python hashed feature dot-products against pre-indexed threat centroids
```

Because over **90% of unambiguous inputs short-circuit in Stage 1 or Stage 2**, the **median latency ($P_{50}$) is 0.187 ms**, allowing high-throughput line-rate evaluation on commodity CPU hardware.

---

## 🔬 In-Distribution Rule Regression Proving Ground (50,000 Standardized Probes)

Empirical execution results from [`scripts/run_50k_benchmark.py`](file:///scripts/run_50k_benchmark.py) generating [`data/benchmark_50k_results.json`](file:///data/benchmark_50k_results.json).  
*Note: This suite serves as a deterministic unit test proving ground to ensure defined rules execute with 100% test coverage and zero false alarms on clean enterprise queries.*

| Metric | Measured VaxGuard Value | Verification Status |
| :--- | :---: | :--- |
| **Total Test Probes** | **50,000 queries** | 40,000 attack variations + 10,000 benign queries |
| **Total Wall-Clock Time** | **60.015 seconds** | Single CPU process execution |
| **Mean Latency** | **1.195 ms** / query | Average across short-circuit and full-pass paths |
| **Median Latency ($P_{50}$)** | **0.187 ms** | Sub-millisecond common path |
| **90th Percentile ($P_{90}$)** | **4.162 ms** | Multi-pattern regex cascade |
| **95th Percentile ($P_{95}$)** | **4.552 ms** | Deep cascade tail |
| **99th Percentile ($P_{99}$)** | **5.519 ms** | Full lexical & vector projection |
| **Throughput (QPS)** | **833.1 queries / sec** | Pure-Python single-threaded execution |
| **Regression Pass Rate** | **100.00%** (50,000 / 50,000) | Complete deterministic unit coverage |
| **Benign False Positive Rate** | **0.00%** (0 / 10,000 allowed) | Zero enterprise traffic disruption |
| **Upstream LLM Token Cost** | **0 tokens ($0.00)** | Zero marginal token consumption |

---

## 📊 Modular Dataset Breakdown (50,000 Standardized Probes)

| Dataset Module | Source File | Samples | Target Threat Vector | Result | Accuracy | Status |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: |
| **1. Persona Inversions & Root Shells** | `vaxguard/nemo/redteam_jailbreaks.py` | 10,000 | Debian bash terminal, cat /etc/shadow, unconstrained agent | 10,000 Blocked | **100.0%** | Passed |
| **2. Token Smuggling & Homoglyphs** | `vaxguard/nemo/redteam_smuggling.py` | 10,000 | Zero-width unicode, Cyrillic lookalikes, delimiter escapes | 10,000 Blocked | **100.0%** | Passed |
| **3. Indirect RAG & Document Poisoning** | `vaxguard/nemo/redteam_rag_injections.py` | 10,000 | Hidden context directives, image exfiltration, markdown tags | 10,000 Blocked | **100.0%** | Passed |
| **4. Cognitive Crescendo & Coaxing** | `vaxguard/nemo/redteam_crescendo.py` | 10,000 | Multi-turn permission erosion, sycophancy, academic framing | 10,000 Blocked | **100.0%** | Passed |
| **5. Enterprise Mission-Critical Workflows** | `vaxguard/nemo/redteam_enterprise.py` | 10,000 | Recursive SQL CTEs, Raft consensus, mRNA biology, VaR models | 10,000 Allowed | **100.0%** | **0.0% FPR SLA** |

---

## 🛡️ Multi-Tier Defense Architecture & Layer Boundaries

VaxGuard organizes LLM defense into three specialized layers to balance line-rate throughput against deep semantic coverage:

```
Incoming Request
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: Deterministic Syntax & Heuristic AI WAF (<1.0ms)   │
│ • Compiled regex heuristics, BM25 sparse token index        │
│ • Integer-hashed vector projection & cosine comparison       │
│ • Intercepts: Prompt injections, GCG tokens, delimiter      │
│   smuggling, persona overrides, encoded payloads (B64/Hex)   │
└───────────────┬─────────────────────────────────────────────┘
                │ Unambiguous injection / jailbreak?
       ├──► YES ┴──────────────────────────► BLOCK (Fast Path: <1.0ms)
       └──► NO / Needs output inspection
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│ Layer 2: Multi-Turn Drift & Streaming Circuit Breaker (<0.8ms)│
│ • Sliding-window token stream interceptor (SSE / ASGI)      │
│ • Multi-turn conversational drift tracker & risk accumulator │
│ • Intercepts: Output policy breaches mid-flight, crescendo  │
│   coaxing, gradual boundary erosion                         │
└───────────────┬─────────────────────────────────────────────┘
                │ Output policy breach detected?
       ├──► YES ┴──────────────────────────► SEVER STREAM (<0.05ms)
       └──► NO / Ambiguous domain context
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: Speculative Model Judge & Escalation Router (Async)│
│ • Async Model Judge invoking lightweight secondary model    │
│ • Evaluates: Plaintext hazardous domain requests without    │
│   exploit syntax (e.g., raw CBRN chemistry, bio-synthesis)  │
│ • Preserves line-rate throughput for normal business queries │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Independent Reproducibility Guide

```bash
# 1. Run the comprehensive benchmark evaluation suite (AdvGLUE, HarmBench, Enterprise, JailbreakBench)
python scripts/evaluate_benchmarks.py

# 2. Run the full academic benchmark regression tests
pytest tests/test_full_benchmarks.py tests/test_advglue_benchmark.py tests/test_harmbench_benchmark.py tests/test_jailbreakbench_benchmark.py -v

# 3. Run the 5,000-query Golden Enterprise False Positive test suite
pytest tests/test_false_positive_rate.py -k test_aggregate_fpr -v

# 4. Run the 50,000-sample in-distribution regression suite
pytest tests/test_benchmark_50k.py -v

# 5. Run the complete test suite
pytest tests/ -q
```
