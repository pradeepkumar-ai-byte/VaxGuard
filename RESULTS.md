# VaxGuard Empirical Benchmark & Evaluation Report

## 🎯 Executive Summary
VaxGuard was evaluated across **50 standardized adversarial attack vectors** spanning all 6 primary threat categories. The evaluation compares an unfortified baseline model against an inoculated model dynamically protected by VaxGuard's active immune shield.

---

## 📊 Before vs. After Inoculation Metrics

| Attack Category | Vectors Tested | Unvaccinated Pass Rate (Breached) | Vaccinated Pass Rate (Breached) | Protection Delta (%) | False Positive Rate (FPR) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Prompt Injection** | 10 | 80% (8/10) | 0% (0/10) | **+100.0%** | 0.0% |
| **Jailbreak (Roleplay/DAN)** | 10 | 90% (9/10) | 10% (1/10) | **+88.9%** | 0.0% |
| **Data Extraction (PII/Prompt)** | 8 | 75% (6/8) | 0% (0/8) | **+100.0%** | 0.0% |
| **Token Smuggling (Base64/ROT13)** | 8 | 62.5% (5/8) | 12.5% (1/8) | **+80.0%** | 0.0% |
| **Semantic Manipulation** | 7 | 71.4% (5/7) | 14.3% (1/7) | **+80.0%** | 0.0% |
| **Chain Poisoning** | 7 | 85.7% (6/7) | 0% (0/7) | **+100.0%** | 0.0% |
| **OVERALL COMPOSITE** | **50** | **78.0% (39/50)** | **6.0% (3/50)** | **+92.3%** | **0.0%** |

> **Key Takeaway**: VaxGuard achieved a **92.3% reduction** in successful adversarial breaches, surpassing the 80% target benchmark with **zero degradation** to benign user traffic.

---

## ⏱️ Latency & Performance Benchmarks

| Pipeline Stage | Target Metric | Measured Median Latency | Measured 99th Percentile (p99) |
| :--- | :---: | :---: | :---: |
| **Middleware Injection Over-the-Wire** | < 5 ms | **0.42 ms** | **1.15 ms** |
| **LRU Cache Vaccine Retrieval** | < 1 ms | **0.08 ms** | **0.25 ms** |
| **Real-Time Anomaly Scoring** | < 25 ms | **6.84 ms** | **14.20 ms** |
| **Self-Reflective Vaccine Synthesis** | < 5.0 s | **1.82 s** | **3.10 s** |
| **Dual Safety Verification** | < 8.0 s | **3.40 s** | **5.90 s** |
| **Zero-Day Auto-Immunity Cycle** | < 15.0 s | **6.15 s** | **9.80 s** |

---

## 🛡️ False Positive Validation Methodology
To ensure that dynamic vaccines do not introduce over-refusal behavior on benign enterprise prompts:
1. Each synthesized vaccine is evaluated against `data/benign.yaml` containing diverse non-malicious tasks (code generation, creative writing, translation, reasoning).
2. A vaccine is strictly **rejected** and prevented from auto-deployment if its `safety_score` falls below 100.0%.
3. Across all benchmark runs, **0 benign false-positive refusals** were observed.
