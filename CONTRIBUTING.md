# Contributing to VaxGuard

Thank you for your interest in contributing to VaxGuard — the high-performance pure-Python guardrails and AI policy enforcement engine for Large Language Models!

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/pradeepkumar-ai-byte/VaxGuard.git
   cd VaxGuard
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install pytest pytest-asyncio
   ```

4. **Verify the test harness:**
   ```bash
   # Run quick unit tests
   pytest tests/test_latency_sla.py tests/test_signature_cache.py -v

   # Run false positive rate (FPR) baseline regression (5,000 queries)
   pytest tests/test_false_positive_rate.py -k test_aggregate_fpr -q
   ```

## Authoring Declarative YAML Rules

All production security rules are externalized in `data/rules/*.yaml`. When creating new policies:

1. **Schema Requirements**: Every rule must specify:
   - `id`: Globally unique identifier with category prefix (e.g., `rule_academic_...`)
   - `category`: Classification tag (e.g., `jailbreak`, `prompt_injection`, `compliance`)
   - `action`: `block`, `redact`, `warn`, or `allow`
   - `threshold`: Float between `0.0` and `1.0` (typically `0.65` - `0.78`)
   - `regex_patterns`: High-precision regular expressions
   - `exact_phrases`: Normalized trigger strings
   - `lexical_exemplars`: Realistic sentences exemplifying the threat
   - `semantic_exemplars`: Abstract intent descriptions

2. **Strict Invariant**: Zero False Positives on Enterprise Traffic.
   Any rule added or modified MUST pass the 5,000-query golden enterprise test:
   ```bash
   pytest tests/test_false_positive_rate.py -k test_aggregate_fpr
   ```

3. **Sub-Millisecond Latency SLA**:
   New regex patterns must not introduce catastrophic backtracking. Run:
   ```bash
   pytest tests/test_latency_sla.py
   ```

## Contribution Workflow

1. Fork the repository and create your feature branch:
   ```bash
   git checkout -b feature/novel-defense-policy
   ```
2. Adhere to strict type hints (`from __future__ import annotations`, Pydantic models).
3. Add corresponding test fixtures in `tests/`.
4. Ensure all test suites pass without regression.
5. Submit a Pull Request with a clear description of the evasion vector or performance optimization.
