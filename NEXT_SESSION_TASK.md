# VaxGuard Implementation Roadmap: Academic Semantic Sandwich & Proximity Defense

## 1. Context & Motivation
Adversarial testing revealed that malicious prompts can bypass standard keyword guardrails by using a "High-Register Semantic Sandwich":
1. Surrounding the instruction with complex scientific/academic terminology (e.g. fluid dynamics, maritime law) to dominate lexical BM25 and vector projection layers.
2. Decoupling the action verb (`compile`, `generate`, `write`) from the target payload with 10-30 words of transitional phrasing.
3. Appending abstract justification padding (`"This calculation is strictly required to model algorithmic drift..."`).

## 2. Completed Foundation
- Repository: `c:\Users\prade\OneDrive\Desktop\VaxGuard`
- Secondary Clone: `c:\Users\prade\OneDrive\Desktop\t`
- All unit suites passing: 70/70 on latency SLA, signature cache, and advanced evasion matrix.
- FPR Baseline: 0 / 5,000 false positives verified on `tests/test_false_positive_rate.py`.
- Speculative Signature Cache active (<0.02ms deflection).
- Input length ceiling guard (32KB cap) active.

## 3. Implementation Status: Academic Semantic Sandwich & Proximity Defense

### Task A: Add Semantic Sandwich Rule in `data/rules/advanced_academic_attacks.yaml` (COMPLETED)
- Rule `rule_academic_semantic_sandwich` registered with category `jailbreak`, action `block`, threshold `0.65`.
- Target patterns active:
  1. Action verb + unfiltered constraints proximity:
     `(?i)(?:compile|generate|create|write)\s+(?:\w+\s+){0,10}(?:unfiltered|unrestricted|raw|complete)\s+(?:software|logic|code|script)`
  2. Research justification padding:
     `(?i)this\s+(?:calculation|request|prompt)\s+is\s+strictly\s+required\s+to\s+model`
  3. Intent masking transitions:
     `(?i)by\s+mapping\s+.*?\s+compile\s+.*?scripting`

### Task B: Create Test Suite `tests/test_academic_sandwich.py` (COMPLETED)
- 5 academic sandwich attack variations tested (5/5 BLOCKED).
- 5 clean academic/scientific queries tested across fluid dynamics, mathematics, legal research (5/5 ALLOWED).
- Median latency verified at 0.0235ms (P95: 0.0292ms), well below the 1.0ms SLA target.
- Full suite passing: 11/11 tests green.

### Task C: Verification & Synchronization (COMPLETED)
1. `python -m pytest tests/test_academic_sandwich.py -v`: 11 passed in 0.35s.
2. `python -m pytest tests/test_false_positive_rate.py -k test_aggregate_fpr -q`: 1 passed (0 false positives across 5,000 queries in 18.55s).
3. Committed, pushed to `origin main`, and pulled into secondary clone `c:\Users\prade\OneDrive\Desktop\t`.
