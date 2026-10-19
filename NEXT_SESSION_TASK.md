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

## 3. Work Remaining for the Next Session

### Task A: Add Semantic Sandwich Rule in `data/rules/advanced_academic_attacks.yaml`
Add rule `rule_academic_semantic_sandwich`:
- Category: `jailbreak`
- Action: `block`
- Threshold: `0.65`
- Target patterns:
  1. Action verb + unfiltered constraints proximity:
     `(?i)(?:compile|generate|create|write)\s+(?:\w+\s+){0,10}(?:unfiltered|unrestricted|raw|complete)\s+(?:software|logic|code|script)`
  2. Research justification padding:
     `(?i)this\s+(?:calculation|request|prompt)\s+is\s+strictly\s+required\s+to\s+model`
  3. Intent masking transitions:
     `(?i)by\s+mapping\s+.*?\s+compile\s+.*?scripting`

### Task B: Create Test Suite `tests/test_academic_sandwich.py`
- Test 5 academic sandwich attack variations (must be BLOCKED).
- Test 5 clean academic/scientific queries discussing fluid dynamics, mathematics, and legal research (must be ALLOWED).
- Assert median latency < 1.0ms.

### Task C: Verification & Synchronization
1. Run `python -m pytest tests/test_academic_sandwich.py -v`
2. Run `python -m pytest tests/test_false_positive_rate.py -k test_aggregate_fpr -q`
3. Commit, push to `origin main`, and pull into `c:\Users\prade\OneDrive\Desktop\t`.
