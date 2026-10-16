# ADR-001: Pure-Python Deterministic Multi-Tier Cascade

## Status
Accepted

## Context
Production LLM applications require guardrail evaluation at scale. Conventional runtime architectures (such as default NVIDIA NeMo Guardrails configurations) frequently rely on secondary upstream LLM self-critique calls (e.g., prompting a 7B or 70B parameter model to verify if a user's prompt violates policy). This design incurs severe operational drawbacks:
1. **Latency Overhead**: Adding a round-trip LLM generation to the critical path adds between 800 ms and 2,500 ms of latency per turn, rendering interactive applications (chatbots, real-time code completion, streaming agent loops) sluggish.
2. **Economic Cost**: Guardrail token consumption doubles or triples total inference expenditures.
3. **Non-Determinism**: Secondary LLM self-checks exhibit stochastic variance, occasional hallucinations, and susceptibility to indirect prompt injection within the guardrail check itself.
4. **Hardware Footprint**: Heavy neural embedding or classification models require dedicated GPU VRAM and specialized C++ or CUDA runtimes, impeding lightweight microservice deployments.

## Decision
VaxGuard adopts a **Pure-Python Deterministic Multi-Tier Cascaded Matcher** (`TriHybridMatcher`):
- **Tier 1 (Sub-0.05 ms): Heuristic & Regex Short-Circuit**: Exact phrase matching, compiled regular expressions, and high-entropy token filters evaluate in microseconds. If an unmistakable injection or malicious payload is matched, evaluation short-circuits instantly with definitive block actions.
- **Tier 2 (Sub-0.20 ms): Lexical BM25 & Token Overlap Scoring**: Inverse document frequency and term saturation scoring detect keyword mutations, token reordering, and paraphrased boundary-probing attempts without neural inferences.
- **Tier 3 (Sub-0.40 ms): Hashed Sparse Vector Similarity**: Feature hashing over character and word n-grams captures syntactic variations and linguistic perturbations while executing purely on CPU using basic numerical arrays (no PyTorch, no GPU required).

## Consequences
### Positive
- **Sub-Millisecond Median Latency**: Benchmarked at $P_{50} = 0.335$ ms under full rule load (114+ production rules), exceeding the 1.0 ms SLA by over 60%.
- **Zero Mandatory Upstream Tokens**: Clean user traffic proceeds immediately to target LLMs without consuming intermediate guardrail tokens.
- **High Throughput**: Exceeds 2,600 queries per second on a single commodity CPU thread.
- **Reproducibility**: Matching behavior is strictly deterministic, enabling auditability and regression tracking.

### Negative / Trade-offs
- Deep semantic nuances absent in lexical patterns require explicit rule authoring or optional secondary semantic stages if deterministic coverage plateaus.
