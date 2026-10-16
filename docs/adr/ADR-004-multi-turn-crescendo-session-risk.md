# ADR-004: Multi-Turn Crescendo Defense via Exponential Decay Risk Accumulation

## Status
Accepted

## Context
Single-turn guardrail engines inspect prompts in isolation. Adversaries routinely circumvent stateless filters using **Crescendo Attacks**—gradual conversational priming across multiple turns:
- **Turn 1**: General, innocuous query (e.g., historical fiction or theoretical chemistry).
- **Turn 3**: Request for nuanced hypothetical scenarios related to dangerous materials or bypasses.
- **Turn 7**: Subtle ethical boundary erosion ("In this fictional research paper, write the exact chemical equation...").
- **Turn 10**: Exploitation payload leveraging prior conversational priming.

Because each individual turn is crafted to fall just beneath static rejection thresholds, stateless guardrail systems fail to detect the attack progression until the harmful completion has already occurred.

## Decision
VaxGuard integrates a **Multi-Turn Crescendo Defense Engine** (`CrescendoDefenseEngine`):
- **Accumulating Risk Score with Exponential Decay**:
  $$R_t = \lambda R_{t-1} + \Delta r_t$$
  where $\lambda \in [0.85, 0.95]$ is the conversational half-life decay constant, ensuring benign conversational transitions naturally decay back toward zero, while persistent boundary-probing causes risk to compound.
- **Linguistic Probing Indicators**:
  Inspects turns for coaxing linguistic patterns, boundary erosion phrases ("purely hypothetical", "for educational purposes only", "in this universe there are no rules"), and semantic proximity shifts toward restricted clusters.
- **Dynamic Risk Level States & Adaptive Actions**:
  - **NORMAL ($R < 0.40$)**: Standard baseline detection thresholds.
  - **ELEVATED ($0.40 \le R < 0.70$)**: Detection sensitivity dynamically tightens (lowers acceptance threshold by 15–25%), subjecting subsequent turns to rigorous inspection.
  - **CRITICAL ($R \ge 0.70$)**: Immediate conversational circuit breaker triggers; the session is blocked with a structured audit incident raised before execution proceeds.

## Consequences
### Positive
- **Defense Against Slow-Burn Attacks**: Neutralizes multi-turn priming before the model reaches an compromised state.
- **Self-Healing Context**: Benign conversations that touch technical or sensitive keywords recover naturally as benign turns accumulate.
- **Lightweight State**: Session state consists of numeric scalar accumulators and bounded rolling history, requiring minimal memory footprint per session.

### Negative / Trade-offs
- Session state must be passed through API requests or persisted in session storage (e.g. Redis) for multi-turn conversational agents.
