"""VaxGuard High-Speed Canonical NeMo Guardrails Actions Library.

Pure Python, high-speed, zero external heavy binary implementations of canonical
NVIDIA NeMo Guardrails actions:
- sensitive_data: detect_sensitive_data, mask_sensitive_data
- self_check: self_check_input, self_check_output, self_check_facts
- topic_safety: topic_safety_check_input, detect_regex_pattern
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vaxguard.nemo.library.sensitive_data import (
    detect_sensitive_data,
    mask_sensitive_data,
    FastSensitiveDataEngine,
    SensitiveDetectionMatch,
)
from vaxguard.nemo.library.self_check import (
    self_check_input,
    self_check_output,
    self_check_facts,
)
from vaxguard.nemo.library.topic_safety import (
    topic_safety_check_input,
    detect_regex_pattern,
)

if TYPE_CHECKING:
    from vaxguard.nemo.actions import ActionRegistry


def register_canonical_actions(registry: ActionRegistry) -> None:
    """Registers all canonical NeMo actions into the given ActionRegistry."""
    from vaxguard.nemo.actions import ActionType

    registry.register(
        "detect_sensitive_data",
        detect_sensitive_data,
        description="Canonical NeMo action: detect PII, PCI, and API keys",
        action_type=ActionType.GUARDRAIL,
    )
    registry.register(
        "mask_sensitive_data",
        mask_sensitive_data,
        description="Canonical NeMo action: redact sensitive entities from text",
        action_type=ActionType.GUARDRAIL,
    )
    registry.register(
        "self_check_input",
        self_check_input,
        description="Canonical NeMo action: evaluate input rail compliance <0.08ms fast-path",
        action_type=ActionType.GUARDRAIL,
    )
    registry.register(
        "self_check_output",
        self_check_output,
        description="Canonical NeMo action: evaluate output rail safety and data leaks",
        action_type=ActionType.GUARDRAIL,
    )
    registry.register(
        "self_check_facts",
        self_check_facts,
        description="Canonical NeMo action: factual grounding and hallucination mitigation",
        action_type=ActionType.GUARDRAIL,
    )
    registry.register(
        "topic_safety_check_input",
        topic_safety_check_input,
        description="Canonical NeMo action: topical boundary containment",
        action_type=ActionType.GUARDRAIL,
    )
    registry.register(
        "detect_regex_pattern",
        detect_regex_pattern,
        description="Canonical NeMo action: arbitrary regex pattern matching",
        action_type=ActionType.GUARDRAIL,
    )


__all__ = [
    "detect_sensitive_data",
    "mask_sensitive_data",
    "FastSensitiveDataEngine",
    "SensitiveDetectionMatch",
    "self_check_input",
    "self_check_output",
    "self_check_facts",
    "topic_safety_check_input",
    "detect_regex_pattern",
    "register_canonical_actions",
]
