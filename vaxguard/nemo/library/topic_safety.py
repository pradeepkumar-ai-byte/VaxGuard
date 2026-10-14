"""VaxGuard High-Speed Topic Safety & Boundary Containment Library.

Canonical NVIDIA NeMo Guardrails actions:
- topic_safety_check_input
- detect_regex_pattern
"""

from __future__ import annotations

import re
import logging
from typing import Any, Dict, List, Optional
from vaxguard.nemo.actions import RailOutcome
from vaxguard.nemo.hybrid_matcher import get_tri_hybrid_matcher

log = logging.getLogger(__name__)
_MATCHER = get_tri_hybrid_matcher()

# Canonical Off-Topic & Harmful Topic Patterns
CANONICAL_OFF_TOPIC_PATTERNS = [
    re.compile(r"(?i)\b(?:how\s+to\s+(?:make|build|cook|synthesize)\s+(?:a\s+bomb|meth|drugs|weapons|explosives))\b"),
    re.compile(r"(?i)\b(?:teach\s+me\s+(?:racial\s+slurs|how\s+to\s+hack|how\s+to\s+stalk))\b"),
    re.compile(r"(?i)\b(?:help\s+me\s+(?:evade\s+taxes|rob\s+a\s+bank|steal\s+an\s+identity))\b"),
    re.compile(r"(?i)\b(?:write\s+a\s+poem\s+about\s+(?:cooking\s+pasta|astrology))\b"),
]


async def topic_safety_check_input(
    context: Optional[Dict[str, Any]] = None,
    user_message: Optional[str] = None,
    config: Optional[Any] = None,
    **kwargs: Any
) -> RailOutcome:
    """Canonical NVIDIA NeMo Guardrails action: topic_safety_check_input.
    
    Verifies that user queries stay within authorized domain scope and do not
    drift into dangerous, disallowed, or irrelevant topics.
    """
    context = context or {}
    text = user_message if user_message is not None else context.get("user_message", "")
    text = str(text or "").strip()

    if not text:
        return RailOutcome.allow(action_name="topic_safety_check_input", data={"on_topic": True})

    # 1. Regex Fast-Path Check
    for pat in CANONICAL_OFF_TOPIC_PATTERNS:
        if pat.search(text):
            return RailOutcome.block(
                action_name="topic_safety_check_input",
                reason="Query falls into disallowed or off-topic scope",
                data={"on_topic": False, "is_blocked": True}
            )

    # 2. Tri-Hybrid Matcher Check
    decision = _MATCHER.evaluate(text)
    if decision.matched and decision.score >= 0.80:
        return RailOutcome.block(
            action_name="topic_safety_check_input",
            reason=f"Off-topic violation matched: {decision.rule_name}",
            data={"on_topic": False, "is_blocked": True, "rule": decision.rule_name}
        )

    return RailOutcome.allow(
        action_name="topic_safety_check_input",
        data={"on_topic": True, "is_blocked": False}
    )


async def detect_regex_pattern(
    source: str = "input",
    text: Optional[str] = None,
    patterns: Optional[List[str]] = None,
    config: Optional[Any] = None,
    context: Optional[Dict[str, Any]] = None,
    **kwargs: Any
) -> RailOutcome:
    """Canonical NVIDIA NeMo Guardrails action: detect_regex_pattern.
    
    Checks text against a list of custom regex patterns.
    """
    if text is None and context:
        if source == "input":
            text = context.get("user_message", "")
        elif source == "output":
            text = context.get("bot_message", "")
        elif source == "retrieval":
            text = str(context.get("relevant_chunks", ""))

    text = str(text or "")
    pattern_list = patterns or []

    matched = []
    for raw_pat in pattern_list:
        try:
            if re.search(raw_pat, text, re.IGNORECASE):
                matched.append(raw_pat)
        except Exception:
            continue

    if matched:
        return RailOutcome.block(
            action_name="detect_regex_pattern",
            reason=f"Regex pattern matched: {matched[0]}",
            data={"is_match": True, "detections": matched, "source": source}
        )

    return RailOutcome.allow(
        action_name="detect_regex_pattern",
        data={"is_match": False, "detections": [], "source": source}
    )
