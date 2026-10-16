"""
VaxGuard Enterprise Ecosystem Adapter: LlamaIndex Integration.

Provides production-grade, zero-external-heavy-dependency integrations for LlamaIndex RAG pipelines:
1. VaxGuardNodePostprocessor: Inspects retrieved nodes/chunks in a RAG pipeline.
   Strips or neutralizes poisoned context chunks containing indirect prompt injections.
2. VaxGuardQueryTransform: Validates and sanitizes user queries before retrieval.

Pure Python 3.10+, deterministic execution SLA < 1.0ms, zero mandatory heavy binaries.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

from vaxguard.core.logger import get_logger
from vaxguard.integrations.langchain_guard import VaxGuardSecurityViolation
from vaxguard.nemo.hybrid_matcher import (
    GuardrailAction,
    TriHybridMatcher,
)

logger = get_logger("LlamaIndexGuard")

# Optional LlamaIndex base imports (graceful fallback if llama_index.core is not installed)
try:
    from llama_index.core.postprocessor.types import BaseNodePostprocessor
except ImportError:
    class BaseNodePostprocessor:  # type: ignore[no-redef]
        """Fallback base class when llama_index.core is not installed."""
        def postprocess_nodes(self, nodes: List[Any], query_bundle: Optional[Any] = None) -> List[Any]:
            return nodes

try:
    from llama_index.core.indices.query.query_transform.base import BaseQueryTransform
except ImportError:
    class BaseQueryTransform:  # type: ignore[no-redef]
        """Fallback base class when llama_index.core is not installed."""
        def run(self, query_bundle_or_str: Union[str, Any], metadata: Optional[Dict[str, Any]] = None) -> Any:
            return query_bundle_or_str

try:
    from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle
except ImportError:
    @dataclass
    class QueryBundle:  # type: ignore[no-redef]
        """Fallback QueryBundle when llama_index.core is not installed."""
        query_str: str
        custom_embedding_strs: Optional[List[str]] = None
        embedding: Optional[List[float]] = None

        def __str__(self) -> str:
            return self.query_str

    @dataclass
    class TextNode:  # type: ignore[no-redef]
        """Fallback TextNode when llama_index.core is not installed."""
        text: str = ""
        node_id: str = ""
        metadata: Dict[str, Any] = field(default_factory=dict)

        def get_content(self, *args: Any, **kwargs: Any) -> str:
            return self.text

        def set_content(self, value: str) -> None:
            self.text = value

    @dataclass
    class NodeWithScore:  # type: ignore[no-redef]
        """Fallback NodeWithScore when llama_index.core is not installed."""
        node: Any
        score: Optional[float] = None


# ─── Specialized Indirect RAG Injection Patterns ──────────────────────────────

INDIRECT_RAG_PATTERNS = [
    (r"\[\s*(?:SYSTEM|INSTRUCTION|ADMIN|DIRECTIVE)[^\]]*\]", "rag_bracket_directive"),
    (r"<!--\s*(?:END|SYSTEM|DIRECTIVE).*?-->", "rag_html_comment_directive"),
    (r"===+\s*(?:END|SYSTEM|RETRIEVED).*?===+", "rag_delimiter_override"),
    (r"(?i)(?:note to assistant|special instruction|developer override)[:\-]?\s*[^.\n]*(?:\.|\n|$)", "rag_assistant_override"),
    (r"(?i)(?:ignore|disregard|forget|override)\s+(?:all\s+)?(?:prior|previous|initial|system)\s+(?:instructions|rules|directives|prompts)[^.\n]*(?:\.|\n|$)", "rag_instruction_override"),
    (r"(?i)(?:you are now in|adopt)\s+(?:developer mode|dan mode|unrestricted mode)[^.\n]*(?:\.|\n|$)", "rag_persona_hijack"),
    (r"(?i)(?:reveal|dump|print|output|leak)\s+(?:your\s+)?(?:system prompt|initial instructions|confidential config|credentials)[^.\n]*(?:\.|\n|$)", "rag_prompt_exfiltration"),
]

COMPILED_INDIRECT_PATTERNS = [
    (re.compile(pat, flags=re.IGNORECASE), rule_id)
    for pat, rule_id in INDIRECT_RAG_PATTERNS
]


# ─── LlamaIndex Node Postprocessor ────────────────────────────────────────────

class VaxGuardNodePostprocessor(BaseNodePostprocessor):
    """
    Production RAG context firewall for LlamaIndex pipelines.
    Inspects retrieved context chunks and protects the LLM from indirect prompt injections.

    Modes:
      - 'filter' (default): Drops poisoned chunks entirely from the returned node list.
      - 'neutralize': Strips/redacts the adversarial instruction span while preserving valid factual context.
      - 'raise': Immediately raises VaxGuardSecurityViolation when an injection is detected.

    Execution SLA: < 0.2ms per retrieved chunk, zero external heavy dependencies.
    """

    DEFAULT_REPLACEMENT = "[POTENTIAL INJECTION REMOVED BY VAXGUARD]"

    def __init__(
        self,
        matcher: Optional[TriHybridMatcher] = None,
        action: str = "filter",  # 'filter', 'neutralize', or 'raise'
        neutralization_replacement: str = DEFAULT_REPLACEMENT,
        drop_empty_after_neutralization: bool = True,
        min_clean_text_length: int = 15,
        on_violation: Optional[Callable[[VaxGuardSecurityViolation], Any]] = None,
    ) -> None:
        super().__init__()
        self.matcher = matcher or TriHybridMatcher()
        self.action = action.lower()
        self.neutralization_replacement = neutralization_replacement
        self.drop_empty_after_neutralization = drop_empty_after_neutralization
        self.min_clean_text_length = min_clean_text_length
        self.on_violation = on_violation

        # Audit & Metrics
        self.total_nodes_inspected: int = 0
        self.poisoned_nodes_detected: int = 0
        self.nodes_neutralized: int = 0
        self.nodes_filtered: int = 0
        self.violations: List[VaxGuardSecurityViolation] = []

    def _extract_node_text(self, item: Any) -> Tuple[str, Any, str]:
        """
        Duck-typed text extraction from NodeWithScore, BaseNode, TextNode, dict, or str.
        Returns (text, target_node_or_dict, node_id).
        """
        if hasattr(item, "node"):
            node_obj = item.node
            node_id = getattr(node_obj, "node_id", getattr(node_obj, "id_", "node_unknown"))
            if hasattr(node_obj, "get_content"):
                return node_obj.get_content(), node_obj, str(node_id)
            if hasattr(node_obj, "text"):
                return str(node_obj.text), node_obj, str(node_id)
            return str(node_obj), node_obj, str(node_id)

        if hasattr(item, "get_content"):
            node_id = getattr(item, "node_id", getattr(item, "id_", "node_unknown"))
            return item.get_content(), item, str(node_id)

        if hasattr(item, "text"):
            node_id = getattr(item, "node_id", getattr(item, "id_", "node_unknown"))
            return str(item.text), item, str(node_id)

        if isinstance(item, dict):
            node_id = str(item.get("node_id", item.get("id", "node_dict")))
            if "node" in item and isinstance(item["node"], dict):
                text = item["node"].get("text", str(item["node"]))
                return text, item["node"], node_id
            text = item.get("text", item.get("content", str(item)))
            return text, item, node_id

        return str(item), item, "node_primitive"

    def _update_node_text(self, item: Any, target_obj: Any, new_text: str, rule_id: str) -> None:
        """Duck-typed text update across various node types."""
        if hasattr(target_obj, "set_content"):
            target_obj.set_content(new_text)
        elif hasattr(target_obj, "text"):
            target_obj.text = new_text
        elif isinstance(target_obj, dict):
            target_obj["text"] = new_text

        # Update metadata if available
        meta = None
        if hasattr(target_obj, "metadata") and isinstance(target_obj.metadata, dict):
            meta = target_obj.metadata
        elif isinstance(target_obj, dict) and "metadata" in target_obj and isinstance(target_obj["metadata"], dict):
            meta = target_obj["metadata"]

        if meta is not None:
            meta["vaxguard_neutralized"] = True
            meta["vaxguard_rule"] = rule_id

    def _inspect_text(self, text: str) -> Tuple[bool, str, str, float]:
        """
        Check if text contains indirect prompt injection or adversarial bypass.
        Returns (is_poisoned, rule_id, category, confidence).
        """
        # 1. Regex inspection for RAG delimiters & injection wrappers
        for pat, rule_id in COMPILED_INDIRECT_PATTERNS:
            if pat.search(text):
                return True, rule_id, "indirect_rag_injection", 1.0

        # 2. TriHybridMatcher evaluation
        match_res = self.matcher.match(text)
        if match_res.matched and match_res.action == GuardrailAction.BLOCK:
            return True, match_res.rule_id, match_res.category, match_res.confidence

        return False, "", "", 0.0

    def _neutralize_text(self, text: str) -> Tuple[str, bool]:
        """
        Neutralizes indirect injection payloads within a chunk, preserving clean factual content.
        """
        sanitized = text
        modified = False

        for pat, rule_id in COMPILED_INDIRECT_PATTERNS:
            if pat.search(sanitized):
                sanitized = pat.sub(self.neutralization_replacement, sanitized)
                modified = True

        return sanitized, modified

    def postprocess_nodes(
        self,
        nodes: List[Any],
        query_bundle: Optional[Any] = None,
    ) -> List[Any]:
        """
        Inspects retrieved nodes. Strips, neutralizes, or filters poisoned chunks.
        """
        cleaned_nodes: List[Any] = []

        for item in nodes:
            self.total_nodes_inspected += 1
            text, target_obj, node_id = self._extract_node_text(item)

            is_poisoned, rule_id, category, conf = self._inspect_text(text)

            if not is_poisoned:
                cleaned_nodes.append(item)
                continue

            # Poisoned node detected
            self.poisoned_nodes_detected += 1
            violation = VaxGuardSecurityViolation(
                message=f"Poisoned RAG chunk detected: {rule_id}",
                category=category,
                rule_id=rule_id,
                confidence=conf,
                payload_sample=text[:120],
                metadata={"node_id": node_id, "action": self.action},
            )
            self.violations.append(violation)
            if self.on_violation:
                try:
                    self.on_violation(violation)
                except Exception as ex:
                    logger.error(f"Error in on_violation: {ex}")

            if self.action == "raise":
                raise violation

            elif self.action == "filter":
                self.nodes_filtered += 1
                logger.info(f"Filtered poisoned chunk [{node_id}] matching {rule_id}")
                continue

            elif self.action == "neutralize":
                neutralized_text, modified = self._neutralize_text(text)
                still_poisoned, _, _, _ = self._inspect_text(neutralized_text)

                # Check if enough legitimate text remains
                clean_residual = neutralized_text.replace(self.neutralization_replacement, "").strip()

                if still_poisoned or (self.drop_empty_after_neutralization and len(clean_residual) < self.min_clean_text_length):
                    # If entire chunk was malicious or cannot be sanitized cleanly, filter it out
                    self.nodes_filtered += 1
                    logger.info(f"Poisoned chunk [{node_id}] had no safe content after neutralization; filtered.")
                    continue

                # Legitimate content preserved with injection removed
                self._update_node_text(item, target_obj, neutralized_text, rule_id)
                self.nodes_neutralized += 1
                cleaned_nodes.append(item)
            else:
                # Default safety fallback: filter out
                self.nodes_filtered += 1

        return cleaned_nodes

    def _postprocess_nodes(
        self,
        nodes: List[Any],
        query_bundle: Optional[Any] = None,
    ) -> List[Any]:
        return self.postprocess_nodes(nodes, query_bundle)

    async def apostprocess_nodes(
        self,
        nodes: List[Any],
        query_bundle: Optional[Any] = None,
    ) -> List[Any]:
        """Asynchronous node postprocessing."""
        return self.postprocess_nodes(nodes, query_bundle)

    async def _apostprocess_nodes(
        self,
        nodes: List[Any],
        query_bundle: Optional[Any] = None,
    ) -> List[Any]:
        return self.postprocess_nodes(nodes, query_bundle)


# ─── LlamaIndex Query Transform ───────────────────────────────────────────────

class VaxGuardQueryTransform(BaseQueryTransform):
    """
    Validates and hardens user queries before retrieval in LlamaIndex.
    Guards the index against adversarial search vectors, jailbreaks, and injection payloads.

    Execution SLA: < 0.2ms per query.
    """

    def __init__(
        self,
        matcher: Optional[TriHybridMatcher] = None,
        raise_exception: bool = True,
        action: str = "block",  # 'block', 'sanitize', 'raise'
        fallback_query: str = "[SAFE_QUERY_EMPTY]",
        inoculate: bool = False,
        on_violation: Optional[Callable[[VaxGuardSecurityViolation], Any]] = None,
    ) -> None:
        super().__init__()
        self.matcher = matcher or TriHybridMatcher()
        self.raise_exception = raise_exception
        self.action = action.lower()
        self.fallback_query = fallback_query
        self.inoculate = inoculate
        self.on_violation = on_violation

        self.queries_inspected: int = 0
        self.queries_blocked: int = 0
        self.violations: List[VaxGuardSecurityViolation] = []

    def _extract_query_str(self, query_bundle_or_str: Union[str, Any]) -> str:
        if isinstance(query_bundle_or_str, str):
            return query_bundle_or_str
        if hasattr(query_bundle_or_str, "query_str"):
            return str(query_bundle_or_str.query_str)
        return str(query_bundle_or_str)

    def _wrap_query(self, original_bundle: Any, new_query_str: str) -> Any:
        if isinstance(original_bundle, str):
            return QueryBundle(query_str=new_query_str)
        if hasattr(original_bundle, "query_str"):
            try:
                new_bundle = copy.copy(original_bundle)
                new_bundle.query_str = new_query_str
                return new_bundle
            except Exception:
                return QueryBundle(query_str=new_query_str)
        return QueryBundle(query_str=new_query_str)

    def run(
        self,
        query_bundle_or_str: Union[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Validates user query before retrieval.
        Raises VaxGuardSecurityViolation or returns safe query bundle.
        """
        self.queries_inspected += 1
        query_str = self._extract_query_str(query_bundle_or_str)
        is_violation = False
        rule_id = ""
        category = ""
        conf = 0.0

        for pat, r_id in COMPILED_INDIRECT_PATTERNS:
            if pat.search(query_str):
                is_violation = True
                rule_id = r_id
                category = "indirect_rag_injection"
                conf = 1.0
                break

        if not is_violation:
            match_res = self.matcher.match(query_str)
            if match_res.matched and match_res.action == GuardrailAction.BLOCK:
                is_violation = True
                rule_id = match_res.rule_id
                category = match_res.category
                conf = match_res.confidence

        if is_violation:
            self.queries_blocked += 1
            violation = VaxGuardSecurityViolation(
                message=f"Pre-retrieval query rejected: {category} ({rule_id})",
                category=category,
                rule_id=rule_id,
                confidence=conf,
                payload_sample=query_str[:120],
                metadata=metadata or {},
            )
            self.violations.append(violation)
            if self.on_violation:
                try:
                    self.on_violation(violation)
                except Exception as ex:
                    logger.error(f"Error in on_violation: {ex}")

            if self.raise_exception or self.action == "raise":
                raise violation

            if self.action == "sanitize":
                # Remove adversarial tokens / override markers
                clean_q = query_str
                for pat, _ in COMPILED_INDIRECT_PATTERNS:
                    clean_q = pat.sub("", clean_q)
                clean_q = re.sub(r"\s+", " ", clean_q).strip()
                if not clean_q:
                    clean_q = self.fallback_query
                return self._wrap_query(query_bundle_or_str, clean_q)

            # Block action
            return self._wrap_query(query_bundle_or_str, self.fallback_query)

        # Benign query
        final_query = query_str
        if self.inoculate:
            # Append defensive scoping constraint if inoculate is enabled
            final_query = f"{query_str.strip()} [defense: factual knowledge only]"

        return self._wrap_query(query_bundle_or_str, final_query)

    def _run(self, query_bundle: Any, metadata: Optional[Dict[str, Any]] = None) -> Any:
        return self.run(query_bundle, metadata)

    async def arun(
        self,
        query_bundle_or_str: Union[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Asynchronously validates query."""
        return self.run(query_bundle_or_str, metadata)

    def __call__(
        self,
        query_bundle_or_str: Union[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        return self.run(query_bundle_or_str, metadata)
