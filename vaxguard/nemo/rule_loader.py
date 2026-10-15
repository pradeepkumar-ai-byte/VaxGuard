"""
VaxGuard Pure-Python YAML Rule Compiler and Hot-Reload Infrastructure.

Enables dynamic, declarative YAML security rules for TriHybridMatcher:
- Reads all `.yaml` and `.yml` files from a given directory (default: `data/rules/`)
- Validates each rule against a strict schema:
  - Required fields: id, category, action, regex_patterns, exact_phrases, lexical_exemplars, semantic_exemplars
  - Optional fields: threshold (default 0.68), description, metadata
- Compiles regex patterns and constructs GuardrailPattern objects
- Supports thread-safe hot-reload via reload()
- Gracefully handles missing PyYAML and malformed rules with clear log warnings
"""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    yaml = None  # type: ignore[assignment]
    YAML_AVAILABLE = False

from vaxguard.core.logger import get_logger
from vaxguard.nemo.hybrid_matcher import GuardrailAction, GuardrailPattern

logger = get_logger("RuleLoader")

REQUIRED_FIELDS: Tuple[str, ...] = (
    "id",
    "category",
    "action",
    "regex_patterns",
    "exact_phrases",
    "lexical_exemplars",
    "semantic_exemplars",
)

ALLOWED_ACTIONS: Set[str] = {"block", "redact", "warn", "allow"}


class ValidationResult(tuple):
    """
    Validation outcome representation.

    Behaves as a Tuple[bool, str] (is_valid, error_msg), evaluates to boolean
    truthiness based on is_valid, and provides convenient property accessors.
    """

    def __new__(cls, is_valid: bool, error: str = ""):
        return super().__new__(cls, (is_valid, error))

    @property
    def is_valid(self) -> bool:
        return self[0]

    @property
    def error(self) -> str:
        return self[1]

    def __bool__(self) -> bool:
        return self[0]

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, bool):
            return self[0] == other
        return super().__eq__(other)

    def __repr__(self) -> str:
        return f"ValidationResult(is_valid={self[0]}, error={self[1]!r})"


class RuleLoader:
    """
    Pure-Python YAML rule compiler and loader for VaxGuard.

    Scans rule directories, validates rules against strict schemas,
    compiles regexes, builds GuardrailPattern objects, and supports
    thread-safe hot-reloading into TriHybridMatcher.
    """

    def __init__(
        self,
        directory: Optional[Union[str, Path]] = None,
        matcher: Optional[Any] = None,
    ) -> None:
        """
        Initialize the RuleLoader.

        Args:
            directory: Directory containing YAML rule files (default: project_root/data/rules).
            matcher: Optional TriHybridMatcher instance to register patterns with on reload.
        """
        if directory is not None:
            self.directory = Path(directory)
        else:
            self.directory = Path(__file__).resolve().parents[2] / "data" / "rules"
        self.matcher = matcher
        self._lock = threading.RLock()
        self._loaded_rules: List[GuardrailPattern] = []

    def validate_rule(self, rule: Any) -> ValidationResult:
        """
        Validate a single rule dictionary against the strict guardrail schema.

        Required fields:
            - id: non-empty string
            - category: non-empty string
            - action: 'block', 'redact', 'warn', or 'allow'
            - regex_patterns: list of valid regex strings
            - exact_phrases: list of strings
            - lexical_exemplars: list of strings
            - semantic_exemplars: list of strings

        Optional fields:
            - threshold: float between 0.0 and 1.0 (default 0.68)
            - description: string
            - metadata: dict

        Returns:
            ValidationResult (is_valid: bool, error: str)
        """
        if not isinstance(rule, dict):
            return ValidationResult(
                False, f"Rule definition must be a dictionary, got {type(rule).__name__}"
            )

        # 1. Validate required fields presence
        for field_name in REQUIRED_FIELDS:
            if field_name not in rule:
                return ValidationResult(False, f"Missing required field: '{field_name}'")

        # 2. Validate 'id'
        rule_id = rule.get("id")
        if not isinstance(rule_id, str) or not rule_id.strip():
            return ValidationResult(False, "Field 'id' must be a non-empty string")

        # 3. Validate 'category'
        category = rule.get("category")
        if not isinstance(category, str) or not category.strip():
            return ValidationResult(False, "Field 'category' must be a non-empty string")

        # 4. Validate 'action'
        action = rule.get("action")
        if isinstance(action, GuardrailAction):
            action_str = action.value
        elif isinstance(action, str):
            action_str = action.strip().lower()
        else:
            return ValidationResult(
                False,
                f"Field 'action' must be a string or GuardrailAction, got {type(action).__name__}",
            )

        if action_str not in ALLOWED_ACTIONS:
            return ValidationResult(
                False,
                f"Invalid action '{action}'; must be one of: {', '.join(sorted(ALLOWED_ACTIONS))}",
            )

        # 5. Validate list fields: exact_phrases, lexical_exemplars, semantic_exemplars
        list_fields = ("exact_phrases", "lexical_exemplars", "semantic_exemplars")
        for f in list_fields:
            val = rule.get(f)
            if not isinstance(val, (list, tuple)):
                return ValidationResult(False, f"Field '{f}' must be a list")
            for i, item in enumerate(val):
                if not isinstance(item, str):
                    return ValidationResult(
                        False,
                        f"Item {i} in '{f}' must be a string, got {type(item).__name__}",
                    )

        # 6. Validate regex_patterns & compile regexes
        regexes = rule.get("regex_patterns")
        if not isinstance(regexes, (list, tuple)):
            return ValidationResult(False, "Field 'regex_patterns' must be a list")
        for i, pat in enumerate(regexes):
            if not isinstance(pat, str):
                return ValidationResult(
                    False,
                    f"Item {i} in 'regex_patterns' must be a string, got {type(pat).__name__}",
                )
            try:
                re.compile(pat)
            except re.error as err:
                return ValidationResult(
                    False,
                    f"Invalid regex pattern '{pat}' in 'regex_patterns': {err}",
                )

        # 7. Validate optional field: threshold
        if "threshold" in rule and rule["threshold"] is not None:
            th = rule["threshold"]
            if isinstance(th, bool) or not isinstance(th, (int, float)):
                return ValidationResult(
                    False,
                    f"Field 'threshold' must be a number between 0.0 and 1.0, got {th!r}",
                )
            th_float = float(th)
            if not (0.0 <= th_float <= 1.0):
                return ValidationResult(
                    False,
                    f"Field 'threshold' must be between 0.0 and 1.0, got {th_float}",
                )

        # 8. Validate optional field: description
        if "description" in rule and rule["description"] is not None:
            if not isinstance(rule["description"], str):
                return ValidationResult(
                    False,
                    f"Field 'description' must be a string, got {type(rule['description']).__name__}",
                )

        # 9. Validate optional field: metadata
        if "metadata" in rule and rule["metadata"] is not None:
            if not isinstance(rule["metadata"], dict):
                return ValidationResult(
                    False,
                    f"Field 'metadata' must be a dictionary, got {type(rule['metadata']).__name__}",
                )

        return ValidationResult(True, "")

    def compile_rule(self, rule_dict: dict) -> GuardrailPattern:
        """
        Compile a validated rule dictionary into a GuardrailPattern.

        Args:
            rule_dict: Validated dictionary describing a guardrail rule.

        Returns:
            GuardrailPattern instance ready for registration with TriHybridMatcher.
        """
        action_val = rule_dict["action"]
        if isinstance(action_val, GuardrailAction):
            action = action_val
        else:
            action = GuardrailAction(str(action_val).strip().lower())

        threshold = float(rule_dict.get("threshold", 0.68))

        metadata = dict(rule_dict.get("metadata") or {})
        if "description" in rule_dict and rule_dict["description"]:
            metadata.setdefault("description", str(rule_dict["description"]))

        weight_heuristic = float(rule_dict.get("weight_heuristic", 1.0))
        weight_lexical = float(rule_dict.get("weight_lexical", 0.4))
        weight_dense = float(rule_dict.get("weight_dense", 0.6))

        return GuardrailPattern(
            id=str(rule_dict["id"]).strip(),
            category=str(rule_dict["category"]).strip(),
            regex_patterns=[str(p) for p in rule_dict.get("regex_patterns", [])],
            exact_phrases=[str(p) for p in rule_dict.get("exact_phrases", [])],
            lexical_exemplars=[str(p) for p in rule_dict.get("lexical_exemplars", [])],
            semantic_exemplars=[str(p) for p in rule_dict.get("semantic_exemplars", [])],
            action=action,
            threshold=threshold,
            weight_heuristic=weight_heuristic,
            weight_lexical=weight_lexical,
            weight_dense=weight_dense,
            metadata=metadata,
        )

    def load_directory(
        self, path: Optional[Union[str, Path]] = None
    ) -> List[GuardrailPattern]:
        """
        Scan a directory for all .yaml and .yml files and compile valid rules into GuardrailPatterns.

        Malformed rules or files are logged with warnings and skipped, while valid rules continue loading.

        Args:
            path: Directory path to load from (updates self.directory if provided).

        Returns:
            List of compiled GuardrailPattern objects.
        """
        if path is not None:
            dir_path = Path(path)
            self.directory = dir_path
        else:
            dir_path = self.directory

        if not YAML_AVAILABLE or yaml is None:
            logger.warning("PyYAML is not installed; cannot load YAML rules.")
            return []

        if not dir_path.exists():
            logger.warning(f"Rules directory does not exist: {dir_path}")
            return []

        if not dir_path.is_dir():
            logger.warning(f"Rules path is not a directory: {dir_path}")
            return []

        loaded_patterns: List[GuardrailPattern] = []
        yaml_files = sorted(
            [p for p in dir_path.rglob("*") if p.is_file() and p.suffix.lower() in (".yaml", ".yml")]
        )

        for file_path in yaml_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                if not content.strip():
                    logger.debug(f"Empty YAML file skipped: {file_path}")
                    continue
                try:
                    docs = list(yaml.safe_load_all(content))
                except yaml.YAMLError as exc:
                    # If invalid escape sequence in double quotes (common with regex \s, \d, \w, \b), retry with auto-escaping
                    if "found unknown escape character" in str(exc):
                        sanitized = re.sub(r'\\([sdwSDW])', r'\\\\\1', content)
                        docs = list(yaml.safe_load_all(sanitized))
                    else:
                        raise
            except yaml.YAMLError as exc:
                logger.warning(f"Malformed YAML in {file_path}: {exc}")
                continue
            except Exception as exc:
                logger.warning(f"Error reading file {file_path}: {exc}")
                continue

            for doc in docs:
                if doc is None:
                    continue

                rules_to_process: List[Any] = []
                if isinstance(doc, list):
                    rules_to_process.extend(doc)
                elif isinstance(doc, dict):
                    if "rules" in doc and isinstance(doc["rules"], list):
                        rules_to_process.extend(doc["rules"])
                    else:
                        rules_to_process.append(doc)
                else:
                    logger.warning(
                        f"Skipping non-mapping/list YAML document in {file_path}"
                    )
                    continue

                for candidate in rules_to_process:
                    if not isinstance(candidate, dict):
                        logger.warning(
                            f"Rule item in {file_path} must be a dictionary, got {type(candidate).__name__}"
                        )
                        continue

                    validation = self.validate_rule(candidate)
                    if not validation.is_valid:
                        rule_id = candidate.get("id", "<unknown>")
                        logger.warning(
                            f"Invalid rule '{rule_id}' in {file_path}: {validation.error}"
                        )
                        continue

                    try:
                        pattern = self.compile_rule(candidate)
                        loaded_patterns.append(pattern)
                    except Exception as exc:
                        logger.warning(
                            f"Failed to compile rule '{candidate.get('id')}' in {file_path}: {exc}"
                        )
                        continue

        with self._lock:
            self._loaded_rules = list(loaded_patterns)

        return loaded_patterns

    def reload(self) -> int:
        """
        Hot-reload all YAML rules from self.directory and re-register them with matcher.
        Thread-safe via simple threading lock.

        Returns:
            Count of loaded rules.
        """
        with self._lock:
            patterns = self.load_directory(self.directory)
            if self.matcher is not None:
                if hasattr(self.matcher, "_reload_from_patterns"):
                    self.matcher._reload_from_patterns(patterns)
                else:
                    for pattern in patterns:
                        self.matcher.register_pattern(pattern)
            return len(patterns)

    def get_loaded_rules(self) -> List[GuardrailPattern]:
        """Return a snapshot of currently loaded rules."""
        with self._lock:
            return list(self._loaded_rules)
