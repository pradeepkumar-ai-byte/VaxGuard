"""
Unit tests for Rule Versioning and Atomic Rollback in RuleLoader.
"""

from pathlib import Path

from vaxguard.nemo.hybrid_matcher import TriHybridMatcher
from vaxguard.nemo.rule_loader import RuleLoader


def test_rule_version_creation_and_hashing(tmp_path: Path):
    """Verify that loading a directory computes SHA-256 fingerprint and creates a RuleSetVersion."""
    rule_file = tmp_path / "test_rules.yaml"
    rule_file.write_text(
        """
id: rule_v1_probe
category: jailbreak
action: block
threshold: 0.70
regex_patterns:
  - "(?i)test_v1_pattern"
exact_phrases:
  - "test_v1_pattern"
lexical_exemplars:
  - "test_v1_pattern sample"
semantic_exemplars:
  - "adversarial probe v1"
""",
        encoding="utf-8",
    )

    loader = RuleLoader(directory=tmp_path)
    rules = loader.load_directory()

    assert len(rules) == 1
    version = loader.get_current_version()
    assert version is not None
    assert version.rule_count == 1
    assert version.version_id.startswith("v_")
    assert len(version.content_sha256) == 64  # Valid SHA-256 hex string


def test_rule_version_history_and_atomic_rollback(tmp_path: Path):
    """Verify version history accumulation and atomic rollback to a prior version."""
    rule_file = tmp_path / "evolving_rules.yaml"

    # Version 1: 1 rule
    rule_file.write_text(
        """
id: rule_initial_v1
category: jailbreak
action: block
threshold: 0.68
regex_patterns:
  - "(?i)initial_v1_trigger"
exact_phrases:
  - "initial_v1_trigger"
lexical_exemplars:
  - "initial_v1_trigger text"
semantic_exemplars:
  - "v1 semantic trigger"
""",
        encoding="utf-8",
    )

    matcher = TriHybridMatcher(load_external_rules=False)
    loader = RuleLoader(directory=tmp_path, matcher=matcher)
    loader.reload()

    v1 = loader.get_current_version()
    assert v1 is not None
    assert v1.rule_count == 1
    v1_id = v1.version_id

    # Version 2: Add second rule
    rule_file.write_text(
        """
- id: rule_initial_v1
  category: jailbreak
  action: block
  threshold: 0.68
  regex_patterns:
    - "(?i)initial_v1_trigger"
  exact_phrases:
    - "initial_v1_trigger"
  lexical_exemplars:
    - "initial_v1_trigger text"
  semantic_exemplars:
    - "v1 semantic trigger"

- id: rule_evolved_v2
  category: injection
  action: block
  threshold: 0.65
  regex_patterns:
    - "(?i)second_v2_trigger"
  exact_phrases:
    - "second_v2_trigger"
  lexical_exemplars:
    - "second_v2_trigger text"
  semantic_exemplars:
    - "v2 semantic trigger"
""",
        encoding="utf-8",
    )

    loader.reload()
    v2 = loader.get_current_version()
    assert v2 is not None
    assert v2.rule_count == 2
    assert v2.version_id != v1_id

    # Verify history contains v1
    history = loader.get_version_history()
    assert len(history) == 1
    assert history[0].version_id == v1_id

    # Matcher has 2 external patterns loaded
    assert "rule_evolved_v2" in matcher.patterns

    # Execute atomic rollback to v1
    rollback_ok = loader.rollback_to_version(v1_id)
    assert rollback_ok is True

    # Active version is now v1
    cur_v = loader.get_current_version()
    assert cur_v is not None
    assert cur_v.version_id == v1_id
    assert cur_v.rule_count == 1

    # Matcher patterns reflect rollback (rule_evolved_v2 is removed)
    assert "rule_evolved_v2" not in matcher.patterns
    assert "rule_initial_v1" in matcher.patterns


def test_rollback_to_invalid_version_fails():
    """Verify rollback gracefully rejects nonexistent version IDs."""
    loader = RuleLoader()
    assert loader.rollback_to_version("non_existent_version_999") is False
