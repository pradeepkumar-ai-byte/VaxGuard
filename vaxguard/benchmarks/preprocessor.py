"""
Unified Benchmark Preprocessor and Data Loader for VaxGuard.

Loads, validates, and normalizes standardized public security benchmarks:
- AdvGLUE (Adversarial linguistic perturbations)
- DoNotAnswer (Safety taxonomy boundary evaluation)
- HarmBench (Functional red-team vectors and role invariance)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field


class BenchmarkSample(BaseModel):
    """Normalized evaluation sample from an independent benchmark dataset."""
    id: str
    prompt: str
    expected_action: str = Field(description="'block' or 'allow'")
    category: str = "general"
    attack_family: str = "benchmark"
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def is_attack(self) -> bool:
        """Return True if this sample is an adversarial/restricted probe expecting 'block'."""
        return self.expected_action.lower() in ("block", "redact", "quarantine")

    @property
    def is_benign(self) -> bool:
        """Return True if this sample is legitimate/benign query expecting 'allow'."""
        return self.expected_action.lower() in ("allow", "permit")


@dataclass
class BenchmarkSummary:
    """Statistical summary of a loaded benchmark dataset."""
    name: str
    total_samples: int
    attack_samples: int
    benign_samples: int
    categories: Dict[str, int] = field(default_factory=dict)
    file_path: Optional[str] = None


def resolve_benchmark_path(name_or_path: Union[str, Path]) -> Path:
    """Resolve a benchmark dataset name or file path into an absolute Path."""
    p = Path(name_or_path)
    if p.exists() and p.is_file():
        return p.resolve()

    # Normalize name (e.g. 'advglue' -> 'advglue.json')
    filename = p.name if p.suffix == ".json" else f"{p.name}.json"
    
    # Check in project data/benchmarks
    project_root = Path(__file__).resolve().parents[2]
    candidate = project_root / "data" / "benchmarks" / filename
    if candidate.exists():
        return candidate.resolve()

    candidate_full = project_root / "data" / "benchmarks" / "full" / filename
    if candidate_full.exists():
        return candidate_full.resolve()

    # Search current working directory
    cwd_candidate = Path.cwd() / "data" / "benchmarks" / filename
    if cwd_candidate.exists():
        return cwd_candidate.resolve()

    cwd_candidate_full = Path.cwd() / "data" / "benchmarks" / "full" / filename
    if cwd_candidate_full.exists():
        return cwd_candidate_full.resolve()

    raise FileNotFoundError(
        f"Benchmark dataset '{name_or_path}' could not be located at {candidate} or {cwd_candidate}."
    )


def load_benchmark(name_or_path: Union[str, Path]) -> List[BenchmarkSample]:
    """
    Load and parse a benchmark dataset into a list of normalized BenchmarkSamples.

    Supports:
    - 'advglue'
    - 'donotanswer'
    - 'harmbench'
    - Any valid JSON benchmark file path
    """
    path = resolve_benchmark_path(name_or_path)
    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, list):
        raise ValueError(f"Benchmark file {path} must contain a JSON array of samples.")

    samples: List[BenchmarkSample] = []
    for item in raw_data:
        # Normalize category from different benchmark schemas
        cat = (
            item.get("category")
            or item.get("risk_category")
            or item.get("functional_category")
            or item.get("perturbation_type")
            or "general"
        )
        sample = BenchmarkSample(
            id=str(item.get("id", f"sample_{len(samples)+1}")),
            prompt=str(item.get("prompt", "")),
            expected_action=str(item.get("expected_action", "block")).lower(),
            category=str(cat),
            attack_family=str(item.get("attack_family", "benchmark")),
            metadata=item.get("metadata", {}),
        )
        samples.append(sample)

    return samples


def split_samples(
    samples: List[BenchmarkSample],
) -> Tuple[List[BenchmarkSample], List[BenchmarkSample]]:
    """
    Split a list of BenchmarkSamples into:
    (attack_samples, benign_samples).
    """
    attacks = [s for s in samples if s.is_attack]
    benigns = [s for s in samples if s.is_benign]
    return attacks, benigns


def summarize_benchmark(name_or_path: Union[str, Path]) -> BenchmarkSummary:
    """Return a statistical BenchmarkSummary for a benchmark dataset."""
    path = resolve_benchmark_path(name_or_path)
    samples = load_benchmark(path)
    attacks, benigns = split_samples(samples)

    categories: Dict[str, int] = {}
    for s in samples:
        categories[s.category] = categories.get(s.category, 0) + 1

    return BenchmarkSummary(
        name=path.stem,
        total_samples=len(samples),
        attack_samples=len(attacks),
        benign_samples=len(benigns),
        categories=categories,
        file_path=str(path),
    )
