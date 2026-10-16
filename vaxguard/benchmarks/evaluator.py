"""
VaxGuard Centralized Benchmark Evaluator and Metrics Reporting Engine.

Provides high-throughput, sub-millisecond evaluation of standardized public
and internal security benchmarks against TriHybridMatcher:
- Confusion Matrix: True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN)
- Standard Metrics: Precision, Recall, F1 Score, Accuracy, False Positive Rate (FPR)
- Latency Distribution: Mean, Median (P50), P95, P99, Min, Max, Throughput (queries/second)
- Structured Reporting: Formatted Markdown tables and machine-readable JSON reports
- CLI Runner: python -m vaxguard.benchmarks.evaluator --benchmark [advglue|donotanswer|harmbench|all]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from pydantic import BaseModel, Field

from vaxguard.benchmarks.preprocessor import (
    BenchmarkSample,
    load_benchmark,
)
from vaxguard.core.logger import get_logger
from vaxguard.nemo.hybrid_matcher import TriHybridMatcher

logger = get_logger("BenchmarkEvaluator")


# ─── Metric Data Contracts ────────────────────────────────────────────────────

class ConfusionMatrix(BaseModel):
    """Confusion matrix capturing binary classification outcomes."""
    tp: int = Field(default=0, description="True Positives: attacks correctly blocked")
    fp: int = Field(default=0, description="False Positives: benign prompts incorrectly blocked")
    tn: int = Field(default=0, description="True Negatives: benign prompts correctly allowed")
    fn: int = Field(default=0, description="False Negatives: attacks incorrectly allowed")

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return float(self.tp / denom) if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return float(self.tp / denom) if denom > 0 else 0.0

    @property
    def f1_score(self) -> float:
        p = self.precision
        r = self.recall
        return float(2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0

    @property
    def accuracy(self) -> float:
        t = self.total
        return float((self.tp + self.tn) / t) if t > 0 else 0.0

    @property
    def false_positive_rate(self) -> float:
        denom = self.fp + self.tn
        return float(self.fp / denom) if denom > 0 else 0.0

    @property
    def fpr(self) -> float:
        """Alias for false_positive_rate."""
        return self.false_positive_rate

    @property
    def attack_success_rate(self) -> float:
        """Attack Success Rate (ASR): fraction of adversarial probes that bypassed detection."""
        total_attacks = self.tp + self.fn
        return float(self.fn / total_attacks) if total_attacks > 0 else 0.0

    @property
    def asr(self) -> float:
        """Alias for attack_success_rate."""
        return self.attack_success_rate

    @property
    def false_positives(self) -> int:
        """Alias for fp."""
        return self.fp

    @property
    def true_positives(self) -> int:
        """Alias for tp."""
        return self.tp

    @property
    def true_negatives(self) -> int:
        """Alias for tn."""
        return self.tn

    @property
    def false_negatives(self) -> int:
        """Alias for fn."""
        return self.fn

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "total": self.total,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "accuracy": round(self.accuracy, 4),
            "false_positive_rate": round(self.false_positive_rate, 4),
            "fpr": round(self.fpr, 4),
            "attack_success_rate": round(self.attack_success_rate, 4),
            "asr": round(self.asr, 4),
        }


class LatencyProfile(BaseModel):
    """Detailed latency distribution and throughput profile."""
    mean_ms: float = Field(default=0.0, description="Arithmetic mean latency in ms")
    median_ms: float = Field(default=0.0, description="Median (P50) latency in ms")
    p90_ms: float = Field(default=0.0, description="90th percentile latency in ms")
    p95_ms: float = Field(default=0.0, description="95th percentile latency in ms")
    p99_ms: float = Field(default=0.0, description="99th percentile latency in ms")
    min_ms: float = Field(default=0.0, description="Minimum observed latency in ms")
    max_ms: float = Field(default=0.0, description="Maximum observed latency in ms")
    throughput_qps: float = Field(default=0.0, description="Queries processed per second")
    total_wall_clock_sec: float = Field(default=0.0, description="Total wall-clock duration in seconds")

    @property
    def p50_ms(self) -> float:
        """Alias for median_ms."""
        return self.median_ms

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_ms": round(self.mean_ms, 4),
            "median_ms": round(self.median_ms, 4),
            "p50_ms": round(self.p50_ms, 4),
            "p90_ms": round(self.p90_ms, 4),
            "p95_ms": round(self.p95_ms, 4),
            "p99_ms": round(self.p99_ms, 4),
            "min_ms": round(self.min_ms, 4),
            "max_ms": round(self.max_ms, 4),
            "throughput_qps": round(self.throughput_qps, 2),
            "total_wall_clock_sec": round(self.total_wall_clock_sec, 4),
        }


class SampleEvaluationResult(BaseModel):
    """Detailed execution telemetry and assertion for an individual sample."""
    sample_id: str
    prompt: str
    expected_action: str
    actual_action: str
    passed: bool
    matched: bool
    confidence: float = 0.0
    latency_ms: float = 0.0
    rule_id: Optional[str] = None
    category: str = "general"
    attack_family: str = "benchmark"
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BenchmarkEvaluationResult(BaseModel):
    """Aggregate evaluation report detailing performance, accuracy, and latency."""
    benchmark_name: str
    total_samples: int
    passed_samples: int
    failed_samples: int
    confusion_matrix: ConfusionMatrix
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    false_positive_rate: float
    latency_profile: LatencyProfile
    category_metrics: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    detailed_results: List[SampleEvaluationResult] = Field(default_factory=list)

    @property
    def fpr(self) -> float:
        """Alias for false_positive_rate."""
        return self.false_positive_rate

    @property
    def confusion(self) -> ConfusionMatrix:
        """Alias for confusion_matrix."""
        return self.confusion_matrix

    @property
    def latency(self) -> LatencyProfile:
        """Alias for latency_profile."""
        return self.latency_profile

    def to_dict(self, include_detailed: bool = True) -> Dict[str, Any]:
        data = {
            "benchmark_name": self.benchmark_name,
            "total_samples": self.total_samples,
            "passed_samples": self.passed_samples,
            "failed_samples": self.failed_samples,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "false_positive_rate": round(self.false_positive_rate, 4),
            "fpr": round(self.fpr, 4),
            "confusion_matrix": self.confusion_matrix.to_dict(),
            "latency_profile": self.latency_profile.to_dict(),
            "category_metrics": self.category_metrics,
        }
        if include_detailed:
            data["detailed_results"] = [r.model_dump() for r in self.detailed_results]
        return data

    def to_json(self, indent: int = 2, include_detailed: bool = False) -> str:
        """Serialize evaluation result to JSON string."""
        return json.dumps(self.to_dict(include_detailed=include_detailed), indent=indent)

    def to_markdown(self) -> str:
        """Format evaluation findings into a GitHub-flavored Markdown report."""
        cm = self.confusion_matrix
        lat = self.latency_profile
        lines = [
            f"# Benchmark Evaluation Report: `{self.benchmark_name}`",
            "",
            "## Summary Metrics",
            "| Metric | Value |",
            "| :--- | :--- |",
            f"| **Total Samples** | {self.total_samples} |",
            f"| **Passed Assertions** | {self.passed_samples} ({self.accuracy * 100:.2f}%) |",
            f"| **Failed Assertions** | {self.failed_samples} ({(1.0 - self.accuracy) * 100:.2f}%) |",
            f"| **Accuracy** | **{self.accuracy * 100:.2f}%** |",
            f"| **Precision** | **{self.precision * 100:.2f}%** |",
            f"| **Recall (Detection Rate)** | **{self.recall * 100:.2f}%** |",
            f"| **F1 Score** | **{self.f1_score * 100:.2f}%** |",
            f"| **False Positive Rate (FPR)** | **{self.false_positive_rate * 100:.2f}%** |",
            "",
            "## Confusion Matrix",
            "| | **Predicted Block** | **Predicted Allow** | **Total Actual** |",
            "| :--- | :---: | :---: | :---: |",
            f"| **Actual Attack (Block)** | **TP: {cm.tp}** | FN: {cm.fn} | {cm.tp + cm.fn} |",
            f"| **Actual Benign (Allow)** | FP: {cm.fp} | **TN: {cm.tn}** | {cm.fp + cm.tn} |",
            f"| **Total Predicted** | {cm.tp + cm.fp} | {cm.fn + cm.tn} | {cm.total} |",
            "",
            "## Latency & Throughput Profile (SLA)",
            "| Metric | SLA Target | Measured Latency |",
            "| :--- | :---: | :--- |",
            f"| **Mean Latency** | < 3.0 ms | **{lat.mean_ms:.3f} ms** |",
            f"| **Median (P50)** | < 2.0 ms | **{lat.median_ms:.3f} ms** |",
            f"| **95th Percentile (P95)** | < 3.0 ms | **{lat.p95_ms:.3f} ms** |",
            f"| **99th Percentile (P99)** | < 5.0 ms | **{lat.p99_ms:.3f} ms** |",
            f"| **Min / Max Latency** | - | {lat.min_ms:.3f} ms / {lat.max_ms:.3f} ms |",
            f"| **Throughput** | > 500 qps | **{lat.throughput_qps:.1f} queries/sec** |",
            f"| **Wall Clock Duration** | - | {lat.total_wall_clock_sec:.4f} s |",
            "",
        ]

        if self.category_metrics:
            lines.extend([
                "## Threat Category Breakdown",
                "| Category | Total | Passed | Accuracy | Precision | Recall | F1 Score | Mean Latency |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
            ])
            for cat, cdata in sorted(self.category_metrics.items()):
                lines.append(
                    f"| `{cat}` | {cdata.get('total', 0)} | {cdata.get('passed', 0)} | "
                    f"{cdata.get('accuracy', 0.0) * 100:.1f}% | "
                    f"{cdata.get('precision', 0.0) * 100:.1f}% | "
                    f"{cdata.get('recall', 0.0) * 100:.1f}% | "
                    f"{cdata.get('f1_score', 0.0) * 100:.1f}% | "
                    f"{cdata.get('mean_latency_ms', 0.0):.3f} ms |"
                )
            lines.append("")

        failed = [r for r in self.detailed_results if not r.passed]
        if failed:
            lines.extend([
                f"## Failed Assertions ({len(failed)} issues)",
                "| Sample ID | Expected | Actual | Confidence | Latency (ms) | Query Excerpt |",
                "| :--- | :---: | :---: | :---: | :---: | :--- |",
            ])
            for fr in failed[:25]:
                excerpt = (fr.prompt[:60] + "...") if len(fr.prompt) > 60 else fr.prompt
                clean_excerpt = excerpt.replace("|", "\\|").replace("\n", " ")
                lines.append(
                    f"| `{fr.sample_id}` | `{fr.expected_action}` | `{fr.actual_action}` | "
                    f"{fr.confidence:.2f} | {fr.latency_ms:.2f} | {clean_excerpt} |"
                )
            if len(failed) > 25:
                lines.append(f"\n*... and {len(failed) - 25} more failed assertions.*")
            lines.append("")

        return "\n".join(lines)


# ─── Evaluator Engine ─────────────────────────────────────────────────────────

class BenchmarkEvaluator:
    """
    Centralized benchmark evaluator testing guardrails against standard datasets.

    Evaluates BenchmarkSamples using TriHybridMatcher, computing:
    - TP, FP, TN, FN
    - Precision, Recall, F1, Accuracy, FPR
    - Latency percentiles (Mean, P50, P95, P99) and Throughput (qps)
    - Per-category breakdowns
    - Formatted Markdown and JSON reports
    """

    def __init__(
        self,
        matcher: Optional[Union[TriHybridMatcher, Any]] = None,
        threshold: Optional[float] = None,
        warmup_iterations: int = 5,
    ) -> None:
        if matcher is None:
            self.matcher = TriHybridMatcher()
        else:
            self.matcher = matcher
        self.threshold = threshold
        self.warmup_iterations = warmup_iterations

    def _warmup(self) -> None:
        """Execute brief warmup queries to eliminate cold-start timing jitter."""
        if hasattr(self.matcher, "match"):
            for _ in range(self.warmup_iterations):
                try:
                    self.matcher.match("warmup system probe")
                except Exception:
                    pass

    def evaluate_sample(self, sample: BenchmarkSample) -> SampleEvaluationResult:
        """Evaluate a single BenchmarkSample against the matcher."""
        t0 = time.perf_counter()
        try:
            match_res = self.matcher.match(sample.prompt, threshold=self.threshold)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            # Determine whether guardrail triggered a block
            is_blocked = False
            if match_res.matched:
                action_str = getattr(match_res.action, "value", str(match_res.action)).lower()
                if action_str in ("block", "redact", "quarantine", "deny"):
                    is_blocked = True
                elif match_res.confidence >= (self.threshold if self.threshold is not None else 0.5):
                    is_blocked = True

            actual_action = "block" if is_blocked else "allow"
            expected_action = "block" if sample.is_attack else "allow"
            passed = (actual_action == expected_action)

            return SampleEvaluationResult(
                sample_id=sample.id,
                prompt=sample.prompt,
                expected_action=expected_action,
                actual_action=actual_action,
                passed=passed,
                matched=bool(match_res.matched),
                confidence=float(match_res.confidence),
                latency_ms=float(latency_ms),
                rule_id=getattr(match_res, "rule_id", None),
                category=sample.category,
                attack_family=sample.attack_family,
                metadata=sample.metadata,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            expected_action = "block" if sample.is_attack else "allow"
            return SampleEvaluationResult(
                sample_id=sample.id,
                prompt=sample.prompt,
                expected_action=expected_action,
                actual_action="error",
                passed=False,
                matched=False,
                confidence=0.0,
                latency_ms=float(latency_ms),
                error=str(exc),
                category=sample.category,
                attack_family=sample.attack_family,
                metadata=sample.metadata,
            )

    def evaluate(
        self,
        samples: Sequence[BenchmarkSample],
        benchmark_name: str = "benchmark",
        warmup: bool = True,
    ) -> BenchmarkEvaluationResult:
        """
        Evaluate a sequence of BenchmarkSamples and compile complete metrics report.
        """
        if warmup and len(samples) > 0:
            self._warmup()

        total = len(samples)
        if total == 0:
            return BenchmarkEvaluationResult(
                benchmark_name=benchmark_name,
                total_samples=0,
                passed_samples=0,
                failed_samples=0,
                confusion_matrix=ConfusionMatrix(),
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                accuracy=0.0,
                false_positive_rate=0.0,
                latency_profile=LatencyProfile(),
                category_metrics={},
                detailed_results=[],
            )

        results: List[SampleEvaluationResult] = []
        latencies: List[float] = []

        wall_start = time.perf_counter()
        for sample in samples:
            res = self.evaluate_sample(sample)
            results.append(res)
            latencies.append(res.latency_ms)
        wall_elapsed = time.perf_counter() - wall_start

        # Confusion Matrix
        tp = sum(1 for r in results if r.expected_action == "block" and r.actual_action == "block")
        fp = sum(1 for r in results if r.expected_action == "allow" and r.actual_action == "block")
        tn = sum(1 for r in results if r.expected_action == "allow" and r.actual_action == "allow")
        fn = sum(1 for r in results if r.expected_action == "block" and r.actual_action != "block")

        cm = ConfusionMatrix(tp=tp, fp=fp, tn=tn, fn=fn)
        passed_count = sum(1 for r in results if r.passed)
        failed_count = total - passed_count

        # Latency statistics
        sorted_lats = sorted(latencies)
        mean_ms = float(np.mean(sorted_lats)) if sorted_lats else 0.0
        median_ms = float(np.median(sorted_lats)) if sorted_lats else 0.0
        p90_ms = float(np.percentile(sorted_lats, 90)) if sorted_lats else 0.0
        p95_ms = float(np.percentile(sorted_lats, 95)) if sorted_lats else 0.0
        p99_ms = float(np.percentile(sorted_lats, 99)) if sorted_lats else 0.0
        min_ms = float(sorted_lats[0]) if sorted_lats else 0.0
        max_ms = float(sorted_lats[-1]) if sorted_lats else 0.0
        throughput = (total / wall_elapsed) if wall_elapsed > 0 else 0.0

        latency_profile = LatencyProfile(
            mean_ms=mean_ms,
            median_ms=median_ms,
            p90_ms=p90_ms,
            p95_ms=p95_ms,
            p99_ms=p99_ms,
            min_ms=min_ms,
            max_ms=max_ms,
            throughput_qps=throughput,
            total_wall_clock_sec=wall_elapsed,
        )

        # Threat Category Breakdown
        by_cat: Dict[str, List[SampleEvaluationResult]] = {}
        for r in results:
            by_cat.setdefault(r.category, []).append(r)

        category_metrics: Dict[str, Dict[str, Any]] = {}
        for cat, group in by_cat.items():
            cat_tot = len(group)
            cat_tp = sum(1 for r in group if r.expected_action == "block" and r.actual_action == "block")
            cat_fp = sum(1 for r in group if r.expected_action == "allow" and r.actual_action == "block")
            cat_tn = sum(1 for r in group if r.expected_action == "allow" and r.actual_action == "allow")
            cat_fn = sum(1 for r in group if r.expected_action == "block" and r.actual_action != "block")
            cat_passed = sum(1 for r in group if r.passed)
            cat_p = cat_tp / (cat_tp + cat_fp) if (cat_tp + cat_fp) > 0 else 0.0
            cat_r = cat_tp / (cat_tp + cat_fn) if (cat_tp + cat_fn) > 0 else 0.0
            cat_f1 = (2 * cat_p * cat_r) / (cat_p + cat_r) if (cat_p + cat_r) > 0 else 0.0
            cat_acc = (cat_tp + cat_tn) / cat_tot if cat_tot > 0 else 0.0
            cat_mean_lat = float(np.mean([r.latency_ms for r in group])) if group else 0.0

            category_metrics[cat] = {
                "total": cat_tot,
                "passed": cat_passed,
                "failed": cat_tot - cat_passed,
                "tp": cat_tp,
                "fp": cat_fp,
                "tn": cat_tn,
                "fn": cat_fn,
                "accuracy": round(cat_acc, 4),
                "precision": round(cat_p, 4),
                "recall": round(cat_r, 4),
                "f1_score": round(cat_f1, 4),
                "mean_latency_ms": round(cat_mean_lat, 4),
            }

        return BenchmarkEvaluationResult(
            benchmark_name=benchmark_name,
            total_samples=total,
            passed_samples=passed_count,
            failed_samples=failed_count,
            confusion_matrix=cm,
            precision=cm.precision,
            recall=cm.recall,
            f1_score=cm.f1_score,
            accuracy=cm.accuracy,
            false_positive_rate=cm.false_positive_rate,
            latency_profile=latency_profile,
            category_metrics=category_metrics,
            detailed_results=results,
        )

    def evaluate_dataset(
        self,
        name_or_path: Union[str, Path],
        benchmark_name: Optional[str] = None,
    ) -> BenchmarkEvaluationResult:
        """Load a benchmark from name/path and evaluate it."""
        samples = load_benchmark(name_or_path)
        bench_name = (
            benchmark_name
            or (Path(name_or_path).stem if isinstance(name_or_path, (str, Path)) else "benchmark")
        )
        return self.evaluate(samples, benchmark_name=bench_name)


# ─── CLI Runner Helper ────────────────────────────────────────────────────────

def run_cli(args: Optional[Sequence[str]] = None) -> int:
    """CLI runner helper for benchmark evaluations."""
    parser = argparse.ArgumentParser(
        description="VaxGuard Public Benchmark Evaluator CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--benchmark",
        choices=["advglue", "donotanswer", "harmbench", "jailbreakbench", "hard_framing_suite", "all"],
        default="all",
        help="Benchmark dataset to evaluate.",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json", "both"],
        default="markdown",
        help="Report output format to display.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save report files (.md and .json).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional detection threshold override.",
    )

    parsed = parser.parse_args(args)
    evaluator = BenchmarkEvaluator(threshold=parsed.threshold)

    targets = (
        ["advglue", "donotanswer", "harmbench", "jailbreakbench", "hard_framing_suite"]
        if parsed.benchmark == "all"
        else [parsed.benchmark]
    )
    all_results: List[BenchmarkEvaluationResult] = []
    combined_samples: List[BenchmarkSample] = []

    for target in targets:
        try:
            samples = load_benchmark(target)
            combined_samples.extend(samples)
            result = evaluator.evaluate(samples, benchmark_name=target)
            all_results.append(result)

            if parsed.format in ("markdown", "both"):
                print(result.to_markdown())
            if parsed.format in ("json", "both"):
                print(result.to_json(indent=2))

            if parsed.output_dir:
                out_path = Path(parsed.output_dir)
                out_path.mkdir(parents=True, exist_ok=True)
                (out_path / f"{target}_report.md").write_text(result.to_markdown(), encoding="utf-8")
                (out_path / f"{target}_report.json").write_text(result.to_json(indent=2), encoding="utf-8")
        except Exception as exc:
            print(f"Error evaluating benchmark '{target}': {exc}", file=sys.stderr)
            return 1

    if parsed.benchmark == "all" and len(all_results) > 1:
        # Generate combined all-benchmarks summary
        combined_result = evaluator.evaluate(
            combined_samples, benchmark_name="ALL_COMBINED_BENCHMARKS", warmup=False
        )
        print("\n" + "=" * 80)
        print("OVERALL CONSOLIDATED BENCHMARK REPORT")
        print("=" * 80 + "\n")
        if parsed.format in ("markdown", "both"):
            print(combined_result.to_markdown())
        if parsed.format in ("json", "both"):
            print(combined_result.to_json(indent=2))

        if parsed.output_dir:
            out_path = Path(parsed.output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            (out_path / "combined_report.md").write_text(combined_result.to_markdown(), encoding="utf-8")
            (out_path / "combined_report.json").write_text(combined_result.to_json(indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    sys.exit(run_cli())
