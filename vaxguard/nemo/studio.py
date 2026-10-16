"""
VaxGuard Inoculation Studio: Developer-Controlled Model Hardening & Colang Rule Compiler.

Central Controller & API managing the full developer-supervised lifecycle:
1. Benchmark Dataset -> Identifies unmitigated adversarial vectors.
2. Synthesize Rule -> Compiles formal Colang (.co) and Input Rail examples.
3. Quarantine Verification -> Executes shadow benchmark with 0% False Positive gate.
4. Developer Approval & Rails Commit -> Updates active NeMo configs with git-ready diffs.
"""

import yaml
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Optional, Union, Any
from pydantic import BaseModel, Field

from vaxguard.core.logger import get_logger
from vaxguard.models.attack import AttackVector, AttackCategory
from vaxguard.nemo.inoculation_compiler import (
    InoculationCompiler,
    SynthesizedColangRule,
)
from vaxguard.nemo.quarantine_sandbox import (
    QuarantineSandbox,
    QuarantineCertificate,
    SandboxTestSuite,
)

logger = get_logger("InoculationStudio")


class DeveloperApprovalRequiredError(PermissionError):
    """Raised when an attempt is made to commit rails without explicit developer approval."""
    pass


class QuarantineVerificationError(RuntimeError):
    """Raised when an unverified rule or one with false positive regression attempts commitment."""
    pass


class BenchmarkReport(BaseModel):
    """Comprehensive benchmark results evaluating adversarial vectors."""
    total_vectors_tested: int
    blocked_count: int
    vulnerable_count: int
    block_rate: float
    vulnerable_vectors: List[AttackVector]
    category_breakdown: Dict[str, Dict[str, int]]
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class CommitResult(BaseModel):
    """Summary of a successful Colang rule deployment to active NeMo Guardrails configuration."""
    rule_id: str
    target_config_dir: str
    target_files_updated: List[str]
    git_commit_message: str
    diff_summary: str
    committed_by: str
    timestamp: str


class InoculationStudio:
    """
    Developer-Controlled Inoculation Studio API & Workflow Orchestrator.
    
    Provides explicit developer endpoints for:
    - `benchmark_dataset(dataset)`
    - `synthesize_rule_for_vector(attack_vector)`
    - `verify_in_quarantine(rule, test_suite)`
    - `commit_rule_to_rails(rule_id, target_config_path, developer_approval=True)`
    """

    def __init__(
        self,
        compiler: Optional[InoculationCompiler] = None,
        sandbox: Optional[QuarantineSandbox] = None,
    ):
        self.compiler = compiler or InoculationCompiler()
        self.sandbox = sandbox or QuarantineSandbox()
        
        # In-memory studio registries
        self._rules_catalog: Dict[str, SynthesizedColangRule] = {}
        self._certificates_catalog: Dict[str, QuarantineCertificate] = {}
        self._vector_map: Dict[str, AttackVector] = {}

    # ─── 1. Benchmark Dataset ──────────────────────────────────────────────────
    def benchmark_dataset(
        self,
        dataset: Union[str, Path, List[Union[AttackVector, Dict[str, Any]]]],
    ) -> BenchmarkReport:
        """
        Runs a comprehensive benchmark across an attack dataset to detect security gaps.
        Accepts:
        - File path to YAML attack library (e.g., 'data/attacks.yaml')
        - List of AttackVector objects or raw dictionaries
        """
        logger.info("Executing benchmark dataset evaluation...")
        vectors: List[AttackVector] = self._resolve_dataset(dataset)

        vulnerable: List[AttackVector] = []
        blocked: List[AttackVector] = []
        category_breakdown: Dict[str, Dict[str, int]] = {}

        for vec in vectors:
            self._vector_map[vec.id] = vec
            cat_name = vec.category.value if isinstance(vec.category, AttackCategory) else str(vec.category)
            if cat_name not in category_breakdown:
                category_breakdown[cat_name] = {"total": 0, "blocked": 0, "vulnerable": 0}
            category_breakdown[cat_name]["total"] += 1

            # Evaluate against currently committed / active rules in catalog
            is_defended = self._check_if_defended(vec)
            if is_defended:
                blocked.append(vec)
                category_breakdown[cat_name]["blocked"] += 1
            else:
                vulnerable.append(vec)
                category_breakdown[cat_name]["vulnerable"] += 1

        total = len(vectors)
        block_rate = (len(blocked) / total * 100.0) if total > 0 else 100.0

        report = BenchmarkReport(
            total_vectors_tested=total,
            blocked_count=len(blocked),
            vulnerable_count=len(vulnerable),
            block_rate=round(block_rate, 2),
            vulnerable_vectors=vulnerable,
            category_breakdown=category_breakdown,
        )

        logger.info(
            f"Benchmark finished: {report.blocked_count}/{report.total_vectors_tested} blocked "
            f"({report.block_rate}%). {report.vulnerable_count} vulnerable vectors flagged."
        )
        return report

    # ─── 2. Synthesize Rule for Vector ─────────────────────────────────────────
    def synthesize_rule_for_vector(
        self,
        attack_vector: Union[AttackVector, Dict[str, Any]],
        author: str = "developer",
    ) -> SynthesizedColangRule:
        """
        Synthesizes formal Colang rule and input rail semantic specifications.
        Always executed with explicit developer context.
        """
        vec = self._to_attack_vector(attack_vector)
        self._vector_map[vec.id] = vec

        rule = self.compiler.compile(
            attack=vec,
            developer_triggered=True,
            author=author,
        )

        # Store in studio catalog
        self._rules_catalog[rule.rule_id] = rule
        return rule

    # ─── 3. Verify in Quarantine ───────────────────────────────────────────────
    def verify_in_quarantine(
        self,
        rule: Union[SynthesizedColangRule, str],
        test_suite: Optional[SandboxTestSuite] = None,
    ) -> QuarantineCertificate:
        """
        Runs candidate Colang rule through shadow Quarantine Sandbox.
        Enforces 0% false positive regression on enterprise benign queries.
        """
        colang_rule: SynthesizedColangRule
        if isinstance(rule, str):
            if rule not in self._rules_catalog:
                raise KeyError(f"Rule ID '{rule}' not found in Inoculation Studio catalog.")
            colang_rule = self._rules_catalog[rule]
        else:
            colang_rule = rule
            self._rules_catalog[colang_rule.rule_id] = colang_rule

        parent_vec = self._vector_map.get(colang_rule.parent_attack_id)

        cert = self.sandbox.run_benchmark(
            rule=colang_rule,
            attack=parent_vec,
            custom_test_suite=test_suite,
        )

        # Update rule status based on quarantine outcome
        if cert.passed:
            colang_rule.status = "verified"
            logger.info(f"Rule {colang_rule.rule_id} marked as VERIFIED (0% FPR). Ready for review.")
        else:
            colang_rule.status = "rejected"
            logger.warning(
                f"Rule {colang_rule.rule_id} REJECTED in quarantine "
                f"(FPR: {cert.false_positive_rate}%, Potency: {cert.potency_score}%)."
            )

        self._certificates_catalog[colang_rule.rule_id] = cert
        return cert

    # ─── 4. Commit Rule to Rails ───────────────────────────────────────────────
    def commit_rule_to_rails(
        self,
        rule_id: str,
        target_config_path: Union[str, Path],
        developer_approval: bool = True,
        approved_by: str = "developer",
    ) -> CommitResult:
        """
        Commits a verified Colang rule to the active NeMo Guardrails configuration.
        
        Strict Safety Controls:
        1. developer_approval must be True.
        2. Rule must possess a valid, passing QuarantineCertificate with 0% FPR.
        """
        if not developer_approval:
            raise DeveloperApprovalRequiredError(
                "Commit aborted: Explicit developer approval is required to update active guardrails."
            )

        if rule_id not in self._rules_catalog:
            raise KeyError(f"Rule ID '{rule_id}' not found in catalog.")

        rule = self._rules_catalog[rule_id]
        cert = self._certificates_catalog.get(rule_id)

        if not cert or not cert.passed or cert.false_positive_rate > 0.0:
            raise QuarantineVerificationError(
                f"Cannot commit unverified or failing rule '{rule_id}'. "
                f"Requires passed quarantine certificate with 0% false positives."
            )

        target_dir = Path(target_config_path)
        if target_dir.is_file():
            target_dir = target_dir.parent

        target_dir.mkdir(parents=True, exist_ok=True)
        rails_dir = target_dir / "rails"
        rails_dir.mkdir(parents=True, exist_ok=True)

        updated_files: List[str] = []

        # 1. Write or Append Colang (.co) File
        colang_file = rails_dir / "vaxguard_inoculated.co"
        header = "# =========================================================================\n# Active NeMo Colang Rails (Managed by VaxGuard Inoculation Studio)\n# =========================================================================\n\n"
        
        if not colang_file.exists():
            colang_file.write_text(header + rule.colang_content + "\n", encoding="utf-8")
        else:
            with open(colang_file, "a", encoding="utf-8") as f:
                f.write("\n" + rule.colang_content + "\n")
        updated_files.append(str(colang_file))

        # 2. Update config.yml for NeMo Guardrails
        config_file = target_dir / "config.yml"
        self._update_or_create_config_yaml(config_file, rule)
        updated_files.append(str(config_file))

        # Mark rule as committed
        rule.status = "committed"

        now_str = datetime.now(timezone.utc).isoformat()
        git_msg = (
            f"security(nemo): inoculate against {rule.category.value} via {rule.flow_name}\n\n"
            f"- Rule ID: {rule.rule_id}\n"
            f"- Quarantine Certificate: {cert.certificate_id} (Safety: {cert.safety_score}%, Potency: {cert.potency_score}%)\n"
            f"- Approved by: {approved_by} at {now_str}"
        )

        result = CommitResult(
            rule_id=rule.rule_id,
            target_config_dir=str(target_dir),
            target_files_updated=updated_files,
            git_commit_message=git_msg,
            diff_summary=cert.diff_content,
            committed_by=approved_by,
            timestamp=now_str,
        )

        logger.info(f"✅ Rule {rule.rule_id} committed to {target_dir}. Ready for git commit.")
        return result

    # ─── Full Pipeline Utility ─────────────────────────────────────────────────
    def run_hardening_cycle(
        self,
        attack_vector: Union[AttackVector, Dict[str, Any]],
        target_config_path: Union[str, Path],
        developer_approval: bool = True,
        author: str = "developer",
    ) -> Dict[str, Any]:
        """
        One-click end-to-end hardening cycle for developer CLI / UI:
        1. Synthesize rule.
        2. Verify in shadow quarantine.
        3. Commit to active config upon developer approval.
        """
        rule = self.synthesize_rule_for_vector(attack_vector, author=author)
        cert = self.verify_in_quarantine(rule)
        
        commit_res = None
        if cert.passed and developer_approval:
            commit_res = self.commit_rule_to_rails(
                rule_id=rule.rule_id,
                target_config_path=target_config_path,
                developer_approval=True,
                approved_by=author,
            )

        return {
            "rule": rule,
            "certificate": cert,
            "commit_result": commit_res,
        }

    # ─── Query & Inspection Helpers ────────────────────────────────────────────
    def list_rules(self) -> List[SynthesizedColangRule]:
        """Returns all rules currently staged or committed in the studio."""
        return list(self._rules_catalog.values())

    def get_rule(self, rule_id: str) -> Optional[SynthesizedColangRule]:
        return self._rules_catalog.get(rule_id)

    def get_certificate(self, rule_id: str) -> Optional[QuarantineCertificate]:
        return self._certificates_catalog.get(rule_id)

    # ─── Internal Helpers ──────────────────────────────────────────────────────
    def _check_if_defended(self, vec: AttackVector) -> bool:
        """Checks if any committed rule defends against this attack vector."""
        committed_rules = [r for r in self._rules_catalog.values() if r.status == "committed"]
        for r in committed_rules:
            if self.sandbox._eval_query_against_rule(vec.payload, r):
                return True
        return False

    def _resolve_dataset(
        self,
        dataset: Union[str, Path, List[Union[AttackVector, Dict[str, Any]]]],
    ) -> List[AttackVector]:
        """Parses and standardizes dataset into AttackVector list."""
        if isinstance(dataset, (str, Path)):
            dataset_path = Path(dataset)
            if not dataset_path.exists():
                raise FileNotFoundError(f"Attack dataset not found at {dataset_path}")
            with open(dataset_path, "r", encoding="utf-8") as f:
                raw_data = yaml.safe_load(f) or []
            return [self._to_attack_vector(item) for item in raw_data]

        return [self._to_attack_vector(item) for item in dataset]

    def _to_attack_vector(self, item: Union[AttackVector, Dict[str, Any]]) -> AttackVector:
        if isinstance(item, AttackVector):
            return item
        return AttackVector(**item)

    def _update_or_create_config_yaml(self, config_path: Path, rule: SynthesizedColangRule) -> None:
        """Updates or creates NeMo Guardrails config.yml to include new flow in input rails."""
        config_data: Dict[str, Any] = {}
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config_data = yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning(f"Could not parse existing config.yml: {e}. Reinitializing.")

        # Ensure rails structure
        rails = config_data.setdefault("rails", {})
        input_rails = rails.setdefault("input", {})
        flows = input_rails.setdefault("flows", [])

        if rule.flow_name not in flows:
            flows.append(rule.flow_name)

        # Write formatted YAML
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
