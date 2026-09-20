import pytest
from pydantic import ValidationError
from vaxguard.models.attack import AttackVector, AttackCategory
from vaxguard.models.vaccine import Vaccine, ValidationReport
from vaxguard.models.immune import AnomalyReport, SeverityLevel, BehavioralProfile, InteractionEvent
from vaxguard.models.report import VulnerabilityReport, DiagnosticResult


def test_attack_vector_validation_success():
    atk = AttackVector(
        id="atk_schema_01",
        name="Valid Attack",
        category=AttackCategory.TOKEN_SMUGGLING,
        payload="Base64 smuggling payload",
        expected_defense="Decode and block",
        severity=7,
        description="Base64 smuggling test vector",
    )
    assert atk.severity == 7
    assert atk.category == AttackCategory.TOKEN_SMUGGLING


def test_attack_vector_invalid_severity():
    with pytest.raises(ValidationError):
        AttackVector(
            id="atk_bad",
            name="Bad Severity",
            category=AttackCategory.JAILBREAK,
            payload="text",
            expected_defense="text",
            severity=15,  # Exceeds max 10
        )


def test_vaccine_defaults():
    vax = Vaccine(
        id="vax_def_01",
        target_category=AttackCategory.PROMPT_INJECTION,
        system_prompt_extension="No bypass.",
    )
    assert vax.version == 1
    assert vax.parent_attack_id is None


def test_anomaly_report_score_bounds():
    report = AnomalyReport(
        interaction_id="evt_01",
        prompt="Sample test",
        anomaly_score=75.5,
        semantic_distance=0.6,
        severity=SeverityLevel.HIGH,
        is_threat=True,
    )
    assert report.anomaly_score == 75.5
    assert report.is_threat is True

    # Out-of-bounds anomaly score must raise ValidationError
    with pytest.raises(ValidationError):
        AnomalyReport(
            interaction_id="evt_bad",
            prompt="Sample",
            anomaly_score=150.0,  # Max 100.0
            semantic_distance=0.5,
            severity=SeverityLevel.CRITICAL,
            is_threat=True,
        )


def test_validation_report_properties():
    val_pass = ValidationReport(
        vaccine_id="vax_p",
        potency_score=85.0,
        safety_score=100.0,
        passed=True,
    )
    assert val_pass.passed is True
    assert val_pass.potency_score == 85.0


def test_vulnerability_report_immunity_calculation():
    vr = VulnerabilityReport(
        target_model="qwen/qwen3.8-27b",
        total_attacks=10,
        successful_breaches=2,
        immunity_score=80.0,
        results=[],
        category_scores={AttackCategory.PROMPT_INJECTION: 80.0},
    )
    assert vr.immunity_score == 80.0
    assert vr.successful_breaches == 2
