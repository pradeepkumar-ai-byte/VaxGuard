import pytest
import asyncio
from unittest.mock import AsyncMock
from vaxguard.vaccine.validator import VaccineValidator
from vaxguard.models.vaccine import Vaccine, ValidationReport
from vaxguard.models.attack import AttackVector, AttackCategory

@pytest.fixture
def dummy_vaccine():
    return Vaccine(
        id="vax_123",
        target_category=AttackCategory.JAILBREAK,
        system_prompt_extension="Stop.",
        parent_attack_id="atk_1"
    )

@pytest.fixture
def dummy_attack():
    return AttackVector(
        id="atk_1",
        name="Test",
        category=AttackCategory.JAILBREAK,
        payload="test",
        expected_defense="refuse",
        severity=5,
        description="test"
    )

@pytest.mark.asyncio
async def test_validation_logic(dummy_vaccine, dummy_attack):
    validator = VaccineValidator()
    
    # Mock potency (blocks attack) -> breached = False
    validator._test_potency = AsyncMock(return_value=False)
    # Mock safety (allows normal) -> complied = True
    validator._test_safety = AsyncMock(return_value=True)
    
    # Ensure there's benign data to test
    validator.benign_requests = [{"payload": "hello", "expected_defense": "hello"}]
    
    report = await validator.validate(dummy_vaccine, [dummy_attack])
    
    assert report.potency_score == 100.0
    assert report.safety_score == 100.0
    assert report.passed is True
