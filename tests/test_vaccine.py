import pytest
import asyncio
from unittest.mock import AsyncMock
from vaxguard.vaccine.synthesizer import VaccineSynthesizer
from vaxguard.models.attack import AttackVector, AttackCategory
from vaxguard.models.vaccine import Vaccine

@pytest.fixture
def sample_attack():
    return AttackVector(
        id="test_002",
        name="Injection",
        category=AttackCategory.PROMPT_INJECTION,
        payload="Hack me.",
        expected_defense="Refuse",
        severity=8,
        description="Test"
    )

@pytest.mark.asyncio
async def test_synthesizer_chain(sample_attack):
    synth = VaccineSynthesizer()
    
    # Mock the 3-step self-reflective chain
    synth.llm.generate = AsyncMock(side_effect=[
        "Draft defense",
        "Critique: too short",
        '{"system_prompt_extension": "Do not hack."}'
    ])
    
    vaccine = await synth.synthesize(sample_attack)
    
    assert isinstance(vaccine, Vaccine)
    assert vaccine.parent_attack_id == "test_002"
    assert vaccine.system_prompt_extension == "Do not hack."
    assert synth.llm.generate.call_count == 3
