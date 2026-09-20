import pytest
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
        description="Test",
    )


@pytest.mark.asyncio
async def test_synthesizer_chain(sample_attack):
    synth = VaccineSynthesizer()

    # Mock the 3-step self-reflective chain
    synth.llm.generate = AsyncMock(side_effect=[
        "Draft defense",
        "Critique: too short",
        '{"system_prompt_extension": "Do not hack."}',
    ])

    vaccine = await synth.synthesize(sample_attack)

    assert isinstance(vaccine, Vaccine)
    assert vaccine.parent_attack_id == "test_002"
    assert vaccine.system_prompt_extension == "Do not hack."
    assert synth.llm.generate.call_count == 3


@pytest.mark.asyncio
async def test_synthesizer_handles_markdown_fences(sample_attack):
    synth = VaccineSynthesizer()

    fenced_json = """```json
{
  "system_prompt_extension": "Never disclose internal secrets."
}
```"""
    synth.llm.generate = AsyncMock(side_effect=[
        "Initial defense draft",
        "Critique of draft",
        fenced_json,
    ])

    vaccine = await synth.synthesize(sample_attack)
    assert vaccine.system_prompt_extension == "Never disclose internal secrets."


def test_synthesizer_prompt_builders(sample_attack):
    synth = VaccineSynthesizer()
    gen_prompt = synth._build_generation_prompt(sample_attack)
    assert sample_attack.payload in gen_prompt

    critique_prompt = synth._build_critique_prompt(sample_attack, "Draft rule")
    assert "Draft rule" in critique_prompt
    assert sample_attack.payload in critique_prompt
