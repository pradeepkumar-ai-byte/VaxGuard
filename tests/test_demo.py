import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient

from vaxguard.api.app import app
from vaxguard.demo.orchestrator import DemoOrchestrator
from vaxguard.db.session import init_db
from vaxguard.models.vaccine import Vaccine, ValidationReport
from vaxguard.models.attack import AttackCategory, AttackVector


@pytest.fixture(scope="module", autouse=True)
def setup_test_db():
    import asyncio
    asyncio.run(init_db())


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_dashboard_route_serves_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "<title>VaxGuard" in response.text
    assert "Autonomous AI Immune System" in response.text


import uuid

@pytest.mark.asyncio
async def test_demo_orchestrator_pipeline():
    orchestrator = DemoOrchestrator()
    vax_id = f"vax_demo_{uuid.uuid4().hex[:6]}"

    # Mock LLM and Classifier to prevent actual network calls during tests
    orchestrator.llm.generate = AsyncMock(return_value="Simulated LLM response for demo.")
    orchestrator.classifier.evaluate = AsyncMock(return_value={
        "breached": True,
        "confidence": 9,
        "reasoning": "Simulated evaluation result.",
    })

    fake_vax = Vaccine(
        id=vax_id,
        target_category=AttackCategory.JAILBREAK,
        system_prompt_extension="Strict demo rule.",
        parent_attack_id="test_demo_parent",
    )
    orchestrator.synthesizer.synthesize = AsyncMock(return_value=fake_vax)
    orchestrator.validator.validate = AsyncMock(return_value=ValidationReport(
        vaccine_id=vax_id,
        potency_score=100.0,
        safety_score=100.0,
        passed=True,
    ))

    result = await orchestrator.run_demo_pipeline()

    assert result["status"] == "success"
    assert "pipeline_results" in result
    assert "step_1" in result["pipeline_results"]
    assert "step_2" in result["pipeline_results"]
    assert "step_3" in result["pipeline_results"]
    assert "step_4" in result["pipeline_results"]
