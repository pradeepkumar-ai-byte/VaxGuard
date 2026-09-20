import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from vaxguard.api.app import app
from vaxguard.db.session import init_db


@pytest.fixture(scope="module", autouse=True)
def setup_test_db():
    import asyncio
    asyncio.run(init_db())


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_api_status(client):
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"
    assert "immunity_score" in data
    assert "active_vaccines" in data
    assert "target_model" in data
    assert len(data["target_model"]) > 0


def test_api_report(client):
    response = client.get("/api/report")
    assert response.status_code == 200
    data = response.json()
    assert "immunity_score" in data
    assert "total_attacks" in data
    assert "category_scores" in data


def test_api_threats(client):
    response = client.get("/api/threats")
    assert response.status_code == 200
    data = response.json()
    assert "threats" in data
    assert isinstance(data["threats"], list)


@patch("vaxguard.core.llm_client.VaxGuardLLM.generate", new_callable=AsyncMock)
def test_api_interact(mock_llm_gen, client):
    mock_llm_gen.return_value = "This is a safe and helpful response."

    payload = {
        "prompt": "Hello! How do I make a cup of tea?",
        "system_prompt": "You are a helpful assistant.",
    }
    response = client.post("/api/interact", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "This is a safe and helpful response."
    assert "anomaly_score" in data
    assert data["is_threat"] is False
    assert "latency_ms" in data


@patch("vaxguard.core.llm_client.VaxGuardLLM.generate", new_callable=AsyncMock)
@patch("vaxguard.engine.classifier.SeverityClassifier.evaluate", new_callable=AsyncMock)
def test_api_demo_attack(mock_eval, mock_llm_gen, client):
    mock_llm_gen.return_value = "Refusing to comply with harmful instruction."
    mock_eval.return_value = {
        "breached": False,
        "confidence": 9,
        "reasoning": "Model refused appropriately.",
    }

    payload = {
        "apply_defense": True,
    }
    response = client.post("/api/demo/attack", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["breached"] is False
    assert data["defense_active"] is True
    assert "attack_name" in data
