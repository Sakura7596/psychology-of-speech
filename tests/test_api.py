# tests/test_api.py
import os
import pytest
from src.api.models import AnalyzeRequest, AnalyzeResponse, HealthResponse

# Set dummy API key so agent initialization doesn't crash during tests
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-for-testing")


# --- Task 34: API Models ---

def test_analyze_request_creation():
    req = AnalyzeRequest(text="今天天气真好")
    assert req.text == "今天天气真好"
    assert req.depth == "standard"
    assert req.output_format == "markdown"


def test_analyze_request_validation():
    with pytest.raises(Exception):
        AnalyzeRequest(text="")


def test_analyze_response_creation():
    resp = AnalyzeResponse(report="报告", analyses={}, confidence=0.8, depth="standard")
    assert resp.confidence == 0.8


def test_health_response():
    resp = HealthResponse(status="ok", version="0.1.0")
    assert resp.status == "ok"


# --- Task 35: FastAPI Routes ---

from fastapi.testclient import TestClient
from src.api.app import create_app


def test_health_endpoint():
    app = create_app()
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_analyze_endpoint():
    app = create_app()
    client = TestClient(app)
    response = client.post("/analyze", json={"text": "今天天气真好", "depth": "standard"})
    assert response.status_code == 200
    data = response.json()
    assert "report" in data
    assert "confidence" in data


def test_analyze_empty_text():
    app = create_app()
    client = TestClient(app)
    response = client.post("/analyze", json={"text": ""})
    assert response.status_code == 422


def test_analyze_invalid_depth():
    app = create_app()
    client = TestClient(app)
    response = client.post("/analyze", json={"text": "测试", "depth": "invalid"})
    assert response.status_code == 400


# --- Task 36: Observability ---

def test_error_handling():
    app = create_app()
    client = TestClient(app)
    response = client.post("/analyze", content="not json", headers={"Content-Type": "application/json"})
    assert response.status_code == 422
