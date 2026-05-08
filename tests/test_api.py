# tests/test_api.py
import pytest
from src.api.models import AnalyzeRequest, AnalyzeResponse, HealthResponse


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
