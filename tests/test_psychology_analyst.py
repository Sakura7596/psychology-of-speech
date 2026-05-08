# tests/test_psychology_analyst.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.agents.base import AnalysisContext, AnalysisDepth
from src.agents.psychology_analyst import PsychologyAnalystAgent


def test_psychology_analyst_name():
    with patch("src.agents.psychology_analyst.get_settings") as mock_settings, \
         patch("src.agents.psychology_analyst.DeepSeekAdapter"), \
         patch("src.agents.psychology_analyst.KnowledgeGraph"), \
         patch("src.agents.psychology_analyst.CaseLibrary"), \
         patch("src.agents.psychology_analyst.KnowledgeRetriever"):
        mock_settings.return_value = MagicMock(
            deepseek_api_key="test",
            deepseek_base_url="https://api.deepseek.com",
            llm_model="deepseek-chat",
            llm_temperature=0.3,
        )
        agent = PsychologyAnalystAgent()
    assert agent.name == "psychology_analyst"


def test_psychology_analyst_description():
    with patch("src.agents.psychology_analyst.get_settings") as mock_settings, \
         patch("src.agents.psychology_analyst.DeepSeekAdapter"), \
         patch("src.agents.psychology_analyst.KnowledgeGraph"), \
         patch("src.agents.psychology_analyst.CaseLibrary"), \
         patch("src.agents.psychology_analyst.KnowledgeRetriever"):
        mock_settings.return_value = MagicMock(
            deepseek_api_key="test",
            deepseek_base_url="https://api.deepseek.com",
            llm_model="deepseek-chat",
            llm_temperature=0.3,
        )
        agent = PsychologyAnalystAgent()
    assert len(agent.description) > 10


def test_analyze_speech_act():
    """测试言语行为分析"""
    with patch("src.agents.psychology_analyst.get_settings") as mock_settings, \
         patch("src.agents.psychology_analyst.DeepSeekAdapter"), \
         patch("src.agents.psychology_analyst.KnowledgeGraph"), \
         patch("src.agents.psychology_analyst.CaseLibrary"), \
         patch("src.agents.psychology_analyst.KnowledgeRetriever"):
        mock_settings.return_value = MagicMock(
            deepseek_api_key="test",
            deepseek_base_url="https://api.deepseek.com",
            llm_model="deepseek-chat",
            llm_temperature=0.3,
        )
        agent = PsychologyAnalystAgent()

    mock_llm = AsyncMock()
    mock_llm.generate.return_value = MagicMock(
        content='{"speech_acts": [{"type": "assertive", "text": "天气好", "confidence": 0.8}], "overall_intent": "表达观点", "confidence": 0.7}'
    )
    agent._llm = mock_llm
    agent._retriever = MagicMock()
    agent._retriever.get_context_string.return_value = "言语行为理论"

    ctx = AnalysisContext(text="今天天气真好", depth=AnalysisDepth.STANDARD)
    result = agent.analyze(ctx)
    assert result.agent_name == "psychology_analyst"
    assert result.confidence > 0


def test_analyze_confidence_range():
    """测试置信度范围"""
    with patch("src.agents.psychology_analyst.get_settings") as mock_settings, \
         patch("src.agents.psychology_analyst.DeepSeekAdapter"), \
         patch("src.agents.psychology_analyst.KnowledgeGraph"), \
         patch("src.agents.psychology_analyst.CaseLibrary"), \
         patch("src.agents.psychology_analyst.KnowledgeRetriever"):
        mock_settings.return_value = MagicMock(
            deepseek_api_key="test",
            deepseek_base_url="https://api.deepseek.com",
            llm_model="deepseek-chat",
            llm_temperature=0.3,
        )
        agent = PsychologyAnalystAgent()

    mock_llm = AsyncMock()
    mock_llm.generate.return_value = MagicMock(
        content='{"analysis": "test", "confidence": 0.6}'
    )
    agent._llm = mock_llm
    agent._retriever = MagicMock()
    agent._retriever.get_context_string.return_value = ""

    ctx = AnalysisContext(text="测试", depth=AnalysisDepth.QUICK)
    result = agent.analyze(ctx)
    assert 0.0 <= result.confidence <= 1.0
