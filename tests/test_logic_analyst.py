# tests/test_logic_analyst.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.agents.base import AnalysisContext, AnalysisDepth
from src.agents.logic_analyst import LogicAnalystAgent


def _make_agent():
    """Helper to create a LogicAnalystAgent with mocked constructor deps."""
    with patch("src.agents.logic_analyst.get_settings") as mock_settings, \
         patch("src.agents.logic_analyst.DeepSeekAdapter"), \
         patch("src.agents.logic_analyst.CaseLibrary"), \
         patch("src.agents.logic_analyst.KnowledgeRetriever"):
        mock_settings.return_value = MagicMock(
            deepseek_api_key="test",
            deepseek_base_url="https://api.deepseek.com",
            llm_model="deepseek-chat",
            llm_temperature=0.3,
        )
        agent = LogicAnalystAgent()
    return agent


def test_logic_analyst_name():
    agent = _make_agent()
    assert agent.name == "logic_analyst"


def test_logic_analyst_description():
    agent = _make_agent()
    assert len(agent.description) > 10


def test_analyze_argument_structure():
    """测试论证结构分析"""
    agent = _make_agent()
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = MagicMock(
        content='{"argument_structure": {"premises": ["所有人都会死"], "conclusion": "苏格拉底会死"}, "argument_strength": 0.9, "confidence": 0.8}'
    )
    agent._llm = mock_llm
    agent._retriever = MagicMock()
    agent._retriever.get_context_string.return_value = ""

    ctx = AnalysisContext(text="所有人都会死，苏格拉底是人，所以苏格拉底会死", depth=AnalysisDepth.STANDARD)
    result = agent.analyze(ctx)
    assert result.agent_name == "logic_analyst"
    assert result.confidence > 0


def test_analyze_fallacy_detection():
    """测试逻辑谬误检测"""
    agent = _make_agent()
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = MagicMock(
        content='{"fallacies": [{"type": "straw_man", "severity": "high"}], "confidence": 0.8}'
    )
    agent._llm = mock_llm
    agent._retriever = MagicMock()
    agent._retriever.get_context_string.return_value = "稻草人谬误"

    ctx = AnalysisContext(text="你支持环保？那你想回到原始社会？", depth=AnalysisDepth.STANDARD)
    result = agent.analyze(ctx)
    assert result.confidence > 0
