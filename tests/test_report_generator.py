# tests/test_report_generator.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.agents.base import AnalysisContext, AnalysisDepth, AgentResult
from src.agents.report_generator import ReportGeneratorAgent


def test_report_generator_name():
    with patch("src.agents.report_generator.get_settings") as mock_settings, \
         patch("src.agents.report_generator.DeepSeekAdapter"), \
         patch("src.agents.report_generator.EthicsGuard"):
        mock_settings.return_value = MagicMock(
            deepseek_api_key="test",
            deepseek_base_url="https://api.deepseek.com",
            llm_model="deepseek-chat",
            llm_temperature=0.3,
        )
        agent = ReportGeneratorAgent()
    assert agent.name == "report_generator"


def test_report_generator_description():
    with patch("src.agents.report_generator.get_settings") as mock_settings, \
         patch("src.agents.report_generator.DeepSeekAdapter"), \
         patch("src.agents.report_generator.EthicsGuard"):
        mock_settings.return_value = MagicMock(
            deepseek_api_key="test",
            deepseek_base_url="https://api.deepseek.com",
            llm_model="deepseek-chat",
            llm_temperature=0.3,
        )
        agent = ReportGeneratorAgent()
    assert len(agent.description) > 10


def test_generate_markdown_report():
    with patch("src.agents.report_generator.get_settings") as mock_settings, \
         patch("src.agents.report_generator.DeepSeekAdapter"), \
         patch("src.agents.report_generator.EthicsGuard"):
        mock_settings.return_value = MagicMock(
            deepseek_api_key="test",
            deepseek_base_url="https://api.deepseek.com",
            llm_model="deepseek-chat",
            llm_temperature=0.3,
        )
        agent = ReportGeneratorAgent()

    mock_llm = AsyncMock()
    mock_llm.generate.return_value = MagicMock(
        content="# 分析报告\n\n## 摘要\n比喻修辞。\n\n---\n**免责声明**：仅供参考。"
    )
    agent._llm = mock_llm

    sibling_results = {
        "text_analyst": AgentResult(
            agent_name="text_analyst",
            analysis={"rhetorical_devices": [{"type": "simile"}]},
            confidence=0.8,
            sources=["HanLP"],
        ),
    }

    ctx = AnalysisContext(text="他的心像冰一样冷", depth=AnalysisDepth.STANDARD, sibling_results=sibling_results)
    result = agent.analyze(ctx)
    assert result.agent_name == "report_generator"
    assert "report" in result.analysis


def test_generate_json_output():
    with patch("src.agents.report_generator.get_settings") as mock_settings, \
         patch("src.agents.report_generator.DeepSeekAdapter"), \
         patch("src.agents.report_generator.EthicsGuard"):
        mock_settings.return_value = MagicMock(
            deepseek_api_key="test",
            deepseek_base_url="https://api.deepseek.com",
            llm_model="deepseek-chat",
            llm_temperature=0.3,
        )
        agent = ReportGeneratorAgent()

    mock_llm = AsyncMock()
    mock_llm.generate.return_value = MagicMock(
        content='{"summary": "比喻", "sections": {}}'
    )
    agent._llm = mock_llm

    ctx = AnalysisContext(text="测试", depth=AnalysisDepth.STANDARD, sibling_results={}, metadata={"output_format": "json"})
    result = agent.analyze(ctx)
    assert result.agent_name == "report_generator"


def test_report_includes_disclaimer():
    with patch("src.agents.report_generator.get_settings") as mock_settings, \
         patch("src.agents.report_generator.DeepSeekAdapter"):
        mock_settings.return_value = MagicMock(
            deepseek_api_key="test",
            deepseek_base_url="https://api.deepseek.com",
            llm_model="deepseek-chat",
            llm_temperature=0.3,
        )
        agent = ReportGeneratorAgent()

    mock_llm = AsyncMock()
    mock_llm.generate.return_value = MagicMock(content="分析结果。")
    agent._llm = mock_llm

    ctx = AnalysisContext(text="测试", depth=AnalysisDepth.STANDARD, sibling_results={})
    result = agent.analyze(ctx)
    report = str(result.analysis)
    assert "免责" in report or "仅供参考" in report
