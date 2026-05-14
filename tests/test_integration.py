# tests/test_integration.py
import pytest
from unittest.mock import AsyncMock
from src.agents.base import AnalysisContext, AnalysisDepth, AgentResult
from src.agents.orchestrator import Orchestrator


@pytest.mark.asyncio
async def test_end_to_end_mock():
    """端到端测试（全部 mock）"""
    orchestrator = Orchestrator()

    def make_mock(name, data):
        mock = AsyncMock()
        mock.analyze.return_value = AgentResult(
            agent_name=name, analysis=data, confidence=0.8, sources=["mock"]
        )
        return mock

    agents = {
        "text_analyst": make_mock("text_analyst", {"tokens": 20, "sentiment": "positive"}),
        "psychology_analyst": make_mock("psychology_analyst", {"intent": "express"}),
        "logic_analyst": make_mock("logic_analyst", {"fallacies": []}),
        "report_generator": make_mock("report_generator", {"report": "完整分析报告"}),
    }

    ctx = AnalysisContext(text="今天天气真好，我们去散步吧", depth=AnalysisDepth.STANDARD)
    result = await orchestrator.run_pipeline(ctx, agents)

    assert result is not None
    assert result.agent_name == "report_generator"
    assert result.confidence > 0


@pytest.mark.asyncio
async def test_end_to_end_with_error():
    """端到端测试（部分 Agent 失败）"""
    orchestrator = Orchestrator()

    good = AsyncMock()
    good.analyze.return_value = AgentResult(
        agent_name="text_analyst", analysis={"tokens": 10}, confidence=0.8, sources=["mock"]
    )

    bad = AsyncMock()
    bad.analyze.side_effect = Exception("LLM 调用失败")

    agents = {
        "text_analyst": good,
        "psychology_analyst": bad,
        "logic_analyst": good,
        "report_generator": AsyncMock(analyze=AsyncMock(return_value=AgentResult(
            agent_name="report_generator", analysis={"report": "部分完成"}, confidence=0.5, sources=["mock"]
        ))),
    }

    ctx = AnalysisContext(text="测试", depth=AnalysisDepth.STANDARD)
    result = await orchestrator.run_pipeline(ctx, agents)
    assert result is not None
