import pytest
from src.agents.base import (
    BaseAgent,
    AnalysisContext,
    AgentResult,
    TextFeatures,
    AnalysisDepth,
)


class MockAgent(BaseAgent):
    """用于测试的模拟 Agent"""

    @property
    def name(self) -> str:
        return "mock_agent"

    @property
    def description(self) -> str:
        return "A mock agent for testing"

    def analyze(self, context: AnalysisContext) -> AgentResult:
        return AgentResult(
            agent_name=self.name,
            analysis={"mock": True},
            confidence=0.8,
            sources=["test_source"],
        )


def test_analysis_context_creation():
    """测试 AnalysisContext 创建"""
    ctx = AnalysisContext(
        text="你好世界",
        depth=AnalysisDepth.STANDARD,
    )
    assert ctx.text == "你好世界"
    assert ctx.depth == AnalysisDepth.STANDARD
    assert ctx.metadata == {}


def test_analysis_context_with_features():
    """测试带特征的 AnalysisContext"""
    features = TextFeatures(
        tokens=["你好", "世界"],
        sentences=["你好世界"],
        sentiment_score=0.5,
    )
    ctx = AnalysisContext(
        text="你好世界",
        depth=AnalysisDepth.STANDARD,
        features=features,
    )
    assert ctx.features.tokens == ["你好", "世界"]
    assert ctx.features.sentiment_score == 0.5


def test_agent_result_creation():
    """测试 AgentResult 创建"""
    result = AgentResult(
        agent_name="test",
        analysis={"key": "value"},
        confidence=0.9,
        sources=["source1"],
    )
    assert result.agent_name == "test"
    assert result.confidence == 0.9
    assert result.is_reliable  # confidence >= 0.3


def test_agent_result_low_confidence():
    """测试低置信度结果"""
    result = AgentResult(
        agent_name="test",
        analysis={},
        confidence=0.2,
        sources=[],
    )
    assert not result.is_reliable  # confidence < 0.3


def test_mock_agent_analyze():
    """测试 MockAgent 的 analyze 方法"""
    agent = MockAgent()
    ctx = AnalysisContext(text="测试文本", depth=AnalysisDepth.QUICK)
    result = agent.analyze(ctx)
    assert result.agent_name == "mock_agent"
    assert result.analysis == {"mock": True}
    assert result.confidence == 0.8


def test_base_agent_is_abstract():
    """测试 BaseAgent 不能直接实例化"""
    with pytest.raises(TypeError):
        BaseAgent()


from src.agents.orchestrator import Orchestrator, AnalysisPlan


async def test_orchestrator_short_text_routing():
    """测试短文本动态路由（<100字）"""
    orchestrator = Orchestrator()
    plan = await orchestrator.plan_analysis("你好")
    assert plan.depth == AnalysisDepth.QUICK
    assert "logic_analyst" not in plan.agents or plan.agent_params.get(
        "logic_analyst", {}
    ).get("skip_deep", False)


async def test_orchestrator_medium_text_routing():
    """测试中等文本路由（100-500字）"""
    text = "这" * 150  # 150 字
    orchestrator = Orchestrator()
    plan = await orchestrator.plan_analysis(text)
    assert "text_analyst" in plan.agents
    assert "psychology_analyst" in plan.agents
    assert "logic_analyst" in plan.agents


async def test_orchestrator_long_text_routing():
    """测试长文本路由（>500字）"""
    text = "这" * 600  # 600 字
    orchestrator = Orchestrator()
    plan = await orchestrator.plan_analysis(text)
    assert plan.depth == AnalysisDepth.DEEP
    assert plan.segment is True


def test_analysis_plan_creation():
    """测试 AnalysisPlan 创建"""
    plan = AnalysisPlan(
        agents=["text_analyst", "psychology_analyst"],
        depth=AnalysisDepth.STANDARD,
        agent_params={"text_analyst": {"language": "zh"}},
    )
    assert len(plan.agents) == 2
    assert plan.depth == AnalysisDepth.STANDARD


@pytest.mark.asyncio
async def test_orchestrator_full_pipeline():
    """测试完整分析管道"""
    from unittest.mock import AsyncMock

    orchestrator = Orchestrator()

    def make_mock(name, data):
        mock = AsyncMock()
        mock.analyze.return_value = AgentResult(
            agent_name=name, analysis=data, confidence=0.8, sources=["mock"]
        )
        return mock

    agents = {
        "text_analyst": make_mock("text_analyst", {"tokens": 10}),
        "psychology_analyst": make_mock("psychology_analyst", {"intent": "express"}),
        "logic_analyst": make_mock("logic_analyst", {"fallacies": []}),
        "report_generator": make_mock("report_generator", {"report": "完整报告"}),
    }

    ctx = AnalysisContext(text="这是一段测试文本，用于验证协调器的完整管道。", depth=AnalysisDepth.STANDARD)
    result = await orchestrator.run_pipeline(ctx, agents)

    assert result is not None
    assert "report" in result.analysis or "analyses" in result
