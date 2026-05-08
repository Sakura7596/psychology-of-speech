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
