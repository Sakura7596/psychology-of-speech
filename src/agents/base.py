from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class AnalysisDepth(str, Enum):
    """分析深度"""
    QUICK = "quick"        # 快速摘要
    STANDARD = "standard"  # 标准报告
    DEEP = "deep"          # 深度分析


@dataclass
class TextFeatures:
    """NLP 特征提取结果"""
    tokens: list[str] = field(default_factory=list)
    sentences: list[str] = field(default_factory=list)
    pos_tags: list[tuple[str, str]] = field(default_factory=list)
    dependency_parse: list[dict] = field(default_factory=list)
    sentiment_score: float = 0.0
    entities: list[dict] = field(default_factory=list)
    rhetorical_devices: list[dict] = field(default_factory=list)
    discourse_markers: list[str] = field(default_factory=list)


@dataclass
class AnalysisContext:
    """Agent 分析上下文"""
    text: str
    depth: AnalysisDepth = AnalysisDepth.STANDARD
    language: str = "zh"
    features: TextFeatures | None = None
    metadata: dict = field(default_factory=dict)
    sibling_results: dict[str, "AgentResult"] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Agent 分析结果"""
    agent_name: str
    analysis: dict
    confidence: float
    sources: list[str]
    errors: list[str] = field(default_factory=list)

    @property
    def is_reliable(self) -> bool:
        """置信度 >= 0.3 视为可靠"""
        return self.confidence >= 0.3


class BaseAgent(ABC):
    """所有 Agent 的基类"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent 名称"""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Agent 描述"""
        ...

    @abstractmethod
    def analyze(self, context: AnalysisContext) -> AgentResult:
        """执行分析，返回结果"""
        ...
