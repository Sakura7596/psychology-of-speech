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


RELIABILITY_THRESHOLD = 0.3


@dataclass
class AgentResult:
    """Agent 分析结果"""
    agent_name: str
    analysis: dict
    confidence: float
    sources: list[str]
    errors: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {self.confidence}")

    @property
    def is_reliable(self) -> bool:
        """置信度 >= RELIABILITY_THRESHOLD 视为可靠"""
        return self.confidence >= RELIABILITY_THRESHOLD


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
