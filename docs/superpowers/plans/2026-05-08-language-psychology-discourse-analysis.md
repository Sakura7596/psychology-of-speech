# 语言心理学话语分析系统 - 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个基于语言心理学的多智能体话语分析系统，从基础框架到完整 Web 服务分 6 个阶段交付

**Architecture:** 插件化单体 Python 应用，5 个 Agent 通过 LangGraph 编排协作，RAG 知识引擎增强分析，FastAPI 提供 Web 接口，Guardrails 层保障安全和伦理

**Tech Stack:** Python 3.11+, LangGraph, FastAPI, HanLP, transformers, ChromaDB, NetworkX, DeepSeek API, Pydantic Settings v2, pytest

---

## 文件结构总览

```
psychology-of-speech/
├── src/
│   ├── __init__.py
│   ├── main.py                     # CLI 入口
│   ├── config.py                   # Pydantic Settings 配置管理
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py                 # BaseAgent 抽象类 + 数据模型
│   │   ├── orchestrator.py         # 总协调器（动态路由）
│   │   ├── text_analyst.py         # 文本解析 Agent
│   │   ├── psychology_analyst.py   # 心理分析 Agent
│   │   ├── logic_analyst.py        # 逻辑推理 Agent
│   │   └── report_generator.py     # 报告生成 Agent
│   ├── nlp/
│   │   ├── __init__.py
│   │   ├── tokenizer.py            # HanLP + jieba 分词
│   │   ├── syntax.py               # HanLP 依存句法
│   │   ├── sentiment.py            # transformers 情感分析
│   │   └── rhetoric.py             # 修辞识别
│   ├── knowledge/
│   │   ├── __init__.py
│   │   ├── vector_store.py         # ChromaDB 向量库
│   │   ├── knowledge_graph.py      # NetworkX 知识图谱
│   │   ├── case_library.py         # 案例库
│   │   └── retriever.py            # 统一检索接口
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── client.py               # LLM 统一调用接口
│   │   ├── prompts.py              # Prompt 模板
│   │   ├── deepseek.py             # DeepSeek 适配器
│   │   └── local.py                # 本地 LLM 适配器
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py               # FastAPI 路由
│   ├── guardrails/
│   │   ├── __init__.py
│   │   ├── hallucination.py        # 幻觉防御
│   │   ├── ethics.py               # 伦理守则
│   │   └── privacy.py              # 隐私保护
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py              # 评测指标
│   │   ├── golden_dataset.py       # 黄金数据集
│   │   ├── human_eval.py           # 人类一致性测试
│   │   └── ab_test.py              # A/B 测试
│   └── utils/
│       ├── __init__.py
│       ├── text_processor.py       # 文本预处理
│       └── validators.py           # 输入验证
├── data/
│   ├── literature/
│   ├── cases/
│   ├── graph/
│   ├── embeddings/
│   └── golden/
├── tests/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_agents.py
│   ├── test_llm.py
│   ├── test_nlp.py
│   ├── test_knowledge.py
│   ├── test_guardrails.py
│   └── test_api.py
├── docs/
│   ├── api.md
│   ├── agent-interface.md
│   └── knowledge-base-contribution.md
├── .env.example
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

# 阶段一：基础框架（1-2 周）

## Task 1: 项目脚手架

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`
- Create: `data/literature/.gitkeep`
- Create: `data/cases/.gitkeep`
- Create: `data/graph/.gitkeep`
- Create: `data/embeddings/.gitkeep`
- Create: `data/golden/.gitkeep`

- [ ] **Step 1: 创建 pyproject.toml**

```toml
[project]
name = "psychology-of-speech"
version = "0.1.0"
description = "语言心理学话语分析多智能体系统"
requires-python = ">=3.11"
dependencies = [
    "langgraph>=0.2.0",
    "fastapi>=0.115.0",
    "uvicorn>=0.32.0",
    "pydantic>=2.9.0",
    "pydantic-settings>=2.6.0",
    "python-dotenv>=1.0.0",
    "httpx>=0.28.0",
    "hanlp>=2.1.0",
    "jieba>=0.42.1",
    "transformers>=4.45.0",
    "torch>=2.5.0",
    "chromadb>=0.5.0",
    "networkx>=3.4.0",
    "sentence-transformers>=3.2.0",
    "python-docx>=1.1.0",
    "PyPDF2>=3.0.0",
    "jinja2>=3.1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3.0",
    "pytest-asyncio>=0.24.0",
    "pytest-cov>=6.0.0",
    "ruff>=0.8.0",
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

- [ ] **Step 2: 创建 requirements.txt**

```
langgraph>=0.2.0
fastapi>=0.115.0
uvicorn>=0.32.0
pydantic>=2.9.0
pydantic-settings>=2.6.0
python-dotenv>=1.0.0
httpx>=0.28.0
hanlp>=2.1.0
jieba>=0.42.1
transformers>=4.45.0
torch>=2.5.0
chromadb>=0.5.0
networkx>=3.4.0
sentence-transformers>=3.2.0
python-docx>=1.1.0
PyPDF2>=3.0.0
jinja2>=3.1.0
```

- [ ] **Step 3: 创建 .env.example**

```bash
# DeepSeek API
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com

# OpenAI API (备选)
OPENAI_API_KEY=your_api_key_here

# 运行环境: dev / test / prod / offline
APP_ENV=dev

# LLM 配置
LLM_MODEL=deepseek-chat
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=4096

# 向量数据库
CHROMA_PERSIST_DIR=./data/embeddings

# 日志级别
LOG_LEVEL=INFO
```

- [ ] **Step 4: 创建目录结构和 __init__.py**

```bash
mkdir -p src/agents src/nlp src/knowledge src/llm src/api src/guardrails src/evaluation src/utils
mkdir -p data/literature data/cases data/graph data/embeddings data/golden
mkdir -p tests docs
```

创建所有 `__init__.py`（空文件）：
- `src/__init__.py`
- `src/agents/__init__.py`
- `src/nlp/__init__.py`
- `src/knowledge/__init__.py`
- `src/llm/__init__.py`
- `src/api/__init__.py`
- `src/guardrails/__init__.py`
- `src/evaluation/__init__.py`
- `src/utils/__init__.py`
- `tests/__init__.py`

创建 `.gitkeep` 文件到 data 子目录。

- [ ] **Step 5: 初始化 git 仓库并提交**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
git init
git add .
git commit -m "chore: initialize project scaffolding"
```

---

## Task 2: 配置管理（Pydantic Settings v2）

**Files:**
- Create: `src/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: 编写配置测试**

```python
# tests/test_config.py
import os
import pytest
from unittest.mock import patch


def test_default_config():
    """测试默认配置值"""
    with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}, clear=False):
        from src.config import Settings
        settings = Settings()
        assert settings.app_env == "dev"
        assert settings.llm_temperature == 0.3
        assert settings.llm_max_tokens == 4096


def test_env_override():
    """测试环境变量覆盖"""
    with patch.dict(os.environ, {
        "DEEPSEEK_API_KEY": "test-key",
        "APP_ENV": "prod",
        "LLM_TEMPERATURE": "0.7",
    }, clear=False):
        from src.config import Settings
        settings = Settings()
        assert settings.app_env == "prod"
        assert settings.llm_temperature == 0.7


def test_offline_mode():
    """测试离线模式配置"""
    with patch.dict(os.environ, {
        "DEEPSEEK_API_KEY": "test-key",
        "APP_ENV": "offline",
    }, clear=False):
        from src.config import Settings
        settings = Settings()
        assert settings.app_env == "offline"
        assert settings.use_local_llm is True


def test_missing_api_key_in_dev():
    """测试 dev 环境缺少 API key 时的处理"""
    with patch.dict(os.environ, {}, clear=True):
        from src.config import Settings
        with pytest.raises(Exception):
            Settings()
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/test_config.py -v
```

预期：FAIL（ImportError: cannot import 'Settings'）

- [ ] **Step 3: 实现配置模块**

```python
# src/config.py
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """应用配置，通过环境变量或 .env 文件加载"""

    # 环境
    app_env: str = Field(default="dev", alias="APP_ENV")

    # DeepSeek API
    deepseek_api_key: str = Field(alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com", alias="DEEPSEEK_BASE_URL"
    )

    # OpenAI API (备选)
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")

    # LLM 配置
    llm_model: str = Field(default="deepseek-chat", alias="LLM_MODEL")
    llm_temperature: float = Field(default=0.3, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=4096, alias="LLM_MAX_TOKENS")

    # 向量数据库
    chroma_persist_dir: str = Field(
        default="./data/embeddings", alias="CHROMA_PERSIST_DIR"
    )

    # 日志
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # 离线模式
    @property
    def use_local_llm(self) -> bool:
        return self.app_env == "offline"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


# 全局单例
_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
DEEPSEEK_API_KEY=test-key python -m pytest tests/test_config.py -v
```

预期：全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/config.py tests/test_config.py
git commit -m "feat: add Pydantic Settings v2 configuration management"
```

---

## Task 3: BaseAgent 接口与数据模型

**Files:**
- Create: `src/agents/base.py`
- Create: `tests/test_agents.py`

- [ ] **Step 1: 编写 BaseAgent 测试**

```python
# tests/test_agents.py
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
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/test_agents.py -v
```

预期：FAIL（ImportError）

- [ ] **Step 3: 实现 BaseAgent 和数据模型**

```python
# src/agents/base.py
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
    # 其他 Agent 的分析结果，用于交叉分析
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
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/test_agents.py -v
```

预期：全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/agents/base.py src/agents/__init__.py tests/test_agents.py
git commit -m "feat: add BaseAgent interface and data models"
```

---

## Task 4: LLM 统一调用接口

**Files:**
- Create: `src/llm/client.py`
- Create: `src/llm/deepseek.py`
- Create: `src/llm/__init__.py`
- Create: `tests/test_llm.py`

- [ ] **Step 1: 编写 LLM 客户端测试**

```python
# tests/test_llm.py
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.llm.client import LLMClient, LLMResponse
from src.llm.deepseek import DeepSeekAdapter


def test_llm_response_creation():
    """测试 LLMResponse 数据模型"""
    resp = LLMResponse(
        content="分析结果",
        model="deepseek-chat",
        tokens_used=100,
        finish_reason="stop",
    )
    assert resp.content == "分析结果"
    assert resp.tokens_used == 100


def test_deepseek_adapter_init():
    """测试 DeepSeek 适配器初始化"""
    adapter = DeepSeekAdapter(
        api_key="test-key",
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
    )
    assert adapter.model == "deepseek-chat"


def test_llm_client_init():
    """测试 LLM 客户端初始化"""
    client = LLMClient(adapter=DeepSeekAdapter(
        api_key="test-key",
        base_url="https://api.deepseek.com",
    ))
    assert client.adapter is not None


@pytest.mark.asyncio
async def test_llm_client_generate():
    """测试 LLM 客户端生成（mock）"""
    mock_adapter = AsyncMock()
    mock_adapter.generate.return_value = LLMResponse(
        content="测试响应",
        model="deepseek-chat",
        tokens_used=50,
        finish_reason="stop",
    )

    client = LLMClient(adapter=mock_adapter)
    response = await client.generate("测试提示词")

    assert response.content == "测试响应"
    mock_adapter.generate.assert_called_once_with("测试提示词", None)


@pytest.mark.asyncio
async def test_llm_client_generate_with_system():
    """测试带系统提示的生成"""
    mock_adapter = AsyncMock()
    mock_adapter.generate.return_value = LLMResponse(
        content="响应",
        model="deepseek-chat",
        tokens_used=30,
        finish_reason="stop",
    )

    client = LLMClient(adapter=mock_adapter)
    response = await client.generate("用户提示", system_prompt="系统指令")

    mock_adapter.generate.assert_called_once_with("用户提示", "系统指令")
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/test_llm.py -v
```

预期：FAIL（ImportError）

- [ ] **Step 3: 实现 LLM 客户端和适配器**

```python
# src/llm/client.py
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """LLM 响应"""
    content: str
    model: str
    tokens_used: int
    finish_reason: str


class LLMAdapter(ABC):
    """LLM 适配器接口"""

    @abstractmethod
    async def generate(
        self, prompt: str, system_prompt: str | None = None
    ) -> LLMResponse:
        """生成响应"""
        ...


class LLMClient:
    """LLM 统一客户端"""

    def __init__(self, adapter: LLMAdapter):
        self.adapter = adapter

    async def generate(
        self, prompt: str, system_prompt: str | None = None
    ) -> LLMResponse:
        return await self.adapter.generate(prompt, system_prompt)
```

```python
# src/llm/deepseek.py
import httpx
from src.llm.client import LLMAdapter, LLMResponse


class DeepSeekAdapter(LLMAdapter):
    """DeepSeek API 适配器"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def generate(
        self, prompt: str, system_prompt: str | None = None
    ) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                },
                timeout=120.0,
            )
            response.raise_for_status()
            data = response.json()

        choice = data["choices"][0]
        usage = data.get("usage", {})

        return LLMResponse(
            content=choice["message"]["content"],
            model=data.get("model", self.model),
            tokens_used=usage.get("total_tokens", 0),
            finish_reason=choice.get("finish_reason", "stop"),
        )
```

```python
# src/llm/__init__.py
from src.llm.client import LLMClient, LLMAdapter, LLMResponse
from src.llm.deepseek import DeepSeekAdapter

__all__ = ["LLMClient", "LLMAdapter", "LLMResponse", "DeepSeekAdapter"]
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/test_llm.py -v
```

预期：全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/llm/ tests/test_llm.py
git commit -m "feat: add LLM unified client interface with DeepSeek adapter"
```

---

## Task 5: Prompt 模板管理

**Files:**
- Create: `src/llm/prompts.py`
- Modify: `tests/test_llm.py`

- [ ] **Step 1: 编写 Prompt 模板测试**

在 `tests/test_llm.py` 中追加：

```python
from src.llm.prompts import PromptTemplates


def test_get_system_prompt():
    """测试获取系统提示词"""
    prompt = PromptTemplates.get_system_prompt("text_analyst")
    assert "语言学" in prompt or "文本" in prompt
    assert len(prompt) > 50


def test_get_analysis_prompt():
    """测试获取分析提示词"""
    prompt = PromptTemplates.get_analysis_prompt(
        "text_analyst",
        text="今天天气真好",
        depth="standard",
    )
    assert "今天天气真好" in prompt
    assert "standard" in prompt.lower() or "标准" in prompt


def test_get_report_prompt():
    """测试获取报告生成提示词"""
    analyses = {
        "text_analyst": {"result": "test"},
        "psychology_analyst": {"result": "test"},
    }
    prompt = PromptTemplates.get_report_prompt(
        text="原始文本",
        analyses=analyses,
        depth="standard",
    )
    assert "原始文本" in prompt
    assert "text_analyst" in prompt


def test_unknown_agent_raises():
    """测试未知 Agent 名称抛出异常"""
    with pytest.raises(KeyError):
        PromptTemplates.get_system_prompt("nonexistent_agent")
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/test_llm.py::test_get_system_prompt -v
```

预期：FAIL（ImportError 或 KeyError）

- [ ] **Step 3: 实现 Prompt 模板**

```python
# src/llm/prompts.py
SYSTEM_PROMPTS = {
    "text_analyst": """你是一位专业的语言学分析师，擅长对文本进行语言学层面的深度解析。

你的分析维度包括：
1. 句法结构：句子类型、从句嵌套、主被动语态
2. 词汇选择：用词正式程度、情感词汇、模糊语识别
3. 语气与情态：情态动词分析、语气强度
4. 修辞手法：比喻、排比、反问、夸张、讽刺等
5. 话语标记：转折、强调、补充等标记词

请以 JSON 格式输出分析结果。""",

    "psychology_analyst": """你是一位语言心理学专家，基于科学理论分析说话者的心理状态和潜在动机。

你的理论工具箱包括：
- 言语行为理论（Austin/Searle）：识别言外行为
- 会话含义（Grice）：分析隐含意义
- 礼貌策略（Brown & Levinson）：分析面子策略
- 儒家面子理论（胡先缙/黄光国）：中国本土面子观
- 高语境沟通（Hall）：中国文化语境特征
- 权力距离（Hofstede）：语言中的权力关系

请以 JSON 格式输出分析结果，每条结论标注理论来源和置信度（0.0-1.0）。""",

    "logic_analyst": """你是一位逻辑分析专家，擅长分析论证结构和识别逻辑问题。

你的分析维度包括：
1. 论证结构提取：前提→推理→结论
2. 逻辑谬误检测：稻草人、滑坡、诉诸权威、人身攻击、循环论证等
3. 隐含前提挖掘：未明说的假设
4. 论证强度评估：证据充分性
5. 反事实分析：前提为假时结论是否成立

请以 JSON 格式输出分析结果。""",

    "report_generator": """你是一位专业的分析报告撰写者，负责整合多个分析模块的结果，生成结构清晰、逻辑严谨的分析报告。

报告结构：
1. 摘要
2. 文本特征分析
3. 心理动机分析
4. 逻辑结构分析
5. 综合洞察
6. 理论依据与局限性
7. 建议与延伸思考

重要：所有输出必须包含免责声明。""",

    "orchestrator": """你是话语分析系统的总协调器，负责理解用户需求并制定分析计划。

你需要判断：
1. 文本类型（简短对话/长文本/对话历史）
2. 适合的分析深度（quick/standard/deep）
3. 需要调用哪些分析 Agent

请以 JSON 格式输出分析计划。""",
}


class PromptTemplates:
    """Prompt 模板管理"""

    @staticmethod
    def get_system_prompt(agent_name: str) -> str:
        """获取 Agent 的系统提示词"""
        if agent_name not in SYSTEM_PROMPTS:
            raise KeyError(f"Unknown agent: {agent_name}")
        return SYSTEM_PROMPTS[agent_name]

    @staticmethod
    def get_analysis_prompt(
        agent_name: str, text: str, depth: str = "standard"
    ) -> str:
        """获取分析提示词"""
        return f"""请对以下文本进行分析：

---
{text}
---

分析深度：{depth}

请以 JSON 格式输出你的分析结果。"""

    @staticmethod
    def get_report_prompt(
        text: str, analyses: dict[str, dict], depth: str = "standard"
    ) -> str:
        """获取报告生成提示词"""
        import json

        analyses_str = json.dumps(analyses, ensure_ascii=False, indent=2)
        return f"""原始文本：
---
{text}
---

各模块分析结果：
{analyses_str}

分析深度：{depth}

请整合以上结果，生成一份完整的分析报告。"""
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/test_llm.py -v -k "prompt"
```

预期：全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/llm/prompts.py tests/test_llm.py
git commit -m "feat: add prompt template management for all agents"
```

---

## Task 6: 协调器骨架（动态路由）

**Files:**
- Create: `src/agents/orchestrator.py`
- Modify: `tests/test_agents.py`

- [ ] **Step 1: 编写协调器测试**

在 `tests/test_agents.py` 中追加：

```python
from src.agents.orchestrator import Orchestrator, AnalysisPlan


def test_orchestrator_short_text_routing():
    """测试短文本动态路由（<100字）"""
    orchestrator = Orchestrator()
    plan = orchestrator.plan_analysis("你好")
    assert plan.depth == AnalysisDepth.QUICK
    # 短文本跳过深度逻辑分析
    assert "logic_analyst" not in plan.agents or plan.agent_params.get(
        "logic_analyst", {}
    ).get("skip_deep", False)


def test_orchestrator_medium_text_routing():
    """测试中等文本路由（100-500字）"""
    text = "这" * 150  # 150 字
    orchestrator = Orchestrator()
    plan = orchestrator.plan_analysis(text)
    assert "text_analyst" in plan.agents
    assert "psychology_analyst" in plan.agents
    assert "logic_analyst" in plan.agents


def test_orchestrator_long_text_routing():
    """测试长文本路由（>500字）"""
    text = "这" * 600  # 600 字
    orchestrator = Orchestrator()
    plan = orchestrator.plan_analysis(text)
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
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/test_agents.py -v -k "orchestrator"
```

预期：FAIL

- [ ] **Step 3: 实现协调器**

```python
# src/agents/orchestrator.py
from dataclasses import dataclass, field
from src.agents.base import AnalysisDepth


@dataclass
class AnalysisPlan:
    """分析计划"""
    agents: list[str]
    depth: AnalysisDepth
    agent_params: dict[str, dict] = field(default_factory=dict)
    segment: bool = False


class Orchestrator:
    """总协调器 - 任务分解与动态路由"""

    # 阈值配置
    SHORT_TEXT_THRESHOLD = 100   # 字
    LONG_TEXT_THRESHOLD = 500    # 字

    def plan_analysis(
        self,
        text: str,
        requested_depth: AnalysisDepth | None = None,
    ) -> AnalysisPlan:
        """根据文本特征生成分析计划"""
        text_len = len(text)

        # 动态确定分析深度
        if requested_depth:
            depth = requested_depth
        elif text_len < self.SHORT_TEXT_THRESHOLD:
            depth = AnalysisDepth.QUICK
        elif text_len > self.LONG_TEXT_THRESHOLD:
            depth = AnalysisDepth.DEEP
        else:
            depth = AnalysisDepth.STANDARD

        # 确定参与的 Agent
        agents = ["text_analyst", "psychology_analyst", "logic_analyst"]
        agent_params: dict[str, dict] = {}

        # 短文本：逻辑分析标记跳过深度分析
        if text_len < self.SHORT_TEXT_THRESHOLD:
            agent_params["logic_analyst"] = {"skip_deep": True}

        # 长文本：启用分段分析
        segment = text_len > self.LONG_TEXT_THRESHOLD

        return AnalysisPlan(
            agents=agents,
            depth=depth,
            agent_params=agent_params,
            segment=segment,
        )

    def merge_results(
        self, results: dict[str, dict], weights: dict[str, float] | None = None
    ) -> dict:
        """合并多 Agent 结果（置信度加权）"""
        if weights is None:
            weights = {
                "text_analyst": 0.3,
                "psychology_analyst": 0.4,
                "logic_analyst": 0.3,
            }

        merged = {
            "analyses": results,
            "confidence_scores": {},
            "overall_confidence": 0.0,
        }

        total_weight = 0.0
        weighted_sum = 0.0

        for agent_name, result in results.items():
            confidence = result.get("confidence", 0.5)
            weight = weights.get(agent_name, 0.3)
            merged["confidence_scores"][agent_name] = confidence
            weighted_sum += confidence * weight
            total_weight += weight

        if total_weight > 0:
            merged["overall_confidence"] = round(weighted_sum / total_weight, 3)

        # 标记低可信度结论
        merged["low_confidence_warnings"] = [
            name for name, conf in merged["confidence_scores"].items()
            if conf < 0.3
        ]

        return merged
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/test_agents.py -v -k "orchestrator or analysis_plan"
```

预期：全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/agents/orchestrator.py tests/test_agents.py
git commit -m "feat: add orchestrator with dynamic routing and confidence fusion"
```

---

## Task 7: CLI 入口与端到端 Demo

**Files:**
- Create: `src/main.py`

- [ ] **Step 1: 实现 CLI 入口**

```python
# src/main.py
import asyncio
import json
import sys

from src.config import get_settings
from src.agents.base import AnalysisContext, AnalysisDepth
from src.agents.orchestrator import Orchestrator
from src.llm.client import LLMClient
from src.llm.deepseek import DeepSeekAdapter
from src.llm.prompts import PromptTemplates


async def analyze_text(text: str, depth: str = "standard") -> dict:
    """执行文本分析（单 Agent demo 版本）"""
    settings = get_settings()

    # 初始化 LLM
    adapter = DeepSeekAdapter(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
    client = LLMClient(adapter=adapter)

    # 协调器规划
    orchestrator = Orchestrator()
    plan = orchestrator.plan_analysis(
        text, AnalysisDepth(depth) if depth else None
    )

    print(f"分析计划：深度={plan.depth.value}, 分段={plan.segment}")
    print(f"参与 Agent: {', '.join(plan.agents)}")

    # Demo: 使用 text_analyst 做单 Agent 分析
    system_prompt = PromptTemplates.get_system_prompt("text_analyst")
    analysis_prompt = PromptTemplates.get_analysis_prompt(
        "text_analyst", text, plan.depth.value
    )

    print("\n正在分析...")
    response = await client.generate(analysis_prompt, system_prompt)

    result = {
        "input_text": text,
        "analysis_plan": {
            "depth": plan.depth.value,
            "agents": plan.agents,
            "segment": plan.segment,
        },
        "text_analysis": response.content,
        "tokens_used": response.tokens_used,
    }

    return result


def main():
    """CLI 入口"""
    if len(sys.argv) < 2:
        print("用法: python -m src.main '要分析的文本' [depth]")
        print("depth: quick / standard / deep (默认 standard)")
        sys.exit(1)

    text = sys.argv[1]
    depth = sys.argv[2] if len(sys.argv) > 2 else "standard"

    result = asyncio.run(analyze_text(text, depth))

    print("\n" + "=" * 60)
    print("分析结果：")
    print("=" * 60)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 测试 CLI 运行**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
DEEPSEEK_API_KEY=your-key python -m src.main "今天天气真不错，我们去公园散步吧" quick
```

预期：输出分析计划和文本分析结果（JSON）

- [ ] **Step 3: 提交**

```bash
git add src/main.py
git commit -m "feat: add CLI entry point with single-agent demo"
```

---

## Task 8: 阶段一总结

- [ ] **Step 1: 运行全部测试**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/ -v --tb=short
```

预期：全部 PASS

- [ ] **Step 2: 创建 README.md**

```markdown
# 语言心理学话语分析系统

基于语言心理学的多智能体话语分析系统，结合 RAG 技术和大语言模型。

## 快速开始

1. 安装依赖：`pip install -r requirements.txt`
2. 配置 API Key：`cp .env.example .env` 并编辑
3. 运行分析：`python -m src.main "要分析的文本"`

## 开发进度

- [x] 阶段一：基础框架
- [ ] 阶段二：NLP 管道 + 文本解析 Agent
- [ ] 阶段三：RAG 知识引擎
- [ ] 阶段四：心理分析 + 逻辑推理 Agent
- [ ] 阶段五：报告生成 + 协调器完善
- [ ] 阶段六：Web 接口 + 可观测性
```

- [ ] **Step 3: 最终提交**

```bash
git add README.md
git commit -m "docs: add README with quick start guide"
```

---

# 阶段二至六概要

> 以下为后续阶段的高层任务清单，每个阶段开始前需细化为 TDD 任务。

## 阶段二：NLP 管道 + 文本解析 Agent（1-2 周）

> **详细计划已生成：** [2026-05-08-phase2-nlp-text-analyst.md](./2026-05-08-phase2-nlp-text-analyst.md)

- [ ] Task 9: HanLP 集成（分词、依存句法、语义角色标注）
- [ ] Task 10: jieba 后备分词 + 自定义词典
- [ ] Task 11: transformers 中文情感/心理模型集成
- [ ] Task 12: 修辞识别模块
- [ ] Task 13: 文本解析 Agent 完整实现（5 个分析维度）
- [ ] Task 14: 评测数据集结构与评分脚本
- [ ] Task 15: 阶段二总结

## 阶段三：RAG 知识引擎（2-3 周）

- [ ] Task 18: ChromaDB 向量库搭建（含元数据过滤）
- [ ] Task 19: text2vec-base-chinese Embedding 集成
- [ ] Task 20: 理论文献库构建（分块、向量化、导入）
- [ ] Task 21: 案例库构建（结构化 JSON + 向量化）
- [ ] Task 22: 知识图谱构建（NetworkX）
- [ ] Task 23: 统一检索接口（向量 + 关键词 + 图谱融合）

## 阶段四：心理分析 + 逻辑推理 Agent（2-3 周）

- [ ] Task 24: 心理分析 Agent 实现（含中西方理论框架）
- [ ] Task 25: 逻辑推理 Agent 实现（含 30+ 种谬误检测）
- [ ] Task 26: Guardrails 模块 - 幻觉防御
- [ ] Task 27: Guardrails 模块 - 伦理守则 + 免责声明
- [ ] Task 28: Guardrails 模块 - 隐私保护

## 阶段五：报告生成 + 协调器完善（1-2 周）

- [ ] Task 29: 报告生成 Agent（Markdown/JSON/HTML 多格式）
- [ ] Task 30: 协调器完善（LangGraph 编排、并行调度）
- [ ] Task 31: 端到端集成测试
- [ ] Task 32: A/B 测试框架（DeepSeek vs OpenAI）

## 阶段六：Web 接口 + 可观测性（1-2 周）

- [ ] Task 33: FastAPI 路由实现（/analyze, /health）
- [ ] Task 34: 请求验证 + 错误处理
- [ ] Task 35: LangFuse 可观测性集成
- [ ] Task 36: Agent 单元测试（mock LLM）
- [ ] Task 37: API 集成测试
- [ ] Task 38: 性能优化 + 缓存策略
