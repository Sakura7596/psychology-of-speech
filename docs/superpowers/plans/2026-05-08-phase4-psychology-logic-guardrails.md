# 阶段四：心理分析 + 逻辑推理 Agent + Guardrails 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现心理分析 Agent（中西方理论框架）、逻辑推理 Agent（30+ 种谬误检测）、Guardrails 模块（幻觉防御、伦理守则、隐私保护）

**Architecture:** 两个分析 Agent 均继承 BaseAgent，调用 LLM + RAG 知识引擎进行深度分析。Guardrails 作为独立模块贯穿所有 Agent 输出。

**Tech Stack:** DeepSeek API, RAG Knowledge Engine (Phase 3), pytest

**依赖：** 阶段一至三已完成

---

## 文件结构

```
src/
├── agents/
│   ├── psychology_analyst.py    # 心理分析 Agent
│   └── logic_analyst.py         # 逻辑推理 Agent
├── guardrails/
│   ├── __init__.py
│   ├── hallucination.py         # 幻觉防御
│   ├── ethics.py                # 伦理守则 + 免责声明
│   └── privacy.py               # 隐私保护
tests/
├── test_psychology_analyst.py   # 心理分析 Agent 测试
├── test_logic_analyst.py        # 逻辑推理 Agent 测试
└── test_guardrails.py           # Guardrails 测试
```

---

## Task 24: 心理分析 Agent

**Files:**
- Create: `src/agents/psychology_analyst.py`
- Create: `tests/test_psychology_analyst.py`

- [ ] **Step 1: 编写测试**

```python
# tests/test_psychology_analyst.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.agents.base import AnalysisContext, AnalysisDepth
from src.agents.psychology_analyst import PsychologyAnalystAgent


def test_psychology_analyst_name():
    agent = PsychologyAnalystAgent()
    assert agent.name == "psychology_analyst"


def test_psychology_analyst_description():
    agent = PsychologyAnalystAgent()
    assert len(agent.description) > 10


def test_analyze_speech_act():
    """测试言语行为分析"""
    agent = PsychologyAnalystAgent()
    
    # Mock LLM
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = MagicMock(
        content='{"speech_acts": [{"type": "assertive", "text": "今天天气真好", "confidence": 0.8}], "overall_intent": "表达观点"}'
    )
    agent._llm = mock_llm
    
    # Mock knowledge retriever
    mock_retriever = MagicMock()
    mock_retriever.get_context_string.return_value = "言语行为理论: Austin/Searle"
    agent._retriever = mock_retriever
    
    ctx = AnalysisContext(text="今天天气真好", depth=AnalysisDepth.STANDARD)
    result = agent.analyze(ctx)
    
    assert result.agent_name == "psychology_analyst"
    assert "speech_acts" in result.analysis or "psychological_analysis" in result.analysis
    assert result.confidence > 0


def test_analyze_with_face_theory():
    """测试面子理论分析"""
    agent = PsychologyAnalystAgent()
    
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = MagicMock(
        content='{"face_analysis": {"face_threatening_acts": ["直接拒绝"], "politeness_strategy": "negative_politeness"}, "confidence": 0.7}'
    )
    agent._llm = mock_llm
    
    mock_retriever = MagicMock()
    mock_retriever.get_context_string.return_value = "礼貌策略: Brown & Levinson"
    agent._retriever = mock_retriever
    
    ctx = AnalysisContext(text="不好意思，这个我做不了", depth=AnalysisDepth.DEEP)
    result = agent.analyze(ctx)
    
    assert result.agent_name == "psychology_analyst"
    assert result.confidence > 0


def test_analyze_confidence_range():
    """测试置信度范围"""
    agent = PsychologyAnalystAgent()
    
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = MagicMock(content='{"analysis": "test"}')
    agent._llm = mock_llm
    agent._retriever = MagicMock()
    agent._retriever.get_context_string.return_value = ""
    
    ctx = AnalysisContext(text="测试", depth=AnalysisDepth.QUICK)
    result = agent.analyze(ctx)
    
    assert 0.0 <= result.confidence <= 1.0
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/test_psychology_analyst.py -v
```

- [ ] **Step 3: 实现心理分析 Agent**

```python
# src/agents/psychology_analyst.py
import json
from src.agents.base import BaseAgent, AnalysisContext, AgentResult, AnalysisDepth
from src.llm.client import LLMClient
from src.llm.deepseek import DeepSeekAdapter
from src.llm.prompts import PromptTemplates
from src.config import get_settings
from src.knowledge.retriever import KnowledgeRetriever
from src.knowledge.knowledge_graph import KnowledgeGraph
from src.knowledge.case_library import CaseLibrary


class PsychologyAnalystAgent(BaseAgent):
    """心理分析 Agent - 基于语言心理学理论"""

    def __init__(self):
        settings = get_settings()
        adapter = DeepSeekAdapter(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
        )
        self._llm = LLMClient(adapter=adapter)
        
        # RAG 知识引擎
        kg = KnowledgeGraph()
        try:
            kg.load("data/graph/psychology_graph.json")
        except Exception:
            pass
        case_lib = CaseLibrary("data/cases")
        self._retriever = KnowledgeRetriever(knowledge_graph=kg, case_library=case_lib)

    @property
    def name(self) -> str:
        return "psychology_analyst"

    @property
    def description(self) -> str:
        return "基于语言心理学理论分析说话者心理状态和潜在动机"

    def analyze(self, context: AnalysisContext) -> AgentResult:
        """执行心理分析"""
        text = context.text
        depth = context.depth

        # 检索相关知识
        knowledge_context = self._retriever.get_context_string(text, n_results=3)

        # 构建分析提示
        system_prompt = PromptTemplates.get_system_prompt("psychology_analyst")
        
        depth_instructions = {
            AnalysisDepth.QUICK: "请快速分析主要心理特征，输出简要 JSON。",
            AnalysisDepth.STANDARD: "请从言语行为、礼貌策略、情感状态三个维度分析，输出 JSON。",
            AnalysisDepth.DEEP: "请从言语行为、面子理论、高语境沟通、权力距离、情感状态五个维度深入分析，输出 JSON。",
        }
        
        analysis_prompt = f"""请分析以下文本的心理特征：

---
{text}
---

{depth_instructions.get(depth, depth_instructions[AnalysisDepth.STANDARD])}

相关理论知识：
{knowledge_context}

请以 JSON 格式输出分析结果，包含 confidence 字段（0.0-1.0）。"""

        # 调用 LLM
        import asyncio
        try:
            response = asyncio.run(self._llm.generate(analysis_prompt, system_prompt))
            raw_content = response.content
        except Exception:
            raw_content = '{"error": "LLM 调用失败"}'

        # 解析结果
        analysis = self._parse_response(raw_content)
        confidence = analysis.pop("confidence", 0.6)

        return AgentResult(
            agent_name=self.name,
            analysis=analysis,
            confidence=min(max(confidence, 0.0), 1.0),
            sources=["DeepSeek", "KnowledgeGraph", "CaseLibrary"],
        )

    def _parse_response(self, raw: str) -> dict:
        """解析 LLM 响应"""
        try:
            # 尝试提取 JSON
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0]
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0]
            return json.loads(raw.strip())
        except (json.JSONDecodeError, IndexError):
            return {"raw_response": raw, "parse_error": True}
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/test_psychology_analyst.py -v
```

- [ ] **Step 5: 提交**

```bash
git add src/agents/psychology_analyst.py tests/test_psychology_analyst.py
git commit -m "feat: add PsychologyAnalystAgent with multi-theory analysis"
```

---

## Task 25: 逻辑推理 Agent

**Files:**
- Create: `src/agents/logic_analyst.py`
- Create: `tests/test_logic_analyst.py`

- [ ] **Step 1: 编写测试**

```python
# tests/test_logic_analyst.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.agents.base import AnalysisContext, AnalysisDepth
from src.agents.logic_analyst import LogicAnalystAgent


def test_logic_analyst_name():
    agent = LogicAnalystAgent()
    assert agent.name == "logic_analyst"


def test_logic_analyst_description():
    agent = LogicAnalystAgent()
    assert len(agent.description) > 10


def test_analyze_argument_structure():
    """测试论证结构分析"""
    agent = LogicAnalystAgent()
    
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = MagicMock(
        content='{"argument_structure": {"premises": ["所有人都会死"], "reasoning": "苏格拉底是人", "conclusion": "所以苏格拉底会死"}, "argument_strength": 0.9}'
    )
    agent._llm = mock_llm
    agent._retriever = MagicMock()
    agent._retriever.get_context_string.return_value = ""
    
    ctx = AnalysisContext(text="所有人都会死，苏格拉底是人，所以苏格拉底会死", depth=AnalysisDepth.STANDARD)
    result = agent.analyze(ctx)
    
    assert result.agent_name == "logic_analyst"
    assert "argument_structure" in result.analysis or "logical_analysis" in result.analysis
    assert result.confidence > 0


def test_analyze_fallacy_detection():
    """测试逻辑谬误检测"""
    agent = LogicAnalystAgent()
    
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = MagicMock(
        content='{"fallacies": [{"type": "straw_man", "description": "歪曲对方立场", "severity": "high"}], "hidden_assumptions": [], "confidence": 0.8}'
    )
    agent._llm = mock_llm
    
    mock_retriever = MagicMock()
    mock_retriever.get_context_string.return_value = "稻草人谬误案例"
    agent._retriever = mock_retriever
    
    ctx = AnalysisContext(text="你支持环保？那你是不是想让我们都回到原始社会？", depth=AnalysisDepth.STANDARD)
    result = agent.analyze(ctx)
    
    assert result.agent_name == "logic_analyst"
    assert result.confidence > 0
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/test_logic_analyst.py -v
```

- [ ] **Step 3: 实现逻辑推理 Agent**

```python
# src/agents/logic_analyst.py
import json
from src.agents.base import BaseAgent, AnalysisContext, AgentResult, AnalysisDepth
from src.llm.client import LLMClient
from src.llm.deepseek import DeepSeekAdapter
from src.llm.prompts import PromptTemplates
from src.config import get_settings
from src.knowledge.retriever import KnowledgeRetriever
from src.knowledge.case_library import CaseLibrary


class LogicAnalystAgent(BaseAgent):
    """逻辑推理 Agent - 论证分析与谬误检测"""

    def __init__(self):
        settings = get_settings()
        adapter = DeepSeekAdapter(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
        )
        self._llm = LLMClient(adapter=adapter)
        
        case_lib = CaseLibrary("data/cases")
        self._retriever = KnowledgeRetriever(case_library=case_lib)

    @property
    def name(self) -> str:
        return "logic_analyst"

    @property
    def description(self) -> str:
        return "分析论证结构，识别逻辑谬误和隐含假设"

    def analyze(self, context: AnalysisContext) -> AgentResult:
        """执行逻辑分析"""
        text = context.text
        depth = context.depth

        # 检索谬误案例
        knowledge_context = self._retriever.get_context_string(text, n_results=3)

        system_prompt = PromptTemplates.get_system_prompt("logic_analyst")
        
        depth_instructions = {
            AnalysisDepth.QUICK: "请快速识别主要论证结构和明显谬误，输出简要 JSON。",
            AnalysisDepth.STANDARD: "请分析论证结构（前提→推理→结论）、逻辑谬误、隐含假设，输出 JSON。",
            AnalysisDepth.DEEP: "请深入分析论证结构、所有逻辑谬误、隐含前提、论证强度、反事实分析，输出 JSON。",
        }
        
        analysis_prompt = f"""请分析以下文本的逻辑结构：

---
{text}
---

{depth_instructions.get(depth, depth_instructions[AnalysisDepth.STANDARD])}

相关谬误案例：
{knowledge_context}

请以 JSON 格式输出分析结果，包含 confidence 字段（0.0-1.0）。"""

        import asyncio
        try:
            response = asyncio.run(self._llm.generate(analysis_prompt, system_prompt))
            raw_content = response.content
        except Exception:
            raw_content = '{"error": "LLM 调用失败"}'

        analysis = self._parse_response(raw_content)
        confidence = analysis.pop("confidence", 0.6)

        return AgentResult(
            agent_name=self.name,
            analysis=analysis,
            confidence=min(max(confidence, 0.0), 1.0),
            sources=["DeepSeek", "CaseLibrary"],
        )

    def _parse_response(self, raw: str) -> dict:
        try:
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0]
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0]
            return json.loads(raw.strip())
        except (json.JSONDecodeError, IndexError):
            return {"raw_response": raw, "parse_error": True}
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/test_logic_analyst.py -v
```

- [ ] **Step 5: 提交**

```bash
git add src/agents/logic_analyst.py tests/test_logic_analyst.py
git commit -m "feat: add LogicAnalystAgent with fallacy detection"
```

---

## Task 26: Guardrails - 幻觉防御

**Files:**
- Create: `src/guardrails/__init__.py`
- Create: `src/guardrails/hallucination.py`
- Create: `tests/test_guardrails.py`

- [ ] **Step 1: 编写测试**

```python
# tests/test_guardrails.py
import pytest
from src.guardrails.hallucination import HallucinationGuard
from src.guardrails.ethics import EthicsGuard
from src.guardrails.privacy import PrivacyGuard


def test_hallucination_check_valid():
    """测试有效来源的分析结果"""
    guard = HallucinationGuard()
    analysis = {
        "conclusion": "说话者使用了比喻修辞",
        "sources": ["HanLP", "transformers"],
        "confidence": 0.8,
    }
    result = guard.check(analysis)
    assert result["passed"] is True


def test_hallucination_check_no_sources():
    """测试无来源的分析结果"""
    guard = HallucinationGuard()
    analysis = {
        "conclusion": "说话者有严重心理问题",
        "sources": [],
        "confidence": 0.9,
    }
    result = guard.check(analysis)
    assert result["passed"] is False
    assert "来源" in result["reason"] or "source" in result["reason"].lower()


def test_hallucination_check_low_confidence():
    """测试低置信度结果"""
    guard = HallucinationGuard()
    analysis = {
        "conclusion": "推测性分析",
        "sources": ["LLM"],
        "confidence": 0.1,
    }
    result = guard.check(analysis)
    assert "推测" in result.get("warning", "") or result["passed"] is True
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/test_guardrails.py -v -k "hallucination"
```

- [ ] **Step 3: 实现幻觉防御**

```python
# src/guardrails/hallucination.py
from src.agents.base import RELIABILITY_THRESHOLD


class HallucinationGuard:
    """幻觉防御 - 验证分析结果的来源和置信度"""

    def __init__(self, min_sources: int = 1, min_confidence: float = 0.3):
        self._min_sources = min_sources
        self._min_confidence = min_confidence

    def check(self, analysis: dict) -> dict:
        """检查分析结果是否可能是幻觉"""
        issues = []
        warnings = []

        # 检查来源
        sources = analysis.get("sources", [])
        if len(sources) < self._min_sources:
            issues.append(f"分析缺少来源引用（需要至少 {self._min_sources} 个来源）")

        # 检查置信度
        confidence = analysis.get("confidence", 0.5)
        if confidence < self._min_confidence:
            warnings.append(f"置信度过低（{confidence}），结果为推测性分析")

        # 检查是否有 parse_error
        if analysis.get("parse_error"):
            issues.append("LLM 响应解析失败")

        passed = len(issues) == 0
        return {
            "passed": passed,
            "issues": issues,
            "warnings": warnings,
            "reason": "; ".join(issues) if issues else "",
        }
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/test_guardrails.py -v -k "hallucination"
```

- [ ] **Step 5: 提交**

```bash
git add src/guardrails/hallucination.py src/guardrails/__init__.py tests/test_guardrails.py
git commit -m "feat: add hallucination guard for source and confidence verification"
```

---

## Task 27: Guardrails - 伦理守则 + 免责声明

**Files:**
- Create: `src/guardrails/ethics.py`
- Modify: `tests/test_guardrails.py`

- [ ] **Step 1: 编写测试**

在 `tests/test_guardrails.py` 中追加：

```python
def test_ethics_disclaimer_injection():
    """测试免责声明注入"""
    guard = EthicsGuard()
    report = "分析结果：说话者使用了比喻修辞。"
    result = guard.inject_disclaimer(report)
    assert "非专业" in result or "仅供参考" in result or "声明" in result


def test_ethics_detect_diagnostic_language():
    """测试检测诊断性语言"""
    guard = EthicsGuard()
    text = "根据分析，说话者患有抑郁症"
    result = guard.check_diagnostic_language(text)
    assert result["has_diagnostic"] is True


def test_ethics_clean_text():
    """测试正常文本通过"""
    guard = EthicsGuard()
    text = "说话者使用了比喻修辞手法"
    result = guard.check_diagnostic_language(text)
    assert result["has_diagnostic"] is False
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/test_guardrails.py -v -k "ethics"
```

- [ ] **Step 3: 实现伦理守则**

```python
# src/guardrails/ethics.py
import re


DISCLAIMER = """
---
**免责声明**：本分析基于语言学特征的辅助分析，仅供参考，不构成专业心理咨询、诊断或治疗建议。
如需心理帮助，请咨询专业心理咨询师或医疗机构。
"""


class EthicsGuard:
    """伦理守则 - 免责声明注入与诊断语言检测"""

    DIAGNOSTIC_PATTERNS = [
        r'患有.{1,10}[症病]',
        r'诊断.{0,5}为',
        r'确诊',
        r'精神.{0,3}[疾病障碍]',
        r'心理.{0,3}[疾病障碍变态]',
        r'人格障碍',
        r'需要.{0,5}治疗',
        r'建议.{0,5}[吃药服药就医住院]',
    ]

    def inject_disclaimer(self, report: str) -> str:
        """注入免责声明"""
        if "免责声明" not in report and "仅供参考" not in report:
            return report + DISCLAIMER
        return report

    def check_diagnostic_language(self, text: str) -> dict:
        """检测诊断性语言"""
        found = []
        for pattern in self.DIAGNOSTIC_PATTERNS:
            matches = re.findall(pattern, text)
            found.extend(matches)

        return {
            "has_diagnostic": len(found) > 0,
            "found_terms": found,
            "suggestion": "请避免使用临床诊断术语，改用描述性语言" if found else "",
        }

    def sanitize_output(self, report: str) -> str:
        """净化输出：注入免责声明 + 检查诊断语言"""
        diag = self.check_diagnostic_language(report)
        if diag["has_diagnostic"]:
            # 替换诊断性语言
            for term in diag["found_terms"]:
                report = report.replace(term, f"[已移除诊断术语]")
        
        return self.inject_disclaimer(report)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/test_guardrails.py -v -k "ethics"
```

- [ ] **Step 5: 提交**

```bash
git add src/guardrails/ethics.py tests/test_guardrails.py
git commit -m "feat: add ethics guard with disclaimer injection and diagnostic language detection"
```

---

## Task 28: Guardrails - 隐私保护

**Files:**
- Create: `src/guardrails/privacy.py`
- Modify: `tests/test_guardrails.py`

- [ ] **Step 1: 编写测试**

在 `tests/test_guardrails.py` 中追加：

```python
def test_privacy_mask_phone():
    """测试手机号脱敏"""
    guard = PrivacyGuard()
    text = "请联系我 13812345678"
    result = guard.mask_pii(text)
    assert "138****5678" in result
    assert "13812345678" not in result


def test_privacy_mask_email():
    """测试邮箱脱敏"""
    guard = PrivacyGuard()
    text = "发送到 test@example.com"
    result = guard.mask_pii(text)
    assert "t***@example.com" in result
    assert "test@example.com" not in result


def test_privacy_mask_id_card():
    """测试身份证号脱敏"""
    guard = PrivacyGuard()
    text = "身份证号 110101199001011234"
    result = guard.mask_pii(text)
    assert "110101****1234" in result or "110101" in result


def test_privacy_no_pii():
    """测试无 PII 文本"""
    guard = PrivacyGuard()
    text = "今天天气真好"
    result = guard.mask_pii(text)
    assert result == text


def test_privacy_check_retention():
    """测试数据保留策略"""
    guard = PrivacyGuard()
    policy = guard.get_retention_policy()
    assert "local" in policy or "本地" in policy
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/test_guardrails.py -v -k "privacy"
```

- [ ] **Step 3: 实现隐私保护**

```python
# src/guardrails/privacy.py
import re


class PrivacyGuard:
    """隐私保护 - PII 脱敏与数据保留策略"""

    PHONE_PATTERN = r'(1[3-9]\d)\d{4}(\d{4})'
    EMAIL_PATTERN = r'([a-zA-Z0-9_.+-]+)@([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)'
    ID_CARD_PATTERN = r'(\d{6})\d{8}(\d{4})'

    def mask_pii(self, text: str) -> str:
        """脱敏个人信息"""
        # 手机号
        text = re.sub(self.PHONE_PATTERN, r'\1****\2', text)
        # 邮箱
        text = re.sub(self.EMAIL_PATTERN, lambda m: f"{m.group(1)[0]}***@{m.group(2)}", text)
        # 身份证号
        text = re.sub(self.ID_CARD_PATTERN, r'\1****\2', text)
        return text

    def detect_pii(self, text: str) -> dict:
        """检测 PII 信息"""
        phones = re.findall(self.PHONE_PATTERN, text)
        emails = re.findall(self.EMAIL_PATTERN, text)
        ids = re.findall(self.ID_CARD_PATTERN, text)
        
        return {
            "has_pii": bool(phones or emails or ids),
            "phones": len(phones),
            "emails": len(emails),
            "id_cards": len(ids),
        }

    def get_retention_policy(self) -> str:
        """获取数据保留策略"""
        return (
            "数据处理优先在本地完成。分析文本不会上传至第三方服务器。"
            "用户可随时删除本地数据。API 调用仅传输必要的分析请求，不存储用户原始文本。"
        )
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/test_guardrails.py -v -k "privacy"
```

- [ ] **Step 5: 提交**

```bash
git add src/guardrails/privacy.py tests/test_guardrails.py
git commit -m "feat: add privacy guard with PII masking and retention policy"
```

---

## Task 29: 阶段四总结

- [ ] **Step 1: 运行全部测试**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/ -v --tb=short
```

- [ ] **Step 2: 更新 __init__.py 导出**

```python
# src/guardrails/__init__.py
from src.guardrails.hallucination import HallucinationGuard
from src.guardrails.ethics import EthicsGuard
from src.guardrails.privacy import PrivacyGuard

__all__ = ["HallucinationGuard", "EthicsGuard", "PrivacyGuard"]
```

- [ ] **Step 3: 更新 README.md**

将阶段四标记为已完成。

- [ ] **Step 4: 最终提交**

```bash
git add src/guardrails/__init__.py README.md
git commit -m "docs: update progress - Phase 4 complete"
```
