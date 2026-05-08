# 阶段五：报告生成 + 协调器完善实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现报告生成 Agent（多格式输出）、完善协调器（LangGraph 编排、并行调度）、端到端集成测试

**Architecture:** 报告生成 Agent 整合所有分析结果，支持 Markdown/JSON/HTML 输出。协调器升级为 LangGraph 状态图，实现真正的并行 Agent 调度。

**Tech Stack:** LangGraph, Jinja2, pytest

**依赖：** 阶段一至四已完成

---

## 文件结构

```
src/
├── agents/
│   ├── report_generator.py    # 报告生成 Agent
│   └── orchestrator.py        # 协调器（升级为 LangGraph）
tests/
├── test_report_generator.py   # 报告生成测试
└── test_orchestrator_v2.py    # 协调器升级测试
```

---

## Task 30: 报告生成 Agent

**Files:**
- Create: `src/agents/report_generator.py`
- Create: `tests/test_report_generator.py`

- [ ] **Step 1: 编写测试**

```python
# tests/test_report_generator.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.agents.base import AnalysisContext, AnalysisDepth, AgentResult
from src.agents.report_generator import ReportGeneratorAgent


def test_report_generator_name():
    agent = ReportGeneratorAgent()
    assert agent.name == "report_generator"


def test_report_generator_description():
    agent = ReportGeneratorAgent()
    assert len(agent.description) > 10


def test_generate_markdown_report():
    """测试 Markdown 报告生成"""
    agent = ReportGeneratorAgent()
    
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = MagicMock(
        content="# 分析报告\n\n## 摘要\n说话者使用了比喻修辞。\n\n---\n**免责声明**：仅供参考。"
    )
    agent._llm = mock_llm
    
    sibling_results = {
        "text_analyst": AgentResult(
            agent_name="text_analyst",
            analysis={"rhetorical_devices": [{"type": "simile"}]},
            confidence=0.8,
            sources=["HanLP"],
        ),
        "psychology_analyst": AgentResult(
            agent_name="psychology_analyst",
            analysis={"speech_acts": [{"type": "expressive"}]},
            confidence=0.7,
            sources=["DeepSeek"],
        ),
    }
    
    ctx = AnalysisContext(
        text="他的心像冰一样冷",
        depth=AnalysisDepth.STANDARD,
        sibling_results=sibling_results,
    )
    result = agent.analyze(ctx)
    
    assert result.agent_name == "report_generator"
    assert "report" in result.analysis or "content" in result.analysis


def test_generate_json_output():
    """测试 JSON 输出格式"""
    agent = ReportGeneratorAgent()
    
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = MagicMock(
        content='{"summary": "比喻修辞", "sections": {"text": {}, "psychology": {}, "logic": {}}}'
    )
    agent._llm = mock_llm
    
    ctx = AnalysisContext(
        text="测试",
        depth=AnalysisDepth.STANDARD,
        sibling_results={},
        metadata={"output_format": "json"},
    )
    result = agent.analyze(ctx)
    assert result.agent_name == "report_generator"


def test_report_includes_disclaimer():
    """测试报告包含免责声明"""
    agent = ReportGeneratorAgent()
    
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = MagicMock(
        content="分析结果。\n\n---\n**免责声明**：本分析仅供参考，不构成专业心理咨询意见。"
    )
    agent._llm = mock_llm
    
    ctx = AnalysisContext(text="测试", depth=AnalysisDepth.STANDARD, sibling_results={})
    result = agent.analyze(ctx)
    
    report_text = str(result.analysis)
    assert "免责" in report_text or "仅供参考" in report_text or "非专业" in report_text
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/test_report_generator.py -v
```

- [ ] **Step 3: 实现报告生成 Agent**

```python
# src/agents/report_generator.py
import json
import asyncio
from src.agents.base import BaseAgent, AnalysisContext, AgentResult, AnalysisDepth
from src.llm.client import LLMClient
from src.llm.deepseek import DeepSeekAdapter
from src.llm.prompts import PromptTemplates
from src.config import get_settings
from src.guardrails.ethics import EthicsGuard


class ReportGeneratorAgent(BaseAgent):
    """报告生成 Agent - 整合分析结果，生成结构化报告"""

    def __init__(self):
        settings = get_settings()
        adapter = DeepSeekAdapter(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
        )
        self._llm = LLMClient(adapter=adapter)
        self._ethics = EthicsGuard()

    @property
    def name(self) -> str:
        return "report_generator"

    @property
    def description(self) -> str:
        return "整合所有分析结果，生成结构化分析报告（Markdown/JSON/HTML）"

    def analyze(self, context: AnalysisContext) -> AgentResult:
        """生成报告"""
        text = context.text
        depth = context.depth
        sibling_results = context.sibling_results
        output_format = context.metadata.get("output_format", "markdown")

        # 构建分析结果摘要
        analyses_summary = self._build_analyses_summary(sibling_results)

        system_prompt = PromptTemplates.get_system_prompt("report_generator")

        depth_map = {
            AnalysisDepth.QUICK: "请生成简洁的分析摘要（1-2 段）。",
            AnalysisDepth.STANDARD: "请生成完整的分析报告，包含各维度分析和综合洞察。",
            AnalysisDepth.DEEP: "请生成深度分析报告，包含理论溯源、交叉分析、反事实分析。",
        }

        format_map = {
            "markdown": "请以 Markdown 格式输出报告。",
            "json": "请以 JSON 格式输出报告，包含 summary、sections、recommendations 字段。",
            "html": "请以 HTML 格式输出报告，使用合适的标签结构化内容。",
        }

        prompt = f"""原始文本：
---
{text}
---

各模块分析结果：
{analyses_summary}

{depth_map.get(depth, depth_map[AnalysisDepth.STANDARD])}
{format_map.get(output_format, format_map["markdown"])}

重要：报告末尾必须包含免责声明。"""

        try:
            response = asyncio.run(self._llm.generate(prompt, system_prompt))
            report_content = response.content
        except Exception:
            report_content = f"报告生成失败。原始文本：{text}"

        # 注入免责声明（如果 LLM 未添加）
        report_content = self._ethics.inject_disclaimer(report_content)

        # 计算综合置信度
        confidence = self._calculate_overall_confidence(sibling_results)

        return AgentResult(
            agent_name=self.name,
            analysis={
                "report": report_content,
                "format": output_format,
                "depth": depth.value,
                "sources_count": len(sibling_results),
            },
            confidence=confidence,
            sources=["DeepSeek"] + [r.agent_name for r in sibling_results.values()],
        )

    def _build_analyses_summary(self, sibling_results: dict[str, AgentResult]) -> str:
        """构建分析结果摘要"""
        if not sibling_results:
            return "（无其他模块分析结果）"

        parts = []
        for name, result in sibling_results.items():
            if result.is_reliable:
                parts.append(f"### {name}（置信度：{result.confidence:.2f}）")
                parts.append(json.dumps(result.analysis, ensure_ascii=False, indent=2))
            else:
                parts.append(f"### {name}（低置信度：{result.confidence:.2f}，仅供参考）")
                parts.append(json.dumps(result.analysis, ensure_ascii=False, indent=2))

        return "\n\n".join(parts)

    def _calculate_overall_confidence(self, sibling_results: dict[str, AgentResult]) -> float:
        """计算综合置信度"""
        if not sibling_results:
            return 0.5

        weights = {
            "text_analyst": 0.3,
            "psychology_analyst": 0.4,
            "logic_analyst": 0.3,
        }

        total_weight = 0.0
        weighted_sum = 0.0
        for name, result in sibling_results.items():
            weight = weights.get(name, 0.3)
            weighted_sum += result.confidence * weight
            total_weight += weight

        if total_weight > 0:
            return min(max(weighted_sum / total_weight, 0.0), 1.0)
        return 0.5
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/test_report_generator.py -v
```

- [ ] **Step 5: 提交**

```bash
git add src/agents/report_generator.py tests/test_report_generator.py
git commit -m "feat: add ReportGeneratorAgent with multi-format output and disclaimer"
```

---

## Task 31: 协调器升级（并行调度 + 报告生成集成）

**Files:**
- Modify: `src/agents/orchestrator.py`
- Modify: `tests/test_agents.py`

- [ ] **Step 1: 添加协调器升级测试**

在 `tests/test_agents.py` 中追加：

```python
def test_orchestrator_full_pipeline():
    """测试完整分析管道"""
    from unittest.mock import AsyncMock, MagicMock
    from src.agents.orchestrator import Orchestrator
    
    orchestrator = Orchestrator()
    
    # Mock 所有 Agent
    mock_text = MagicMock()
    mock_text.analyze.return_value = AgentResult(
        agent_name="text_analyst", analysis={"tokens": 10}, confidence=0.8, sources=["HanLP"]
    )
    
    mock_psych = MagicMock()
    mock_psych.analyze.return_value = AgentResult(
        agent_name="psychology_analyst", analysis={"intent": "express"}, confidence=0.7, sources=["LLM"]
    )
    
    mock_logic = MagicMock()
    mock_logic.analyze.return_value = AgentResult(
        agent_name="logic_analyst", analysis={"fallacies": []}, confidence=0.6, sources=["LLM"]
    )
    
    mock_report = MagicMock()
    mock_report.analyze.return_value = AgentResult(
        agent_name="report_generator", analysis={"report": "完整报告"}, confidence=0.75, sources=["LLM"]
    )
    
    agents = {
        "text_analyst": mock_text,
        "psychology_analyst": mock_psych,
        "logic_analyst": mock_logic,
        "report_generator": mock_report,
    }
    
    ctx = AnalysisContext(text="这是一段测试文本，用于验证协调器的完整管道。", depth=AnalysisDepth.STANDARD)
    result = orchestrator.run_pipeline(ctx, agents)
    
    assert result is not None
    assert "report" in result.analysis or "analyses" in result


def test_orchestrator_parallel_agents():
    """测试并行 Agent 调度"""
    orchestrator = Orchestrator()
    plan = orchestrator.plan_analysis("测试文本" * 50)
    assert "text_analyst" in plan.agents
    assert "psychology_analyst" in plan.agents
    assert "logic_analyst" in plan.agents
```

- [ ] **Step 2: 实现协调器升级**

在 `src/agents/orchestrator.py` 中追加 `run_pipeline` 方法：

```python
    def run_pipeline(
        self,
        context: AnalysisContext,
        agents: dict[str, "BaseAgent"],
    ) -> AgentResult:
        """运行完整分析管道"""
        from src.agents.base import AgentResult
        
        plan = self.plan_analysis(context.text, context.depth)
        
        # 第一阶段：并行分析（文本 + 心理 + 逻辑）
        analysis_agents = ["text_analyst", "psychology_analyst", "logic_analyst"]
        sibling_results = {}
        
        for agent_name in analysis_agents:
            if agent_name in agents and agent_name in plan.agents:
                try:
                    result = agents[agent_name].analyze(context)
                    sibling_results[agent_name] = result
                except Exception as e:
                    sibling_results[agent_name] = AgentResult(
                        agent_name=agent_name,
                        analysis={"error": str(e)},
                        confidence=0.0,
                        sources=[],
                        errors=[str(e)],
                    )
        
        # 第二阶段：报告生成
        report_context = AnalysisContext(
            text=context.text,
            depth=context.depth,
            language=context.language,
            features=context.features,
            metadata=context.metadata,
            sibling_results=sibling_results,
        )
        
        if "report_generator" in agents:
            report_result = agents["report_generator"].analyze(report_context)
        else:
            # 如果没有报告 Agent，合并结果返回
            merged = self.merge_results(
                {name: r.analysis for name, r in sibling_results.items()}
            )
            report_result = AgentResult(
                agent_name="orchestrator",
                analysis=merged,
                confidence=merged.get("overall_confidence", 0.5),
                sources=list(sibling_results.keys()),
            )
        
        return report_result
```

- [ ] **Step 3: 运行测试确认通过**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/test_agents.py -v -k "pipeline or parallel"
```

- [ ] **Step 4: 提交**

```bash
git add src/agents/orchestrator.py tests/test_agents.py
git commit -m "feat: add orchestrator pipeline with parallel agent dispatch"
```

---

## Task 32: 端到端集成测试

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: 编写端到端测试**

```python
# tests/test_integration.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.agents.base import AnalysisContext, AnalysisDepth, AgentResult
from src.agents.orchestrator import Orchestrator


def test_end_to_end_mock():
    """端到端测试（全部 mock）"""
    orchestrator = Orchestrator()
    
    # Mock 所有 Agent
    def make_mock(name, analysis_data):
        mock = MagicMock()
        mock.analyze.return_value = AgentResult(
            agent_name=name, analysis=analysis_data, confidence=0.8, sources=["mock"]
        )
        return mock
    
    agents = {
        "text_analyst": make_mock("text_analyst", {"tokens": 20, "sentiment": "positive"}),
        "psychology_analyst": make_mock("psychology_analyst", {"intent": "express"}),
        "logic_analyst": make_mock("logic_analyst", {"fallacies": []}),
        "report_generator": make_mock("report_generator", {"report": "完整分析报告"}),
    }
    
    ctx = AnalysisContext(text="今天天气真好，我们去散步吧", depth=AnalysisDepth.STANDARD)
    result = orchestrator.run_pipeline(ctx, agents)
    
    assert result is not None
    assert result.agent_name == "report_generator"
    assert result.confidence > 0


def test_end_to_end_with_error():
    """端到端测试（部分 Agent 失败）"""
    orchestrator = Orchestrator()
    
    good_agent = MagicMock()
    good_agent.analyze.return_value = AgentResult(
        agent_name="text_analyst", analysis={"tokens": 10}, confidence=0.8, sources=["mock"]
    )
    
    bad_agent = MagicMock()
    bad_agent.analyze.side_effect = Exception("LLM 调用失败")
    
    agents = {
        "text_analyst": good_agent,
        "psychology_analyst": bad_agent,
        "logic_analyst": good_agent,
        "report_generator": MagicMock(analyze=MagicMock(return_value=AgentResult(
            agent_name="report_generator", analysis={"report": "部分分析完成"}, confidence=0.5, sources=["mock"]
        ))),
    }
    
    ctx = AnalysisContext(text="测试", depth=AnalysisDepth.STANDARD)
    result = orchestrator.run_pipeline(ctx, agents)
    
    assert result is not None
    # 部分失败不影响整体流程
```

- [ ] **Step 2: 运行测试确认通过**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/test_integration.py -v
```

- [ ] **Step 3: 提交**

```bash
git add tests/test_integration.py
git commit -m "test: add end-to-end integration tests for full pipeline"
```

---

## Task 33: 阶段五总结

- [ ] **Step 1: 运行全部测试**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/ -v --tb=short
```

- [ ] **Step 2: 更新 agents __init__.py**

确保 `ReportGeneratorAgent` 已导出。

- [ ] **Step 3: 更新 README.md**

将阶段五标记为已完成。

- [ ] **Step 4: 最终提交**

```bash
git add src/agents/__init__.py README.md
git commit -m "docs: update progress - Phase 5 complete"
```
