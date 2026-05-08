# src/agents/logic_analyst.py
import json
import asyncio
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
        self._retriever = KnowledgeRetriever(
            case_library=CaseLibrary("data/cases"),
        )

    @property
    def name(self) -> str:
        return "logic_analyst"

    @property
    def description(self) -> str:
        return "分析论证结构，识别逻辑谬误和隐含假设"

    def analyze(self, context: AnalysisContext) -> AgentResult:
        text = context.text
        depth = context.depth
        knowledge_context = self._retriever.get_context_string(text, n_results=3)
        system_prompt = PromptTemplates.get_system_prompt("logic_analyst")

        depth_map = {
            AnalysisDepth.QUICK: "请快速识别主要论证结构和明显谬误，输出简要 JSON。",
            AnalysisDepth.STANDARD: "请分析论证结构（前提→推理→结论）、逻辑谬误、隐含假设，输出 JSON。",
            AnalysisDepth.DEEP: "请深入分析论证结构、所有逻辑谬误、隐含前提、论证强度、反事实分析，输出 JSON。",
        }

        prompt = f"""请分析以下文本的逻辑结构：

---
{text}
---

{depth_map.get(depth, depth_map[AnalysisDepth.STANDARD])}

相关谬误案例：
{knowledge_context}

请以 JSON 格式输出分析结果，包含 confidence 字段（0.0-1.0）。"""

        try:
            response = asyncio.run(self._llm.generate(prompt, system_prompt))
            raw = response.content
        except Exception:
            raw = '{"error": "LLM 调用失败", "confidence": 0.0}'

        analysis = self._parse(raw)
        confidence = analysis.pop("confidence", 0.6)

        return AgentResult(
            agent_name=self.name,
            analysis=analysis,
            confidence=min(max(confidence, 0.0), 1.0),
            sources=["DeepSeek", "CaseLibrary"],
        )

    def _parse(self, raw: str) -> dict:
        """解析 LLM 返回的 JSON"""
        try:
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0]
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0]
            return json.loads(raw.strip())
        except (json.JSONDecodeError, IndexError):
            return {"raw_response": raw, "parse_error": True}
