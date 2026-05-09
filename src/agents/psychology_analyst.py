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
        kg = KnowledgeGraph()
        try:
            kg.load("data/graph/psychology_graph.json")
        except Exception:
            pass
        self._retriever = KnowledgeRetriever(
            knowledge_graph=kg,
            case_library=CaseLibrary("data/cases"),
        )

    @property
    def name(self) -> str:
        return "psychology_analyst"

    @property
    def description(self) -> str:
        return "基于语言心理学理论分析说话者心理状态和潜在动机"

    async def analyze(self, context: AnalysisContext) -> AgentResult:
        """执行心理分析"""
        text = context.text
        depth = context.depth

        knowledge_context = self._retriever.get_context_string(text, n_results=3)
        system_prompt = PromptTemplates.get_system_prompt("psychology_analyst")

        depth_map = {
            AnalysisDepth.QUICK: "请快速分析主要心理特征，输出简要 JSON。",
            AnalysisDepth.STANDARD: "请从言语行为、礼貌策略、情感状态三个维度分析，输出 JSON。",
            AnalysisDepth.DEEP: "请从言语行为、面子理论、高语境沟通、权力距离、情感状态五个维度深入分析，输出 JSON。",
        }

        prompt = f"""请分析以下文本的心理特征：

---
{text}
---

{depth_map.get(depth, depth_map[AnalysisDepth.STANDARD])}

相关理论知识：
{knowledge_context}

请以 JSON 格式输出分析结果，包含 confidence 字段（0.0-1.0）。"""

        try:
            response = await self._llm.generate(prompt, system_prompt)
            raw = response.content
        except Exception:
            raw = '{"error": "LLM 调用失败", "confidence": 0.0}'

        analysis = self._parse(raw)
        confidence = analysis.pop("confidence", 0.6)

        return AgentResult(
            agent_name=self.name,
            analysis=analysis,
            confidence=min(max(confidence, 0.0), 1.0),
            sources=["DeepSeek", "KnowledgeGraph", "CaseLibrary"],
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
