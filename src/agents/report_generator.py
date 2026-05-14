# src/agents/report_generator.py
import json

from src.agents.base import BaseAgent, AnalysisContext, AgentResult, AnalysisDepth
from src.llm.client import LLMClient
from src.llm.deepseek import DeepSeekAdapter
from src.llm.prompts import PromptTemplates
from src.config import get_settings
from src.guardrails.ethics import EthicsGuard


class ReportGeneratorAgent(BaseAgent):
    """报告生成 Agent - 整合所有分析结果，生成结构化分析报告"""

    def __init__(self, llm_client: LLMClient | None = None):
        if llm_client is None:
            settings = get_settings()
            adapter = DeepSeekAdapter(
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
                model=settings.llm_model,
                temperature=settings.llm_temperature,
            )
            llm_client = LLMClient(adapter=adapter)
        self._llm = llm_client
        self._ethics = EthicsGuard()

    @property
    def name(self) -> str:
        return "report_generator"

    @property
    def description(self) -> str:
        return "整合所有分析结果，生成结构化分析报告（Markdown/JSON/HTML）"

    async def analyze(self, context: AnalysisContext) -> AgentResult:
        """执行报告生成"""
        text = context.text
        depth = context.depth
        sibling_results = context.sibling_results
        output_format = context.metadata.get("output_format", "markdown")

        analyses_summary = self._build_summary(sibling_results)
        system_prompt = PromptTemplates.get_system_prompt("report_generator")

        depth_map = {
            AnalysisDepth.QUICK: "请生成简洁的分析摘要（1-2 段）。",
            AnalysisDepth.STANDARD: "请生成完整的分析报告，包含各维度分析和综合洞察。",
            AnalysisDepth.DEEP: "请生成深度分析报告，包含理论溯源、交叉分析。",
        }
        format_map = {
            "markdown": "请以 Markdown 格式输出报告。",
            "json": "请以 JSON 格式输出报告。",
            "html": "请以 HTML 格式输出报告。",
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
            response = await self._llm.generate(prompt, system_prompt)
            report = response.content
        except Exception:
            report = f"报告生成失败。原始文本：{text}"

        report = self._ethics.inject_disclaimer(report)
        confidence = self._calc_confidence(sibling_results)

        return AgentResult(
            agent_name=self.name,
            analysis={"report": report, "format": output_format, "depth": depth.value},
            confidence=confidence,
            sources=["DeepSeek"] + [r.agent_name for r in sibling_results.values()],
        )

    def _build_summary(self, sibling_results: dict[str, AgentResult]) -> str:
        """构建各模块分析结果摘要"""
        if not sibling_results:
            return "（无其他模块分析结果）"
        parts = []
        for name, result in sibling_results.items():
            parts.append(
                f"### {name}（置信度：{result.confidence:.2f}）\n"
                f"{json.dumps(result.analysis, ensure_ascii=False, indent=2)}"
            )
        return "\n\n".join(parts)

    def _calc_confidence(self, sibling_results: dict[str, AgentResult]) -> float:
        """基于各模块置信度加权计算报告置信度"""
        if not sibling_results:
            return 0.5
        weights = {
            "text_analyst": 0.3,
            "psychology_analyst": 0.4,
            "logic_analyst": 0.3,
        }
        total_w = sum(weights.get(n, 0.3) for n in sibling_results)
        weighted = sum(
            r.confidence * weights.get(n, 0.3) for n, r in sibling_results.items()
        )
        return min(max(weighted / total_w, 0.0), 1.0) if total_w > 0 else 0.5
