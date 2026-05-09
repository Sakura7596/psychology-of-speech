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

    async def run_pipeline(
        self,
        context: "AnalysisContext",
        agents: dict[str, "BaseAgent"],
    ) -> "AgentResult":
        """运行完整分析管道"""
        from src.agents.base import AgentResult, AnalysisContext

        plan = self.plan_analysis(context.text, context.depth)

        # 第一阶段：分析 Agent
        analysis_agents = ["text_analyst", "psychology_analyst", "logic_analyst"]
        sibling_results = {}

        for agent_name in analysis_agents:
            if agent_name in agents and agent_name in plan.agents:
                try:
                    result = await agents[agent_name].analyze(context)
                    sibling_results[agent_name] = result
                except Exception as e:
                    sibling_results[agent_name] = AgentResult(
                        agent_name=agent_name,
                        analysis={"error": str(e)},
                        confidence=0.0, sources=[], errors=[str(e)],
                    )

        # 第二阶段：报告生成
        report_context = AnalysisContext(
            text=context.text, depth=context.depth,
            language=context.language, features=context.features,
            metadata=context.metadata, sibling_results=sibling_results,
        )

        if "report_generator" in agents:
            return await agents["report_generator"].analyze(report_context)
        else:
            merged = self.merge_results({n: r.analysis for n, r in sibling_results.items()})
            return AgentResult(
                agent_name="orchestrator", analysis=merged,
                confidence=merged.get("overall_confidence", 0.5),
                sources=list(sibling_results.keys()),
            )
