import asyncio
import json
import logging
from dataclasses import dataclass, field
from src.agents.base import AnalysisDepth

logger = logging.getLogger(__name__)


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

    def __init__(self, llm_client=None):
        self._llm = llm_client

    async def plan_analysis(
        self,
        text: str,
        requested_depth: AnalysisDepth | None = None,
    ) -> AnalysisPlan:
        """根据文本特征生成分析计划（可选 LLM 驱动）"""
        # 如果有 LLM，尝试用 LLM 规划
        if self._llm and len(text) > 50:
            try:
                plan = await self._llm_plan(text, requested_depth)
                if plan:
                    return plan
            except Exception as e:
                logger.warning(f"LLM 规划失败，回退到规则规划: {e}")

        # 规则驱动的后备规划
        return self._rule_based_plan(text, requested_depth)

    def _rule_based_plan(
        self, text: str, requested_depth: AnalysisDepth | None = None
    ) -> AnalysisPlan:
        """基于规则的分析计划"""
        text_len = len(text)

        if requested_depth:
            depth = requested_depth
        elif text_len < self.SHORT_TEXT_THRESHOLD:
            depth = AnalysisDepth.QUICK
        elif text_len > self.LONG_TEXT_THRESHOLD:
            depth = AnalysisDepth.DEEP
        else:
            depth = AnalysisDepth.STANDARD

        agents = ["text_analyst", "psychology_analyst", "logic_analyst"]
        agent_params: dict[str, dict] = {}

        if text_len < self.SHORT_TEXT_THRESHOLD:
            agent_params["logic_analyst"] = {"skip_deep": True}

        segment = text_len > self.LONG_TEXT_THRESHOLD

        return AnalysisPlan(
            agents=agents,
            depth=depth,
            agent_params=agent_params,
            segment=segment,
        )

    async def _llm_plan(
        self, text: str, requested_depth: AnalysisDepth | None = None
    ) -> AnalysisPlan | None:
        """用 LLM 判断文本类型和最佳分析策略"""
        from src.llm.prompts import PromptTemplates
        system_prompt = PromptTemplates.get_system_prompt("orchestrator")
        prompt = f"""请分析以下文本特征，返回 JSON 格式的分析计划：

文本：
---
{text[:500]}
---

请返回：
{{"text_type": "对话/独白/论述/其他", "emotional_intensity": "低/中/高", "recommended_depth": "quick/standard/deep", "priority_agents": ["text_analyst", "psychology_analyst", "logic_analyst"], "reasoning": "简要说明"}}"""

        response = await self._llm.generate(prompt, system_prompt, use_cache=True)
        try:
            raw = response.content
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0]
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0]
            plan_data = json.loads(raw.strip())

            depth_str = plan_data.get("recommended_depth", "standard")
            depth = requested_depth or AnalysisDepth(depth_str)
            priority_agents = plan_data.get("priority_agents", ["text_analyst", "psychology_analyst", "logic_analyst"])

            # 确保所有 agent 名称有效
            valid_agents = {"text_analyst", "psychology_analyst", "logic_analyst"}
            agents = [a for a in priority_agents if a in valid_agents]
            if not agents:
                agents = ["text_analyst", "psychology_analyst", "logic_analyst"]

            logger.info(f"LLM 规划: type={plan_data.get('text_type')}, depth={depth.value}, agents={agents}")
            return AnalysisPlan(agents=agents, depth=depth)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"LLM 规划结果解析失败: {e}")
            return None

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

        merged["low_confidence_warnings"] = [
            name for name, conf in merged["confidence_scores"].items()
            if conf < 0.3
        ]

        return merged

    async def _run_agent(
        self, agent_name: str, agent: "BaseAgent", context: "AnalysisContext"
    ) -> tuple[str, "AgentResult"]:
        """运行单个 Agent，捕获异常降级"""
        from src.agents.base import AgentResult
        try:
            result = await agent.analyze(context)
            return agent_name, result
        except Exception as e:
            logger.warning(f"Agent {agent_name} 执行失败: {e}")
            return agent_name, AgentResult(
                agent_name=agent_name,
                analysis={"error": str(e)},
                confidence=0.0, sources=[], errors=[str(e)],
            )

    async def run_pipeline(
        self,
        context: "AnalysisContext",
        agents: dict[str, "BaseAgent"],
    ) -> "AgentResult":
        """运行完整分析管道（并行执行分析 Agent）"""
        from src.agents.base import AgentResult, AnalysisContext
        from src.guardrails.privacy import PrivacyGuard

        # 隐私脱敏
        privacy = PrivacyGuard()
        context.text = privacy.mask_pii(context.text)

        plan = await self.plan_analysis(context.text, context.depth)

        # 第一阶段：分析 Agent 并行执行
        analysis_agents = ["text_analyst", "psychology_analyst", "logic_analyst"]
        tasks = [
            self._run_agent(name, agents[name], context)
            for name in analysis_agents
            if name in agents and name in plan.agents
        ]

        results = await asyncio.gather(*tasks)
        sibling_results = dict(results)

        # 将分析结果挂到原始 context 上，供调用方读取
        context.sibling_results = sibling_results

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
