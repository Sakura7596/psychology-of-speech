import asyncio
import json
import logging

from src.scraper.base import RawContent

logger = logging.getLogger(__name__)


ANALYSIS_SYSTEM_PROMPT = """你是一个心理学话语分析专家。你的任务是分析一段关于恋爱关系的中文文本，并将其转化为结构化的案例数据。

你需要输出严格的 JSON 格式，包含以下字段：
- "type": 案例类型，必须是以下之一：romantic, dating, romantic_breakup, romantic_conflict, romantic_chase, attachment, emotional, dating_app
- "subtype": 子类型，如 ambiguous_flirting, breadcrumbing, love_bombing, ghosting, breakup_rhetoric, conflict_escalation, anxious_attachment, avoidant_attachment, silent_treatment, jealousy_conflict, soft_rejection, push_pull, situationship, rebound 等
- "text": 原文中最具代表性的1-2句话（如果原文很长，提取关键片段）
- "keywords": 3-5个关键语言标记词列表
- "analysis": 100-200字的心理语言学分析，必须基于具体文本特征
- "psychological_state": 说话者的心理状态（1-2句话）
- "theories": 适用的理论列表（从以下选择：言语行为理论, 会话含义理论, 礼貌策略, 儒家面子理论, 高语境沟通, 权力距离, 关联理论, 面子理论（Goffman）, 框架理论, 评价理论, 适应理论, 认知语言学, 批判性话语分析, 防御机制理论, 归因理论, 情绪感染理论, 认知扭曲理论, 依恋理论, 悲伤五阶段理论, 社会交换理论, 间歇性强化理论, 煤气灯效应理论, 情感操控理论, 共生依赖理论, 自我边界理论）

如果文本不适合转化为案例（如广告、无关内容、过短），返回 {"skip": true, "reason": "..."}

重要：
- 只输出 JSON，不要有其他文字
- analysis 必须基于具体文本特征，不能泛泛而谈
- theories 必须与分析内容有实质关联
- text 必须是原文的精确引用"""


class ContentAnalyzer:
    def __init__(self, llm_client):
        self._llm = llm_client

    async def analyze_to_case(self, text: str, source: str, url: str) -> dict | None:
        prompt = f"来源：{source}\n\n请分析以下文本并转化为案例数据：\n\n---\n{text[:2000]}\n---\n\n请输出 JSON："

        try:
            response = await self._llm.generate(prompt, ANALYSIS_SYSTEM_PROMPT)
            case_data = self._parse_llm_response(response.content)
            case_data["_source"] = source
            case_data["_url"] = url
            return case_data
        except Exception as e:
            logger.warning(f"Failed to analyze content from {source}: {e}")
            return None

    async def batch_analyze(self, contents: list[RawContent], concurrency: int = 3) -> list[dict]:
        semaphore = asyncio.Semaphore(concurrency)

        async def _analyze_one(content: RawContent):
            async with semaphore:
                return await self.analyze_to_case(content.text, content.source, content.url)

        tasks = [_analyze_one(c) for c in contents]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if isinstance(r, dict)]

    def _parse_llm_response(self, content: str) -> dict:
        raw = content.strip()
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]
        data = json.loads(raw.strip())
        if data.get("skip"):
            raise ValueError(f"LLM skipped: {data.get('reason', 'unknown')}")
        return data
