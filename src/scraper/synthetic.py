"""合成数据生成器 - 使用 LLM 生成真实恋爱对话数据"""

import asyncio
import json
import logging

from src.scraper.base import RawContent

logger = logging.getLogger(__name__)

GENERATION_PROMPTS = [
    # 暧昧期
    "生成5段真实中国年轻人暧昧期的聊天对话片段，要体现暧昧试探、欲擒故纵、已读不回等特征。每段2-4句话，要有具体细节。",
    "生成5段暧昧期被忽冷忽热对待后的真实内心独白，要体现焦虑、猜测、自我怀疑。",
    "生成5段暧昧期制造嫉妒、推拉、面包屑策略的对话或独白。",
    # 恋爱冲突
    "生成5段恋爱中冷暴力、吵架、翻旧账的真实对话，要体现绝对化指责、伪道歉、情感操控。",
    "生成5段恋爱中缺乏安全感、查手机、追问行踪的真实对话，要体现焦虑型依恋。",
    "生成5段恋爱中感到疲惫、想分手但又舍不得的真实内心独白。",
    # 分手
    "生成5段分手时的委婉话术、直接拒绝、或挽回尝试的对话，要真实。",
    "生成5段分手后看到前任有新欢、想联系前任、或终于释怀的内心独白。",
    "生成5段分手后自我反思、自责、或愤怒阶段的真实独白。",
    # 依恋风格
    "生成5段体现焦虑型依恋的真实对话或独白：害怕被抛弃、需要频繁确认、过度解读信号。",
    "生成5段体现回避型依恋的真实对话或独白：需要空间、害怕亲密、情感不可及。",
    "生成5段体现安全型依恋的真实对话或独白：信任伴侣、表达清晰、给对方空间。",
    # 约会软件
    "生成5段使用约会软件的真实体验：刷到照骗、开场白焦虑、聊着聊着就没了、被取消配对。",
    "生成5段同时和多个人聊天、不知道选谁、或目的不匹配的真实独白。",
    # 特殊场景
    "生成5段异地恋的真实对话或独白：时差、信任、想见面、坚持不下去。",
    "生成5段被PUA或情感操控的真实对话，要体现煤气灯效应、内疚诱导、间歇性强化。",
    "生成5段复合尝试的真实对话：承诺改变、试探对方态度、或最终放弃。",
    "生成5段暗恋的真实独白：想表白又怕、关注对方动态、解读每一个细节。",
]

SYSTEM_PROMPT = """你是一个擅长观察和记录中国年轻人恋爱心理的作家。你需要生成非常真实、接地气的恋爱对话或独白片段。

要求：
1. 语言要自然、口语化，像真实聊天记录或内心独白
2. 要有具体细节（时间、场景、表情符号等）
3. 每段之间要有明显差异，覆盖不同情感状态
4. 不要太文艺，要像普通人说话
5. 每段用 --- 分隔
6. 每段标注：【类型】暧昧/冲突/分手/依恋/约会/其他
"""


class SyntheticDataGenerator:
    def __init__(self, llm_client):
        self._llm = llm_client

    async def generate_batch(self, concurrency: int = 3) -> list[RawContent]:
        semaphore = asyncio.Semaphore(concurrency)

        async def _gen_one(prompt: str) -> list[RawContent]:
            async with semaphore:
                try:
                    response = await self._llm.generate(prompt, SYSTEM_PROMPT)
                    return self._parse_response(response.content)
                except Exception as e:
                    logger.warning(f"Generation failed: {e}")
                    return []

        tasks = [_gen_one(p) for p in GENERATION_PROMPTS]
        results = await asyncio.gather(*tasks)
        all_items = []
        for items in results:
            all_items.extend(items)
        logger.info(f"Generated {len(all_items)} synthetic items")
        return all_items

    def _parse_response(self, content: str) -> list[RawContent]:
        items = []
        segments = content.split("---")
        for i, seg in enumerate(segments):
            seg = seg.strip()
            if len(seg) < 10:
                continue

            # Extract type tag
            content_type = "未分类"
            if "【类型】" in seg:
                parts = seg.split("【类型】")
                seg = parts[0].strip()
                content_type = parts[1].strip().split("\n")[0].strip()

            items.append(RawContent(
                source="synthetic",
                url=f"synthetic://generated/{len(items)}",
                title=content_type,
                text=seg,
            ))
        return items
