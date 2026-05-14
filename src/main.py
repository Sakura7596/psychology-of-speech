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
    plan = await orchestrator.plan_analysis(
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

    await client.close()
    return result


def serve():
    """启动 Web 服务"""
    import uvicorn
    from src.api.app import create_app
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)


def main():
    """CLI 入口"""
    if len(sys.argv) < 2:
        print("用法:")
        print("  python -m src.main '要分析的文本' [depth]    # CLI 分析")
        print("  python -m src.main serve                     # 启动 Web 服务")
        print("depth: quick / standard / deep (默认 standard)")
        sys.exit(1)

    if sys.argv[1] == "serve":
        serve()
        return

    text = sys.argv[1]
    depth = sys.argv[2] if len(sys.argv) > 2 else "standard"

    result = asyncio.run(analyze_text(text, depth))

    print("\n" + "=" * 60)
    print("分析结果：")
    print("=" * 60)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
