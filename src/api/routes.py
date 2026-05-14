# src/api/routes.py
import asyncio
import json
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from src.api.models import (
    AnalyzeRequest, AnalyzeResponse, HealthResponse,
    ScrapeRequest, ScrapeResponse, ScrapeStatusResponse,
)
from src.agents.base import AnalysisContext, AnalysisDepth, AgentResult
from src.agents.orchestrator import Orchestrator

router = APIRouter()

VALID_DEPTHS = {"quick", "standard", "deep"}
VALID_FORMATS = {"markdown", "json", "html"}

ANALYSIS_AGENTS = ["text_analyst", "psychology_analyst", "logic_analyst"]


def _get_or_create_agents(req: Request) -> dict:
    """从 app state 获取已初始化的 agents，未初始化则创建（共享 LLMClient）"""
    agents = getattr(req.app.state, "agents", None)
    if agents is None:
        from src.config import get_settings
        from src.llm.client import LLMClient
        from src.llm.deepseek import DeepSeekAdapter
        from src.agents.text_analyst import TextAnalystAgent
        from src.agents.psychology_analyst import PsychologyAnalystAgent
        from src.agents.logic_analyst import LogicAnalystAgent
        from src.agents.report_generator import ReportGeneratorAgent

        settings = get_settings()
        adapter = DeepSeekAdapter(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
        shared_llm = LLMClient(adapter=adapter)

        agents = {
            "text_analyst": TextAnalystAgent(),
            "psychology_analyst": PsychologyAnalystAgent(llm_client=shared_llm),
            "logic_analyst": LogicAnalystAgent(llm_client=shared_llm),
            "report_generator": ReportGeneratorAgent(llm_client=shared_llm),
        }
        req.app.state.agents = agents
    return agents


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", version="0.1.0")


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest, req: Request):
    if request.depth not in VALID_DEPTHS:
        raise HTTPException(status_code=400, detail=f"无效的分析深度: {request.depth}")
    if request.output_format not in VALID_FORMATS:
        raise HTTPException(status_code=400, detail=f"无效的输出格式: {request.output_format}")

    try:
        agents = _get_or_create_agents(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent 初始化失败: {str(e)}")

    depth = AnalysisDepth(request.depth)
    context = AnalysisContext(
        text=request.text, depth=depth,
        metadata={"output_format": request.output_format},
    )

    orchestrator = Orchestrator()
    try:
        result = await orchestrator.run_pipeline(context, agents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")

    return AnalyzeResponse(
        report=result.analysis.get("report", str(result.analysis)),
        analyses={n: r.analysis for n, r in context.sibling_results.items()} if context.sibling_results else {},
        confidence=result.confidence,
        depth=request.depth,
    )


@router.post("/analyze/stream")
async def analyze_stream(request: AnalyzeRequest, req: Request):
    """SSE 流式分析端点"""
    if request.depth not in VALID_DEPTHS:
        raise HTTPException(status_code=400, detail=f"无效的分析深度: {request.depth}")

    try:
        agents = _get_or_create_agents(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent 初始化失败: {str(e)}")

    async def event_generator():
        yield f"data: {json.dumps({'type': 'start', 'message': '开始分析'})}\n\n"

        depth = AnalysisDepth(request.depth)
        context = AnalysisContext(
            text=request.text, depth=depth,
            metadata={"output_format": request.output_format},
        )

        # 用 asyncio.Queue 收集进度事件，实现并行执行 + 实时进度
        progress_queue: asyncio.Queue[str] = asyncio.Queue()

        async def run_with_progress(name: str, agent) -> tuple[str, AgentResult | None]:
            await progress_queue.put(
                f"data: {json.dumps({'type': 'progress', 'agent': name, 'status': 'running'})}\n\n"
            )
            try:
                result = await agent.analyze(context)
                await progress_queue.put(
                    f"data: {json.dumps({'type': 'progress', 'agent': name, 'status': 'done', 'confidence': result.confidence})}\n\n"
                )
                return name, result
            except Exception as e:
                await progress_queue.put(
                    f"data: {json.dumps({'type': 'progress', 'agent': name, 'status': 'error', 'error': str(e)})}\n\n"
                )
                return name, None

        # 并行执行三个分析 Agent
        async def _run_all():
            return await asyncio.gather(*(run_with_progress(n, agents[n]) for n in ANALYSIS_AGENTS))

        analysis_task = asyncio.create_task(_run_all())

        # 持续 drain 进度队列，直到分析任务完成
        while not analysis_task.done():
            try:
                event = await asyncio.wait_for(progress_queue.get(), timeout=0.1)
                yield event
            except asyncio.TimeoutError:
                pass

        # 收集剩余进度事件
        results_list = analysis_task.result()
        while not progress_queue.empty():
            yield progress_queue.get_nowait()

        sibling_results: dict[str, AgentResult] = {name: r for name, r in results_list if r is not None}

        # 阶段4: 报告生成（流式输出）
        yield f"data: {json.dumps({'type': 'progress', 'agent': 'report_generator', 'status': 'running'})}\n\n"

        try:
            from src.llm.prompts import PromptTemplates

            report_agent = agents["report_generator"]
            analyses_summary = report_agent._build_summary(sibling_results)
            system_prompt = PromptTemplates.get_system_prompt("report_generator")
            prompt = f"原始文本：\n---\n{context.text}\n---\n\n各模块分析结果：\n{analyses_summary}\n\n请生成完整的分析报告，包含各维度分析和综合洞察。请以 Markdown 格式输出。重要：报告末尾必须包含免责声明。"

            full_report = ""
            if hasattr(report_agent._llm, "generate_stream"):
                async for chunk in report_agent._llm.generate_stream(prompt, system_prompt):
                    full_report += chunk
                    yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
            else:
                response = await report_agent._llm.generate(prompt, system_prompt)
                full_report = response.content
                yield f"data: {json.dumps({'type': 'chunk', 'content': full_report})}\n\n"

            full_report = report_agent._ethics.inject_disclaimer(full_report)

            analyses = {n: r.analysis for n, r in sibling_results.items()}
            yield f"data: {json.dumps({'type': 'done', 'report': full_report, 'analyses': analyses, 'confidence': report_agent._calc_confidence(sibling_results), 'depth': request.depth})}\n\n"
            yield f"data: {json.dumps({'type': 'progress', 'agent': 'report_generator', 'status': 'done'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _init_scrape_pipeline():
    """初始化爬取管道"""
    from src.config import get_settings
    from src.llm.client import LLMClient
    from src.llm.deepseek import DeepSeekAdapter
    from src.scraper.pipeline import ScrapePipeline
    from src.scraper.http_client import ResilientHttpClient
    from src.scraper.cleaners import ContentCleaner
    from src.scraper.analyzer import ContentAnalyzer
    from src.scraper.validator import ContentValidator
    from src.scraper.storage import StorageManager
    from src.scraper.sources.zhihu import ZhihuScraper
    from src.scraper.sources.douban import DoubanScraper
    from src.scraper.sources.xiaohongshu import XiaohongshuScraper
    from src.scraper.sources.psychology_blog import PsychologyBlogScraper

    settings = get_settings()
    http = ResilientHttpClient(respect_robots=True)
    llm_adapter = DeepSeekAdapter(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
    llm_client = LLMClient(adapter=llm_adapter)

    scrapers = {
        "zhihu": ZhihuScraper(http),
        "douban": DoubanScraper(http),
        "xiaohongshu": XiaohongshuScraper(http),
        "psychology_blog": PsychologyBlogScraper(http),
    }

    return ScrapePipeline(
        scrapers=scrapers,
        cleaner=ContentCleaner(),
        analyzer=ContentAnalyzer(llm_client),
        validator=ContentValidator(),
        storage=StorageManager(),
    )


@router.get("/scrape/sources", response_model=ScrapeStatusResponse)
async def list_scrape_sources():
    return ScrapeStatusResponse(
        status="ok",
        sources_available=["zhihu", "douban", "xiaohongshu", "psychology_blog"],
    )


@router.post("/scrape", response_model=ScrapeResponse)
async def trigger_scrape(request: ScrapeRequest, req: Request):
    pipeline = getattr(req.app.state, "scrape_pipeline", None)
    if pipeline is None:
        try:
            pipeline = _init_scrape_pipeline()
            req.app.state.scrape_pipeline = pipeline
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"爬取管道初始化失败: {str(e)}")

    try:
        stats = await pipeline.run(
            query=request.query,
            sources=request.sources,
            max_items_per_source=request.max_items_per_source,
            dry_run=request.dry_run,
        )
        return ScrapeResponse(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"采集失败: {str(e)}")
