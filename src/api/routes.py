# src/api/routes.py
from fastapi import APIRouter, HTTPException
from src.api.models import AnalyzeRequest, AnalyzeResponse, HealthResponse
from src.agents.base import AnalysisContext, AnalysisDepth
from src.agents.orchestrator import Orchestrator

router = APIRouter()

VALID_DEPTHS = {"quick", "standard", "deep"}
VALID_FORMATS = {"markdown", "json", "html"}


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", version="0.1.0")


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    if request.depth not in VALID_DEPTHS:
        raise HTTPException(status_code=400, detail=f"无效的分析深度: {request.depth}")
    if request.output_format not in VALID_FORMATS:
        raise HTTPException(status_code=400, detail=f"无效的输出格式: {request.output_format}")

    depth = AnalysisDepth(request.depth)
    context = AnalysisContext(
        text=request.text, depth=depth,
        metadata={"output_format": request.output_format},
    )

    try:
        from src.agents.text_analyst import TextAnalystAgent
        from src.agents.psychology_analyst import PsychologyAnalystAgent
        from src.agents.logic_analyst import LogicAnalystAgent
        from src.agents.report_generator import ReportGeneratorAgent

        agents = {
            "text_analyst": TextAnalystAgent(),
            "psychology_analyst": PsychologyAnalystAgent(),
            "logic_analyst": LogicAnalystAgent(),
            "report_generator": ReportGeneratorAgent(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent 初始化失败: {str(e)}")

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
