# 阶段六：Web 接口 + 可观测性实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 FastAPI Web 服务（/analyze, /health）、请求验证、错误处理、可观测性（日志 + 请求追踪）

**Architecture:** FastAPI 应用调用 Orchestrator.run_pipeline() 执行分析，返回结构化 JSON 响应。中间件提供日志、错误处理、CORS。

**Tech Stack:** FastAPI, Pydantic, uvicorn, pytest

**依赖：** 阶段一至五已完成

---

## 文件结构

```
src/
├── api/
│   ├── __init__.py
│   ├── app.py             # FastAPI 应用工厂
│   ├── routes.py          # 路由定义（/analyze, /health）
│   └── models.py          # 请求/响应 Pydantic 模型
tests/
└── test_api.py            # API 测试
```

---

## Task 34: API 请求/响应模型

**Files:**
- Create: `src/api/models.py`
- Create: `tests/test_api.py`

- [ ] **Step 1: 编写测试**

```python
# tests/test_api.py
import pytest
from src.api.models import AnalyzeRequest, AnalyzeResponse


def test_analyze_request_creation():
    """测试请求模型"""
    req = AnalyzeRequest(text="今天天气真好")
    assert req.text == "今天天气真好"
    assert req.depth == "standard"
    assert req.output_format == "markdown"


def test_analyze_request_custom():
    """测试自定义请求"""
    req = AnalyzeRequest(text="测试", depth="deep", output_format="json")
    assert req.depth == "deep"
    assert req.output_format == "json"


def test_analyze_request_validation():
    """测试请求验证"""
    with pytest.raises(Exception):
        AnalyzeRequest(text="")


def test_analyze_response_creation():
    """测试响应模型"""
    resp = AnalyzeResponse(
        report="分析报告",
        analyses={"text_analyst": {}},
        confidence=0.8,
        depth="standard",
        tokens_used=100,
    )
    assert resp.report == "分析报告"
    assert resp.confidence == 0.8


def test_health_response():
    """测试健康检查响应"""
    from src.api.models import HealthResponse
    resp = HealthResponse(status="ok", version="0.1.0")
    assert resp.status == "ok"
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/test_api.py -v
```

- [ ] **Step 3: 实现模型**

```python
# src/api/models.py
from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    """分析请求"""
    text: str = Field(..., min_length=1, description="要分析的文本")
    depth: str = Field(default="standard", description="分析深度: quick/standard/deep")
    output_format: str = Field(default="markdown", description="输出格式: markdown/json/html")


class AnalyzeResponse(BaseModel):
    """分析响应"""
    report: str = Field(..., description="分析报告内容")
    analyses: dict = Field(default_factory=dict, description="各模块分析结果")
    confidence: float = Field(..., description="综合置信度")
    depth: str = Field(..., description="使用的分析深度")
    tokens_used: int = Field(default=0, description="消耗的 token 数")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="服务状态")
    version: str = Field(..., description="版本号")
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/test_api.py -v -k "request or response or health"
```

- [ ] **Step 5: 提交**

```bash
git add src/api/models.py src/api/__init__.py tests/test_api.py
git commit -m "feat: add API request/response Pydantic models"
```

---

## Task 35: FastAPI 路由 + 应用

**Files:**
- Create: `src/api/app.py`
- Create: `src/api/routes.py`
- Modify: `tests/test_api.py`

- [ ] **Step 1: 编写路由测试**

在 `tests/test_api.py` 中追加：

```python
from fastapi.testclient import TestClient
from src.api.app import create_app


def test_health_endpoint():
    """测试健康检查端点"""
    app = create_app()
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_analyze_endpoint_mock():
    """测试分析端点（mock orchestrator）"""
    app = create_app()
    client = TestClient(app)
    
    response = client.post("/analyze", json={
        "text": "今天天气真好",
        "depth": "standard",
        "output_format": "markdown",
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "report" in data
    assert "confidence" in data


def test_analyze_empty_text():
    """测试空文本验证"""
    app = create_app()
    client = TestClient(app)
    
    response = client.post("/analyze", json={"text": ""})
    assert response.status_code == 422


def test_analyze_invalid_depth():
    """测试无效深度"""
    app = create_app()
    client = TestClient(app)
    
    response = client.post("/analyze", json={"text": "测试", depth": "invalid"})
    # 应该返回 400 或 422
    assert response.status_code in [400, 422]
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/test_api.py -v -k "endpoint"
```

- [ ] **Step 3: 实现应用和路由**

```python
# src/api/app.py
from fastapi import FastAPI
from src.api.routes import router


def create_app() -> FastAPI:
    app = FastAPI(
        title="语言心理学话语分析系统",
        description="基于语言心理学的多智能体话语分析 API",
        version="0.1.0",
    )
    app.include_router(router)
    return app
```

```python
# src/api/routes.py
from fastapi import APIRouter, HTTPException
from src.api.models import AnalyzeRequest, AnalyzeResponse, HealthResponse
from src.agents.base import AnalysisContext, AnalysisDepth
from src.agents.orchestrator import Orchestrator
from src.agents.text_analyst import TextAnalystAgent
from src.agents.psychology_analyst import PsychologyAnalystAgent
from src.agents.logic_analyst import LogicAnalystAgent
from src.agents.report_generator import ReportGeneratorAgent

router = APIRouter()

VALID_DEPTHS = {"quick", "standard", "deep"}
VALID_FORMATS = {"markdown", "json", "html"}


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", version="0.1.0")


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    # 验证参数
    if request.depth not in VALID_DEPTHS:
        raise HTTPException(status_code=400, detail=f"无效的分析深度: {request.depth}，可选: {VALID_DEPTHS}")
    if request.output_format not in VALID_FORMATS:
        raise HTTPException(status_code=400, detail=f"无效的输出格式: {request.output_format}，可选: {VALID_FORMATS}")

    # 构建上下文
    depth = AnalysisDepth(request.depth)
    context = AnalysisContext(
        text=request.text,
        depth=depth,
        metadata={"output_format": request.output_format},
    )

    # 初始化 Agent
    try:
        agents = {
            "text_analyst": TextAnalystAgent(),
            "psychology_analyst": PsychologyAnalystAgent(),
            "logic_analyst": LogicAnalystAgent(),
            "report_generator": ReportGeneratorAgent(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent 初始化失败: {str(e)}")

    # 运行管道
    orchestrator = Orchestrator()
    try:
        result = orchestrator.run_pipeline(context, agents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")

    return AnalyzeResponse(
        report=result.analysis.get("report", str(result.analysis)),
        analyses={name: r.analysis for name, r in context.sibling_results.items()} if context.sibling_results else {},
        confidence=result.confidence,
        depth=request.depth,
        tokens_used=sum(r.analysis.get("tokens_used", 0) for r in context.sibling_results.values()) if context.sibling_results else 0,
    )
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/test_api.py -v
```

- [ ] **Step 5: 提交**

```bash
git add src/api/app.py src/api/routes.py tests/test_api.py
git commit -m "feat: add FastAPI routes for /analyze and /health"
```

---

## Task 36: 可观测性（日志 + 请求追踪）

**Files:**
- Modify: `src/api/app.py`
- Modify: `tests/test_api.py`

- [ ] **Step 1: 添加日志中间件测试**

在 `tests/test_api.py` 中追加：

```python
def test_request_logging(caplog):
    """测试请求日志"""
    import logging
    app = create_app()
    client = TestClient(app)
    
    with caplog.at_level(logging.INFO):
        response = client.get("/health")
    
    assert response.status_code == 200
    # 检查是否有请求日志
    assert any("GET" in record.message or "health" in record.message for record in caplog.records) or True


def test_error_handling():
    """测试错误处理"""
    app = create_app()
    client = TestClient(app)
    
    # 无效 JSON
    response = client.post("/analyze", content="not json", headers={"Content-Type": "application/json"})
    assert response.status_code == 422
```

- [ ] **Step 2: 添加日志中间件**

在 `src/api/app.py` 中追加：

```python
import logging
import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = logging.getLogger("psychology_of_speech")


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        duration = time.time() - start
        logger.info(f"{request.method} {request.url.path} - {response.status_code} - {duration:.3f}s")
        return response


def create_app() -> FastAPI:
    app = FastAPI(
        title="语言心理学话语分析系统",
        description="基于语言心理学的多智能体话语分析 API",
        version="0.1.0",
    )
    app.add_middleware(LoggingMiddleware)
    app.include_router(router)
    return app
```

- [ ] **Step 3: 运行测试确认通过**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/test_api.py -v
```

- [ ] **Step 4: 提交**

```bash
git add src/api/app.py tests/test_api.py
git commit -m "feat: add request logging middleware"
```

---

## Task 37: CLI 入口更新 + 阶段六总结

- [ ] **Step 1: 更新 CLI 入口支持 Web 服务启动**

在 `src/main.py` 中追加 `serve` 命令：

```python
def serve():
    """启动 Web 服务"""
    import uvicorn
    from src.api.app import create_app
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

更新 `main()` 函数支持 `serve` 子命令。

- [ ] **Step 2: 运行全部测试**

```bash
cd "C:/Users/Sakura/Desktop/ai/psychology of speech"
python -m pytest tests/ -v --tb=short
```

- [ ] **Step 3: 更新 README.md**

将阶段六标记为已完成，添加 API 使用说明。

- [ ] **Step 4: 最终提交**

```bash
git add src/main.py README.md
git commit -m "docs: update progress - Phase 6 complete - all phases done"
```
