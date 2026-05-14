# src/api/models.py
from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="要分析的文本")
    depth: str = Field(default="standard", description="分析深度: quick/standard/deep")
    output_format: str = Field(default="markdown", description="输出格式: markdown/json/html")


class AnalyzeResponse(BaseModel):
    report: str
    analyses: dict = Field(default_factory=dict)
    confidence: float
    depth: str
    tokens_used: int = 0


class HealthResponse(BaseModel):
    status: str
    version: str


class ScrapeRequest(BaseModel):
    query: str = Field(..., min_length=1, description="搜索关键词")
    sources: list[str] = Field(
        default=["zhihu", "douban", "xiaohongshu", "psychology_blog"],
        description="数据源列表",
    )
    max_items_per_source: int = Field(default=10, ge=1, le=50)
    dry_run: bool = Field(default=False, description="仅预览不写入")


class ScrapeResponse(BaseModel):
    scraped: int
    cleaned: int
    analyzed: int
    validated: int
    stored: int
    errors: list[str] = []
    dry_run_cases: list[dict] | None = None


class ScrapeStatusResponse(BaseModel):
    status: str
    sources_available: list[str]
