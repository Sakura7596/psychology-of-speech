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
