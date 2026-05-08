from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """LLM 响应"""
    content: str
    model: str
    tokens_used: int
    finish_reason: str


class LLMAdapter(ABC):
    """LLM 适配器接口"""

    @abstractmethod
    async def generate(
        self, prompt: str, system_prompt: str | None = None
    ) -> LLMResponse:
        """生成响应"""
        ...


class LLMClient:
    """LLM 统一客户端"""

    def __init__(self, adapter: LLMAdapter):
        self.adapter = adapter

    async def generate(
        self, prompt: str, system_prompt: str | None = None
    ) -> LLMResponse:
        return await self.adapter.generate(prompt, system_prompt)
