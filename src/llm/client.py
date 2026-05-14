import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


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


class LLMCache:
    """基于 prompt hash 的 LLM 响应缓存"""

    def __init__(self, ttl_seconds: int = 3600, max_size: int = 256):
        self._cache: dict[str, tuple[float, LLMResponse]] = {}
        self._ttl = ttl_seconds
        self._max_size = max_size

    def _make_key(self, prompt: str, system_prompt: str | None) -> str:
        content = f"{system_prompt or ''}\n{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get(self, prompt: str, system_prompt: str | None = None) -> LLMResponse | None:
        key = self._make_key(prompt, system_prompt)
        if key in self._cache:
            ts, resp = self._cache[key]
            if time.time() - ts < self._ttl:
                logger.debug(f"LLM 缓存命中: {key}")
                return resp
            else:
                del self._cache[key]
        return None

    def set(self, prompt: str, response: LLMResponse, system_prompt: str | None = None) -> None:
        if len(self._cache) >= self._max_size:
            # 淘汰最旧的条目
            oldest_key = min(self._cache, key=lambda k: self._cache[k][0])
            del self._cache[oldest_key]
        key = self._make_key(prompt, system_prompt)
        self._cache[key] = (time.time(), response)

    def clear(self) -> None:
        self._cache.clear()


class LLMClient:
    """LLM 统一客户端（带缓存）"""

    def __init__(self, adapter: LLMAdapter, enable_cache: bool = True, cache_ttl: int = 3600):
        self.adapter = adapter
        self._cache = LLMCache(ttl_seconds=cache_ttl) if enable_cache else None

    async def generate(
        self, prompt: str, system_prompt: str | None = None, use_cache: bool = True
    ) -> LLMResponse:
        # 尝试缓存
        if use_cache and self._cache:
            cached = self._cache.get(prompt, system_prompt)
            if cached:
                return cached

        response = await self.adapter.generate(prompt, system_prompt)

        # 写入缓存
        if use_cache and self._cache:
            self._cache.set(prompt, response, system_prompt)

        return response

    async def generate_stream(self, prompt: str, system_prompt: str | None = None):
        """流式生成（不缓存）"""
        if hasattr(self.adapter, "generate_stream"):
            async for chunk in self.adapter.generate_stream(prompt, system_prompt):
                yield chunk
        else:
            # 回退到非流式
            response = await self.generate(prompt, system_prompt, use_cache=False)
            yield response.content

    async def close(self):
        """关闭适配器底层资源"""
        if hasattr(self.adapter, "close"):
            await self.adapter.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
