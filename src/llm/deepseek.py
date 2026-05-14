import asyncio
import json
import logging
import httpx

from src.llm.client import LLMAdapter, LLMResponse
from src.llm.exceptions import LLMConnectionError, LLMResponseError

logger = logging.getLogger(__name__)


class DeepSeekAdapter(LLMAdapter):
    """DeepSeek API 适配器（带指数退避重试）"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        temperature: float = 0.3,
        max_tokens: int = 4096,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.base_delay = base_delay
        self._client = httpx.AsyncClient(timeout=120.0)

    async def close(self):
        """关闭底层 HTTP 客户端"""
        await self._client.aclose()

    async def generate_stream(
        self, prompt: str, system_prompt: str | None = None
    ):
        """流式生成响应（带重试，返回 async generator of str）"""
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                async with self._client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens,
                        "stream": True,
                    },
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                delta = data["choices"][0].get("delta", {})
                                content = delta.get("content") or ""
                                if content:
                                    yield content
                            except (json.JSONDecodeError, KeyError, IndexError):
                                continue
                    return  # 成功完成，退出重试循环
            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code == 429 or e.response.status_code >= 500:
                    delay = self.base_delay * (2 ** attempt)
                    logger.warning(f"流式请求失败 (HTTP {e.response.status_code}), {delay}s 后重试 ({attempt+1}/{self.max_retries})")
                    await asyncio.sleep(delay)
                    continue
                raise LLMConnectionError(f"HTTP {e.response.status_code}: {e}") from e
            except httpx.RequestError as e:
                last_error = e
                delay = self.base_delay * (2 ** attempt)
                logger.warning(f"流式请求异常, {delay}s 后重试 ({attempt+1}/{self.max_retries}): {e}")
                await asyncio.sleep(delay)
                continue

        raise LLMConnectionError(f"流式重试 {self.max_retries} 次后仍失败: {last_error}") from last_error

    async def generate(
        self, prompt: str, system_prompt: str | None = None
    ) -> LLMResponse:
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                choice = data["choices"][0]
                usage = data.get("usage", {})
                # MiMo 返回 reasoning_content + content，只取 content
                message = choice["message"]
                content = message.get("content") or ""
                return LLMResponse(
                    content=content,
                    model=data.get("model", self.model),
                    tokens_used=usage.get("total_tokens", 0),
                    finish_reason=choice.get("finish_reason", "stop"),
                )
            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code == 429 or e.response.status_code >= 500:
                    delay = self.base_delay * (2 ** attempt)
                    logger.warning(f"LLM 请求失败 (HTTP {e.response.status_code}), {delay}s 后重试 ({attempt+1}/{self.max_retries})")
                    await asyncio.sleep(delay)
                    continue
                raise LLMConnectionError(f"HTTP {e.response.status_code}: {e}") from e
            except httpx.RequestError as e:
                last_error = e
                delay = self.base_delay * (2 ** attempt)
                logger.warning(f"LLM 请求异常, {delay}s 后重试 ({attempt+1}/{self.max_retries}): {e}")
                await asyncio.sleep(delay)
                continue

        raise LLMConnectionError(f"重试 {self.max_retries} 次后仍失败: {last_error}") from last_error
