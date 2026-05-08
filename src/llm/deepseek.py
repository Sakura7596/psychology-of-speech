import httpx

from src.llm.client import LLMAdapter, LLMResponse
from src.llm.exceptions import LLMConnectionError, LLMResponseError


class DeepSeekAdapter(LLMAdapter):
    """DeepSeek API 适配器"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = httpx.AsyncClient(timeout=120.0)

    async def close(self):
        """关闭底层 HTTP 客户端"""
        await self._client.aclose()

    async def generate(
        self, prompt: str, system_prompt: str | None = None
    ) -> LLMResponse:
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await self._client.post(
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
                },
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise LLMConnectionError(f"HTTP {e.response.status_code}: {e}") from e
        except httpx.RequestError as e:
            raise LLMConnectionError(f"请求失败: {e}") from e

        try:
            data = response.json()
            choice = data["choices"][0]
            usage = data.get("usage", {})
        except (KeyError, IndexError, ValueError) as e:
            raise LLMResponseError(f"响应解析失败: {e}") from e

        return LLMResponse(
            content=choice["message"]["content"],
            model=data.get("model", self.model),
            tokens_used=usage.get("total_tokens", 0),
            finish_reason=choice.get("finish_reason", "stop"),
        )
