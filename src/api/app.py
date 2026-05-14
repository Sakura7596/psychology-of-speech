# src/api/app.py
import logging
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from src.api.routes import router

logger = logging.getLogger("psychology_of_speech")


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        duration = time.time() - start
        logger.info(f"{request.method} {request.url.path} - {response.status_code} - {duration:.3f}s")
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """简单的内存速率限制（按 IP），带自动过期清理"""

    def __init__(self, app, requests_per_minute: int = 20):
        super().__init__(app)
        self._requests_per_minute = requests_per_minute
        self._requests: dict[str, list[float]] = {}
        self._last_cleanup: float = time.time()

    def _cleanup(self, now: float) -> None:
        """定期清理所有过期 IP 记录（每 60 秒一次）"""
        if now - self._last_cleanup < 60:
            return
        self._last_cleanup = now
        expired_ips = [
            ip for ip, timestamps in self._requests.items()
            if all(now - t >= 60 for t in timestamps)
        ]
        for ip in expired_ips:
            del self._requests[ip]

    async def dispatch(self, request: Request, call_next):
        # 限制 /analyze 和 /scrape 端点
        if request.url.path not in ("/analyze", "/scrape"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        self._cleanup(now)

        # 清理当前 IP 的过期记录
        if client_ip in self._requests:
            self._requests[client_ip] = [
                t for t in self._requests[client_ip] if now - t < 60
            ]
        else:
            self._requests[client_ip] = []

        if len(self._requests[client_ip]) >= self._requests_per_minute:
            return JSONResponse(
                status_code=429,
                content={"detail": "请求过于频繁，请稍后再试"},
            )

        self._requests[client_ip].append(now)
        return await call_next(request)


def create_app() -> FastAPI:
    app = FastAPI(
        title="语言心理学话语分析系统",
        description="基于语言心理学的多智能体话语分析 API",
        version="0.1.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RateLimitMiddleware, requests_per_minute=20)
    app.add_middleware(LoggingMiddleware)
    app.include_router(router)
    return app
