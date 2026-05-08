# src/api/app.py
import logging
import time
from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from src.api.routes import router

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
