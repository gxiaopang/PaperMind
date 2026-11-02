import asyncio
import secrets
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

import structlog
from arq import create_pool
from arq.connections import RedisSettings
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.api.v1 import auth, chat, collections, memory, papers, retrieval
from app.core.config import Settings, get_settings
from app.core.observability import setup_logging, tracing_client
from app.db.models import SystemConfig

REQUESTS = Counter("papermind_http_requests_total", "HTTP requests", ["method", "route", "status"])
LATENCY = Histogram("papermind_http_duration_seconds", "HTTP duration", ["method", "route"])
log = structlog.get_logger()


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        setup_logging()
        app.state.settings = config
        app.state.engine = create_async_engine(
            config.database_url.get_secret_value(), pool_size=5, max_overflow=5, pool_pre_ping=True
        )
        app.state.session_factory = async_sessionmaker(app.state.engine, expire_on_commit=False)
        app.state.redis = await create_pool(
            RedisSettings.from_dsn(config.redis_url.get_secret_value())
        )
        app.state.tracing = tracing_client(config)
        try:
            async with app.state.session_factory() as db:
                current = await db.get(SystemConfig, "embedding")
                if current is None or current.value != {
                    "model": config.embedding_model,
                    "dimension": config.embedding_dim,
                }:
                    raise RuntimeError("Embedding 模型或维度与数据库不一致，请先执行迁移或重建索引")
                extensions = set(await db.scalars(text("SELECT extname FROM pg_extension")))
                if not {"vector", "pg_search"}.issubset(extensions):
                    raise RuntimeError("数据库缺少 vector 或 pg_search 扩展")
            yield
        finally:
            if app.state.tracing:
                await asyncio.to_thread(app.state.tracing.shutdown)
            await app.state.redis.aclose()
            await app.state.engine.dispose()

    app = FastAPI(
        title="PaperMind",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs" if config.app_env != "production" else None,
        redoc_url=None,
        openapi_url="/openapi.json" if config.app_env != "production" else None,
    )
    app.state.settings = config

    @app.middleware("http")
    async def security(request: Request, call_next):  # type: ignore[no-untyped-def]
        started = time.monotonic()
        request_id = uuid4().hex
        if request.method not in {"GET", "HEAD", "OPTIONS"} and request.url.path.startswith(
            "/api/"
        ):
            cookie, header = (
                request.cookies.get("pm_csrf", ""),
                request.headers.get("X-CSRF-Token", ""),
            )
            if (
                request.headers.get("Origin") != config.origin
                or not cookie
                or not secrets.compare_digest(cookie, header)
            ):
                return JSONResponse({"detail": "来源或安全校验失败，请刷新页面"}, status_code=403)
        try:
            response = await call_next(request)
        except Exception as exc:
            log.error("request_failed", request_id=request_id, error_type=type(exc).__name__)
            response = JSONResponse(
                {"detail": "服务暂时不可用", "request_id": request_id}, status_code=503
            )
        route = request.scope.get("route")
        route_name = route.path if route else "unmatched"
        REQUESTS.labels(request.method, route_name, response.status_code).inc()
        LATENCY.labels(request.method, route_name).observe(time.monotonic() - started)
        response.headers.update(
            {
                "X-Request-ID": request_id,
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer",
                "Cache-Control": "no-store",
            }
        )
        return response

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def ready(request: Request) -> JSONResponse:
        try:
            async with request.app.state.engine.connect() as connection:
                await connection.execute(select(1))
            await request.app.state.redis.ping()
            return JSONResponse({"status": "ready"})
        except Exception:
            return JSONResponse({"status": "unavailable"}, status_code=503)

    app.include_router(auth.router, prefix="/api/v1")
    app.include_router(collections.router, prefix="/api/v1")
    app.include_router(papers.router, prefix="/api/v1")
    app.include_router(retrieval.router, prefix="/api/v1")
    app.include_router(chat.router, prefix="/api/v1")
    app.include_router(memory.router, prefix="/api/v1")
    # Operational access is authenticated and admin-only even on internal deployments.
    from app.api.v1 import admin

    app.include_router(admin.router, prefix="/api/v1")
    return app


app = create_app()
