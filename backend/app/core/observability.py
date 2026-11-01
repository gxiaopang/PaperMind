import logging
from typing import Any

import structlog
from langfuse import Langfuse

from app.core.config import Settings


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # HTTP client debug logs can contain account tokens or provider response bodies.
    for name in ("httpx", "httpx2", "httpcore", "httpcore2", "openai", "aiosmtplib"):
        logging.getLogger(name).setLevel(logging.WARNING)
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ]
    )


def tracing_client(settings: Settings) -> Langfuse | None:
    if not settings.tracing_enabled:
        return None

    def mask(data: Any, **_: Any) -> Any:
        return data if settings.trace_content_enabled else "[content disabled]"

    return Langfuse(
        public_key=settings.langfuse_public_key.get_secret_value(),
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        base_url=settings.langfuse_base_url,
        mask=mask,
        environment=settings.app_env,
    )
