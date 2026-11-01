from typing import Any

from arq import cron
from arq.connections import RedisSettings
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.core.observability import setup_logging
from app.services.accounts.mail import send_outbox


async def startup(ctx: dict[str, Any]) -> None:
    setup_logging()
    settings = get_settings()
    ctx["settings"] = settings
    ctx["engine"] = create_async_engine(
        settings.database_url.get_secret_value(),
        pool_size=settings.provider_concurrency + 2,
        max_overflow=2,
    )
    ctx["session_factory"] = async_sessionmaker(ctx["engine"], expire_on_commit=False)


async def shutdown(ctx: dict[str, Any]) -> None:
    await ctx["engine"].dispose()


class WorkerSettings:
    functions = [send_outbox]
    cron_jobs = [cron(send_outbox, second=set(range(0, 60, 10)), run_at_startup=True)]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url.get_secret_value())
    queue_name = "papermind:mail"
    max_jobs = 2
    job_timeout = 600
    health_check_interval = 15
