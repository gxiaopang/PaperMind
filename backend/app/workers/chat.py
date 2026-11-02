import asyncio
import hashlib
from datetime import timedelta
from typing import Any
from uuid import UUID

import structlog
from arq import cron
from arq.connections import RedisSettings
from fastapi import HTTPException
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.errors import NodeCancelledError
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from sqlalchemy import delete, select, text

from app.core.config import get_settings
from app.core.observability import (
    Observation,
    SafeCallbackHandler,
    run_trace,
    trace_context,
    tracing_client,
)
from app.db.models import ChatEvent, ChatRun, TaskOutbox, User, utcnow
from app.graph.builder import build_graph
from app.graph.nodes import GraphServices
from app.graph.state import initial_state
from app.services.chat.events import append, mirror
from app.services.chat.llm import ChatModels
from app.services.chat.store import (
    PROMPT_VERSION,
    assert_live,
    execution_config,
    finish_task,
    owned_run,
    scope_for,
    thread_id,
)
from app.services.chat.telemetry import record_result
from app.services.memory.consolidation import consolidate
from app.services.memory.history import load_memory
from app.services.retrieval.store import RetrievalStore
from app.workers.mail import shutdown as db_shutdown
from app.workers.mail import startup as db_startup

log = structlog.get_logger()


def lock_key(user_id: UUID, conversation_id: UUID) -> int:
    return int.from_bytes(
        hashlib.sha256(user_id.bytes + conversation_id.bytes).digest()[:8], "big", signed=True
    )


async def fail(ctx: dict[str, Any], user_id: UUID, rid: UUID, error: Exception) -> None:
    detail = (
        str(error.detail) if isinstance(error, HTTPException) else "问答服务暂时不可用，请稍后重试"
    )
    async with ctx["session_factory"]() as db:
        # Disabled users must still have reservations released and a terminal run.
        await db.scalar(select(User).where(User.id == user_id).with_for_update())
        run = await owned_run(db, user_id, rid, lock=True)
        if run.status in {"completed", "failed"}:
            return
        run.status, run.error, run.finished_at = "failed", detail, utcnow()
        await finish_task(db, run)
        event = append(
            db,
            run,
            "error",
            {
                "message": detail,
                "status": "failed",
                "http_status": error.status_code if isinstance(error, HTTPException) else 503,
            },
            run.generation_count,
        )
        await db.commit()
    await mirror(ctx["redis"], ctx["settings"], event)
    log.warning("chat_run_failed", run_id=str(rid), error_type=type(error).__name__)


def graph_config(ctx: dict[str, Any], run: ChatRun, *, kind: str = "chat") -> dict:
    config: dict = {
        "configurable": {"thread_id": thread_id(run)},
        "recursion_limit": 60,
        "run_name": f"papermind:{run.mode}:{PROMPT_VERSION}",
        "metadata": {
            "langfuse_user_id": str(run.user_id),
            "langfuse_session_id": str(run.conversation_id),
            "run_id": str(run.id),
            "prompt_version": PROMPT_VERSION,
            "mode": run.mode,
        },
    }
    if ctx.get("tracing"):
        try:
            handler = SafeCallbackHandler(
                public_key=ctx["settings"].langfuse_public_key.get_secret_value(),
                trace_context=trace_context(run, kind),
            )
            config["callbacks"] = [handler]
        except Exception as exc:
            log.warning("chat_tracing_unavailable", error_type=type(exc).__name__)
    return config


async def execute(ctx: dict[str, Any], owner: str, resource: str) -> None:
    uid, rid = UUID(owner), UUID(resource)
    async with ctx["session_factory"]() as db:
        run = await owned_run(db, uid, rid)
    attempt = run.recoveries + (run.status not in {"completed", "failed"})
    with run_trace(ctx.get("tracing"), run, attempt=attempt) as root:
        await _execute(ctx, run, root)
        if ctx.get("tracing"):
            try:
                await record_result(ctx, uid, rid, root)
            except Exception as exc:
                log.warning("chat_result_tracing_failed", error_type=type(exc).__name__)


async def _execute(ctx: dict[str, Any], run: ChatRun, root: Observation) -> None:
    if run.status in {"completed", "failed"}:
        return
    uid, rid, cid = run.user_id, run.id, run.conversation_id
    # Dedicated transaction: process death/cancellation releases serialization.
    # Full UUID hash avoids the user-prefix collision of per-user provider slots.
    async with ctx["engine"].connect() as connection:
        if not await connection.scalar(
            text("SELECT pg_try_advisory_xact_lock(:key)"), {"key": lock_key(uid, cid)}
        ):
            return
        try:
            run = await assert_live(ctx["session_factory"], uid, rid)
            await RetrievalStore(ctx["session_factory"], ctx["settings"]).validate_scope(
                scope_for(uid, run.scope)
            )
            async with ctx["session_factory"]() as db:
                earlier = await db.scalar(
                    select(ChatRun.id)
                    .where(
                        ChatRun.user_id == uid,
                        ChatRun.conversation_id == cid,
                        ChatRun.status.in_(["pending", "running"]),
                        (ChatRun.created_at < run.created_at)
                        | ((ChatRun.created_at == run.created_at) & (ChatRun.id < run.id)),
                    )
                    .limit(1)
                )
                if earlier:
                    return
                run = await owned_run(db, uid, rid, lock=True)
                # A bounded restart budget prevents a repeatedly killed node from
                # consuming unbounded internal-model calls.
                run.recoveries += 1
                if run.recoveries > ctx["settings"].chat_max_recoveries:
                    raise HTTPException(503, "问答恢复次数已耗尽，请提交新请求")
                run.status, run.started_at = "running", run.started_at or utcnow()
                await db.commit()
            settings = ctx["settings"]
            expected = execution_config(settings)
            if run.config != expected:
                raise HTTPException(409, "问答配置已改变，请提交新请求")
            graph = ctx["graphs"][run.mode]
            config = graph_config(ctx, run)
            try:
                # Cover old messages before budget trimming, even if the async task lags.
                await consolidate(ctx["services"], run, config, pressure_only=True)
            except Exception as exc:
                log.warning("chat_memory_consolidation_deferred", error_type=type(exc).__name__)
            saved = await graph.aget_state(config)
            resume = bool(saved.values and saved.values.get("run_id") == str(run.id) and saved.next)
            root.update(attempt=run.recoveries, resumed=resume)
            if resume:
                # Business history always wins over an older checkpoint; do not
                # append it. Nodes recheck sources immediately before every use.
                memory = await load_memory(
                    ctx["session_factory"], settings, uid, cid, run.scope, run.question
                )
                await graph.aupdate_state(config, memory)
            await graph.ainvoke(None if resume else initial_state(run), config, durability="sync")
        except NodeCancelledError:
            # LangGraph wraps an interrupted node; keep it resumable just like a
            # worker process cancellation, rather than persisting a false failure.
            raise asyncio.CancelledError() from None
        except Exception as exc:
            root.update(status="failed", error_type=type(exc).__name__)
            await fail(ctx, uid, rid, exc)


async def summarize(ctx: dict[str, Any], owner: str, resource: str) -> None:
    uid, rid = UUID(owner), UUID(resource)
    async with ctx["session_factory"]() as db:
        run = await owned_run(db, uid, rid)
    async with ctx["engine"].connect() as connection:
        if not await connection.scalar(
            text("SELECT pg_try_advisory_xact_lock(:key)"),
            {"key": lock_key(uid, run.conversation_id)},
        ):
            return
        try:
            await assert_live(ctx["session_factory"], uid, rid)
            async with ctx["session_factory"]() as db:
                task = await db.scalar(
                    select(TaskOutbox).where(
                        TaskOutbox.user_id == uid,
                        TaskOutbox.kind == "summarize",
                        TaskOutbox.resource_id == rid,
                    )
                )
                if not task or task.completed_at:
                    return
                task.attempts += 1
                await db.commit()
            with run_trace(ctx.get("tracing"), run, kind="summarize", attempt=task.attempts):
                await consolidate(ctx["services"], run, graph_config(ctx, run, kind="summarize"))
            async with ctx["session_factory"]() as db:
                task = await db.scalar(
                    select(TaskOutbox).where(
                        TaskOutbox.user_id == uid,
                        TaskOutbox.kind == "summarize",
                        TaskOutbox.resource_id == rid,
                    )
                )
                assert task
                task.completed_at = utcnow()
                await db.commit()
        except Exception as exc:
            log.warning("chat_summary_failed", run_id=str(rid), error_type=type(exc).__name__)
            async with ctx["session_factory"]() as db:
                task = await db.scalar(
                    select(TaskOutbox).where(
                        TaskOutbox.user_id == uid,
                        TaskOutbox.kind == "summarize",
                        TaskOutbox.resource_id == rid,
                    )
                )
                if task:
                    task.available_at = utcnow() + timedelta(
                        seconds=30 * 2 ** min(task.attempts, 4)
                    )
                    if task.attempts >= 3:
                        task.completed_at = utcnow()
                    task.delivered_at = None
                    await db.commit()


async def dispatch(ctx: dict[str, Any]) -> None:
    async with ctx["session_factory"]() as db:
        rows = list(
            await db.scalars(
                select(TaskOutbox)
                .where(
                    TaskOutbox.kind.in_(["chat", "summarize"]),
                    TaskOutbox.completed_at.is_(None),
                    TaskOutbox.available_at <= utcnow(),
                    (TaskOutbox.delivered_at.is_(None))
                    | (TaskOutbox.delivered_at < utcnow() - timedelta(seconds=30)),
                )
                .order_by(TaskOutbox.available_at)
                .limit(100)
                .with_for_update(skip_locked=True)
            )
        )
        for row in rows:
            await ctx["redis"].enqueue_job(
                "execute" if row.kind == "chat" else "summarize",
                str(row.user_id),
                str(row.resource_id),
                _job_id=f"pm:{row.user_id}:{row.kind}:{row.resource_id}",
                _queue_name="papermind:chat",
            )
            row.delivered_at = utcnow()
        await db.commit()


async def prune_events(ctx: dict[str, Any]) -> None:
    # Never expire resumable runs; final messages/citations survive stream expiry.
    cutoff = utcnow() - timedelta(seconds=ctx["settings"].chat_event_ttl_s)
    async with ctx["session_factory"]() as db:
        finished = select(ChatRun.id).where(
            ChatRun.finished_at < cutoff, ChatRun.status.in_(["completed", "failed"])
        )
        await db.execute(delete(ChatEvent).where(ChatEvent.run_id.in_(finished)))
        await db.commit()
    # Checkpoints for idle completed conversations may be pruned. Acquire the same
    # conversation lock, so a queued/resuming run cannot lose its checkpoint.
    async with ctx["session_factory"]() as db:
        candidates = (
            await db.execute(
                select(ChatRun.user_id, ChatRun.conversation_id)
                .where(ChatRun.id.in_(finished))
                .distinct()
                .limit(100)
            )
        ).all()
    for uid, cid in candidates:
        async with ctx["engine"].connect() as conn:
            if not await conn.scalar(
                text("SELECT pg_try_advisory_xact_lock(:key)"), {"key": lock_key(uid, cid)}
            ):
                continue
            async with ctx["session_factory"]() as db:
                recent = await db.scalar(
                    select(ChatRun.id)
                    .where(
                        ChatRun.user_id == uid,
                        ChatRun.conversation_id == cid,
                        (ChatRun.finished_at.is_(None)) | (ChatRun.finished_at >= cutoff),
                    )
                    .limit(1)
                )
            if not recent:
                await ctx["checkpointer"].adelete_thread(f"pm:{uid}:{cid}")


async def startup(ctx: dict[str, Any]) -> None:
    await db_startup(ctx)
    settings = ctx["settings"]
    ctx["tracing"] = tracing_client(settings)
    pool = AsyncConnectionPool[AsyncConnection[dict[str, Any]]](
        settings.database_url.get_secret_value().replace("postgresql+asyncpg://", "postgresql://"),
        min_size=1,
        max_size=settings.checkpoint_pool_size,
        open=False,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
            "options": "-c search_path=checkpoints,public",
        },
    )
    await pool.open(wait=True)
    ctx["checkpoint_pool"] = pool
    ctx["checkpointer"] = AsyncPostgresSaver(pool)
    # Schema setup belongs to the deploy migration task, never to a chat request.
    services = GraphServices(
        settings,
        ctx["session_factory"],
        ctx["engine"],
        ctx["redis"],
        ChatModels(settings, ctx["session_factory"], ctx["engine"]),
    )
    ctx["services"] = services
    ctx["graphs"] = {
        mode: build_graph(services, mode, ctx["checkpointer"]) for mode in ("fast", "agentic")
    }


async def shutdown(ctx: dict[str, Any]) -> None:
    if ctx.get("tracing"):
        await asyncio.to_thread(ctx["tracing"].shutdown)
    await ctx["checkpoint_pool"].close()
    await db_shutdown(ctx)


class WorkerSettings:
    functions = [execute, summarize, dispatch, prune_events]
    cron_jobs = [
        cron(dispatch, second=set(range(0, 60, 5)), run_at_startup=True),
        cron(prune_events, minute={0, 30}, second=0),
    ]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url.get_secret_value())
    queue_name = "papermind:chat"
    max_jobs = get_settings().user_chat_concurrency + 2
    job_timeout = 1800
    max_tries = 100
    keep_result = 0
    health_check_interval = 15
