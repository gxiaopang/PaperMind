import hashlib
import json
from typing import Any
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.db.models import (
    ChatRun,
    Conversation,
    QuotaReservation,
    TaskOutbox,
    User,
    UserSetting,
    utcnow,
)
from app.graph.context import CONTEXT_VERSION
from app.schemas.chat import RunInput
from app.services.ingestion.quota import lock_user, totals, user_settings
from app.services.retrieval.types import RetrievalScope

PROMPT_VERSION = "m4-v6-node-context"
CONTEXT_STRATEGY = "paper-map-tools-v5"


def execution_config(settings: Settings) -> dict:
    return {
        "prompt_version": PROMPT_VERSION,
        "context_strategy": CONTEXT_STRATEGY,
        "node_context_version": CONTEXT_VERSION,
        **{
            name: getattr(settings, name)
            for name in (
                "llm_model",
                "llm_fast_model",
                "llm_base_url",
                "llm_fast_base_url",
                "rag_max_retrieval_retries",
                "rag_max_generation_retries",
                "chat_max_output_tokens",
                "embedding_model",
                "embedding_base_url",
                "embedding_dim",
                "rerank_model",
                "rerank_base_url",
                "retrieval_top_k",
                "rerank_top_n",
                "rrf_k",
                "rrf_vector_weight",
                "rrf_bm25_weight",
                "rrf_metadata_weight",
                "rerank_score_threshold",
                "retrieval_max_subqueries",
                "retrieval_merge_min_hits",
                "retrieval_cross_ref_max_chunks",
                "retrieval_background_weight",
                "retrieval_table_weight",
                "retrieval_experiment_weight",
                "retrieval_query_bytes",
                "rerank_candidate_bytes",
                "retrieval_context_bytes",
                "retrieval_parent_bytes",
                "retrieval_max_sources",
                "retrieval_parent_max_chunks",
                "retrieval_neighbor_window",
                "paper_tool_max_bytes",
                "memory_recent_token_budget",
                "memory_summarize_after_turns",
            )
        },
        "retrieval_retries": settings.rag_max_retrieval_retries,
        "generation_retries": settings.rag_max_generation_retries,
    }


def scope_for(user_id: UUID, scope: dict) -> RetrievalScope:
    return RetrievalScope(
        user_id,
        **{
            key: tuple(UUID(v) for v in scope[key]) if scope.get(key) is not None else None
            for key in ("collection_ids", "paper_ids")
        },
    )


def thread_id(run: ChatRun) -> str:
    return f"pm:{run.user_id}:{run.conversation_id}"


async def owned_conversation(db: AsyncSession, user_id: UUID, cid: UUID) -> Conversation:
    item = await db.scalar(
        select(Conversation).where(
            Conversation.id == cid,
            Conversation.user_id == user_id,
            Conversation.deleted_at.is_(None),
        )
    )
    if item is None:
        raise HTTPException(404, "会话不存在")
    return item


async def owned_run(db: AsyncSession, user_id: UUID, rid: UUID, *, lock: bool = False) -> ChatRun:
    stmt = select(ChatRun).where(ChatRun.id == rid, ChatRun.user_id == user_id)
    if lock:
        stmt = stmt.with_for_update().execution_options(populate_existing=True)
    run = await db.scalar(stmt)
    if run is None:
        raise HTTPException(404, "问答任务不存在")
    return run


async def create_run(
    db: AsyncSession, settings: Settings, user_id: UUID, cid: UUID, body: RunInput
) -> ChatRun:
    await lock_user(db, user_id)
    conversation = await owned_conversation(db, user_id, cid)
    # Hash the submitted intent, including inheritance. Retrying an inherited scope
    # request still resolves the original run after the conversation selection changes.
    digest = hashlib.sha256(
        json.dumps(
            body.model_dump(mode="json", exclude={"idempotency_key"}), sort_keys=True
        ).encode()
    ).hexdigest()
    existing = await db.scalar(
        select(ChatRun).where(
            ChatRun.user_id == user_id,
            ChatRun.conversation_id == cid,
            ChatRun.idempotency_key == body.idempotency_key,
        )
    )
    if existing:
        if existing.request_hash != digest:
            raise HTTPException(409, "幂等键已用于不同请求")
        return existing
    if not body.question.strip() or len(body.question.encode()) > settings.retrieval_query_bytes:
        raise HTTPException(422, "问题为空或超过 UTF-8 字节限制")
    scope = body.scope.model_dump(mode="json") if body.scope is not None else conversation.scope
    # Use this session to avoid waiting for another pool connection while holding the user lock.
    from app.db.models import Collection, Paper

    for key, model in (("collection_ids", Collection), ("paper_ids", Paper)):
        ids = scope.get(key)
        if ids:
            stmt = select(model.id).where(
                model.user_id == user_id, model.id.in_([UUID(x) for x in ids])
            )
            if model is Paper:
                stmt = stmt.where(Paper.deleted_at.is_(None))
            if set(await db.scalars(stmt)) != {UUID(x) for x in ids}:
                raise HTTPException(404, "检索范围不存在")
    count = await db.scalar(
        select(func.count())
        .select_from(ChatRun)
        .where(ChatRun.user_id == user_id, ChatRun.status.in_(["pending", "running"]))
    )
    if (count or 0) >= settings.user_chat_concurrency:
        raise HTTPException(429, "同时进行的问答过多，请稍后重试")
    limits = await user_settings(db, settings, user_id)
    await db.execute(text("SELECT pg_advisory_xact_lock(620021)"))
    amount = settings.chat_initial_token_reserve
    for owner, limit in (
        (user_id, limits.user_daily_model_tokens),
        (None, limits.global_daily_model_tokens),
    ):
        used, held = await totals(db, "model_tokens", owner)
        if used + held + amount > limit:
            raise HTTPException(429, "今日模型 token 额度不足")
    pref = await db.get(UserSetting, user_id)
    preferences = {
        "answer_language": pref.answer_language if pref else "zh",
        "detail_level": pref.detail_level if pref else "standard",
    }
    run = ChatRun(
        user_id=user_id,
        conversation_id=cid,
        idempotency_key=body.idempotency_key,
        request_hash=digest,
        question=body.question.strip(),
        mode=body.mode or settings.rag_mode,
        scope=scope,
        config=execution_config(settings),
        preferences=preferences,
    )
    db.add(run)
    await db.flush()
    db.add(QuotaReservation(user_id=user_id, run_id=run.id, metric="model_tokens", amount=amount))
    db.add(TaskOutbox(user_id=user_id, kind="chat", resource_id=run.id))
    return run


async def finish_task(db: AsyncSession, run: ChatRun) -> None:
    await db.execute(
        update(QuotaReservation)
        .where(
            QuotaReservation.user_id == run.user_id,
            QuotaReservation.run_id == run.id,
            QuotaReservation.status == "reserved",
        )
        .values(status="released")
    )
    await db.execute(
        update(TaskOutbox)
        .where(
            TaskOutbox.user_id == run.user_id,
            TaskOutbox.kind == "chat",
            TaskOutbox.resource_id == run.id,
        )
        .values(completed_at=utcnow())
    )


async def assert_live(sessions: Any, user_id: UUID, rid: UUID) -> ChatRun:
    async with sessions() as db:
        run = await owned_run(db, user_id, rid)
        user = await db.get(User, user_id)
        if not user or user.status != "active" or user.email_verified_at is None:
            raise HTTPException(403, "账户不可用或邮箱未验证")
        return run
