import json
from typing import Any
from uuid import UUID

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ChatEvent, ChatRun, Citation, Paper

log = structlog.get_logger()


def stream_key(user_id: UUID, run_id: UUID) -> str:
    return f"pm:{user_id}:chat:{run_id}:events"


def append(db: AsyncSession, run: ChatRun, kind: str, data: dict, attempt: int = 0) -> ChatEvent:
    # Caller holds the run row lock. Terminal events share the final message transaction.
    run.event_seq += 1
    event = ChatEvent(
        user_id=run.user_id, run_id=run.id, seq=run.event_seq, kind=kind, attempt=attempt, data=data
    )
    db.add(event)
    return event


def envelope(event: ChatEvent) -> dict:
    return {"id": event.seq, "run_id": str(event.run_id), "attempt": event.attempt, **event.data}


async def mirror(redis: Any, settings: Any, event: ChatEvent) -> None:
    try:
        key = stream_key(event.user_id, event.run_id)
        await redis.xadd(
            key,
            {"kind": event.kind, "data": json.dumps(envelope(event), ensure_ascii=False)},
            id=f"{event.seq}-0",
        )
        await redis.expire(key, settings.chat_event_ttl_s)
    except Exception as exc:
        # SQL ledger and final result remain authoritative, including Redis outages.
        log.warning(
            "chat_event_cache_unavailable", error_type=type(exc).__name__, run_id=str(event.run_id)
        )


async def emit(
    ctx: Any, user_id: UUID, run_id: UUID, kind: str, data: dict, attempt: int = 0
) -> None:
    from app.services.chat.store import owned_run

    async with ctx.sessions() as db:
        run = await owned_run(db, user_id, run_id, lock=True)
        if run.status in {"completed", "failed"}:
            return
        event = append(db, run, kind, data, attempt)
        await db.commit()
    await mirror(ctx.redis, ctx.settings, event)


async def citations_for(db: AsyncSession, user_id: UUID, message_id: UUID) -> list[dict]:
    rows = (
        await db.execute(
            select(Citation, Paper)
            .join(Paper, (Paper.user_id == Citation.user_id) & (Paper.id == Citation.paper_id))
            .where(
                Citation.user_id == user_id,
                Citation.message_id == message_id,
            )
            .order_by(Citation.number)
        )
    ).all()
    return [
        {
            "id": str(c.id),
            "number": c.number,
            "paper_id": str(c.paper_id),
            "version_id": str(c.version_id),
            "chunk_id": str(c.chunk_id),
            "title": "来源已删除" if c.deleted_at or p.deleted_at else p.title,
            "text": "" if c.deleted_at or p.deleted_at else c.source_text,
            "locations": [] if c.deleted_at or p.deleted_at else c.locations,
            "deleted": bool(c.deleted_at or p.deleted_at),
        }
        for c, p in rows
    ]


async def public_event(sessions: Any, settings: Any, event: ChatEvent) -> dict:
    data = envelope(event)
    if event.kind == "citations":
        async with sessions() as db:
            data["citations"] = await citations_for(db, event.user_id, UUID(data["message_id"]))
    elif event.kind == "sources":
        # Never replay cached source text. A deleted paper is immediately inaccessible.
        from langchain_core.documents import Document

        from app.services.chat.store import owned_run, scope_for
        from app.services.retrieval.store import RetrievalStore

        async with sessions() as db:
            run = await owned_run(db, event.user_id, event.run_id)
        documents = [
            Document(page_content="", metadata={"chunk_id": x["chunk_id"], "number": x["number"]})
            for x in data.pop("references", [])
        ]
        refreshed = await RetrievalStore(sessions, settings).refresh(
            scope_for(event.user_id, run.scope), documents
        )
        numbers = {x.metadata["chunk_id"]: x.metadata["number"] for x in documents}
        data["sources"] = [
            {
                "number": numbers[d.metadata["chunk_id"]],
                "text": d.page_content,
                **{k: v for k, v in d.metadata.items() if k not in {"user_id", "embedding_input"}},
            }
            for d in refreshed
        ]
    return data


def sse(kind: str, data: dict) -> str:
    return f"id: {data['id']}\nevent: {kind}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
