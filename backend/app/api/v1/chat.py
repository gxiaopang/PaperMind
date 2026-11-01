import asyncio
import time
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import select

from app.api.deps import CurrentIdentity, VerifiedUser, require_identity
from app.db.models import ChatEvent, ChatRun, Conversation, Message, User, utcnow
from app.schemas.chat import ConversationInput, ConversationUpdate, FeedbackInput, RunInput
from app.services.chat.events import citations_for, public_event, sse, stream_key
from app.services.chat.results import result_for
from app.services.chat.store import create_run, owned_conversation, owned_run, scope_for
from app.services.retrieval.store import RetrievalStore

router = APIRouter(tags=["chat"])


def conversation_json(item: Conversation) -> dict:
    return {
        "id": item.id,
        "title": item.title,
        "scope": item.scope,
        "active_paper_ids": item.active_paper_ids,
        "created_at": item.created_at,
    }


@router.post("/conversations", status_code=201)
async def create_conversation(
    body: ConversationInput, request: Request, user: VerifiedUser
) -> dict:
    state = request.app.state
    scope = body.scope.model_dump(mode="json")
    await RetrievalStore(state.session_factory, state.settings).validate_scope(
        scope_for(user.id, scope)
    )
    async with state.session_factory() as db:
        item = Conversation(user_id=user.id, title=body.title, scope=scope)
        db.add(item)
        await db.commit()
        return conversation_json(item)


@router.get("/conversations")
async def conversations(
    request: Request,
    identity: CurrentIdentity,
    limit: int = Query(50, ge=1, le=100),
    before: UUID | None = None,
) -> list[dict]:
    async with request.app.state.session_factory() as db:
        stmt = select(Conversation).where(
            Conversation.user_id == identity.user.id, Conversation.deleted_at.is_(None)
        )
        if before:
            item = await owned_conversation(db, identity.user.id, before)
            stmt = stmt.where(
                (Conversation.created_at < item.created_at)
                | ((Conversation.created_at == item.created_at) & (Conversation.id < item.id))
            )
        return [
            conversation_json(c)
            for c in await db.scalars(
                stmt.order_by(Conversation.created_at.desc(), Conversation.id.desc()).limit(limit)
            )
        ]


@router.patch("/conversations/{cid}")
async def update_conversation(
    cid: UUID, body: ConversationUpdate, request: Request, identity: CurrentIdentity
) -> dict:
    state = request.app.state
    scope = body.scope.model_dump(mode="json") if body.scope is not None else None
    if scope is not None:
        await RetrievalStore(state.session_factory, state.settings).validate_scope(
            scope_for(identity.user.id, scope)
        )
    async with state.session_factory() as db:
        await db.scalar(select(User).where(User.id == identity.user.id).with_for_update())
        item = await owned_conversation(db, identity.user.id, cid)
        if body.title is not None:
            item.title = body.title.strip() or "新会话"
        if scope is not None:
            item.scope = scope
        await db.commit()
        return conversation_json(item)


@router.delete("/conversations/{cid}", status_code=204)
async def delete_conversation(cid: UUID, request: Request, identity: CurrentIdentity) -> None:
    async with request.app.state.session_factory() as db:
        # Serialize deletion with run submission so a new task cannot race the tombstone.
        await db.scalar(select(User).where(User.id == identity.user.id).with_for_update())
        item = await owned_conversation(db, identity.user.id, cid)
        active = await db.scalar(
            select(ChatRun.id)
            .where(
                ChatRun.user_id == identity.user.id,
                ChatRun.conversation_id == cid,
                ChatRun.status.in_(["pending", "running"]),
            )
            .limit(1)
        )
        if active:
            raise HTTPException(409, "会话正在生成回答，请等待完成后再删除")
        item.deleted_at = utcnow()
        await db.commit()


@router.get("/conversations/{cid}/runs")
async def list_runs(
    cid: UUID,
    request: Request,
    identity: CurrentIdentity,
    limit: int = Query(20, ge=1, le=50),
) -> list[dict]:
    async with request.app.state.session_factory() as db:
        await owned_conversation(db, identity.user.id, cid)
        rows = list(
            await db.scalars(
                select(ChatRun)
                .where(ChatRun.user_id == identity.user.id, ChatRun.conversation_id == cid)
                .order_by(ChatRun.created_at.desc(), ChatRun.id.desc())
                .limit(limit)
            )
        )
        return [
            {
                "id": str(r.id),
                "question": r.question,
                "mode": r.mode,
                "status": r.status,
                "result_status": r.result_status,
                "error": r.error,
                "event_seq": r.event_seq,
                "retrieval_count": r.retrieval_count,
                "generation_count": r.generation_count,
                "created_at": r.created_at,
                "finished_at": r.finished_at,
            }
            for r in rows
        ]


@router.get("/conversations/{cid}/messages")
async def messages(
    cid: UUID,
    request: Request,
    identity: CurrentIdentity,
    after: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
) -> list[dict]:
    async with request.app.state.session_factory() as db:
        await owned_conversation(db, identity.user.id, cid)
        rows = list(
            await db.scalars(
                select(Message)
                .where(
                    Message.user_id == identity.user.id,
                    Message.conversation_id == cid,
                    Message.seq > after,
                )
                .order_by(Message.seq)
                .limit(limit)
            )
        )
        return [
            {
                "id": str(m.id),
                "run_id": str(m.run_id),
                "seq": m.seq,
                "role": m.role,
                "content": m.content,
                "rewritten_query": m.rewritten_query,
                "route": m.route,
                "verify_result": m.verify_result,
                "feedback": m.feedback,
                "citations": await citations_for(db, identity.user.id, m.id),
            }
            for m in rows
        ]


@router.post("/conversations/{cid}/runs", status_code=202)
async def submit(cid: UUID, body: RunInput, request: Request, user: VerifiedUser) -> dict:
    async with request.app.state.session_factory() as db:
        run = await create_run(db, request.app.state.settings, user.id, cid, body)
        await db.commit()
        return {
            "run_id": run.id,
            "status": run.status,
            "events_url": f"/api/v1/chat/runs/{run.id}/events",
            "result_url": f"/api/v1/chat/runs/{run.id}",
        }


@router.get("/chat/runs/{rid}")
async def result(rid: UUID, request: Request, identity: CurrentIdentity) -> dict:
    async with request.app.state.session_factory() as db:
        run = await owned_run(db, identity.user.id, rid)
        await owned_conversation(db, identity.user.id, run.conversation_id)
        return await result_for(db, run)


@router.patch("/messages/{mid}/feedback")
async def feedback(
    mid: UUID, body: FeedbackInput, request: Request, identity: CurrentIdentity
) -> dict:
    async with request.app.state.session_factory() as db:
        message = await db.scalar(
            select(Message).where(
                Message.id == mid, Message.user_id == identity.user.id, Message.role == "assistant"
            )
        )
        if not message:
            raise HTTPException(404, "消息不存在")
        await owned_conversation(db, identity.user.id, message.conversation_id)
        message.feedback = body.feedback
        await db.commit()
        return {"id": message.id, "feedback": message.feedback}


def cursor_for(request: Request, after: int) -> int:
    raw = request.headers.get("Last-Event-ID")
    if raw is None:
        return after
    if not raw.isascii() or not raw.isdigit() or len(raw) > 18:
        raise HTTPException(422, "Last-Event-ID 必须是递增整数")
    return int(raw)


async def event_response(request: Request, uid: UUID, rid: UUID, after: int) -> StreamingResponse:
    state = request.app.state
    async with state.session_factory() as db:
        run = await owned_run(db, uid, rid)
        await owned_conversation(db, uid, run.conversation_id)
        if after > run.event_seq:
            raise HTTPException(409, "事件游标超过当前任务进度")

    async def iterate() -> Any:
        cursor, heartbeat = after, time.monotonic()
        while not await request.is_disconnected():
            async with state.session_factory() as db:
                # Session revocation, expiry and disabling also revoke active streams.
                try:
                    identity = await require_identity(request, db)
                    if identity.user.id != uid:
                        return
                except HTTPException:
                    return
                current = await owned_run(db, uid, rid)
                try:
                    await owned_conversation(db, uid, current.conversation_id)
                except HTTPException:
                    return
                rows = list(
                    await db.scalars(
                        select(ChatEvent)
                        .where(
                            ChatEvent.user_id == uid,
                            ChatEvent.run_id == rid,
                            ChatEvent.seq > cursor,
                        )
                        .order_by(ChatEvent.seq)
                        .limit(200)
                    )
                )
                final = (
                    await result_for(db, current)
                    if not rows and current.status in {"completed", "failed"}
                    else None
                )
            for event in rows:
                data = await public_event(state.session_factory, state.settings, event)
                yield sse(event.kind, data)
                cursor = event.seq
                if event.kind in {"done", "error"}:
                    return
            if final is not None:
                # Recover after TTL expiry with original terminal IDs for deduplication.
                common = {
                    "run_id": str(rid),
                    "attempt": current.generation_count,
                    "recovered": True,
                }
                if final["status"] == "failed":
                    if cursor < current.event_seq:
                        yield sse(
                            "error",
                            {
                                **common,
                                "id": current.event_seq,
                                "message": final["error"],
                                "status": "failed",
                            },
                        )
                elif final["message"]:
                    msg = final["message"]
                    terminal = [
                        ("answer", {"text": msg["content"], "status": final["result_status"]}),
                        ("citations", {"citations": msg["citations"]}),
                        ("done", {"status": final["result_status"], "usage": final["usage"]}),
                    ]
                    for seq, (kind, data) in enumerate(terminal, current.event_seq - 2):
                        if seq > cursor:
                            yield sse(kind, {**common, "id": seq, "message_id": msg["id"], **data})
                return
            if rows:
                continue
            if time.monotonic() - heartbeat >= 10:
                yield ": heartbeat\n\n"
                heartbeat = time.monotonic()
            try:
                await state.redis.xread({stream_key(uid, rid): f"{cursor}-0"}, count=1, block=1000)
            except Exception:
                await asyncio.sleep(0.25)

    return StreamingResponse(
        iterate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-store", "X-Accel-Buffering": "no"},
    )


@router.get("/chat/runs/{rid}/events")
async def events(rid: UUID, request: Request, after: int = Query(0, ge=0)) -> StreamingResponse:
    # Close the initial authorization session before entering a long subscription.
    async with request.app.state.session_factory() as db:
        identity = await require_identity(request, db)
        uid = identity.user.id
    return await event_response(request, uid, rid, cursor_for(request, after))


@router.post("/conversations/{cid}/stream")
async def submit_stream(
    cid: UUID, body: RunInput, request: Request, after: int = Query(0, ge=0)
) -> StreamingResponse:
    async with request.app.state.session_factory() as db:
        identity = await require_identity(request, db)
        if identity.user.email_verified_at is None:
            raise HTTPException(403, "请先验证邮箱")
        uid = identity.user.id
        run = await create_run(db, request.app.state.settings, uid, cid, body)
        await db.commit()
        rid = run.id
    return await event_response(request, uid, rid, cursor_for(request, after))
