import hashlib
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import delete, false, select, update

from app.api.deps import DB, CurrentIdentity
from app.db.models import (
    ChatRun,
    Conversation,
    MemoryCoreBlock,
    MemoryEntry,
    MemoryEvent,
    Message,
    Paper,
    ResearchProject,
    User,
    UserSetting,
    utcnow,
)
from app.schemas.memory import (
    BulkMemoryInput,
    ConversationMemoryUpdate,
    CoreMemoryInput,
    MemoryInput,
    MemorySettingsInput,
    MemoryUpdate,
    ProjectInput,
    ReadingPositionInput,
)
from app.services.chat.store import owned_conversation
from app.services.ingestion.quota import lock_user
from app.services.memory.history import compact_text
from app.services.memory.store import (
    evidence_map,
    evidence_status,
    memory_json,
    owned_memory,
    owned_project,
    validate_memory,
)
from app.services.memory.tools import (
    get_core_block,
    get_write_mode,
    list_memory_events,
    save_core_block,
    session_search,
    undo_memory_event,
)

router = APIRouter(tags=["研究记忆"])


def project_json(item: ResearchProject) -> dict:
    return {
        "id": item.id,
        "name": item.name,
        "description": item.description,
        "created_at": item.created_at,
    }


async def idle_conversation(db: DB, uid: UUID, cid: UUID) -> Conversation:
    await lock_user(db, uid)
    item = await owned_conversation(db, uid, cid)
    active = await db.scalar(
        select(ChatRun.id)
        .where(
            ChatRun.user_id == uid,
            ChatRun.conversation_id == cid,
            ChatRun.status.in_(["pending", "running"]),
        )
        .limit(1)
    )
    if active:
        raise HTTPException(409, "请等待当前回答完成后再修改会话记忆或项目")
    return item


@router.get("/projects")
async def projects(db: DB, identity: CurrentIdentity) -> list[dict]:
    return [
        project_json(p)
        for p in await db.scalars(
            select(ResearchProject)
            .where(
                ResearchProject.user_id == identity.user.id,
                ResearchProject.deleted_at.is_(None),
            )
            .order_by(ResearchProject.created_at.desc())
            .limit(100)
        )
    ]


@router.post("/projects", status_code=201)
async def create_project(body: ProjectInput, db: DB, identity: CurrentIdentity) -> dict:
    item = ResearchProject(user_id=identity.user.id, **body.model_dump())
    db.add(item)
    await db.commit()
    return project_json(item)


@router.patch("/projects/{pid}")
async def edit_project(pid: UUID, body: ProjectInput, db: DB, identity: CurrentIdentity) -> dict:
    item = await owned_project(db, identity.user.id, pid)
    item.name, item.description = body.name, body.description
    await db.commit()
    return project_json(item)


@router.delete("/projects/{pid}", status_code=204)
async def delete_project(pid: UUID, db: DB, identity: CurrentIdentity) -> None:
    uid = identity.user.id
    await lock_user(db, uid)
    item = await owned_project(db, uid, pid)
    cids = select(Conversation.id).where(
        Conversation.user_id == uid, Conversation.project_id == pid
    )
    active = await db.scalar(
        select(ChatRun.id)
        .where(
            ChatRun.user_id == uid,
            ChatRun.conversation_id.in_(cids),
            ChatRun.status.in_(["pending", "running"]),
        )
        .limit(1)
    )
    if active:
        raise HTTPException(409, "请等待项目中的回答完成后再删除")
    item.deleted_at = utcnow()
    await db.execute(
        delete(MemoryCoreBlock).where(
            MemoryCoreBlock.user_id == uid, MemoryCoreBlock.project_id == pid
        )
    )
    await db.execute(
        update(MemoryEntry)
        .where(MemoryEntry.user_id == uid, MemoryEntry.project_id == pid)
        .values(deleted_at=utcnow())
    )
    await db.execute(
        update(Conversation)
        .where(Conversation.user_id == uid, Conversation.project_id == pid)
        .values(
            project_id=None,
            summary="",
            research_state={},
            active_paper_ids=[],
            summarized_seq=Conversation.next_seq - 1,
            memory_revision=Conversation.memory_revision + 1,
        )
    )
    await db.commit()


@router.get("/memories")
async def memories(
    db: DB,
    identity: CurrentIdentity,
    conversation_id: UUID | None = None,
    project_id: UUID | None = None,
    level: str | None = None,
    kind: str | None = None,
    status: str | None = None,
    evidence_filter: str | None = Query(default=None, alias="evidence_status"),
    q: str | None = None,
    limit: int = Query(100, ge=1, le=200),
) -> list[dict]:
    stmt = select(MemoryEntry).where(
        MemoryEntry.user_id == identity.user.id, MemoryEntry.deleted_at.is_(None)
    )
    if conversation_id:
        conv = await owned_conversation(db, identity.user.id, conversation_id)
        from sqlalchemy import or_

        stmt = stmt.where(
            or_(
                MemoryEntry.level == "user",
                MemoryEntry.conversation_id == conversation_id,
                (
                    (MemoryEntry.project_id == conv.project_id)
                    & MemoryEntry.level.in_(["project", "paper"])
                )
                if conv.project_id
                else false(),
                (MemoryEntry.level == "paper") & MemoryEntry.project_id.is_(None),
            )
        )
    if project_id:
        await owned_project(db, identity.user.id, project_id)
        stmt = stmt.where(MemoryEntry.project_id == project_id)
    if level:
        stmt = stmt.where(MemoryEntry.level == level)
    if kind:
        stmt = stmt.where(MemoryEntry.kind == kind)
    if status:
        stmt = stmt.where(MemoryEntry.status == status)
    if q and q.strip():
        stmt = stmt.where(MemoryEntry.content.ilike(f"%{q.strip()}%"))
    entries = list(
        await db.scalars(
            stmt.order_by(MemoryEntry.pinned.desc(), MemoryEntry.updated_at.desc()).limit(limit)
        )
    )
    evidence = await evidence_map(db, identity.user.id, entries)
    if evidence_filter:
        entries = [item for item in entries if evidence_status(item, evidence) == evidence_filter]
    return [memory_json(item, evidence) for item in entries]


@router.post("/memories", status_code=201)
async def create_memory(body: MemoryInput, db: DB, identity: CurrentIdentity) -> dict:
    await lock_user(db, identity.user.id)
    await validate_memory(db, identity.user.id, body)
    item = MemoryEntry(user_id=identity.user.id, **body.model_dump(mode="json"))
    # SQLAlchemy UUID columns expect UUID objects; JSON conversion is only for citation IDs.
    for field in ("project_id", "conversation_id", "paper_id", "source_message_id"):
        setattr(item, field, getattr(body, field))
    db.add(item)
    await db.flush()
    db.add(
        MemoryEvent(
            user_id=identity.user.id,
            project_id=item.project_id,
            conversation_id=item.conversation_id,
            memory_id=item.id,
            target="entry",
            action="add",
            actor="user",
            summary=f"手动保存记忆：{compact_text(item.content, 48)}",
            before_state={},
            after_state={
                "id": str(item.id),
                "content": item.content,
                "kind": item.kind,
                "status": item.status,
            },
        )
    )
    await db.commit()
    return memory_json(item, await evidence_map(db, identity.user.id, [item]))


@router.get("/memories/{mid}/source")
async def memory_source(mid: UUID, db: DB, identity: CurrentIdentity) -> dict:
    item = await owned_memory(db, identity.user.id, mid)
    if not item.source_message_id:
        raise HTTPException(404, "此记忆没有来源消息")
    message = await db.scalar(
        select(Message).where(
            Message.id == item.source_message_id,
            Message.user_id == identity.user.id,
        )
    )
    if not message:
        raise HTTPException(404, "来源消息不可用")
    await owned_conversation(db, identity.user.id, message.conversation_id)
    return {
        "content": message.content,
        "role": message.role,
        "seq": message.seq,
        "conversation_id": message.conversation_id,
    }


@router.patch("/memories/{mid}")
async def edit_memory(mid: UUID, body: MemoryUpdate, db: DB, identity: CurrentIdentity) -> dict:
    await lock_user(db, identity.user.id)
    item = await owned_memory(db, identity.user.id, mid)
    merged = {key: getattr(item, key) for key in MemoryInput.model_fields}
    changes = body.model_dump(exclude_unset=True)
    if any(value is None for value in changes.values()):
        raise HTTPException(422, "记忆字段不能为空")
    merged.update(changes)
    checked = MemoryInput.model_validate(merged)
    if checked.epistemic == "evidence_linked" and not checked.citation_ids:
        raise HTTPException(422, "关联证据的记忆需要提供引用")
    # Existing provenance stays editable even when a source is deleted or moved.
    # Only newly supplied links need current ownership/scope validation again.
    if (
        "citation_ids" in changes
        and checked.citation_ids
        and [str(v) for v in checked.citation_ids] != item.citation_ids
    ):
        await validate_memory(db, identity.user.id, checked)
    for key in changes:
        value = getattr(checked, key)
        setattr(item, key, [str(v) for v in value] if key == "citation_ids" else value)
    item.updated_at = utcnow()
    await db.commit()
    return memory_json(item, await evidence_map(db, identity.user.id, [item]))


@router.delete("/memories/{mid}", status_code=204)
async def delete_memory(mid: UUID, db: DB, identity: CurrentIdentity) -> None:
    await lock_user(db, identity.user.id)
    item = await owned_memory(db, identity.user.id, mid)
    before = {
        "content": item.content,
        "status": item.status,
        "kind": item.kind,
        "epistemic": item.epistemic,
        "pinned": item.pinned,
    }
    item.deleted_at = utcnow()
    db.add(
        MemoryEvent(
            user_id=identity.user.id,
            project_id=item.project_id,
            conversation_id=item.conversation_id,
            memory_id=item.id,
            target="entry",
            action="delete",
            actor="user",
            summary=f"删除记忆：{compact_text(item.content, 48)}",
            before_state=before,
            after_state={"deleted": True},
        )
    )
    await db.commit()


@router.post("/memories/bulk")
async def bulk_memories(body: BulkMemoryInput, db: DB, identity: CurrentIdentity) -> dict:
    uid = identity.user.id
    await lock_user(db, uid)
    if body.action == "move_project":
        if not body.project_id:
            raise HTTPException(422, "迁移项目需要提供目标项目")
        await owned_project(db, uid, body.project_id)
    items = list(
        await db.scalars(
            select(MemoryEntry).where(
                MemoryEntry.user_id == uid,
                MemoryEntry.id.in_(body.ids),
                MemoryEntry.deleted_at.is_(None),
            )
        )
    )
    if len(items) != len(set(body.ids)):
        raise HTTPException(404, "部分记忆不存在或无权操作")
    now = utcnow()
    for item in items:
        if body.action == "confirm":
            item.status = "active"
        elif body.action == "supersede":
            item.status = "superseded"
        elif body.action == "delete":
            item.deleted_at = now
        elif body.action == "pin":
            item.pinned = True
        elif body.action == "unpin":
            item.pinned = False
        elif body.action == "move_project":
            item.level = "project"
            item.project_id = body.project_id
            item.conversation_id = None
            item.paper_id = None
        item.updated_at = now
    await db.commit()
    return {"updated": len(items)}


@router.post("/memories/cleanup-discussed")
async def cleanup_discussed_candidates(
    db: DB, identity: CurrentIdentity, project_id: UUID | None = None
) -> dict:
    uid = identity.user.id
    await lock_user(db, uid)
    stmt = select(MemoryEntry).where(
        MemoryEntry.user_id == uid,
        MemoryEntry.deleted_at.is_(None),
        MemoryEntry.kind == "discussed",
        MemoryEntry.status == "candidate",
    )
    if project_id:
        await owned_project(db, uid, project_id)
        stmt = stmt.where(MemoryEntry.project_id == project_id)
    rows = list(await db.scalars(stmt))
    now = utcnow()
    for row in rows:
        row.deleted_at = now
        row.updated_at = now
    await db.commit()
    return {"cleaned": len(rows)}


@router.get("/memory/settings")
async def memory_settings(db: DB, identity: CurrentIdentity) -> dict:
    return {"memory_write_mode": await get_write_mode(db, identity.user.id)}


@router.put("/memory/settings")
async def update_memory_settings(
    body: MemorySettingsInput, db: DB, identity: CurrentIdentity
) -> dict:
    uid = identity.user.id
    await db.scalar(select(User).where(User.id == uid).with_for_update())
    pref = await db.get(UserSetting, uid)
    if pref is None:
        pref = UserSetting(user_id=uid, memory_write_mode=body.memory_write_mode)
        db.add(pref)
    else:
        pref.memory_write_mode = body.memory_write_mode
        pref.updated_at = utcnow()
    await db.commit()
    return {"memory_write_mode": pref.memory_write_mode}


@router.get("/memory/core")
async def core_memory(db: DB, identity: CurrentIdentity, project_id: UUID | None = None) -> dict:
    uid = identity.user.id
    profile = await get_core_block(db, uid, "user")
    project = await get_core_block(db, uid, "project", project_id) if project_id else None
    return {"profile": profile, "project": project}


@router.put("/memory/core")
async def edit_core_memory(
    body: CoreMemoryInput,
    db: DB,
    identity: CurrentIdentity,
    level: Literal["user", "project"] = "user",
    project_id: UUID | None = None,
) -> dict:
    uid = identity.user.id
    await lock_user(db, uid)
    block, _ = await save_core_block(db, uid, level, body.content, project_id, actor="user")
    await db.commit()
    return block


@router.get("/memory/events")
async def memory_events(
    db: DB,
    identity: CurrentIdentity,
    project_id: UUID | None = None,
    conversation_id: UUID | None = None,
    limit: int = Query(50, ge=1, le=100),
) -> list[dict]:
    return await list_memory_events(
        db,
        identity.user.id,
        project_id=project_id,
        conversation_id=conversation_id,
        limit=limit,
    )


@router.post("/memory/events/{eid}/undo")
async def undo_event(eid: UUID, db: DB, identity: CurrentIdentity) -> dict:
    await lock_user(db, identity.user.id)
    result = await undo_memory_event(db, identity.user.id, eid)
    await db.commit()
    return result


@router.get("/memory/sessions/search")
async def search_sessions(
    db: DB,
    identity: CurrentIdentity,
    q: str = Query(min_length=1, max_length=200),
    project_id: UUID | None = None,
    project_only: bool = False,
    limit: int = Query(10, ge=1, le=20),
) -> list[dict]:
    if project_id:
        await owned_project(db, identity.user.id, project_id)
    return await session_search(
        db,
        identity.user.id,
        q,
        project_id=project_id,
        project_only=project_only,
        limit=limit,
    )


@router.get("/conversations/{cid}/memory")
async def conversation_memory(cid: UUID, db: DB, identity: CurrentIdentity) -> dict:
    item = await owned_conversation(db, identity.user.id, cid)
    return {
        "summary": item.summary,
        "research_state": item.research_state,
        "summarized_seq": item.summarized_seq,
        "project_id": item.project_id,
        "active_paper_ids": item.active_paper_ids,
    }


@router.patch("/conversations/{cid}/memory")
async def edit_conversation_memory(
    cid: UUID, body: ConversationMemoryUpdate, db: DB, identity: CurrentIdentity
) -> dict:
    item = await idle_conversation(db, identity.user.id, cid)
    seqs = {v["source_seq"] for values in body.research_state.model_dump().values() for v in values}
    if seqs:
        found = set(
            await db.scalars(
                select(Message.seq).where(
                    Message.user_id == identity.user.id,
                    Message.conversation_id == cid,
                    Message.seq.in_(seqs),
                )
            )
        )
        if found != seqs:
            raise HTTPException(422, "研究状态的来源消息不存在")
    item.summary, item.research_state = body.summary, body.research_state.model_dump()
    item.memory_revision += 1
    await db.commit()
    return {"summary": item.summary, "research_state": item.research_state}


@router.delete("/conversations/{cid}/memory", status_code=204)
async def clear_conversation_memory(cid: UUID, db: DB, identity: CurrentIdentity) -> None:
    item = await idle_conversation(db, identity.user.id, cid)
    item.summary, item.research_state, item.active_paper_ids = "", {}, []
    item.summarized_seq = item.next_seq - 1
    item.memory_revision += 1
    await db.execute(
        update(MemoryEntry)
        .where(
            MemoryEntry.user_id == identity.user.id,
            MemoryEntry.conversation_id == cid,
        )
        .values(deleted_at=utcnow())
    )
    await db.commit()


@router.get("/papers/{pid}/reading-position")
async def reading_position(pid: UUID, db: DB, identity: CurrentIdentity) -> dict:
    paper = await db.scalar(
        select(Paper).where(
            Paper.id == pid, Paper.user_id == identity.user.id, Paper.deleted_at.is_(None)
        )
    )
    if not paper:
        raise HTTPException(404, "论文不存在")
    item = await db.scalar(
        select(MemoryEntry).where(
            MemoryEntry.user_id == identity.user.id,
            MemoryEntry.paper_id == pid,
            MemoryEntry.fingerprint
            == hashlib.sha256(f"reading:{identity.user.id}:{pid}".encode()).hexdigest(),
            MemoryEntry.deleted_at.is_(None),
            MemoryEntry.status == "active",
        )
    )
    return {
        "position": item.reading_anchor if item else None,
        "current": bool(
            item and item.reading_anchor.get("version_id") == str(paper.active_version_id)
        ),
    }


@router.put("/papers/{pid}/reading-position")
async def save_reading_position(
    pid: UUID, body: ReadingPositionInput, db: DB, identity: CurrentIdentity
) -> dict:
    await lock_user(db, identity.user.id)
    paper = await db.scalar(
        select(Paper).where(
            Paper.id == pid, Paper.user_id == identity.user.id, Paper.deleted_at.is_(None)
        )
    )
    if not paper:
        raise HTTPException(404, "论文不存在")
    if paper.active_version_id != body.version_id:
        raise HTTPException(409, "论文版本已更新，请刷新后保存阅读位置")
    if body.page_idx >= paper.page_count:
        raise HTTPException(422, "阅读位置超出论文页数")
    fingerprint = hashlib.sha256(f"reading:{identity.user.id}:{pid}".encode()).hexdigest()
    item = await db.scalar(
        select(MemoryEntry).where(
            MemoryEntry.user_id == identity.user.id, MemoryEntry.fingerprint == fingerprint
        )
    )
    if not item:
        item = MemoryEntry(
            user_id=identity.user.id,
            level="paper",
            paper_id=pid,
            kind="focus",
            fingerprint=fingerprint,
        )
        db.add(item)
    item.content = f"正在阅读《{paper.title[:100]}》第 {body.page_idx + 1} 页"
    item.reading_anchor = {"version_id": str(body.version_id), "page_idx": body.page_idx}
    item.status, item.epistemic, item.deleted_at, item.updated_at = (
        "active",
        "user_statement",
        None,
        utcnow(),
    )
    await db.commit()
    return {"position": item.reading_anchor, "current": True}
