"""Hermes-inspired core memory blocks, cross-session search, deduplicated writes and undo."""

import hashlib
import re
from typing import Annotated, Any, Literal
from uuid import UUID

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    Conversation,
    MemoryCoreBlock,
    MemoryEntry,
    MemoryEvent,
    Message,
    ResearchProject,
    UserSetting,
    utcnow,
)
from app.schemas.memory import MemoryKind
from app.services.memory.history import compact_text
from app.services.memory.store import owned_project, terms

PROFILE_CHAR_LIMIT = 1500
PROJECT_CHAR_LIMIT = 3000


class MemorySearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=500)
    kind: MemoryKind | None = None
    limit: int = Field(default=8, ge=1, le=20)


class SessionSearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=500)
    project_only: bool = True
    limit: int = Field(default=6, ge=1, le=20)


class MemoryUpdateArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    target: Literal["profile", "project", "entry"] = "entry"
    action: Literal["add", "update", "supersede", "delete"] = "add"
    kind: MemoryKind = "note"
    content: str = Field(min_length=1, max_length=1000)
    replace_id: UUID | None = None


class MemorySearchToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["memory_search"]
    args: MemorySearchArgs


class SessionSearchToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["session_search"]
    args: SessionSearchArgs


class MemoryUpdateToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["memory_update"]
    args: MemoryUpdateArgs


MemoryToolCall = Annotated[
    MemorySearchToolCall | SessionSearchToolCall | MemoryUpdateToolCall,
    Field(discriminator="name"),
]

MEMORY_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": schema.model_json_schema(),
        },
    }
    for name, description, schema in (
        (
            "memory_search",
            "按关键词与研究类型检索用户的长期档案记忆（目标、约束、假设、研究判断、阅读位置）。",
            MemorySearchArgs,
        ),
        (
            "session_search",
            "跨会话全文检索历史问答原文片段，用于找回此前会话讨论过的细节。",
            SessionSearchArgs,
        ),
        (
            "memory_update",
            "新增、合并或修正研究者画像（profile）、项目核心备忘（project）或长期档案记忆（entry）。",
            MemoryUpdateArgs,
        ),
    )
]


def _norm(text: str) -> str:
    cleaned = re.sub(r"[\s，。！？、；：,.!?;:]+$", "", text.strip().lower())
    return re.sub(r"\s+", " ", cleaned)


def similarity_score(a: str, b: str) -> float:
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return 0.0
    if na == nb or (len(na) >= 8 and (na in nb or nb in na)):
        return 1.0
    ta, tb = terms(na), terms(nb)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, min(len(ta), len(tb)))


def merge_core_line(existing: str, line: str, limit: int) -> tuple[str, bool]:
    clean = re.sub(r"^[-*•\s]+", "", line.strip())
    if not clean:
        return existing, False
    bullet = f"- {clean}"
    raw_lines = [ln.strip() for ln in existing.splitlines() if ln.strip()]
    updated = False
    new_lines: list[str] = []
    for ln in raw_lines:
        body = re.sub(r"^[-*•\s]+", "", ln)
        if _norm(body) == _norm(clean):
            return existing, False
        if similarity_score(body, clean) >= 0.78 and not updated:
            new_lines.append(bullet)
            updated = True
        else:
            new_lines.append(ln if ln.startswith(("-", "*", "•")) else f"- {ln}")
    if not updated:
        new_lines.append(bullet)
    joined = "\n".join(new_lines)
    while len(joined) > limit and len(new_lines) > 1:
        new_lines.pop(0)
        joined = "\n".join(new_lines)
    if len(joined) > limit:
        joined = joined[:limit]
    return joined, True


def event_json(event: MemoryEvent) -> dict:
    return {
        "id": str(event.id),
        "project_id": str(event.project_id) if event.project_id else None,
        "conversation_id": str(event.conversation_id) if event.conversation_id else None,
        "run_id": str(event.run_id) if event.run_id else None,
        "memory_id": str(event.memory_id) if event.memory_id else None,
        "target": event.target,
        "action": event.action,
        "actor": event.actor,
        "summary": event.summary,
        "undone_at": event.undone_at.isoformat() if event.undone_at else None,
        "created_at": event.created_at.isoformat() if event.created_at else None,
    }


async def get_write_mode(db: AsyncSession, user_id: UUID) -> Literal["auto", "review"]:
    pref = await db.get(UserSetting, user_id)
    if pref and pref.memory_write_mode in ("auto", "review"):
        return pref.memory_write_mode  # type: ignore[return-value]
    return "auto"


async def get_core_block(
    db: AsyncSession,
    user_id: UUID,
    level: Literal["user", "project"],
    project_id: UUID | None = None,
) -> dict:
    limit = PROFILE_CHAR_LIMIT if level == "user" else PROJECT_CHAR_LIMIT
    if level == "project":
        if not project_id:
            raise HTTPException(422, "项目核心备忘需要指定项目")
        project = await owned_project(db, user_id, project_id)
        stmt = select(MemoryCoreBlock).where(
            MemoryCoreBlock.user_id == user_id,
            MemoryCoreBlock.level == "project",
            MemoryCoreBlock.project_id == project_id,
        )
    else:
        project = None
        stmt = select(MemoryCoreBlock).where(
            MemoryCoreBlock.user_id == user_id,
            MemoryCoreBlock.level == "user",
            MemoryCoreBlock.project_id.is_(None),
        )
    block = await db.scalar(stmt)
    content = block.content if block is not None else (project.description if project else "")
    return {
        "level": level,
        "project_id": str(project_id) if project_id else None,
        "project_name": project.name if project else None,
        "content": content,
        "char_count": len(content),
        "char_limit": limit,
        "revision": block.revision if block else 0,
        "updated_at": (
            block.updated_at if block else (project.created_at if project else utcnow())
        ).isoformat(),
    }


async def save_core_block(
    db: AsyncSession,
    user_id: UUID,
    level: Literal["user", "project"],
    content: str,
    project_id: UUID | None = None,
    *,
    actor: Literal["user", "agent", "review"] = "user",
    conversation_id: UUID | None = None,
    run_id: UUID | None = None,
) -> tuple[dict, MemoryEvent | None]:
    limit = PROFILE_CHAR_LIMIT if level == "user" else PROJECT_CHAR_LIMIT
    if len(content) > limit:
        raise HTTPException(422, f"核心记忆内容不能超过 {limit} 字符")
    if level == "project":
        if not project_id:
            raise HTTPException(422, "项目核心备忘需要指定项目")
        project = await owned_project(db, user_id, project_id)
        stmt = select(MemoryCoreBlock).where(
            MemoryCoreBlock.user_id == user_id,
            MemoryCoreBlock.level == "project",
            MemoryCoreBlock.project_id == project_id,
        )
    else:
        project = None
        stmt = select(MemoryCoreBlock).where(
            MemoryCoreBlock.user_id == user_id,
            MemoryCoreBlock.level == "user",
            MemoryCoreBlock.project_id.is_(None),
        )
    block = await db.scalar(stmt)
    before = block.content if block is not None else (project.description if project else "")
    if before == content and block is not None:
        return await get_core_block(db, user_id, level, project_id), None
    if block is None:
        block = MemoryCoreBlock(
            user_id=user_id,
            level=level,
            project_id=project_id if level == "project" else None,
            content=content,
            revision=1,
            updated_at=utcnow(),
        )
        db.add(block)
    else:
        block.content = content
        block.revision += 1
        block.updated_at = utcnow()
    if project is not None:
        project.description = content[:2000]
    label = "研究者画像" if level == "user" else f"项目备忘（{project.name if project else ''}）"
    ev = MemoryEvent(
        user_id=user_id,
        project_id=project_id if level == "project" else None,
        conversation_id=conversation_id,
        run_id=run_id,
        target="profile" if level == "user" else "project",
        action="update",
        actor=actor,
        summary=f"更新{label}：{compact_text(content, 48) or '已清空'}",
        before_state={"content": before},
        after_state={"content": content},
    )
    db.add(ev)
    await db.flush()
    return await get_core_block(db, user_id, level, project_id), ev


async def load_core_snapshot(
    db: AsyncSession, user_id: UUID, project_id: UUID | None = None
) -> dict:
    profile = await get_core_block(db, user_id, "user")
    project = await get_core_block(db, user_id, "project", project_id) if project_id else None
    result: dict[str, Any] = {}
    if profile["content"].strip():
        result["profile"] = profile["content"].strip()
    if project and project["content"].strip():
        result["project"] = {
            "name": project["project_name"],
            "notebook": project["content"].strip(),
        }
    return result


async def smart_save_entry(
    db: AsyncSession,
    user_id: UUID,
    *,
    level: Literal["user", "project", "paper", "conversation"],
    kind: str,
    content: str,
    project_id: UUID | None = None,
    conversation_id: UUID | None = None,
    paper_id: UUID | None = None,
    status: Literal["candidate", "active", "superseded"] = "active",
    epistemic: Literal["user_statement", "hypothesis", "evidence_linked"] = "user_statement",
    pinned: bool = False,
    source_message_id: UUID | None = None,
    citation_ids: list[str] | None = None,
    fingerprint: str | None = None,
    actor: Literal["user", "agent", "review"] = "agent",
    run_id: UUID | None = None,
    replace_id: UUID | None = None,
) -> tuple[MemoryEntry | None, MemoryEvent | None, str]:
    clean = content.strip()
    if not clean:
        return None, None, "noop"
    if fingerprint:
        tombstone = await db.scalar(
            select(MemoryEntry).where(
                MemoryEntry.user_id == user_id, MemoryEntry.fingerprint == fingerprint
            )
        )
        if tombstone is not None:
            return tombstone, None, "noop"

    stmt = select(MemoryEntry).where(
        MemoryEntry.user_id == user_id,
        MemoryEntry.deleted_at.is_(None),
        MemoryEntry.level == level,
        MemoryEntry.project_id == project_id if project_id else MemoryEntry.project_id.is_(None),
    )
    if level == "conversation" and conversation_id:
        stmt = stmt.where(MemoryEntry.conversation_id == conversation_id)
    if level == "paper" and paper_id:
        stmt = stmt.where(MemoryEntry.paper_id == paper_id)
    candidates = list(await db.scalars(stmt.order_by(MemoryEntry.updated_at.desc()).limit(100)))

    if replace_id:
        target_item = next((c for c in candidates if c.id == replace_id), None)
        if target_item:
            before = {
                "content": target_item.content,
                "status": target_item.status,
                "kind": target_item.kind,
                "epistemic": target_item.epistemic,
                "pinned": target_item.pinned,
            }
            target_item.content = clean
            target_item.kind = kind
            target_item.status = status
            target_item.updated_at = utcnow()
            ev = MemoryEvent(
                user_id=user_id,
                project_id=project_id,
                conversation_id=conversation_id,
                run_id=run_id,
                memory_id=target_item.id,
                target="entry",
                action="update",
                actor=actor,
                summary=f"更新记忆：{compact_text(clean, 48)}",
                before_state=before,
                after_state={
                    "content": clean,
                    "status": status,
                    "kind": kind,
                    "epistemic": target_item.epistemic,
                    "pinned": target_item.pinned,
                },
            )
            db.add(ev)
            await db.flush()
            return target_item, ev, "updated"

    superseded_ids: list[str] = []
    if kind == "correction":
        for existing in candidates:
            if existing.status == "active" and existing.kind in {
                "goal",
                "conclusion",
                "hypothesis",
                "focus",
            }:
                if similarity_score(existing.content, clean) >= 0.35:
                    existing.status = "superseded"
                    existing.updated_at = utcnow()
                    superseded_ids.append(str(existing.id))

    for existing in candidates:
        if existing.status == "superseded" or existing.reading_anchor:
            continue
        sim = similarity_score(existing.content, clean)
        if _norm(existing.content) == _norm(clean):
            if existing.status == "candidate" and status == "active":
                before = {
                    "content": existing.content,
                    "status": existing.status,
                    "kind": existing.kind,
                    "epistemic": existing.epistemic,
                    "pinned": existing.pinned,
                }
                existing.status = "active"
                existing.updated_at = utcnow()
                ev = MemoryEvent(
                    user_id=user_id,
                    project_id=project_id,
                    conversation_id=conversation_id,
                    run_id=run_id,
                    memory_id=existing.id,
                    target="entry",
                    action="update",
                    actor=actor,
                    summary=f"确认记忆：{compact_text(clean, 48)}",
                    before_state=before,
                    after_state={**before, "status": "active"},
                )
                db.add(ev)
                await db.flush()
                return existing, ev, "updated"
            return existing, None, "noop"
        if existing.kind == kind and sim >= 0.8:
            before = {
                "content": existing.content,
                "status": existing.status,
                "kind": existing.kind,
                "epistemic": existing.epistemic,
                "pinned": existing.pinned,
            }
            existing.content = clean
            if status == "active":
                existing.status = "active"
            existing.updated_at = utcnow()
            ev = MemoryEvent(
                user_id=user_id,
                project_id=project_id,
                conversation_id=conversation_id,
                run_id=run_id,
                memory_id=existing.id,
                target="entry",
                action="update",
                actor=actor,
                summary=f"合并更新记忆：{compact_text(clean, 48)}",
                before_state=before,
                after_state={
                    "content": existing.content,
                    "status": existing.status,
                    "kind": existing.kind,
                    "epistemic": existing.epistemic,
                    "pinned": existing.pinned,
                },
            )
            db.add(ev)
            await db.flush()
            return existing, ev, "updated"

    item = MemoryEntry(
        user_id=user_id,
        level=level,
        project_id=project_id,
        conversation_id=conversation_id if level == "conversation" else None,
        paper_id=paper_id if level == "paper" else None,
        kind=kind,
        content=clean,
        status=status,
        epistemic=epistemic,
        pinned=pinned,
        source_message_id=source_message_id,
        citation_ids=citation_ids or [],
        fingerprint=fingerprint,
    )
    db.add(item)
    await db.flush()
    action: Literal["add", "supersede"] = "supersede" if superseded_ids else "add"
    ev = MemoryEvent(
        user_id=user_id,
        project_id=project_id,
        conversation_id=conversation_id,
        run_id=run_id,
        memory_id=item.id,
        target="entry",
        action=action,
        actor=actor,
        summary=f"{'纠正并记住' if superseded_ids else '已记住'}：{compact_text(clean, 48)}",
        before_state={"superseded_ids": superseded_ids},
        after_state={
            "id": str(item.id),
            "content": item.content,
            "kind": item.kind,
            "status": item.status,
            "level": item.level,
        },
    )
    db.add(ev)
    await db.flush()
    return item, ev, "added"


async def session_search(
    db: AsyncSession,
    user_id: UUID,
    query: str,
    *,
    current_conversation_id: UUID | None = None,
    project_id: UUID | None = None,
    project_only: bool = False,
    limit: int = 6,
) -> list[dict]:
    """Cross-session recall over raw message transcripts (pg_trgm + lexical ranking)."""
    q_terms = terms(query)
    raw_words = [
        w for w in re.findall(r"[\u3400-\u9fff]{2,6}|[a-zA-Z0-9_]{2,}", query) if len(w) >= 2
    ]
    stmt = (
        select(Message, Conversation)
        .join(
            Conversation,
            (Conversation.id == Message.conversation_id)
            & (Conversation.user_id == Message.user_id),
        )
        .where(
            Message.user_id == user_id,
            Conversation.deleted_at.is_(None),
        )
    )
    if project_only and project_id:
        stmt = stmt.where(Conversation.project_id == project_id)
    if current_conversation_id:
        # Exclude recent unsummarized turns of the active conversation to avoid duplicating history.
        stmt = stmt.where(
            or_(
                Message.conversation_id != current_conversation_id,
                Message.seq <= Conversation.summarized_seq,
            )
        )
    if raw_words:
        stmt = stmt.where(or_(*(Message.content.ilike(f"%{w}%") for w in raw_words[:6])))
    rows = (
        await db.execute(stmt.order_by(Message.created_at.desc(), Message.seq.desc()).limit(120))
    ).all()
    ranked: list[tuple[int, str, dict]] = []
    for msg, conv in rows:
        overlap = len(q_terms & terms(msg.content))
        exact_bonus = sum(4 for w in raw_words[:6] if w.lower() in msg.content.lower())
        project_bonus = 3 if (project_id and conv.project_id == project_id) else 0
        score = overlap * 5 + exact_bonus + project_bonus + (2 if msg.role == "user" else 1)
        if q_terms and overlap == 0 and exact_bonus == 0:
            continue
        ranked.append(
            (
                score,
                msg.created_at.isoformat() if msg.created_at else "",
                {
                    "message_id": str(msg.id),
                    "conversation_id": str(conv.id),
                    "conversation_title": conv.title,
                    "project_id": str(conv.project_id) if conv.project_id else None,
                    "seq": msg.seq,
                    "role": msg.role,
                    "snippet": compact_text(msg.content, 160),
                    "created_at": msg.created_at.isoformat() if msg.created_at else None,
                },
            )
        )
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in ranked[:limit]]


_TURN_PATTERNS: list[tuple[MemoryKind, re.Pattern[str]]] = [
    (
        "correction",
        re.compile(
            r"(?:不对|纠正|更正|理解错了|并不是|其实不是|应该是|请更正|correction|actually\s+it\s+is)",
            re.I,
        ),
    ),
    (
        "constraint",
        re.compile(
            r"(?:不要|只看|限定在|不考虑|排除|必须包含|只关注|请勿|only\s+consider|do\s+not\s+include|exclude)",
            re.I,
        ),
    ),
    (
        "preference",
        re.compile(
            r"(?:我习惯|我的背景是|回答时请|请用.*格式|我偏好|记住我|我是.*研究|i\s+prefer|my\s+background\s+is)",
            re.I,
        ),
    ),
    (
        "hypothesis",
        re.compile(
            r"(?:我的假设是|我猜想|我怀疑|我推测|有没有可能是因为|待验证假设|my\s+hypothesis|i\s+suspect\s+that)",
            re.I,
        ),
    ),
    (
        "goal",
        re.compile(
            r"(?:研究目标是|我的目标是|我正在研究|本课题旨在|接下来重点研究|我要对比|our\s+goal\s+is|i\s+am\s+researching)",
            re.I,
        ),
    ),
]


async def curate_user_turn(
    db: AsyncSession, conversation: Conversation, user_message: Message, run_id: UUID
) -> list[dict]:
    """Extract durable user statements without spamming every question as a candidate."""
    text = user_message.content.strip()
    if len(text) < 6 or len(text) > 1200:
        return []
    matched_kind: MemoryKind | None = None
    for kind, pattern in _TURN_PATTERNS:
        if pattern.search(text):
            matched_kind = kind
            break
    if not matched_kind:
        return []
    mode = await get_write_mode(db, conversation.user_id)
    target_status: Literal["active", "candidate"] = "active" if mode == "auto" else "candidate"
    target_level: Literal["user", "project", "conversation"] = (
        "user"
        if matched_kind == "preference"
        else "project"
        if conversation.project_id
        else "conversation"
    )
    epistemic: Literal["user_statement", "hypothesis"] = (
        "hypothesis" if matched_kind == "hypothesis" else "user_statement"
    )
    fingerprint = hashlib.sha256(
        f"{conversation.id}:{user_message.id}:{matched_kind}".encode()
    ).hexdigest()
    _, ev, outcome = await smart_save_entry(
        db,
        conversation.user_id,
        level=target_level,
        kind=matched_kind,
        content=compact_text(text, 220),
        project_id=None if target_level == "user" else conversation.project_id,
        conversation_id=conversation.id if target_level == "conversation" else None,
        status=target_status,
        epistemic=epistemic,
        source_message_id=user_message.id,
        fingerprint=fingerprint,
        actor="agent",
        run_id=run_id,
    )
    events: list[dict] = []
    if ev is not None and outcome in {"added", "updated"}:
        events.append(event_json(ev))
    if mode == "auto" and matched_kind in {"preference", "goal", "constraint", "correction"}:
        core_level: Literal["user", "project"] | None = (
            "user"
            if matched_kind == "preference"
            else ("project" if conversation.project_id else None)
        )
        if core_level:
            pid = conversation.project_id if core_level == "project" else None
            current = await get_core_block(db, conversation.user_id, core_level, pid)
            merged, changed = merge_core_line(
                current["content"], compact_text(text, 120), current["char_limit"]
            )
            if changed:
                await save_core_block(
                    db,
                    conversation.user_id,
                    core_level,
                    merged,
                    pid,
                    actor="agent",
                    conversation_id=conversation.id,
                    run_id=run_id,
                )
    return events


async def list_memory_events(
    db: AsyncSession,
    user_id: UUID,
    *,
    project_id: UUID | None = None,
    conversation_id: UUID | None = None,
    limit: int = 50,
) -> list[dict]:
    stmt = select(MemoryEvent).where(MemoryEvent.user_id == user_id)
    if project_id:
        stmt = stmt.where(MemoryEvent.project_id == project_id)
    if conversation_id:
        stmt = stmt.where(MemoryEvent.conversation_id == conversation_id)
    rows = list(await db.scalars(stmt.order_by(MemoryEvent.created_at.desc()).limit(limit)))
    return [event_json(r) for r in rows]


async def undo_memory_event(db: AsyncSession, user_id: UUID, event_id: UUID) -> dict:
    ev = await db.scalar(
        select(MemoryEvent).where(MemoryEvent.id == event_id, MemoryEvent.user_id == user_id)
    )
    if not ev:
        raise HTTPException(404, "记忆操作记录不存在")
    if ev.undone_at is not None:
        raise HTTPException(409, "该记忆操作已撤销")
    if ev.target in ("profile", "project"):
        level: Literal["user", "project"] = "user" if ev.target == "profile" else "project"
        prev_content = str(ev.before_state.get("content", ""))
        stmt = select(MemoryCoreBlock).where(
            MemoryCoreBlock.user_id == user_id,
            MemoryCoreBlock.level == level,
            MemoryCoreBlock.project_id == ev.project_id
            if level == "project"
            else MemoryCoreBlock.project_id.is_(None),
        )
        block = await db.scalar(stmt)
        if block:
            block.content = prev_content
            block.revision += 1
            block.updated_at = utcnow()
        if level == "project" and ev.project_id:
            proj = await db.get(ResearchProject, ev.project_id)
            if proj and proj.user_id == user_id:
                proj.description = prev_content[:2000]
    elif ev.target == "entry" and ev.memory_id:
        entry = await db.scalar(
            select(MemoryEntry).where(
                MemoryEntry.id == ev.memory_id, MemoryEntry.user_id == user_id
            )
        )
        if entry:
            if ev.action in ("add", "supersede"):
                entry.deleted_at = utcnow()
                entry.updated_at = utcnow()
                superseded_ids = ev.before_state.get("superseded_ids", [])
                if superseded_ids:
                    await db.execute(
                        update(MemoryEntry)
                        .where(
                            MemoryEntry.user_id == user_id,
                            MemoryEntry.id.in_([UUID(x) for x in superseded_ids]),
                        )
                        .values(status="active", updated_at=utcnow())
                    )
            elif ev.action in ("update", "delete"):
                before = ev.before_state
                if "content" in before:
                    entry.content = before["content"]
                if "status" in before:
                    entry.status = before["status"]
                if "kind" in before:
                    entry.kind = before["kind"]
                if "epistemic" in before:
                    entry.epistemic = before["epistemic"]
                if "pinned" in before:
                    entry.pinned = bool(before["pinned"])
                entry.deleted_at = None
                entry.updated_at = utcnow()
    ev.undone_at = utcnow()
    await db.flush()
    return event_json(ev)
