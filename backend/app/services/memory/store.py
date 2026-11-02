"""Owner/project-scoped memory with live evidence checks and bounded recall."""

import hashlib
import json
import re
from typing import Any
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    Chunk,
    Citation,
    Conversation,
    MemoryEntry,
    Message,
    Paper,
    ResearchProject,
)
from app.schemas.memory import MemoryInput, ResearchState
from app.services.memory.history import estimate_tokens


async def owned_project(db: AsyncSession, user_id: UUID, project_id: UUID) -> ResearchProject:
    item = await db.scalar(
        select(ResearchProject).where(
            ResearchProject.id == project_id,
            ResearchProject.user_id == user_id,
            ResearchProject.deleted_at.is_(None),
        )
    )
    if not item:
        raise HTTPException(404, "研究项目不存在")
    return item


async def owned_memory(db: AsyncSession, user_id: UUID, mid: UUID) -> MemoryEntry:
    item = await db.scalar(
        select(MemoryEntry).where(
            MemoryEntry.id == mid,
            MemoryEntry.user_id == user_id,
            MemoryEntry.deleted_at.is_(None),
        )
    )
    if not item:
        raise HTTPException(404, "记忆不存在")
    return item


async def validate_memory(db: AsyncSession, user_id: UUID, body: MemoryInput) -> None:
    if body.level == "user":
        valid = not (body.project_id or body.conversation_id or body.paper_id)
    elif body.level == "project":
        valid = bool(body.project_id) and not (body.conversation_id or body.paper_id)
    elif body.level == "paper":
        valid = bool(body.paper_id) and not body.conversation_id
    else:
        valid = bool(body.conversation_id) and not body.paper_id
    if not valid:
        raise HTTPException(422, "记忆层级与归属不匹配")
    if body.project_id:
        await owned_project(db, user_id, body.project_id)
    if body.paper_id:
        paper = await db.scalar(
            select(Paper).where(
                Paper.id == body.paper_id,
                Paper.user_id == user_id,
                Paper.deleted_at.is_(None),
            )
        )
        if not paper:
            raise HTTPException(404, "论文不存在")
    if body.conversation_id:
        conv = await db.get(Conversation, body.conversation_id)
        if not conv or conv.user_id != user_id or conv.deleted_at:
            raise HTTPException(404, "会话不存在")
        if body.project_id != conv.project_id:
            raise HTTPException(422, "会话与研究项目不匹配")
    if body.source_message_id:
        msg = await db.scalar(
            select(Message)
            .join(Conversation, Conversation.id == Message.conversation_id)
            .where(
                Message.id == body.source_message_id,
                Message.user_id == user_id,
                Conversation.deleted_at.is_(None),
            )
        )
        if not msg:
            raise HTTPException(404, "来源消息不存在")
        conv = await db.get(Conversation, msg.conversation_id)
        assert conv
        if (body.project_id and conv.project_id != body.project_id) or (
            body.conversation_id and msg.conversation_id != body.conversation_id
        ):
            raise HTTPException(422, "来源消息超出记忆归属")
    if body.epistemic == "evidence_linked" and not body.citation_ids:
        raise HTTPException(422, "关联证据的记忆需要提供引用")
    if body.citation_ids:
        citations = list(
            await db.scalars(
                select(Citation)
                .join(Message, Message.id == Citation.message_id)
                .join(Conversation, Conversation.id == Message.conversation_id)
                .where(
                    Citation.user_id == user_id,
                    Citation.id.in_(body.citation_ids),
                    Citation.deleted_at.is_(None),
                    Conversation.deleted_at.is_(None),
                )
            )
        )
        if {c.id for c in citations} != set(body.citation_ids):
            raise HTTPException(404, "引用不存在")
        for citation in citations:
            msg = await db.get(Message, citation.message_id)
            assert msg
            conv = await db.get(Conversation, msg.conversation_id)
            assert conv
            if (
                (body.source_message_id and citation.message_id != body.source_message_id)
                or (body.paper_id and citation.paper_id != body.paper_id)
                or (body.project_id and conv.project_id != body.project_id)
                or (body.conversation_id and msg.conversation_id != body.conversation_id)
            ):
                raise HTTPException(422, "引用超出记忆归属")


async def evidence_map(db: AsyncSession, user_id: UUID, entries: list[MemoryEntry]) -> dict:
    ids = {UUID(v) for e in entries for v in e.citation_ids}
    anchors = [e for e in entries if e.reading_anchor]
    result: dict = {}
    if anchors:
        papers = {
            p.id: p
            for p in await db.scalars(
                select(Paper).where(
                    Paper.user_id == user_id,
                    Paper.id.in_([e.paper_id for e in anchors]),
                )
            )
        }
        for entry in anchors:
            paper = papers.get(entry.paper_id) if entry.paper_id else None
            status = (
                "unavailable"
                if not paper or paper.deleted_at
                else "current"
                if str(paper.active_version_id) == entry.reading_anchor.get("version_id")
                else "outdated"
            )
            result[f"anchor:{entry.id}"] = {"status": status}
    if not ids:
        return result
    rows = (
        await db.execute(
            select(Citation, Paper, Chunk)
            .join(Paper, (Paper.id == Citation.paper_id) & (Paper.user_id == Citation.user_id))
            .join(Message, Message.id == Citation.message_id)
            .join(Conversation, Conversation.id == Message.conversation_id)
            .outerjoin(Chunk, (Chunk.id == Citation.chunk_id) & (Chunk.user_id == Citation.user_id))
            .where(
                Citation.user_id == user_id,
                Citation.id.in_(ids),
                Conversation.deleted_at.is_(None),
            )
        )
    ).all()
    for c, paper, chunk in rows:
        if c.deleted_at or paper.deleted_at:
            status = "unavailable"
        elif (
            paper.active_version_id != c.version_id
            or not chunk
            or chunk.text != c.source_text
            or chunk.paper_id != c.paper_id
            or chunk.version_id != c.version_id
        ):
            status = "outdated"
        else:
            status = "current"
        result[str(c.id)] = {
            "id": str(c.id),
            "paper_id": str(c.paper_id),
            "version_id": str(c.version_id),
            "chunk_id": str(c.chunk_id),
            "status": status,
            "title": paper.title if status != "unavailable" else "来源不可用",
            "locations": c.locations if status == "current" else [],
        }
    return result


def evidence_status(item: MemoryEntry, evidence: dict) -> str:
    if not item.citation_ids:
        return (
            evidence.get(f"anchor:{item.id}", {}).get("status", "unavailable")
            if item.reading_anchor
            else "unlinked"
        )
    states = [evidence.get(v, {}).get("status", "unavailable") for v in item.citation_ids]
    return (
        "unavailable"
        if "unavailable" in states
        else "outdated"
        if "outdated" in states
        else "current"
    )


def memory_json(item: MemoryEntry, evidence: dict) -> dict:
    return {
        "id": str(item.id),
        "level": item.level,
        "project_id": item.project_id,
        "conversation_id": item.conversation_id,
        "paper_id": item.paper_id,
        "kind": item.kind,
        "content": item.content,
        "status": item.status,
        "epistemic": item.epistemic,
        "pinned": item.pinned,
        "source_message_id": item.source_message_id,
        "citation_ids": item.citation_ids,
        "reading_anchor": item.reading_anchor,
        "evidence_status": evidence_status(item, evidence),
        "evidence": [
            evidence.get(v, {"id": v, "status": "unavailable"}) for v in item.citation_ids
        ],
        "created_at": item.created_at,
        "updated_at": item.updated_at,
    }


def terms(text: str) -> set[str]:
    text = text.lower()
    words = set(re.findall(r"[a-z0-9_]{2,}", text))
    for segment in re.findall(r"[\u3400-\u9fff]+", text):
        words.update(segment[i : i + 2] for i in range(max(1, len(segment) - 1)))
    return words


def select_memories(entries: list[dict], question: str, budget: int) -> list[dict]:
    """Transparent lexical relevance; pinned goals/constraints survive vague follow-ups."""
    query = terms(question)
    resume = bool(re.search(r"继续|上次|之前|刚才|continue|resume|last time", question, re.I))
    ranked = []
    for item in entries:
        overlap = len(query & terms(item["content"]))
        durable = item["kind"] in {
            "goal",
            "constraint",
            "correction",
            "next_step",
            "preference",
            "focus",
        }
        if not (
            item["pinned"]
            or durable
            or overlap
            or not query
            or (resume and item["kind"] == "discussed")
        ):
            continue
        score = (
            200 * (item["kind"] == "correction")
            + 100 * item["pinned"]
            + 12 * overlap
            + 4 * durable
            + 2 * (item["status"] == "active")
        )
        ranked.append((score, str(item["updated_at"]), item))
    chosen = []
    for _, _, item in sorted(ranked, key=lambda row: (row[0], row[1]), reverse=True):
        cost = estimate_tokens(json.dumps(item, ensure_ascii=False, default=str)) + 8
        if cost <= budget:
            chosen.append(item)
            budget -= cost
        if len(chosen) == 12:
            break
    return chosen


async def recall(
    db: AsyncSession, settings: Any, conversation: Conversation, scope: dict, question: str
) -> list[dict]:
    uid, pid = conversation.user_id, conversation.project_id
    papers_stmt = select(Paper.id).where(Paper.user_id == uid, Paper.deleted_at.is_(None))
    for key, column in (("paper_ids", Paper.id), ("collection_ids", Paper.collection_id)):
        if scope.get(key) is not None:
            papers_stmt = papers_stmt.where(column.in_([UUID(v) for v in scope[key]]))
    clauses = [
        MemoryEntry.level == "user",
        (MemoryEntry.level == "conversation")
        & (MemoryEntry.conversation_id == conversation.id)
        & (MemoryEntry.project_id == pid if pid else MemoryEntry.project_id.is_(None)),
    ]
    if pid:
        clauses.append((MemoryEntry.level == "project") & (MemoryEntry.project_id == pid))
    clauses.append(
        (MemoryEntry.level == "paper")
        & MemoryEntry.paper_id.in_(papers_stmt)
        & (
            or_(MemoryEntry.project_id.is_(None), MemoryEntry.project_id == pid)
            if pid
            else MemoryEntry.project_id.is_(None)
        )
    )
    entries = list(
        await db.scalars(
            select(MemoryEntry)
            .where(
                MemoryEntry.user_id == uid,
                MemoryEntry.deleted_at.is_(None),
                MemoryEntry.status.in_(["active", "candidate"]),
                or_(
                    MemoryEntry.status != "candidate",
                    MemoryEntry.kind != "discussed",
                    MemoryEntry.source_message_id.is_(None),
                    MemoryEntry.source_message_id.not_in(
                        select(Message.id).where(
                            Message.user_id == uid,
                            Message.conversation_id == conversation.id,
                        )
                    ),
                ),
                or_(*clauses),
            )
            .order_by(
                (MemoryEntry.kind == "correction").desc(),
                MemoryEntry.pinned.desc(),
                (MemoryEntry.status == "active").desc(),
                MemoryEntry.updated_at.desc(),
            )
            .limit(200)
        )
    )
    evidence = await evidence_map(db, uid, entries)
    selected = []
    allowed_papers = set(await db.scalars(papers_stmt))
    for item in entries:
        if evidence_status(item, evidence) not in {"current", "unlinked"}:
            continue
        # Source provenance must also stay within the current paper/collection scope.
        if item.citation_ids and any(
            UUID(evidence[v]["paper_id"]) not in allowed_papers for v in item.citation_ids
        ):
            continue
        selected.append(
            {
                "id": str(item.id),
                "level": item.level,
                "kind": item.kind,
                "content": item.content,
                "status": item.status,
                "epistemic": item.epistemic,
                "pinned": item.pinned,
                "citation_ids": item.citation_ids,
                "evidence": [evidence[v] for v in item.citation_ids],
                "reading_anchor": item.reading_anchor,
                "updated_at": item.updated_at.isoformat(),
            }
        )
    if pid:
        project = await owned_project(db, uid, pid)
        selected.append(
            {
                "id": f"project:{pid}",
                "level": "project",
                "kind": "goal",
                "content": f"研究项目：{project.name}。{project.description}",
                "status": "active",
                "epistemic": "user_statement",
                "pinned": False,
                "citation_ids": [],
                "updated_at": project.created_at.isoformat(),
            }
        )
    return select_memories(selected, question, settings.memory_long_term_token_budget)


async def save_candidates(
    db: AsyncSession,
    conversation: Conversation,
    state: ResearchState,
    messages: list[Message],
    *,
    status: str = "candidate",
) -> None:
    from app.services.memory.tools import similarity_score

    by_seq = {m.seq: m for m in messages}
    level = "project" if conversation.project_id else "conversation"
    live_entries = list(
        await db.scalars(
            select(MemoryEntry).where(
                MemoryEntry.user_id == conversation.user_id,
                MemoryEntry.deleted_at.is_(None),
                MemoryEntry.level == level,
                MemoryEntry.project_id == conversation.project_id
                if conversation.project_id
                else MemoryEntry.conversation_id == conversation.id,
            )
        )
    )
    for kind, items in state.model_dump().items():
        for item in items:
            source = by_seq.get(item["source_seq"])
            # Only user-origin statements become long-term candidates automatically.
            if not source or source.role != "user":
                continue
            digest = hashlib.sha256(f"{conversation.id}:{source.id}:{kind}".encode()).hexdigest()
            existing = await db.scalar(
                select(MemoryEntry.id).where(
                    MemoryEntry.user_id == conversation.user_id,
                    MemoryEntry.fingerprint == digest,
                )
            )
            if existing:
                continue  # Includes tombstones: deleted suggestions never resurrect.
            duplicate = next(
                (
                    e
                    for e in live_entries
                    if e.kind == kind
                    and e.status != "superseded"
                    and similarity_score(e.content, item["content"]) >= 0.85
                ),
                None,
            )
            if duplicate:
                if duplicate.status == "candidate" and status == "active":
                    duplicate.status = "active"
                continue
            entry = MemoryEntry(
                user_id=conversation.user_id,
                level=level,
                project_id=conversation.project_id,
                conversation_id=None if conversation.project_id else conversation.id,
                kind=kind,
                content=item["content"],
                status=status,
                epistemic="user_statement",
                source_message_id=source.id,
                citation_ids=[],
                fingerprint=digest,
            )
            db.add(entry)
            live_entries.append(entry)


async def record_project_question(
    db: AsyncSession, conversation: Conversation, message: Message
) -> None:
    """Keep short sessions resumable without adding a model call to every answer."""
    if not conversation.project_id:
        return
    fingerprint = hashlib.sha256(f"{conversation.id}:{message.id}:discussed".encode()).hexdigest()
    db.add(
        MemoryEntry(
            user_id=conversation.user_id,
            level="project",
            project_id=conversation.project_id,
            kind="discussed",
            content="此前用户提出的问题：" + message.content,
            status="candidate",
            epistemic="user_statement",
            source_message_id=message.id,
            fingerprint=fingerprint,
        )
    )
