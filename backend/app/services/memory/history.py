"""Bounded conversational context. Original messages remain in PostgreSQL."""

import math
from typing import Any
from uuid import UUID

from sqlalchemy import select

from app.db.models import Conversation, Message, Paper

OMISSION = "\n[…原始消息过长，已压缩；完整内容保留在聊天记录中…]\n"


def estimate_tokens(value: str) -> int:
    """Offline language-aware estimate, not a provider-specific tokenizer.

    Reserve one token per CJK character and one per three ASCII characters;
    other Unicode uses a conservative UTF-8 estimate. Count overhead separately.
    """
    ascii_count = sum(ord(c) < 128 for c in value)
    cjk = sum("\u3400" <= c <= "\u9fff" or "\uf900" <= c <= "\ufaff" for c in value)
    other_bytes = sum(
        len(c.encode())
        for c in value
        if ord(c) >= 128 and not ("\u3400" <= c <= "\u9fff" or "\uf900" <= c <= "\ufaff")
    )
    return math.ceil(ascii_count / 3) + cjk + math.ceil(other_bytes / 2)


def history_tokens(messages: list[dict]) -> int:
    return sum(estimate_tokens(m["content"]) + 8 for m in messages)


def compact_text(value: str, budget: int) -> str:
    if estimate_tokens(value) <= budget:
        return value
    if budget <= estimate_tokens(OMISSION):
        return ""
    low, high = 0, len(value)
    while low < high:
        size = (low + high + 1) // 2
        candidate = (
            value[: (size + 1) // 2] + OMISSION + (value[-(size // 2) :] if size // 2 else "")
        )
        if estimate_tokens(candidate) <= budget:
            low = size
        else:
            high = size - 1
    return value[: (low + 1) // 2] + OMISSION + (value[-(low // 2) :] if low // 2 else "")


def trim_history(messages: list[dict], budget: int) -> list[dict]:
    """Keep complete recent turns; compact an oversized latest turn explicitly."""
    turns: list[list[dict]] = []
    for item in messages:
        if item["role"] == "user":
            turns.append([item])
        elif turns and len(turns[-1]) == 1:
            turns[-1].append(item)
    chosen: list[dict] = []
    for turn in reversed(turns):
        cost = history_tokens(turn)
        if cost <= budget:
            chosen = turn + chosen
            budget -= cost
        elif not chosen and budget >= 96:
            available = budget - 8 * len(turn)
            user_budget = min(estimate_tokens(turn[0]["content"]), available // 2)
            chosen = [{**turn[0], "content": compact_text(turn[0]["content"], user_budget)}]
            if len(turn) == 2:
                chosen.append(
                    {
                        **turn[1],
                        "content": compact_text(
                            turn[1]["content"], available - estimate_tokens(chosen[0]["content"])
                        ),
                    }
                )
            break
        else:
            break
    return chosen


async def load_memory(
    sessions: Any,
    settings: Any,
    user_id: UUID,
    conversation_id: UUID,
    scope: dict | None = None,
    question: str = "",
) -> dict:
    async with sessions() as db:
        conversation = await db.get(Conversation, conversation_id)
        assert conversation and conversation.user_id == user_id and not conversation.deleted_at
        rows = list(
            await db.scalars(
                select(Message)
                .where(
                    Message.user_id == user_id,
                    Message.conversation_id == conversation_id,
                    Message.seq > conversation.summarized_seq,
                )
                .order_by(Message.seq)
            )
        )
        active_stmt = select(Paper.id).where(
            Paper.user_id == user_id,
            Paper.id.in_([UUID(v) for v in conversation.active_paper_ids]),
            Paper.deleted_at.is_(None),
        )
        for key, column in (("paper_ids", Paper.id), ("collection_ids", Paper.collection_id)):
            if scope is not None and scope.get(key) is not None:
                active_stmt = active_stmt.where(column.in_([UUID(v) for v in scope[key]]))
        active = list(await db.scalars(active_stmt))
        history = [{"role": m.role, "content": m.content} for m in rows]
        from app.services.memory.store import recall
        from app.services.memory.tools import load_core_snapshot, session_search

        memories = await recall(db, settings, conversation, scope or {}, question)
        core_memory = await load_core_snapshot(db, user_id, conversation.project_id)
        session_recall = (
            await session_search(
                db,
                user_id,
                question,
                current_conversation_id=conversation_id,
                project_id=conversation.project_id,
                project_only=bool(conversation.project_id),
                limit=4,
            )
            if question.strip()
            else []
        )
        # Budget structured state independently from recent messages and recalled entries.
        state_budget = 1200
        research_state: dict = {}
        for kind in (
            "correction",
            "goal",
            "constraint",
            "focus",
            "open_question",
            "next_step",
            "discussed",
        ):
            for item in conversation.research_state.get(kind, []):
                cost = estimate_tokens(item["content"]) + 20
                if cost <= state_budget:
                    research_state.setdefault(kind, []).append(item)
                    state_budget -= cost
        recent = trim_history(history, settings.memory_recent_token_budget)
        return {
            "core_memory": core_memory,
            "session_recall": session_recall,
            "research_state": research_state,
            "memories": memories,
            "memory_coverage": {
                "summarized_seq": conversation.summarized_seq,
                "recent_compacted": history_tokens(history) > settings.memory_recent_token_budget,
                "history_omitted": len(history) - len(recent),
                "first_recent_seq": rows[-len(recent)].seq if recent else None,
                "recalled_count": len(memories),
                "recalled_token_estimate": estimate_tokens(str(memories)),
                "core_loaded": bool(core_memory),
                "session_recall_count": len(session_recall),
            },
            "history": recent,
            "summary": conversation.summary,
            "active_paper_ids": [str(x) for x in active],
        }
