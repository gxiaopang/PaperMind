from typing import Any
from uuid import UUID

from sqlalchemy import select

from app.db.models import Conversation, Message, Paper


def trim_history(messages: list[dict], budget: int) -> list[dict]:
    # UTF-8 bytes overestimate token use for both selected provider families.
    chosen = []
    for item in reversed(messages):
        cost = len(item["content"].encode()) + 8
        if cost > budget:
            break
        chosen.append(item)
        budget -= cost
    return list(reversed(chosen))


async def load_memory(
    sessions: Any, settings: Any, user_id: UUID, conversation_id: UUID, scope: dict | None = None
) -> dict:
    async with sessions() as db:
        conversation = await db.get(Conversation, conversation_id)
        assert conversation and conversation.user_id == user_id
        rows = list(
            await db.scalars(
                select(Message)
                .where(
                    Message.user_id == user_id,
                    Message.conversation_id == conversation_id,
                    Message.seq > conversation.summarized_seq,
                )
                .order_by(Message.seq.desc())
                .limit(200)
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
        return {
            "history": trim_history(
                [{"role": m.role, "content": m.content} for m in reversed(rows)],
                settings.memory_recent_token_budget,
            ),
            "summary": conversation.summary,
            "active_paper_ids": [str(x) for x in active],
        }
