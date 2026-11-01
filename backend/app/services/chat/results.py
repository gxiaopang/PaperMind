from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ChatRun, Message, UsageRecord
from app.services.chat.events import citations_for


async def usage_for(db: AsyncSession, user_id: UUID, rid: UUID) -> dict:
    records = list(
        await db.scalars(
            select(UsageRecord).where(UsageRecord.user_id == user_id, UsageRecord.run_id == rid)
        )
    )
    return {
        "tokens": sum(r.amount for r in records),
        "estimated": any(r.estimated for r in records),
        "nodes": [
            {
                "node": r.node,
                "model": r.model,
                "tokens": r.amount,
                "estimated": r.estimated,
                "outcome": r.outcome,
            }
            for r in records
        ],
    }


async def result_for(db: AsyncSession, run: ChatRun) -> dict:
    message = await db.scalar(
        select(Message).where(
            Message.user_id == run.user_id, Message.run_id == run.id, Message.role == "assistant"
        )
    )
    return {
        "id": str(run.id),
        "conversation_id": str(run.conversation_id),
        "status": run.status,
        "result_status": run.result_status,
        "error": run.error,
        "mode": run.mode,
        "last_event_id": run.event_seq,
        "retrieval_count": run.retrieval_count,
        "generation_count": run.generation_count,
        "message": {
            "id": str(message.id),
            "content": message.content,
            "verify_result": message.verify_result,
            "citations": await citations_for(db, run.user_id, message.id),
        }
        if message
        else None,
        "usage": await usage_for(db, run.user_id, run.id),
    }
