"""Shared summary writer; callers hold the conversation advisory lock."""

from typing import Any

from sqlalchemy import select

from app.db.models import ChatRun, Conversation, Message
from app.graph.prompts import Summary
from app.schemas.memory import ResearchState
from app.services.ingestion.quota import lock_user
from app.services.memory.history import compact_text, history_tokens
from app.services.memory.store import save_candidates


async def consolidate(
    services: Any, run: ChatRun, config: dict, *, pressure_only: bool = False
) -> bool:
    changed = False
    # Drain the prefix in bounded requests. A later call can continue a large backlog.
    for _ in range(16):
        async with services.sessions() as db:
            conversation = await db.get(Conversation, run.conversation_id)
            assert conversation and conversation.user_id == run.user_id
            if conversation.deleted_at:
                return changed
            base = conversation.summarized_seq
            rows = list(
                await db.scalars(
                    select(Message)
                    .where(
                        Message.user_id == run.user_id,
                        Message.conversation_id == run.conversation_id,
                        Message.seq > base,
                    )
                    .order_by(Message.seq)
                    .limit(64)
                )
            )
            if (
                pressure_only
                and len(rows) < 64
                and history_tokens([{"content": m.content} for m in rows])
                <= services.settings.memory_recent_token_budget
            ):
                return changed
            # Keep two complete recent turns verbatim in the persistent log.
            cutoff = conversation.next_seq - 1 - 4
            if (
                rows
                and rows[-1].seq == conversation.next_seq - 1
                and history_tokens([{"content": m.content} for m in rows[-4:]])
                > services.settings.memory_recent_token_budget
            ):
                cutoff = conversation.next_seq - 1 - 2
            prefix = [m for m in rows if m.seq <= cutoff]
            if len(prefix) % 2:
                prefix.pop()
            if not prefix:
                return changed
            history: list[dict] = []
            target = base
            for i in range(0, len(prefix), 2):
                pair = [
                    {"seq": m.seq, "role": m.role, "content": compact_text(m.content, 2500)}
                    for m in prefix[i : i + 2]
                ]
                if history and history_tokens(history + pair) > 6000:
                    break
                history.extend(pair)
                target = prefix[i + 1].seq
            previous = conversation.summary
            previous_state = conversation.research_state
            revision = conversation.memory_revision
        output = await services.models.invoke(
            run,
            "summarize",
            {"summary": previous, "research_state": previous_state, "history": history},
            config,
            schema=Summary,
        )
        async with services.sessions() as db:
            await lock_user(db, run.user_id)
            conversation = await db.scalar(
                select(Conversation)
                .where(
                    Conversation.id == run.conversation_id,
                    Conversation.user_id == run.user_id,
                )
                .with_for_update()
            )
            assert conversation
            if (
                conversation.deleted_at
                or conversation.summarized_seq != base
                or conversation.memory_revision != revision
            ):
                return changed
            # Pydantic enforces a character limit; never cut through a sentence by bytes.
            conversation.summary = output.summary
            allowed = {m.seq for m in prefix if m.seq <= target}
            allowed.update(
                item["source_seq"] for items in previous_state.values() for item in items
            )
            user_seqs = set(
                await db.scalars(
                    select(Message.seq).where(
                        Message.user_id == run.user_id,
                        Message.conversation_id == run.conversation_id,
                        Message.seq.in_(allowed),
                        Message.role == "user",
                    )
                )
            )
            safe_state = ResearchState.model_validate(
                {
                    kind: [
                        item
                        for item in items
                        if item["source_seq"] in allowed
                        and (
                            kind not in {"goal", "constraint", "correction", "next_step"}
                            or item["source_seq"] in user_seqs
                        )
                    ]
                    for kind, items in output.research_state.model_dump().items()
                }
            )
            conversation.research_state = safe_state.model_dump()
            conversation.summarized_seq = target
            await save_candidates(
                db, conversation, safe_state, [m for m in prefix if m.seq <= target]
            )
            await db.commit()
            changed = True
    return changed
