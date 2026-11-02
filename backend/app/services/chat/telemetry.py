"""Build final diagnostics from committed business records, including resumed runs."""

from typing import Any
from uuid import UUID

from sqlalchemy import func, select

from app.core.observability import Observation, publish_result
from app.db.models import ChatRun, Citation, Message
from app.services.chat.results import usage_for


def result_metrics(run: Any, verification: dict, usage: dict, citation_count: int) -> dict:
    citation_validation = "not_run"
    if run.status == "completed":
        citation_validation = "not_applicable"
        if run.result_status == "answered" and isinstance(
            verification.get("citations_valid"), bool
        ):
            citation_validation = "passed" if verification["citations_valid"] else "failed"
    grounding = verification.get("grounding", "not_run")
    if grounding not in {"passed", "failed", "not_run", "not_applicable"}:
        grounding = "not_applicable"
    metrics = {
        "status": run.status,
        "result_status": run.result_status or run.status,
        "citation_validation": citation_validation,
        "grounding": grounding,
        "sources_revoked": bool(verification.get("sources_revoked")),
        "safe_fallback": run.result_status == "safe_fallback",
        "degraded": bool(verification.get("degraded")),
        "citation_count": citation_count,
        "retrieval_count": run.retrieval_count,
        "generation_count": run.generation_count,
        "retrieval_retries": max(0, run.retrieval_count - 1),
        "generation_retries": max(0, run.generation_count - 1),
        "recoveries": max(0, run.recoveries - 1),
        "tokens": usage["tokens"],
        "tokens_estimated": usage["estimated"],
    }
    if run.finished_at and run.created_at:
        metrics["duration_seconds"] = max(0, (run.finished_at - run.created_at).total_seconds())
    return metrics


async def record_result(
    ctx: dict[str, Any], user_id: UUID, run_id: UUID, root: Observation
) -> None:
    async with ctx["session_factory"]() as db:
        run = await db.scalar(
            select(ChatRun).where(ChatRun.user_id == user_id, ChatRun.id == run_id)
        )
        if run is None or run.status not in {"completed", "failed"}:
            return
        verification = await db.scalar(
            select(Message.verify_result).where(
                Message.user_id == user_id, Message.run_id == run_id, Message.role == "assistant"
            )
        )
        citation_count = await db.scalar(
            select(func.count())
            .select_from(Citation)
            .join(Message, Message.id == Citation.message_id)
            .where(Message.user_id == user_id, Message.run_id == run_id)
        )
        usage = await usage_for(db, user_id, run_id)
    publish_result(
        ctx.get("tracing"),
        run,
        result_metrics(run, verification or {}, usage, citation_count or 0),
        root,
    )
