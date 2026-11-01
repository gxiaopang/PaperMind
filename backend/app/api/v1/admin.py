import json
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import func, select, update

from app.api.deps import DB, CurrentIdentity
from app.db.models import (
    ChatRun,
    MailOutbox,
    Paper,
    PaperVersion,
    Session,
    TaskOutbox,
    UsageRecord,
    User,
    UserQuota,
    utcnow,
)
from app.schemas.accounts import AdminQuotaUpdate, AdminUserUpdate
from app.services.ingestion.quota import today, totals

router = APIRouter(tags=["运维"])


def ensure_admin(identity: CurrentIdentity) -> None:
    if identity.user.role != "admin":
        raise HTTPException(404, "资源不存在")


@router.get("/admin/metrics")
async def metrics(identity: CurrentIdentity) -> Response:
    ensure_admin(identity)
    return Response(generate_latest(), headers={"Content-Type": CONTENT_TYPE_LATEST})


def _load_report(filename: str) -> dict | None:
    candidates = [
        Path("/app/reports") / filename,
        Path(__file__).resolve().parents[4] / "reports" / filename,
    ]
    for path in candidates:
        if path.is_file():
            try:
                return json.loads(path.read_text())
            except Exception:
                return None
    return None


@router.get("/admin/overview")
async def overview(request: Request, identity: CurrentIdentity, db: DB) -> dict:
    ensure_admin(identity)
    settings = request.app.state.settings
    user_total, user_active, user_disabled = (
        await db.execute(
            select(
                func.count(User.id),
                func.count(User.id).filter(User.status == "active"),
                func.count(User.id).filter(User.status == "disabled"),
            )
        )
    ).one()
    paper_count, storage_bytes = (
        await db.execute(
            select(func.count(Paper.id), func.coalesce(func.sum(Paper.size_bytes), 0)).where(
                Paper.deleted_at.is_(None)
            )
        )
    ).one()
    pages_used, pages_reserved = await totals(db, "parse_pages", None)
    tokens_used, tokens_reserved = await totals(db, "model_tokens", None)
    tokens_estimated = bool(
        await db.scalar(
            select(UsageRecord.id)
            .where(
                UsageRecord.metric == "model_tokens",
                UsageRecord.created_at >= today(),
                UsageRecord.estimated.is_(True),
                UsageRecord.amount > 0,
            )
            .limit(1)
        )
    )
    outbox_rows = (
        await db.execute(
            select(TaskOutbox.kind, func.count(TaskOutbox.id))
            .where(TaskOutbox.completed_at.is_(None))
            .group_by(TaskOutbox.kind)
        )
    ).all()
    outbox_counts = {row[0]: int(row[1]) for row in outbox_rows}
    mail_pending = int(
        await db.scalar(
            select(func.count())
            .select_from(MailOutbox)
            .where(MailOutbox.sent_at.is_(None), MailOutbox.expires_at > utcnow())
        )
        or 0
    )
    running_ingest = int(
        await db.scalar(
            select(func.count(PaperVersion.id)).where(
                PaperVersion.status.in_(
                    ["pending", "uploading", "parsing", "chunking", "embedding"]
                )
            )
        )
        or 0
    )
    running_chat = int(
        await db.scalar(
            select(func.count(ChatRun.id)).where(ChatRun.status.in_(["pending", "running"]))
        )
        or 0
    )
    redis_queues: dict[str, int] = {}
    for q in ("papermind:mail", "papermind:ingest", "papermind:chat"):
        try:
            redis_queues[q] = int(await request.app.state.redis.zcard(q) or 0)
        except Exception:
            redis_queues[q] = 0

    return {
        "system": {
            "app_env": settings.app_env,
            "tracing_enabled": settings.tracing_enabled,
            "llm_model": settings.llm_model,
            "llm_fast_model": settings.llm_fast_model,
            "embedding_model": settings.embedding_model,
            "rerank_model": settings.rerank_model,
            "provider_concurrency": settings.provider_concurrency,
        },
        "users": {
            "total": int(user_total or 0),
            "active": int(user_active or 0),
            "disabled": int(user_disabled or 0),
        },
        "global_usage": {
            "papers": int(paper_count or 0),
            "storage_bytes": int(storage_bytes or 0),
            "parse_pages": pages_used,
            "parse_pages_reserved": pages_reserved,
            "daily_parse_pages_limit": settings.global_daily_parse_pages,
            "model_tokens": tokens_used,
            "model_tokens_reserved": tokens_reserved,
            "model_tokens_estimated": tokens_estimated,
            "daily_model_tokens_limit": settings.global_daily_model_tokens,
            "quota_timezone": "UTC",
        },
        "queues": {
            "outbox": {
                "ingest": outbox_counts.get("ingest", 0),
                "cleanup": outbox_counts.get("cleanup", 0),
                "chat": outbox_counts.get("chat", 0),
                "summarize": outbox_counts.get("summarize", 0),
                "mail": mail_pending,
            },
            "active_tasks": {
                "ingest": running_ingest,
                "chat": running_chat,
            },
            "redis_queued": redis_queues,
        },
        "evaluations": {
            "retrieval": _load_report("m3-retrieval-v7-final.json"),
            "chat": _load_report("pre-m6-v7-live-summary.json"),
        },
    }


async def _user_admin_row(db: DB, settings: object, user: User) -> dict:
    from app.core.config import Settings

    assert isinstance(settings, Settings)
    quota = await db.get(UserQuota, user.id)
    paper_count, storage_bytes = (
        await db.execute(
            select(func.count(Paper.id), func.coalesce(func.sum(Paper.size_bytes), 0)).where(
                Paper.user_id == user.id, Paper.deleted_at.is_(None)
            )
        )
    ).one()
    pages_used, pages_reserved = await totals(db, "parse_pages", user.id)
    tokens_used, tokens_reserved = await totals(db, "model_tokens", user.id)
    return {
        "id": str(user.id),
        "email": user.email,
        "role": user.role,
        "status": user.status,
        "email_verified_at": user.email_verified_at,
        "created_at": user.created_at,
        "usage": {
            "papers": int(paper_count or 0),
            "storage_bytes": int(storage_bytes or 0),
            "parse_pages": pages_used,
            "parse_pages_reserved": pages_reserved,
            "model_tokens": tokens_used,
            "model_tokens_reserved": tokens_reserved,
        },
        "quota_overrides": {
            "max_papers": quota.max_papers if quota else None,
            "storage_mb": quota.storage_mb if quota else None,
            "daily_parse_pages": quota.daily_parse_pages if quota else None,
            "daily_model_tokens": quota.daily_model_tokens if quota else None,
            "upload_concurrency": quota.upload_concurrency if quota else None,
        },
        "effective_limits": {
            "max_papers": (
                quota.max_papers if quota and quota.max_papers else settings.user_max_papers
            ),
            "storage_mb": (
                quota.storage_mb if quota and quota.storage_mb else settings.user_max_storage_mb
            ),
            "daily_parse_pages": (
                quota.daily_parse_pages
                if quota and quota.daily_parse_pages
                else settings.user_daily_parse_pages
            ),
            "daily_model_tokens": (
                quota.daily_model_tokens
                if quota and quota.daily_model_tokens
                else settings.user_daily_model_tokens
            ),
            "upload_concurrency": (
                quota.upload_concurrency
                if quota and quota.upload_concurrency
                else settings.user_upload_concurrency
            ),
        },
    }


@router.get("/admin/users")
async def list_users(request: Request, identity: CurrentIdentity, db: DB) -> list[dict]:
    ensure_admin(identity)
    users = list(await db.scalars(select(User).order_by(User.created_at.desc()).limit(200)))
    return [await _user_admin_row(db, request.app.state.settings, u) for u in users]


@router.patch("/admin/users/{target_id}")
async def update_user_status(
    target_id: UUID,
    body: AdminUserUpdate,
    request: Request,
    identity: CurrentIdentity,
    db: DB,
) -> dict:
    ensure_admin(identity)
    user = await db.scalar(select(User).where(User.id == target_id).with_for_update())
    if user is None:
        raise HTTPException(404, "用户不存在")
    if user.id == identity.user.id and body.status == "disabled":
        raise HTTPException(400, "不能停用当前登录的管理员账户")
    user.status = body.status
    if body.status == "disabled":
        await db.execute(
            update(Session)
            .where(Session.user_id == user.id, Session.revoked_at.is_(None))
            .values(revoked_at=utcnow())
        )
    await db.commit()
    return await _user_admin_row(db, request.app.state.settings, user)


@router.put("/admin/users/{target_id}/quota")
async def update_user_quota(
    target_id: UUID,
    body: AdminQuotaUpdate,
    request: Request,
    identity: CurrentIdentity,
    db: DB,
) -> dict:
    ensure_admin(identity)
    user = await db.scalar(select(User).where(User.id == target_id).with_for_update())
    if user is None:
        raise HTTPException(404, "用户不存在")
    quota = await db.get(UserQuota, user.id)
    values = body.model_dump()
    if all(v is None for v in values.values()):
        if quota is not None:
            await db.delete(quota)
    else:
        if quota is None:
            quota = UserQuota(user_id=user.id)
            db.add(quota)
        for key, value in values.items():
            setattr(quota, key, value)
    await db.commit()
    return await _user_admin_row(db, request.app.state.settings, user)
