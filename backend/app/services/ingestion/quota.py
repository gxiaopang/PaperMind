from datetime import UTC, datetime
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.db.models import QuotaReservation, UsageRecord, User, UserQuota


def today() -> datetime:
    return datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)


async def user_settings(db: AsyncSession, settings: Settings, user_id: UUID) -> Settings:
    quota = await db.get(UserQuota, user_id)
    if quota is None:
        return settings
    fields = (
        "max_papers",
        "storage_mb",
        "daily_parse_pages",
        "daily_model_tokens",
        "upload_concurrency",
    )
    return settings.model_copy(
        update={
            "user_" + key.replace("storage_mb", "max_storage_mb"): getattr(quota, key)
            for key in fields
            if getattr(quota, key) is not None
        }
    )


async def lock_user(db: AsyncSession, user_id: UUID) -> User:
    user = await db.scalar(
        select(User)
        .where(User.id == user_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    if user is None or user.status != "active" or user.email_verified_at is None:
        raise HTTPException(403, "账户不可用或邮箱未验证")
    return user


async def totals(db: AsyncSession, metric: str, user_id: UUID | None) -> tuple[int, int]:
    used = select(func.coalesce(func.sum(UsageRecord.amount), 0)).where(
        UsageRecord.metric == metric, UsageRecord.created_at >= today()
    )
    held = select(func.coalesce(func.sum(QuotaReservation.amount), 0)).where(
        QuotaReservation.metric == metric, QuotaReservation.status == "reserved"
    )
    if user_id is not None:
        used, held = (
            used.where(UsageRecord.user_id == user_id),
            held.where(QuotaReservation.user_id == user_id),
        )
    return int(await db.scalar(used) or 0), int(await db.scalar(held) or 0)


async def reserve(
    db: AsyncSession, settings: Settings, user_id: UUID, version_id: UUID, metric: str, amount: int
) -> QuotaReservation:
    settings = await user_settings(db, settings, user_id)
    # Serialize the global budget in all callers; no network call holds this lock.
    await db.execute(text("SELECT pg_advisory_xact_lock(620021)"))
    existing = await db.scalar(
        select(QuotaReservation).where(
            QuotaReservation.user_id == user_id,
            QuotaReservation.version_id == version_id,
            QuotaReservation.metric == metric,
        )
    )
    if existing and existing.status == "reserved":
        return existing
    user_limit = (
        settings.user_daily_parse_pages
        if metric == "parse_pages"
        else settings.user_daily_model_tokens
    )
    global_limit = (
        settings.global_daily_parse_pages
        if metric == "parse_pages"
        else settings.global_daily_model_tokens
    )
    for owner, limit in ((user_id, user_limit), (None, global_limit)):
        used, held = await totals(db, metric, owner)
        if used + held + amount > limit:
            raise HTTPException(
                429,
                "今日解析页数额度不足" if metric == "parse_pages" else "今日模型 token 额度不足",
            )
    if existing:
        existing.amount, existing.status = amount, "reserved"
        return existing
    item = QuotaReservation(user_id=user_id, version_id=version_id, metric=metric, amount=amount)
    db.add(item)
    await db.flush()
    return item


async def release(db: AsyncSession, user_id: UUID, version_id: UUID) -> None:
    from sqlalchemy import update

    await db.execute(
        update(QuotaReservation)
        .where(
            QuotaReservation.user_id == user_id,
            QuotaReservation.version_id == version_id,
            QuotaReservation.status == "reserved",
        )
        .values(status="released")
    )
