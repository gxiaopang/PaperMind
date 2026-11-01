from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import TypeVar
from uuid import UUID, uuid4

import httpx
from fastapi import HTTPException
from openai import APIStatusError
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.db.models import QuotaReservation, UsageRecord
from app.services.ingestion.quota import lock_user, totals, user_settings

T = TypeVar("T")


class RetrievalAccounting:
    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        engine: AsyncEngine,
        settings: Settings,
        *,
        run_id: UUID | None = None,
    ):
        self.sessions, self.engine, self.settings = sessions, engine, settings
        self.run_id = run_id

    @asynccontextmanager
    async def slot(self, user_id: UUID) -> AsyncIterator[None]:
        # Dedicated transaction, released on cancellation/connection loss; shares
        # the same provider namespace as ingestion. Per-user retrieval slots are finite.
        async with self.engine.connect() as connection:
            for namespace, slots in (
                (620022, self.settings.provider_concurrency),
                (
                    int.from_bytes(user_id.bytes[:4], "big", signed=True),
                    self.settings.user_chat_concurrency,
                ),
            ):
                acquired = False
                for slot in range(slots):
                    key = slot if namespace == 620022 else 10000 + slot
                    if await connection.scalar(
                        text("SELECT pg_try_advisory_xact_lock(:ns, :slot)"),
                        {"ns": namespace, "slot": key},
                    ):
                        acquired = True
                        break
                if not acquired:
                    raise HTTPException(429, "检索服务繁忙，请稍后重试")
            yield

    async def call(
        self,
        user_id: UUID,
        operation_id: UUID,
        node: str,
        amount: int,
        callback: Callable[[], Awaitable[T]],
        usage: Callable[[], int | None],
        *,
        provider: str | None = None,
        model: str | None = None,
        version_id: UUID | None = None,
        record_id: UUID | None = None,
        validate: Callable[[AsyncSession], Awaitable[None]] | None = None,
    ) -> T:
        provider = provider or (
            self.settings.embedding_base_url
            if node == "query_embedding"
            else self.settings.rerank_base_url
        )
        model = model or (
            self.settings.embedding_model
            if node == "query_embedding"
            else self.settings.rerank_model
        )
        record_id = record_id or uuid4()
        async with self.sessions() as db:
            await lock_user(db, user_id)
            if validate is not None:
                await validate(db)
            limits = await user_settings(db, self.settings, user_id)
            await db.execute(text("SELECT pg_advisory_xact_lock(620021)"))
            reservation = None
            credit = 0
            if self.run_id is not None:
                reservation = await db.scalar(
                    select(QuotaReservation).where(
                        QuotaReservation.user_id == user_id,
                        QuotaReservation.run_id == self.run_id,
                        QuotaReservation.metric == "model_tokens",
                        QuotaReservation.status == "reserved",
                    )
                )
                credit = min(reservation.amount, amount) if reservation else 0
            for owner, limit in (
                (user_id, limits.user_daily_model_tokens),
                (None, limits.global_daily_model_tokens),
            ):
                used, held = await totals(db, "model_tokens", owner)
                if used + held + amount - credit > limit:
                    raise HTTPException(429, "今日模型 token 额度不足")
            if reservation:
                reservation.amount -= credit
            db.add(
                UsageRecord(
                    id=record_id,
                    user_id=user_id,
                    version_id=version_id,
                    operation_id=operation_id,
                    run_id=self.run_id,
                    node=node,
                    provider=provider.rstrip("/"),
                    model=model,
                    metric="model_tokens",
                    amount=amount,
                    estimated=True,
                    outcome="uncertain",
                )
            )
            await db.commit()
        # Durable conservative intent accounts even for a killed process or unknown
        # remote result. Only explicit provider rejection releases this call's spend.
        try:
            result = await callback()
        except (APIStatusError, httpx.HTTPStatusError) as error:
            status = (
                error.status_code
                if isinstance(error, APIStatusError)
                else error.response.status_code
            )
            if status < 500:
                async with self.sessions() as db:
                    record = await db.get(UsageRecord, record_id)
                    assert record is not None
                    record.amount, record.outcome = 0, "rejected"
                    await db.commit()
            raise
        await self.settle(user_id, record_id, usage())
        return result

    async def settle(self, user_id: UUID, record_id: UUID, actual: int | None) -> None:
        """Idempotently settle a response, including one recovered from private storage."""
        if not isinstance(actual, int) or isinstance(actual, bool) or actual < 0:
            actual = None
        async with self.sessions() as db:
            record = await db.scalar(
                select(UsageRecord)
                .where(UsageRecord.id == record_id, UsageRecord.user_id == user_id)
                .with_for_update()
            )
            if record is None:
                raise ValueError("Usage record not found for owner")
            if actual is not None:
                record.amount = actual
            record.estimated, record.outcome = actual is None, "confirmed"
            await db.commit()
