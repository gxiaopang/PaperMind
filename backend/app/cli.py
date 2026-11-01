import argparse
import asyncio
from typing import Any
from uuid import UUID

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.db.models import Paper, PaperVersion, ParseJob, Session, User, UserQuota, utcnow


async def checkpoints() -> None:
    settings = get_settings()
    url = settings.database_url.get_secret_value().replace("postgresql+asyncpg://", "postgresql://")
    async with AsyncConnectionPool[AsyncConnection[dict[str, Any]]](
        url,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
            "options": "-c search_path=checkpoints,public",
        },
        open=False,
    ) as pool:
        await AsyncPostgresSaver(pool).setup()
    print("Checkpoint schema ready")


async def upgrade_maps() -> None:
    import json

    from app.services.ingestion.paper_maps import backfill_paper_maps

    settings = get_settings()
    engine = create_async_engine(settings.database_url.get_secret_value())
    try:
        print(
            json.dumps(
                await backfill_paper_maps(settings, async_sessionmaker(engine)), ensure_ascii=False
            )
        )
    finally:
        await engine.dispose()


async def set_user(email: str, action: str) -> None:
    from sqlalchemy import update

    engine = create_async_engine(get_settings().database_url.get_secret_value())
    try:
        async with async_sessionmaker(engine, expire_on_commit=False)() as db:
            user = await db.scalar(
                select(User).where(User.email == email.strip().casefold()).with_for_update()
            )
            if user is None:
                raise SystemExit("账户不存在")
            if action == "admin":
                user.role = "admin"
            else:
                user.status = "disabled" if action == "disable" else "active"
                if action == "disable":
                    await db.execute(
                        update(Session)
                        .where(Session.user_id == user.id)
                        .values(revoked_at=utcnow())
                    )
            await db.commit()
            print("账户设置已更新")
    finally:
        await engine.dispose()


async def set_quota(email: str, values: dict[str, int]) -> None:
    engine = create_async_engine(get_settings().database_url.get_secret_value())
    try:
        async with async_sessionmaker(engine, expire_on_commit=False)() as db:
            user = await db.scalar(
                select(User).where(User.email == email.strip().casefold()).with_for_update()
            )
            if not user:
                raise SystemExit("账户不存在")
            quota = await db.get(UserQuota, user.id)
            if quota is None:
                quota = UserQuota(user_id=user.id)
                db.add(quota)
            for key, value in values.items():
                setattr(quota, key, value)
            await db.commit()
            print("用户额度已更新")
    finally:
        await engine.dispose()


async def reconcile_version(version_id: UUID, batch_id: str) -> None:
    import json
    import os

    import httpx

    from app.services.parsing.mineru_client import MinerUClient
    from app.services.parsing.pipeline import private_json
    from app.services.storage.private import resolve_private
    from app.workers.ingestion import task_lock

    settings = get_settings()
    engine = create_async_engine(settings.database_url.get_secret_value())
    descriptor = None
    try:
        async with async_sessionmaker(engine, expire_on_commit=False)() as db:
            version = await db.get(PaperVersion, version_id)
            if not version or version.status != "failed":
                raise SystemExit("仅可核对失败版本")
            paper = await db.get(Paper, version.paper_id)
            job = await db.get(ParseJob, version.parse_job_id)
            if not paper or paper.deleted_at or not job:
                raise SystemExit("论文不可用")
            descriptor = task_lock(settings, paper.user_id, paper.id)
            if descriptor is None:
                raise SystemExit("任务仍在运行，请稍后核对")
            async with httpx.AsyncClient() as http:
                result = await MinerUClient(settings, http).result(batch_id, paper.sha256)
            if result.state in {"waiting-file", "failed"}:
                raise SystemExit("远端任务尚未上传或已失败，不能恢复；请明确重新解析")
            await db.scalar(select(User).where(User.id == paper.user_id).with_for_update())
            await db.refresh(paper)
            if paper.deleted_at:
                raise SystemExit("论文已删除")
            path = resolve_private(settings, job.result_path) / "job.json"
            manifest = json.loads(path.read_text())
            if manifest.get("identity", {}).get("sha256") != paper.sha256:
                raise SystemExit("恢复清单与 PDF 不匹配")
            manifest.update(batch_id=batch_id, stage="polling")
            manifest.pop("upload_url", None)
            private_json(path, manifest)
            job.batch_id, job.error = batch_id, None
            await db.commit()
            print("远端批次已核对，可在论文页面点击重试")
    finally:
        if descriptor is not None:
            os.close(descriptor)
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=[
            "checkpoints",
            "admin",
            "disable",
            "enable",
            "quota",
            "reconcile",
            "backfill-paper-maps",
        ],
    )
    parser.add_argument("email", nargs="?")
    for key in (
        "max-papers",
        "storage-mb",
        "daily-parse-pages",
        "daily-model-tokens",
        "upload-concurrency",
    ):
        parser.add_argument("--" + key, type=int)
    parser.add_argument("--batch-id")
    args = parser.parse_args()
    if args.action == "checkpoints":
        asyncio.run(checkpoints())
    elif args.action == "backfill-paper-maps":
        asyncio.run(upgrade_maps())
    elif args.action == "reconcile" and args.email and args.batch_id:
        asyncio.run(reconcile_version(UUID(args.email), args.batch_id))
    elif args.action == "quota" and args.email:
        fields = (
            "max_papers",
            "storage_mb",
            "daily_parse_pages",
            "daily_model_tokens",
            "upload_concurrency",
        )
        values = {key: getattr(args, key) for key in fields if getattr(args, key) is not None}
        if not values or any(value <= 0 for value in values.values()):
            parser.error("至少提供一个正整数额度")
        asyncio.run(set_quota(args.email, values))
    elif args.action in {"admin", "disable", "enable"} and args.email:
        asyncio.run(set_user(args.email, args.action))
    else:
        parser.error("请提供邮箱")


if __name__ == "__main__":
    main()
