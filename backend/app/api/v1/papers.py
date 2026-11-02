import asyncio
import mimetypes
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select

from app.api.deps import DB, CurrentIdentity, VerifiedUser
from app.api.v1.collections import owned_collection
from app.db.models import (
    Chunk,
    Citation,
    Paper,
    PaperVersion,
    ParseJob,
    UsageRecord,
    User,
    UserSetting,
    utcnow,
)
from app.schemas.accounts import UserSettingsInput
from app.services.ingestion.lifecycle import accept_upload, new_version, owned_paper, tombstone
from app.services.ingestion.quota import lock_user, today, totals, user_settings
from app.services.parsing.errors import ParsingError
from app.services.parsing.metadata import PaperMetadata
from app.services.parsing.mineru_result import ParsedDocument
from app.services.parsing.pdf import inspect_pdf
from app.services.storage.private import receive_pdf, resolve_private

router = APIRouter(tags=["论文入库"])


class PaperView(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    collection_id: UUID | None
    title: str
    authors: list[str]
    year: int | None
    venue: str | None
    doi: str | None
    arxiv_id: str | None
    abstract: str | None
    page_count: int
    size_bytes: int
    status: str
    active_version_id: UUID | None
    pending_version_id: UUID | None
    error: str | None


class MetadataEdit(PaperMetadata):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1, max_length=2000)


@router.post("/collections/{collection_id}/papers", response_model=PaperView, status_code=202)
async def upload(
    collection_id: UUID, request: Request, user: VerifiedUser, db: DB, cloud_consent: bool = False
) -> Paper:
    await owned_collection(db, user.id, collection_id)
    if not cloud_consent:
        raise HTTPException(400, "请确认 PDF 将发送至 MinerU 云服务解析")
    settings = request.app.state.settings
    temporary = await receive_pdf(request, settings)
    try:
        pdf = await asyncio.to_thread(inspect_pdf, temporary, settings)
        await lock_user(db, user.id)
        await owned_collection(db, user.id, collection_id)
        paper = await accept_upload(db, settings, user.id, collection_id, temporary, pdf)
        await db.commit()
        return paper
    except ParsingError as error:
        raise HTTPException(400, str(error)) from None
    finally:
        temporary.unlink(missing_ok=True)


@router.get("/collections/{collection_id}/papers", response_model=list[PaperView])
async def list_papers(collection_id: UUID, identity: CurrentIdentity, db: DB) -> list[Paper]:
    await owned_collection(db, identity.user.id, collection_id)
    return list(
        await db.scalars(
            select(Paper)
            .where(
                Paper.user_id == identity.user.id,
                Paper.collection_id == collection_id,
                Paper.deleted_at.is_(None),
            )
            .order_by(Paper.created_at.desc())
        )
    )


@router.get("/papers", response_model=list[PaperView])
async def list_all_papers(identity: CurrentIdentity, db: DB) -> list[Paper]:
    return list(
        await db.scalars(
            select(Paper)
            .where(
                Paper.user_id == identity.user.id,
                Paper.deleted_at.is_(None),
            )
            .order_by(Paper.created_at.desc())
            .limit(500)
        )
    )


@router.get("/papers/{paper_id}", response_model=PaperView)
async def get_paper(paper_id: UUID, identity: CurrentIdentity, db: DB) -> Paper:
    return await owned_paper(db, identity.user.id, paper_id)


@router.patch("/papers/{paper_id}", response_model=PaperView)
async def edit_paper(
    paper_id: UUID, body: MetadataEdit, identity: CurrentIdentity, db: DB
) -> Paper:
    await db.scalar(select(User).where(User.id == identity.user.id).with_for_update())
    paper = await owned_paper(db, identity.user.id, paper_id)
    for key in ("title", "authors", "year", "venue", "doi", "arxiv_id", "abstract"):
        setattr(paper, key, getattr(body, key))
    paper.metadata_edited = True
    await db.commit()
    return paper


@router.post("/papers/{paper_id}/{operation}", response_model=PaperView, status_code=202)
async def rebuild(
    paper_id: UUID, operation: str, request: Request, user: VerifiedUser, db: DB
) -> Paper:
    if operation not in {"reparse", "rechunk", "retry"}:
        raise HTTPException(404, "操作不存在")
    await lock_user(db, user.id)
    paper = await owned_paper(db, user.id, paper_id)
    if operation == "retry":
        failed = await db.scalar(
            select(PaperVersion)
            .where(
                PaperVersion.user_id == user.id,
                PaperVersion.paper_id == paper_id,
                PaperVersion.status == "failed",
            )
            .order_by(PaperVersion.created_at.desc())
        )
        if not failed or paper.pending_version_id:
            raise HTTPException(409, "没有可重试的失败任务")
        job = await db.get(ParseJob, failed.parse_job_id)
        assert job is not None
        if job.error == "submission_uncertain":
            raise HTTPException(409, "提交结果未知，请运维核对远端批次后恢复")
        settings = await user_settings(db, request.app.state.settings, user.id)
        running = await db.scalar(
            select(func.count())
            .select_from(Paper)
            .where(
                Paper.user_id == user.id,
                Paper.deleted_at.is_(None),
                Paper.pending_version_id.is_not(None),
            )
        )
        if (running or 0) >= settings.user_upload_concurrency:
            raise HTTPException(429, "同时入库的论文过多，请稍后重试")
        # Retry resumes the SAME parse and version; reparse always creates a new job.
        from app.db.models import TaskOutbox
        from app.services.ingestion.quota import reserve

        if not job.charged and failed.operation != "rechunk":
            await reserve(
                db, request.app.state.settings, user.id, failed.id, "parse_pages", paper.page_count
            )
        failed.status, failed.error, failed.attempts = "pending", None, 0
        paper.pending_version_id, paper.status, paper.error = failed.id, "pending", None
        outbox = await db.scalar(
            select(TaskOutbox).where(
                TaskOutbox.user_id == user.id,
                TaskOutbox.resource_id == failed.id,
                TaskOutbox.kind == "ingest",
            )
        )
        assert outbox is not None
        from app.db.models import utcnow

        outbox.completed_at, outbox.delivered_at, outbox.available_at = None, None, utcnow()
    else:
        await new_version(db, request.app.state.settings, paper, operation)
    await db.commit()
    return paper


@router.delete("/papers/{paper_id}", status_code=204)
async def delete_paper(paper_id: UUID, identity: CurrentIdentity, db: DB) -> None:
    await db.scalar(select(User).where(User.id == identity.user.id).with_for_update())
    await tombstone(db, await owned_paper(db, identity.user.id, paper_id))
    await db.commit()


@router.get("/papers/{paper_id}/versions")
async def versions(paper_id: UUID, identity: CurrentIdentity, db: DB) -> list[dict]:
    await owned_paper(db, identity.user.id, paper_id)
    rows = await db.scalars(
        select(PaperVersion)
        .where(PaperVersion.user_id == identity.user.id, PaperVersion.paper_id == paper_id)
        .order_by(PaperVersion.created_at.desc())
    )
    return [
        {
            "id": row.id,
            "status": row.status,
            "operation": row.operation,
            "error": row.error,
            "chunker_version": row.chunker_version,
            "embedding_model": row.embedding_model,
            "attempts": row.attempts,
            "created_at": row.created_at,
        }
        for row in rows
    ]


@router.get("/papers/{paper_id}/chunks")
async def chunks(paper_id: UUID, identity: CurrentIdentity, db: DB) -> list[dict]:
    paper = await owned_paper(db, identity.user.id, paper_id)
    rows = await db.scalars(
        select(Chunk)
        .where(Chunk.user_id == identity.user.id, Chunk.version_id == paper.active_version_id)
        .order_by(Chunk.ordinal)
    )
    return [
        {
            "id": row.id,
            "kind": row.kind,
            "parent_id": row.parent_id,
            "source_spans": row.source_spans,
            "chunk_metadata": row.chunk_metadata,
            "text": row.text,
            "locations": row.locations,
            "section_path": row.section_path,
            "indexable": row.indexable,
            "unindexed_reason": row.unindexed_reason,
            "token_count": row.token_count,
        }
        for row in rows
    ]


@router.get("/papers/{paper_id}/pdf")
async def pdf(paper_id: UUID, request: Request, identity: CurrentIdentity, db: DB) -> FileResponse:
    paper = await owned_paper(db, identity.user.id, paper_id)
    path = resolve_private(request.app.state.settings, paper.file_path)
    if not path.is_file():
        raise HTTPException(404, "文件不存在")
    return FileResponse(
        path, media_type="application/pdf", filename="paper.pdf", content_disposition_type="inline"
    )


async def parsed(
    paper_id: UUID, request: Request, identity: CurrentIdentity, db: DB
) -> tuple[ParsedDocument, ParseJob]:
    paper = await owned_paper(db, identity.user.id, paper_id)
    version = (
        await db.get(PaperVersion, paper.active_version_id) if paper.active_version_id else None
    )
    if not version:
        raise HTTPException(409, "论文尚未完成入库")
    job = await db.get(ParseJob, version.parse_job_id)
    assert job is not None
    root = resolve_private(request.app.state.settings, job.result_path)
    return ParsedDocument.model_validate_json((root / "document.json").read_text()), job


@router.get("/papers/{paper_id}/document")
async def document_info(
    paper_id: UUID, request: Request, identity: CurrentIdentity, db: DB
) -> dict:
    document, _ = await parsed(paper_id, request, identity, db)
    return {
        "page_count": document.page_count,
        "sections": [s.model_dump(mode="json") for s in document.sections],
        "blocks": [b.model_dump(mode="json") for b in document.blocks],
        "assets": document.assets,
        "warnings": document.warnings,
    }


@router.get("/papers/{paper_id}/markdown")
async def markdown(
    paper_id: UUID, request: Request, identity: CurrentIdentity, db: DB
) -> FileResponse:
    document, job = await parsed(paper_id, request, identity, db)
    if not document.markdown_path:
        raise HTTPException(404, "Markdown 不存在")
    path = resolve_private(
        request.app.state.settings, job.result_path + "/raw/" + document.markdown_path
    )
    return FileResponse(path, media_type="text/plain; charset=utf-8")


@router.get("/papers/{paper_id}/assets/{asset_path:path}")
async def asset(
    paper_id: UUID, asset_path: str, request: Request, identity: CurrentIdentity, db: DB
) -> FileResponse:
    document, job = await parsed(paper_id, request, identity, db)
    if asset_path not in document.assets or not asset_path.lower().endswith(
        (".png", ".jpg", ".jpeg", ".webp")
    ):
        raise HTTPException(404, "图片不存在")
    path = resolve_private(request.app.state.settings, job.result_path + "/raw/" + asset_path)
    return FileResponse(path, media_type=mimetypes.guess_type(path.name)[0])


@router.get("/citations/{citation_id}")
async def citation(citation_id: UUID, identity: CurrentIdentity, db: DB) -> dict:
    item = await db.scalar(
        select(Citation).where(Citation.id == citation_id, Citation.user_id == identity.user.id)
    )
    if not item:
        raise HTTPException(404, "引用不存在")
    return {
        "id": item.id,
        "paper_id": item.paper_id,
        "version_id": item.version_id,
        "source_text": item.source_text,
        "locations": item.locations,
        "status": "来源已删除" if item.deleted_at else "available",
    }


@router.get("/usage")
async def usage(request: Request, identity: CurrentIdentity, db: DB) -> dict:
    settings = await user_settings(db, request.app.state.settings, identity.user.id)
    count, size = (
        await db.execute(
            select(func.count(), func.coalesce(func.sum(Paper.size_bytes), 0)).where(
                Paper.user_id == identity.user.id, Paper.deleted_at.is_(None)
            )
        )
    ).one()
    pages, pages_held = await totals(db, "parse_pages", identity.user.id)
    tokens, tokens_held = await totals(db, "model_tokens", identity.user.id)
    estimated = bool(
        await db.scalar(
            select(UsageRecord.id)
            .where(
                UsageRecord.user_id == identity.user.id,
                UsageRecord.metric == "model_tokens",
                UsageRecord.created_at >= today(),
                UsageRecord.estimated.is_(True),
                UsageRecord.amount > 0,
            )
            .limit(1)
        )
    )
    return {
        "papers": count,
        "storage_bytes": int(size),
        "parse_pages": pages,
        "parse_pages_reserved": pages_held,
        "model_tokens": tokens,
        "model_tokens_estimated": estimated,
        "quota_timezone": "UTC",
        "model_tokens_reserved": tokens_held,
        "limits": {
            "papers": settings.user_max_papers,
            "storage_bytes": settings.user_max_storage_mb * 1024**2,
            "daily_parse_pages": settings.user_daily_parse_pages,
            "daily_model_tokens": settings.user_daily_model_tokens,
        },
        "by_node": [
            {
                "node": row[0],
                "model": row[1],
                "tokens": int(row[2] or 0),
                "calls": int(row[3] or 0),
                "estimated": bool(row[4]),
            }
            for row in (
                await db.execute(
                    select(
                        UsageRecord.node,
                        UsageRecord.model,
                        func.coalesce(func.sum(UsageRecord.amount), 0),
                        func.count(UsageRecord.id),
                        func.bool_or(UsageRecord.estimated),
                    )
                    .where(
                        UsageRecord.user_id == identity.user.id,
                        UsageRecord.metric == "model_tokens",
                        UsageRecord.created_at >= today(),
                    )
                    .group_by(UsageRecord.node, UsageRecord.model)
                    .order_by(UsageRecord.node)
                )
            ).all()
        ],
    }


@router.get("/settings")
async def get_user_settings(identity: CurrentIdentity, db: DB) -> dict:
    pref = await db.get(UserSetting, identity.user.id)
    return {
        "answer_language": pref.answer_language if pref else "zh",
        "detail_level": pref.detail_level if pref else "standard",
    }


@router.put("/settings")
async def update_user_settings(body: UserSettingsInput, identity: CurrentIdentity, db: DB) -> dict:
    await db.scalar(select(User).where(User.id == identity.user.id).with_for_update())
    pref = await db.get(UserSetting, identity.user.id)
    if pref is None:
        pref = UserSetting(
            user_id=identity.user.id,
            answer_language=body.answer_language,
            detail_level=body.detail_level,
            memory_write_mode=body.memory_write_mode,
        )
        db.add(pref)
    else:
        pref.answer_language = body.answer_language
        pref.detail_level = body.detail_level
        if "memory_write_mode" in body.model_fields_set:
            pref.memory_write_mode = body.memory_write_mode
        pref.updated_at = utcnow()
    await db.commit()
    return {
        "answer_language": pref.answer_language,
        "detail_level": pref.detail_level,
    }


@router.get("/papers/{paper_id}/references")
async def references(paper_id: UUID, identity: CurrentIdentity, db: DB) -> list[dict]:
    paper = await owned_paper(db, identity.user.id, paper_id)
    version = await db.scalar(
        select(PaperVersion).where(
            PaperVersion.user_id == identity.user.id,
            PaperVersion.paper_id == paper.id,
            PaperVersion.id == paper.active_version_id,
        )
    )
    return version.reference_entries if version else []
