"""Idempotent navigation-only upgrade using stored parsing results; no model I/O."""

import asyncio
from typing import Any

from sqlalchemy import select

from app.core.config import Settings
from app.db.models import Paper, PaperVersion, ParseJob, Section, User
from app.services.chunking.paper_map import bind_section_ids, build_paper_map, section_std
from app.services.parsing.mineru_result import ParsedDocument
from app.services.storage.private import resolve_private


async def backfill_paper_maps(settings: Settings, sessions: Any) -> dict[str, int]:
    counts = {"updated": 0, "skipped": 0, "unavailable": 0}
    async with sessions() as db:
        rows = (
            await db.execute(
                select(Paper.id, Paper.user_id).where(
                    Paper.deleted_at.is_(None), Paper.active_version_id.is_not(None)
                )
            )
        ).all()
    for paper_id, user_id in rows:
        async with sessions() as db:
            # Same owner lock as ingestion/deletion; never republish deleted data.
            user = await db.scalar(select(User).where(User.id == user_id).with_for_update())
            if user is None or user.status != "active" or user.email_verified_at is None:
                counts["skipped"] += 1
                continue
            row = (
                await db.execute(
                    select(Paper, PaperVersion, ParseJob)
                    .join(PaperVersion, PaperVersion.id == Paper.active_version_id)
                    .join(ParseJob, ParseJob.id == PaperVersion.parse_job_id)
                    .where(
                        Paper.id == paper_id,
                        Paper.user_id == user_id,
                        Paper.deleted_at.is_(None),
                        PaperVersion.status == "ready",
                    )
                )
            ).first()
            if not row:
                counts["skipped"] += 1
                continue
            paper, version, job = row
            if version.paper_map:
                counts["skipped"] += 1
                continue
            path = resolve_private(settings, job.result_path) / "document.json"
            if not path.is_file():
                counts["unavailable"] += 1
                continue
            document = ParsedDocument.model_validate_json(await asyncio.to_thread(path.read_text))
            stored = list(
                await db.scalars(
                    select(Section).where(
                        Section.user_id == user_id, Section.version_id == version.id
                    )
                )
            )
            by_path = {tuple(section.path): section.id for section in stored}
            mapping = {s.id: by_path.get(tuple(s.path)) for s in document.sections}
            if any(value is None for value in mapping.values()):
                counts["unavailable"] += 1
                continue
            value = build_paper_map(document, paper.title, paper.abstract)

            bind_section_ids(value, mapping)
            for section in stored:
                section.section_std = section_std(section.path)
            version.paper_map = value
            await db.commit()
            counts["updated"] += 1
    return counts
