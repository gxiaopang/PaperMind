import asyncio
import json
from uuid import UUID, uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from pypdf import PdfWriter
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError

from app.db.models import (
    Chunk,
    ChunkParent,
    Citation,
    EmbeddingCache,
    Paper,
    PaperVersion,
    ParseJob,
    QuotaReservation,
    TaskOutbox,
    UsageRecord,
    User,
    utcnow,
)
from app.db.models import (
    Section as StoredSection,
)
from app.services.ingestion.lifecycle import active_chunks, snapshot
from app.services.parsing.errors import ParsingError
from app.services.parsing.metadata import (
    MetadataExtractionSkipped,
    PaperMetadata,
    apply_evidence,
)
from app.services.parsing.mineru_result import Location, ParsedBlock, ParsedDocument, Section
from app.services.storage.private import resolve_private
from app.workers.ingestion import cleanup, dispatch_outbox, ingest
from tests.conftest import login, post, register

pytestmark = pytest.mark.integration


class Embedder:
    def __init__(self):
        self.calls = []
        self.error = None
        self.gate = None
        self.entered = asyncio.Event()

    async def aembed_documents(self, texts):
        self.calls.append(texts)
        self.entered.set()
        if self.gate:
            await self.gate.wait()
        if self.error:
            raise self.error
        return [[0.1] * 1024 for _ in texts]


@pytest.fixture
async def setup_ingestion(app, client, tmp_path):
    app.state.settings.storage_root = tmp_path
    await register(client)
    await login(client)
    user_id = UUID((await client.get("/api/v1/auth/me")).json()["id"])
    collection = (await post(client, "/collections", {"name": "Research"})).json()["id"]
    source = tmp_path / "source.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_blank_page(width=612, height=792)
    writer.write(source)
    embedder = Embedder()
    parser_calls = []

    async def parser(pdf_path, output, settings, **kwargs):
        parser_calls.append(str(output))
        output.mkdir(parents=True, exist_ok=True)
        await kwargs["checkpoint"]({"stage": "polling", "batch_id": "remote-batch"})
        doc = ParsedDocument(
            page_count=2,
            blocks=[
                ParsedBlock(
                    id="b0",
                    kind="text",
                    text="A method across pages.",
                    section_id="s1",
                    section_path=["Methods"],
                    locations=[Location(page_idx=0, bbox=(0.1, 0.1, 0.9, 0.8))],
                ),
                ParsedBlock(
                    id="b1",
                    kind="text",
                    text="Experimental evaluation and evidence.",
                    section_id="s1",
                    section_path=["Methods"],
                    locations=[Location(page_idx=1)],
                ),
            ],
            sections=[Section(id="s1", parent_id=None, title="Methods", level=1, path=["Methods"])],
            markdown_path="full.md",
            content_list_path="content_list.json",
            layout_path=None,
            assets=["images/plot.png"],
            warnings=[],
        )
        (output / "document.json").write_text(doc.model_dump_json())
        (output / "metadata.json").write_text(
            json.dumps(
                {"title": "Test Research", "authors": ["Alice"], "abstract": "A useful abstract."}
            )
        )
        (output / "job.json").write_text(
            json.dumps({"stage": "complete", "batch_id": "remote-batch"})
        )
        (output / "raw/images").mkdir(parents=True, exist_ok=True)
        (output / "raw/full.md").write_text("# Research\nBody")
        (output / "raw/images/plot.png").write_bytes(b"png fixture")
        await kwargs["checkpoint"]({"stage": "complete", "batch_id": "remote-batch"})

    ctx = {
        "settings": app.state.settings,
        "session_factory": app.state.session_factory,
        "engine": app.state.engine,
        "redis": app.state.redis,
        "embedder": embedder,
        "parse_pdf": parser,
    }
    return {
        "ctx": ctx,
        "user": user_id,
        "collection": collection,
        "source": source,
        "embedder": embedder,
        "parser_calls": parser_calls,
    }


async def upload(client, setup, content=None, consent=True):
    csrf = (await client.get("/api/v1/auth/csrf")).json()["csrf_token"]
    return await client.post(
        f"/api/v1/collections/{setup['collection']}/papers?cloud_consent={str(consent).lower()}",
        content=setup["source"].read_bytes() if content is None else content,
        headers={"X-CSRF-Token": csrf, "Content-Type": "application/pdf"},
    )


async def run(setup, paper):
    await ingest(setup["ctx"], str(setup["user"]), paper["pending_version_id"])


async def state(client, paper):
    return (await client.get(f"/api/v1/papers/{paper['id']}")).json()


async def test_upload_dedup_outbox_and_idempotent_worker(client, app, setup_ingestion):
    s = setup_ingestion
    response = await upload(client, s)
    assert response.status_code == 202, response.text
    paper = response.json()
    assert (await upload(client, s)).json()["id"] == paper["id"]
    async with app.state.session_factory() as db:
        assert await db.scalar(select(func.count()).select_from(TaskOutbox)) == 1
        assert (await db.scalar(select(QuotaReservation))).amount == 2
    await dispatch_outbox(s["ctx"])
    await asyncio.gather(run(s, paper), run(s, paper))
    result = await state(client, paper)
    assert result["status"] == "ready", result
    assert result["title"] == "Test Research"
    assert result["active_version_id"] == paper["pending_version_id"]
    assert result["pending_version_id"] is None
    calls = len(s["embedder"].calls)
    await run(s, paper)
    assert len(s["embedder"].calls) == calls
    async with app.state.session_factory() as db:
        assert len(await active_chunks(db, s["user"])) == 3
        assert await db.scalar(select(func.count()).select_from(PaperVersion)) == 1
        assert (await db.scalar(select(ParseJob))).batch_id == "remote-batch"
        assert (await db.scalar(select(TaskOutbox))).completed_at
        assert await db.scalar(select(func.count()).select_from(ChunkParent)) == 3
        chunks = await active_chunks(db, s["user"])
        assert all(c.parent_id for c in chunks)
        assert all(c.chunk_metadata["context_prefix"] for c in chunks)
        overview = next(c for c in chunks if c.kind == "document_meta")
        assert overview.chunk_metadata["level"] == 0
        assert "Alice" in overview.text and overview.locations == []
        assert next(c for c in chunks if c.kind == "text").source_spans
    usage = (await client.get("/api/v1/usage")).json()
    assert usage["parse_pages"] == 2 and usage["parse_pages_reserved"] == 0
    assert usage["model_tokens"] > 0 and len(usage["by_node"]) == 1
    doc_res = await client.get(f"/api/v1/papers/{paper['id']}/document")
    assert doc_res.status_code == 200 and doc_res.json()["page_count"] == 2


async def test_rechunk_uses_cache_keeps_snapshot_and_never_calls_mineru(
    client, app, setup_ingestion
):
    s = setup_ingestion
    paper = (await upload(client, s)).json()
    await run(s, paper)
    before = await state(client, paper)
    async with app.state.session_factory() as db:
        chunk = (await active_chunks(db, s["user"]))[0]
        citation = await snapshot(db, s["user"], chunk.id)
        citation_id, source_text = citation.id, citation.source_text
        await db.commit()
    parser_count, embed_count = len(s["parser_calls"]), len(s["embedder"].calls)
    new = (await post(client, f"/papers/{paper['id']}/rechunk")).json()
    assert new["active_version_id"] == before["active_version_id"]
    await run(s, new)
    after = await state(client, paper)
    assert after["status"] == "ready" and after["active_version_id"] != before["active_version_id"]
    assert len(s["parser_calls"]) == parser_count and len(s["embedder"].calls) == embed_count
    old = (await client.get(f"/api/v1/citations/{citation_id}")).json()
    assert old["source_text"] == source_text and old["status"] == "available"
    async with app.state.session_factory() as db:
        assert len(await active_chunks(db, s["user"])) == 3
        assert await db.scalar(select(func.count()).select_from(Chunk)) == 6
    assert (await client.get("/api/v1/usage")).json()["parse_pages"] == 2


async def test_failed_new_version_preserves_old_then_manual_retry_recovers(
    client, app, setup_ingestion
):
    s = setup_ingestion
    paper = (await upload(client, s)).json()
    await run(s, paper)
    before = await state(client, paper)
    await post(client, f"/papers/{paper['id']}", {"title": "Changed prefix"}, method="PATCH")
    new = (await post(client, f"/papers/{paper['id']}/rechunk")).json()
    s["embedder"].error = ValueError("secret vendor message")
    await run(s, new)
    after = await state(client, paper)
    assert after["status"] == "ready" and after["active_version_id"] == before["active_version_id"]
    assert "secret" not in after["error"]
    versions = (await client.get(f"/api/v1/papers/{paper['id']}/versions")).json()
    assert versions[0]["status"] == "failed"
    s["embedder"].error = None
    retry = await post(client, f"/papers/{paper['id']}/retry")
    assert retry.status_code == 202, retry.text
    assert retry.json()["pending_version_id"] == new["pending_version_id"]
    await run(s, retry.json())
    assert (await state(client, paper))["active_version_id"] == new["pending_version_id"]


async def test_reparse_creates_new_cloud_job(client, app, setup_ingestion):
    s = setup_ingestion
    paper = (await upload(client, s)).json()
    await run(s, paper)
    new = (await post(client, f"/papers/{paper['id']}/reparse")).json()
    await run(s, new)
    assert len(s["parser_calls"]) == 2 and s["parser_calls"][0] != s["parser_calls"][1]
    assert (await client.get("/api/v1/usage")).json()["parse_pages"] == 4


@pytest.mark.parametrize("mode", ["success", "quota", "rejected", "edited"])
async def test_worker_wires_accounted_metadata_and_rechunk_reuses_it(
    client, app, setup_ingestion, monkeypatch, mode
):
    import httpx
    from openai import APIStatusError

    from app.services.parsing.pipeline import private_json
    from tests.test_ingestion_metadata import Model

    s = setup_ingestion
    model = Model()
    monkeypatch.setattr("app.services.ingestion.metadata.chat_model", lambda *a, **kw: model)
    original = s["ctx"]["parse_pdf"]

    async def parser(pdf_path, output, settings, **kwargs):
        assert kwargs["use_llm"] is True
        await original(pdf_path, output, settings, **kwargs)
        metadata = PaperMetadata.model_validate_json((output / "metadata.json").read_text())
        try:
            extraction = await kwargs["extractor"]("Published in 2024.")
            apply_evidence(metadata, extraction, "Published in 2024.")
        except MetadataExtractionSkipped as error:
            metadata.warnings.append(str(error))
        private_json(output / "metadata.json", metadata.model_dump())

    s["ctx"]["parse_pdf"] = parser
    if mode == "quota":
        s["ctx"]["settings"].user_daily_model_tokens = 1000
    elif mode == "rejected":
        model.replies = [
            APIStatusError(
                "private vendor error",
                response=httpx.Response(
                    401, request=httpx.Request("POST", "https://model.invalid")
                ),
                body=None,
            )
        ]
    paper = (await upload(client, s)).json()
    if mode == "edited":
        edited = await post(
            client,
            f"/papers/{paper['id']}",
            {"title": paper["title"], "year": 2020},
            method="PATCH",
        )
        assert edited.status_code == 200, edited.text
        assert edited.json()["year"] == 2020
    await run(s, paper)
    result = await state(client, paper)
    assert result["status"] == "ready", result
    assert result["year"] == (2024 if mode == "success" else 2020 if mode == "edited" else None)
    assert len(model.calls) == int(mode in {"success", "rejected"})
    usage = (await client.get("/api/v1/usage")).json()
    metadata_nodes = [r for r in usage["by_node"] if r["node"] == "metadata_extract"]
    assert len(metadata_nodes) == int(mode in {"success", "rejected"})
    async with app.state.session_factory() as db:
        rows = list(
            await db.scalars(select(UsageRecord).where(UsageRecord.node == "metadata_extract"))
        )
        assert all(
            r.user_id == s["user"] and str(r.version_id) == paper["pending_version_id"]
            for r in rows
        )
        if mode == "success":
            assert rows[0].amount == 20 and not rows[0].estimated
        elif mode == "rejected":
            assert rows[0].amount == 0 and rows[0].outcome == "rejected"
    await run(s, paper)
    rechunk = (await post(client, f"/papers/{paper['id']}/rechunk")).json()
    await run(s, rechunk)
    assert (await state(client, paper))["status"] == "ready"
    assert len(model.calls) == int(mode in {"success", "rejected"})


async def test_delete_during_embedding_cannot_resurrect_and_clears_citations(
    client, app, setup_ingestion
):
    s = setup_ingestion
    paper = (await upload(client, s)).json()
    await run(s, paper)
    async with app.state.session_factory() as db:
        citation = await snapshot(db, s["user"], (await active_chunks(db, s["user"]))[0].id)
        citation_id = citation.id
        await db.commit()
    await post(client, f"/papers/{paper['id']}", {"title": "Different input"}, method="PATCH")
    new = (await post(client, f"/papers/{paper['id']}/rechunk")).json()
    s["embedder"].entered.clear()
    s["embedder"].gate = asyncio.Event()
    task = asyncio.create_task(run(s, new))
    await asyncio.wait_for(s["embedder"].entered.wait(), 5)
    assert (await post(client, f"/papers/{paper['id']}", method="DELETE")).status_code == 204
    # Cleanup defers while the worker owns the paper file lock.
    await cleanup(s["ctx"], str(s["user"]), paper["id"])
    s["embedder"].gate.set()
    await task
    await cleanup(s["ctx"], str(s["user"]), paper["id"])
    assert (await client.get(f"/api/v1/papers/{paper['id']}/pdf")).status_code == 404
    assert not (s["ctx"]["settings"].storage_root / str(s["user"]) / paper["id"]).exists()
    item = (await client.get(f"/api/v1/citations/{citation_id}")).json()
    assert item["status"] == "来源已删除" and item["source_text"] == "" and item["locations"] == []
    async with app.state.session_factory() as db:
        assert await db.scalar(select(func.count()).select_from(Chunk)) == 0
        assert await db.scalar(select(func.count()).select_from(ChunkParent)) == 0
        assert await db.scalar(select(func.count()).select_from(EmbeddingCache)) == 0
        row = await db.get(Paper, UUID(paper["id"]))
        assert row.active_version_id is None and row.status == "deleted"


async def test_user_scope_files_range_assets_and_cache(client, app, setup_ingestion):
    s = setup_ingestion
    paper = (await upload(client, s)).json()
    await run(s, paper)
    path = f"/api/v1/papers/{paper['id']}"
    result = await client.get(path + "/pdf", headers={"Range": "bytes=0-4"})
    assert result.status_code == 206 and result.content == b"%PDF-"
    assert (await client.get(path + "/markdown")).text.startswith("# Research")
    assert (await client.get(path + "/assets/images/plot.png")).status_code == 200
    assert (await client.get(path + "/assets/document.json")).status_code == 404
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost:8080",
        headers={"Origin": "http://localhost:8080"},
    ) as b:
        await register(b, "b@example.org")
        await login(b, "b@example.org")
        for suffix in (
            "",
            "/pdf",
            "/chunks",
            "/versions",
            "/references",
            "/markdown",
            "/assets/images/plot.png",
        ):
            assert (await b.get(path + suffix)).status_code == 404
        for operation in ("reparse", "rechunk", "retry"):
            assert (await post(b, f"/papers/{paper['id']}/{operation}")).status_code == 404
        assert (await post(b, f"/papers/{paper['id']}", method="DELETE")).status_code == 404
        other_user = UUID((await b.get("/api/v1/auth/me")).json()["id"])
        other_col = (await post(b, "/collections", {"name": "B"})).json()["id"]
        s2 = {**s, "user": other_user, "collection": other_col}
        other_paper = (await upload(b, s2)).json()
        count = len(s["embedder"].calls)
        await run(s2, other_paper)
        assert len(s["embedder"].calls) > count  # no cache across users


async def test_atomic_quota_and_concurrency_reservation(client, app, setup_ingestion):
    s = setup_ingestion
    app.state.settings.user_daily_parse_pages = 2
    other = s["source"].read_bytes() + b"\n%other"
    csrf = (await client.get("/api/v1/auth/csrf")).json()["csrf_token"]
    results = await asyncio.gather(
        *(
            client.post(
                f"/api/v1/collections/{s['collection']}/papers?cloud_consent=true",
                content=content,
                headers={"X-CSRF-Token": csrf},
            )
            for content in (s["source"].read_bytes(), other)
        )
    )
    assert sorted(r.status_code for r in results) == [202, 429], [r.text for r in results]
    async with app.state.session_factory() as db:
        assert await db.scalar(select(func.count()).select_from(Paper)) == 1
        assert (
            await db.scalar(
                select(func.sum(QuotaReservation.amount)).where(
                    QuotaReservation.status == "reserved"
                )
            )
            == 2
        )


async def test_quota_before_embedding_and_failure_releases_pages(client, app, setup_ingestion):
    s = setup_ingestion
    app.state.settings.user_daily_model_tokens = 1
    paper = (await upload(client, s)).json()
    await run(s, paper)
    assert len(s["embedder"].calls) == 0
    result = await state(client, paper)
    assert result["status"] == "failed" and "额度" in result["error"]
    assert (await client.get("/api/v1/usage")).json()["parse_pages"] == 2


async def test_invalid_pdf_consent_verification_and_count_limits(client, app, setup_ingestion):
    s = setup_ingestion
    assert (await upload(client, s, consent=False)).status_code == 400
    assert (await upload(client, s, content=b"not pdf")).status_code == 400
    app.state.settings.user_max_papers = 1
    paper = (await upload(client, s)).json()
    assert (
        await upload(client, s, content=s["source"].read_bytes() + b"\n%another")
    ).status_code == 429
    assert (await post(client, f"/papers/{paper['id']}/rechunk")).status_code == 409
    async with app.state.session_factory() as db:
        await db.execute(update(User).where(User.id == s["user"]).values(email_verified_at=None))
        await db.commit()
    assert (await upload(client, s)).status_code == 403


async def test_failure_before_cloud_releases_reservation_and_outbox_can_replay(
    client, app, setup_ingestion
):
    s = setup_ingestion

    async def broken(*args, **kwargs):
        raise ParsingError("Token 不可用", code="authentication")

    s["ctx"]["parse_pdf"] = broken
    paper = (await upload(client, s)).json()
    await dispatch_outbox(s["ctx"])
    await app.state.redis.flushdb()  # simulate lost Redis queue
    async with app.state.session_factory() as db:
        await db.execute(update(TaskOutbox).values(delivered_at=None))
        await db.commit()
    await dispatch_outbox(s["ctx"])
    assert await app.state.redis.zcard("papermind:ingest") == 1
    await run(s, paper)
    usage = (await client.get("/api/v1/usage")).json()
    assert usage["parse_pages"] == usage["parse_pages_reserved"] == 0
    assert (await state(client, paper))["status"] == "failed"


async def test_transient_failure_retries_same_version_and_remote_work(client, app, setup_ingestion):
    s = setup_ingestion
    original = s["ctx"]["parse_pdf"]
    count = 0

    async def flaky(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 1:
            raise ParsingError("暂时不可用", code="network", retryable=True)
        return await original(*args, **kwargs)

    s["ctx"]["parse_pdf"] = flaky
    paper = (await upload(client, s)).json()
    await run(s, paper)
    result = await state(client, paper)
    assert (
        result["status"] == "pending"
        and result["pending_version_id"] == paper["pending_version_id"]
    )
    await run(s, paper)
    assert (await state(client, paper))["status"] == "ready"
    async with app.state.session_factory() as db:
        assert await db.scalar(select(func.count()).select_from(PaperVersion)) == 1
        assert (await db.scalar(select(PaperVersion))).attempts == 2


async def test_disabled_user_worker_makes_no_external_calls(client, app, setup_ingestion):
    s = setup_ingestion
    paper = (await upload(client, s)).json()
    async with app.state.session_factory() as db:
        await db.execute(update(User).where(User.id == s["user"]).values(status="disabled"))
        await db.commit()
    await run(s, paper)
    assert s["parser_calls"] == [] and s["embedder"].calls == []
    async with app.state.session_factory() as db:
        assert (await db.get(Paper, UUID(paper["id"]))).status == "failed"


async def test_collection_delete_tombstones_papers(client, app, setup_ingestion):
    s = setup_ingestion
    paper = (await upload(client, s)).json()
    await run(s, paper)
    assert (
        await post(client, f"/collections/{s['collection']}", method="DELETE")
    ).status_code == 204
    assert (await client.get(f"/api/v1/papers/{paper['id']}")).status_code == 404
    await cleanup(s["ctx"], str(s["user"]), paper["id"])
    async with app.state.session_factory() as db:
        assert (await db.get(Paper, UUID(paper["id"]))).collection_id is None


async def test_database_blocks_cross_user_version_and_snapshot(client, app, setup_ingestion):
    s = setup_ingestion
    paper = (await upload(client, s)).json()
    await run(s, paper)
    async with app.state.session_factory() as db:
        db.add(
            Citation(
                user_id=uuid4(),
                paper_id=UUID(paper["id"]),
                version_id=UUID(paper["pending_version_id"]),
                chunk_id=uuid4(),
                source_text="forged",
                locations=[],
            )
        )
        with pytest.raises(IntegrityError):
            await db.commit()


async def test_invalid_embedding_dimension_is_not_activated(client, app, setup_ingestion):
    s = setup_ingestion

    async def wrong(texts):
        return [[0.1, 0.2] for _ in texts]

    s["embedder"].aembed_documents = wrong
    paper = (await upload(client, s)).json()
    await run(s, paper)
    result = await state(client, paper)
    assert result["status"] == "failed" and result["active_version_id"] is None
    assert "维度" in result["error"]


async def test_arq_consumes_durable_outbox_job(client, app, setup_ingestion):
    from arq import create_pool
    from arq.connections import RedisSettings
    from arq.worker import Worker

    s = setup_ingestion
    paper = (await upload(client, s)).json()
    await dispatch_outbox(s["ctx"])
    pool = await create_pool(
        RedisSettings.from_dsn(app.state.settings.redis_url.get_secret_value())
    )
    worker = Worker(
        functions=[ingest],
        redis_pool=pool,
        queue_name="papermind:ingest",
        ctx=s["ctx"],
        burst=True,
        handle_signals=False,
        keep_result=0,
        max_jobs=1,
    )
    try:
        await worker.async_run()
        assert worker.jobs_complete == 1
        assert (await state(client, paper))["status"] == "ready"
    finally:
        await worker.close()


async def test_worker_cancellation_resumes_completed_embedding_batches(
    client, app, setup_ingestion
):
    s = setup_ingestion
    app.state.settings.embedding_batch_size = 1
    paper = (await upload(client, s)).json()
    calls = []
    waiting = asyncio.Event()

    async def interrupted(texts):
        calls.append(texts)
        if len(calls) == 2:
            waiting.set()
            await asyncio.Event().wait()
        return [[0.1] * 1024 for _ in texts]

    s["embedder"].aembed_documents = interrupted
    task = asyncio.create_task(run(s, paper))
    await asyncio.wait_for(waiting.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    async with app.state.session_factory() as db:
        assert await db.scalar(select(func.count()).select_from(EmbeddingCache)) == 1
    parser_calls = len(s["parser_calls"])
    await run(s, paper)
    assert (await state(client, paper))["status"] == "ready"
    assert len(calls) == 4 and calls[1] == calls[2] and calls[0] != calls[2]
    assert len({tuple(batch) for batch in calls}) == 3
    assert len(s["parser_calls"]) == parser_calls
    async with app.state.session_factory() as db:
        records = list(
            await db.scalars(select(UsageRecord).where(UsageRecord.metric == "model_tokens"))
        )
        assert len(records) == 4 and sum(r.outcome == "uncertain" for r in records) == 1


async def test_per_user_quota_override(client, app, setup_ingestion):
    from app.db.models import UserQuota

    s = setup_ingestion
    async with app.state.session_factory() as db:
        db.add(
            UserQuota(user_id=s["user"], max_papers=1, daily_parse_pages=1, daily_model_tokens=3000)
        )
        await db.commit()
    result = await upload(client, s)
    assert result.status_code == 429 and "页数" in result.json()["detail"]
    usage = (await client.get("/api/v1/usage")).json()
    assert usage["limits"]["papers"] == 1 and usage["limits"]["daily_model_tokens"] == 3000


async def test_storage_includes_parsed_files_and_prunes_oversized_new_results(
    client, app, setup_ingestion
):
    s = setup_ingestion
    app.state.settings.user_max_storage_mb = 1
    original = s["ctx"]["parse_pdf"]

    async def oversized(pdf_path, output, settings, **kwargs):
        await original(pdf_path, output, settings, **kwargs)
        (output / "metadata-extraction.json").write_text('{"already_paid": true}')
        (output / "raw/large.bin").write_bytes(b"a" * 2 * 1024**2)

    s["ctx"]["parse_pdf"] = oversized
    paper = (await upload(client, s)).json()
    await run(s, paper)
    result = await state(client, paper)
    assert result["status"] == "failed" and "存储" in result["error"]
    assert result["size_bytes"] < 1024**2
    async with app.state.session_factory() as db:
        job = await db.scalar(select(ParseJob))
        directory = resolve_private(app.state.settings, job.result_path)
        assert not (directory / "raw").exists()
        assert json.loads((directory / "job.json").read_text())["stage"] == "polling"
        assert json.loads((directory / "metadata-extraction.json").read_text())["already_paid"]
        assert job.batch_id == "remote-batch"


async def test_uncertain_submission_is_budgeted_and_requires_reconciliation(
    client, app, setup_ingestion
):
    s = setup_ingestion

    async def uncertain(*args, **kwargs):
        raise ParsingError("提交结果未知", code="submission_uncertain")

    s["ctx"]["parse_pdf"] = uncertain
    paper = (await upload(client, s)).json()
    await run(s, paper)
    assert (await client.get("/api/v1/usage")).json()["parse_pages"] == 2
    assert (await post(client, f"/papers/{paper['id']}/retry")).status_code == 409
    async with app.state.session_factory() as db:
        record = await db.scalar(select(UsageRecord))
        assert record.outcome == "uncertain"


async def test_global_page_limit_applies_before_submission(client, app, setup_ingestion):
    s = setup_ingestion
    app.state.settings.global_daily_parse_pages = 1
    result = await upload(client, s)
    assert result.status_code == 429 and s["parser_calls"] == []


async def test_pending_version_and_usage_have_database_owner_constraints(
    client, app, setup_ingestion
):
    from app.db.models import Collection

    s = setup_ingestion
    paper = (await upload(client, s)).json()
    async with app.state.session_factory() as db:
        other = User(email="other@example.org", password_hash="unused", email_verified_at=utcnow())
        db.add(other)
        await db.flush()
        collection = Collection(user_id=other.id, name="Other")
        db.add(collection)
        await db.flush()
        other_paper = Paper(
            user_id=other.id,
            collection_id=collection.id,
            title="Other",
            sha256="0" * 64,
            file_path="unused",
            size_bytes=1,
            page_count=1,
            pending_version_id=UUID(paper["pending_version_id"]),
        )
        db.add(other_paper)
        with pytest.raises(IntegrityError):
            await db.commit()


async def test_file_cleanup_failure_remains_replayable(client, app, setup_ingestion, monkeypatch):
    import shutil

    s = setup_ingestion
    paper = (await upload(client, s)).json()
    assert (await post(client, f"/papers/{paper['id']}", method="DELETE")).status_code == 204
    original = shutil.rmtree

    def unavailable(*args, **kwargs):
        raise PermissionError("temporary filesystem failure")

    monkeypatch.setattr(shutil, "rmtree", unavailable)
    with pytest.raises(PermissionError):
        await cleanup(s["ctx"], str(s["user"]), paper["id"])
    async with app.state.session_factory() as db:
        outbox = await db.scalar(select(TaskOutbox).where(TaskOutbox.kind == "cleanup"))
        assert outbox.completed_at is None
    monkeypatch.setattr(shutil, "rmtree", original)
    await cleanup(s["ctx"], str(s["user"]), paper["id"])
    # Replayed cleanup after successful removal is also safe.
    await cleanup(s["ctx"], str(s["user"]), paper["id"])
    async with app.state.session_factory() as db:
        assert (
            await db.scalar(select(TaskOutbox).where(TaskOutbox.kind == "cleanup"))
        ).completed_at


async def test_structured_references_are_versioned_and_attached_to_body(
    client, app, setup_ingestion
):
    s = setup_ingestion
    original_parser = s["ctx"]["parse_pdf"]

    async def parser(pdf_path, output, settings, **kwargs):
        await original_parser(pdf_path, output, settings, **kwargs)
        path = output / "document.json"
        doc = ParsedDocument.model_validate_json(path.read_text())
        doc.blocks[0].text += " Based on prior work [12]."
        doc.blocks.append(
            ParsedBlock(
                id="ref12",
                kind="reference",
                text="[12] Alice and Bob. Routing. Conference, 2024.",
                section_path=["7 References"],
                locations=[Location(page_idx=1)],
                indexable=False,
            )
        )
        path.write_text(doc.model_dump_json())

    s["ctx"]["parse_pdf"] = parser
    paper = (await upload(client, s)).json()
    await run(s, paper)
    assert (await state(client, paper))["status"] == "ready"
    response = await client.get(f"/api/v1/papers/{paper['id']}/references")
    assert response.status_code == 200
    refs = response.json()
    assert len(refs) == 1 and refs[0]["ref_id"] == "[12]" and refs[0]["year"] == 2024
    chunks = (await client.get(f"/api/v1/papers/{paper['id']}/chunks")).json()
    body = next(c for c in chunks if c["kind"] == "text")
    assert body["chunk_metadata"]["citations"] == refs
    assert not any("Conference, 2024" in c["text"] for c in chunks)
    new = (await post(client, f"/papers/{paper['id']}/rechunk")).json()
    await run(s, new)
    assert (await client.get(f"/api/v1/papers/{paper['id']}/references")).json() == refs
    async with app.state.session_factory() as db:
        versions = list(await db.scalars(select(PaperVersion)))
        assert len(versions) == 2 and all(v.reference_entries == refs for v in versions)
        assert all(v.paper_map["schema_version"] == "papermind.map.v1" for v in versions)
        assert all(v.paper_map["toc"] for v in versions)
        section_ids = set(await db.scalars(select(StoredSection.id)))
        assert all(UUID(v.paper_map["toc"][0]["section_id"]) in section_ids for v in versions)


async def test_navigation_backfill_is_idempotent_and_uses_saved_parse_without_embedding(
    client, app, setup_ingestion
):
    from app.services.ingestion.paper_maps import backfill_paper_maps

    s = setup_ingestion
    paper = (await upload(client, s)).json()
    await run(s, paper)
    async with app.state.session_factory() as db:
        version = await db.scalar(select(PaperVersion))
        original = version.paper_map
        version.paper_map = {}
        await db.commit()
    before = len(s["embedder"].calls) if "embedder" in s else None
    counts = await backfill_paper_maps(app.state.settings, app.state.session_factory)
    assert counts == {"updated": 1, "skipped": 0, "unavailable": 0}
    async with app.state.session_factory() as db:
        assert (await db.scalar(select(PaperVersion))).paper_map == original
    assert await backfill_paper_maps(app.state.settings, app.state.session_factory) == {
        "updated": 0,
        "skipped": 1,
        "unavailable": 0,
    }
    if before is not None:
        assert len(s["embedder"].calls) == before
