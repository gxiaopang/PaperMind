"""Current frozen gold integrity, navigation, isolation and real graph harness checks."""

import copy
import hashlib
import importlib.util
import json
import re
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID

import pytest
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.models import Chunk, PaperVersion, Section
from app.eval.retrieval import load_dataset, validate_dataset
from app.services.chat.store import CONTEXT_STRATEGY
from app.services.chunking.chunker import CHUNKER_VERSION, chunk_document
from app.services.parsing.mineru_result import ParsedDocument
from app.services.retrieval.store import RetrievalStore
from app.services.retrieval.types import RetrievalScope
from tests.test_ingestion_metadata import AsyncSessionFacade
from tests.test_parent_retrieval_store import parent_store as parent_store

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "eval/datasets/m3-gold-v7.json"


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def snapshot():
    return load_dataset(DATASET)


def test_frozen_review_preserves_confirmed_answers_and_source_identity(snapshot):
    frozen, corpora = snapshot
    assert frozen["status"] == "frozen" and frozen["human_review"]["status"] == "completed"
    assert frozen["chunker_version"] == CHUNKER_VERSION
    assert frozen["retrieval_strategy"] == CONTEXT_STRATEGY
    assert len(frozen["cases"]) == 34 and len(frozen["answerability_cases"]) == 2
    audit = json.loads((ROOT / "eval/datasets/m3-gold-v7-review.json").read_bytes())
    assert audit["confirmation_text"] == "我已核对完暂无需要修改的地方"
    assert not audit["changes_to_questions_answers_evidence_or_corpora"]
    source = (ROOT / audit["source_worksheet"]).read_bytes()
    assert hashlib.sha256(source).hexdigest() == audit["source_worksheet_sha256"]
    assert hashlib.sha256(DATASET.read_bytes()).hexdigest() == audit["dataset_sha256"]
    assert audit["scope"] == dict(
        positive_cases=34, insufficient_cases=2, paper_tool_mechanisms=4, real_formula_hop=1
    )
    sections = dict(re.findall(r"^## (m3-\d+)\n(.*?)(?=^## |\Z)", source.decode(), re.M | re.S))
    for case in frozen["cases"] + frozen["answerability_cases"]:
        answer = re.search(
            r"^\*\*参考答案：\*\*[^\S\n]*(.*?)(?=^\*\*)", sections[case["id"]], re.M | re.S
        )
        assert answer and answer[1].strip() == case["standard_answer"]
    for entry in frozen["supplemental_review"].values():
        supplement = json.loads((ROOT / entry["path"]).read_bytes())
        assert supplement["status"] == "completed"
        assert supplement["human_review"]["confirmation_text"] == audit["confirmation_text"]
    assert {name: len(c["chunks"]) for name, c in corpora.items()} == {"attention": 50, "lora": 103}
    assert all(c["chunker_version"] == CHUNKER_VERSION for c in corpora.values())
    assert all(c["paper_map"]["toc"] and c["paper_map"]["abstract"] for c in corpora.values())
    assert all("section_std" in s for c in corpora.values() for s in c["sections"])
    script("m3_eval").require_reviewed(frozen)


async def test_pending_review_stops_evaluation_before_infrastructure_or_provider(
    snapshot, tmp_path, monkeypatch, parent_store
):
    module = script("m3_eval")
    data, corpora = copy.deepcopy(snapshot)
    data["status"] = "draft"
    data["human_review"]["status"] = "pending"
    monkeypatch.setattr(module, "load_dataset", lambda _: (data, corpora))

    def unexpected(*args, **kwargs):
        raise AssertionError("Pending draft started provider or database")

    monkeypatch.setattr(module, "create_async_engine", unexpected)
    monkeypatch.setattr(module, "AccountedEmbeddings", unexpected)
    with pytest.raises(ValueError, match="pending human-review draft"):
        await module.evaluate(parent_store.settings, DATASET, tmp_path / "cache.json")


@pytest.mark.parametrize("damage", ["target", "duplicate", "negative"])
def test_snapshot_rejects_broken_graph_and_mixed_answerability_labels(snapshot, damage):
    data, corpora = copy.deepcopy(snapshot)
    if damage == "target":
        corpora["attention"]["chunks"][0]["chunk_metadata"]["cross_refs"] = [
            {"label": "table:1", "target_ordinals": [999]}
        ]
    elif damage == "duplicate":
        data["answerability_cases"][0]["id"] = data["cases"][0]["id"]
    else:
        data["answerability_cases"][0]["relevant"] = data["cases"][0]["relevant"]
    with pytest.raises(ValueError):
        validate_dataset(data, corpora)


def test_source_backed_unindexed_evidence_requires_explicit_policy(snapshot):
    data, corpora = copy.deepcopy(snapshot)
    ref = data["cases"][0]["relevant"][0]
    corpora[ref["paper"]]["chunks"][ref["ordinal"]]["indexable"] = False
    validate_dataset(data, corpora)
    del data["evidence_policy"]
    with pytest.raises(ValueError, match="Evidence label differs"):
        validate_dataset(data, corpora)


async def test_snapshot_import_preserves_unindexed_targets_metadata_sections_and_isolation(
    snapshot, parent_store
):
    data, corpora = copy.deepcopy(snapshot)
    c = parent_store
    module = script("m3_eval")
    corpus = corpora["attention"]
    # Exercise context-only targets and metadata-only overview locations without
    # changing a frozen file or requiring a real embedding/provider.
    corpus["chunks"][25].update(indexable=False, unindexed_reason="atomic_span_exceeds_input_limit")
    corpus["chunks"][0]["locations"] = []
    vectors = {row["content_hash"]: [0.1] * 1024 for row in corpus["chunks"] if row["indexable"]}

    class Facade(AsyncSessionFacade):
        async def flush(self):
            self.db.flush()

    @asynccontextmanager
    async def sessions():
        with Session(c.engine) as db:
            yield Facade(db)

    identity = {
        "provider": c.settings.embedding_base_url.rstrip("/"),
        "model": c.settings.embedding_model,
        "dimension": 1024,
    }
    papers, keys = await module.import_corpus(
        sessions, data, {"attention": corpus}, identity, vectors, c.owner, c.foreign
    )
    store = RetrievalStore(sessions, c.settings)
    scope = RetrievalScope(c.owner, paper_ids=(papers["attention"],))
    with Session(c.engine) as db:
        rows = list(
            db.scalars(
                select(Chunk).where(Chunk.paper_id == papers["attention"]).order_by(Chunk.ordinal)
            )
        )
        assert len(rows) == len(corpus["chunks"]) == len(keys)
        assert rows[25].embedding is None and not rows[25].indexable
        assert rows[25].chunk_metadata == corpus["chunks"][25]["chunk_metadata"]
        assert rows[25].source_spans == corpus["chunks"][25]["source_spans"]
        assert rows[0].locations == []
        assert db.get(Section, rows[30].section_id).path == corpus["chunks"][30]["section_path"]
        anchor = store.document(rows[30], corpus["title"])
    docs = await store.expand_context(scope, [anchor])
    assert any(
        d.metadata["ordinal"] == 25 and d.metadata["retrieval_role"] == "cross_ref" for d in docs
    )
    assert {d.metadata["user_id"] for d in docs} == {str(c.owner)}
    assert {d.metadata["paper_id"] for d in docs} == {str(papers["attention"])}
    assert (await store.acronym_dictionary(scope))["WSJ"] == ["Wall Street Journal"]
    case = next(q for q in data["cases"] if q["id"] == "m3-31")
    assert module.case_scope(
        c.owner, {"attention": c.papers[0], "lora": c.papers[1]}, case
    ).paper_ids == tuple(c.papers)


def test_map_with_nonexistent_section_is_rejected_before_import():
    data, corpora = copy.deepcopy(load_dataset(DATASET))
    corpora["attention"]["paper_map"]["toc"][0]["section_id"] = "missing"
    with pytest.raises(ValueError, match="unknown section"):
        validate_dataset(data, corpora)


async def import_current(c, data, corpora):
    class Facade(AsyncSessionFacade):
        async def flush(self):
            self.db.flush()

    @asynccontextmanager
    async def sessions():
        with Session(c.engine) as db:
            yield Facade(db)

    identity = dict(
        provider=c.settings.embedding_base_url.rstrip("/"),
        model=c.settings.embedding_model,
        dimension=1024,
    )
    vectors = {
        r["content_hash"]: [0.1] * 1024
        for p in corpora.values()
        for r in p["chunks"]
        if r["indexable"]
    }
    papers, _ = await script("m3_eval").import_corpus(
        sessions, data, corpora, identity, vectors, c.owner, c.foreign
    )
    return papers, RetrievalStore(sessions, c.settings)


async def test_current_snapshot_import_keeps_real_maps_section_tools_and_exact_tables(parent_store):
    c = parent_store
    data, corpora = load_dataset(DATASET)
    papers, store = await import_current(c, data, corpora)
    scope = RetrievalScope(c.owner, paper_ids=tuple(papers.values()))
    maps = await store.paper_maps(scope)
    assert len(maps) == 2 and all(not m.get("legacy") for m in maps)
    assert {m["title"] for m in maps} == {p["title"] for p in corpora.values()}
    with Session(c.engine) as db:
        versions = list(
            db.scalars(select(PaperVersion).where(PaperVersion.paper_id.in_(papers.values())))
        )
        for version in versions:

            def check(nodes, version_id):
                for node in nodes:
                    section = db.get(Section, UUID(node["section_id"]))
                    assert section.version_id == version_id and section.user_id == c.owner
                    assert section.section_std == node["section_std"]
                    check(node["children"], version_id)

            check(version.paper_map["toc"], version.id)
        case = next(q for q in data["cases"] if q["id"] == "m3-13")
        seed = db.scalar(
            select(Chunk).where(
                Chunk.paper_id == papers["lora"], Chunk.ordinal == case["relevant"][0]["ordinal"]
            )
        )
        section_id = seed.section_id
    docs, parents = await store.read_section(
        RetrievalScope(c.owner, paper_ids=(papers["lora"],)), "unused", section_id=section_id
    )
    assert docs and parents and all(d.metadata["parent_complete"] for d in docs)
    assert any(d.metadata["ordinal"] == case["relevant"][0]["ordinal"] for d in docs)
    table_docs = await store.inspect(
        RetrievalScope(c.owner, paper_ids=(papers["attention"],)), "table:2"
    )
    assert table_docs and any("27.3" in d.page_content for d in table_docs)
    assert any(d.metadata["chunk_metadata"].get("markdown_table") for d in table_docs)
    assert not await store.inspect(scope, "table:999")
    with pytest.raises(HTTPException) as denied:
        await store.paper_maps(RetrievalScope(c.foreign, paper_ids=(papers["attention"],)))
    assert denied.value.status_code == 404


async def test_real_pdf_formula_reference_expands_across_sections_and_is_inspectable(parent_store):
    c = parent_store
    fixture = json.loads((ROOT / "eval/fixtures/rmsnorm-equation-hop-v7.json").read_bytes())
    document = ParsedDocument.model_validate(fixture["document"])
    drafts = chunk_document(document, fixture["title"])
    rows = [
        dict(
            ordinal=i,
            parent_ordinal=None,
            kind=d.kind,
            text=d.text,
            embedding_input=d.embedding_input,
            section_id=d.section_id,
            section_path=d.section_path,
            locations=d.locations,
            source_spans=d.source_spans,
            chunk_metadata=d.chunk_metadata,
            content_hash=d.content_hash,
            token_count=d.token_count,
            indexable=d.indexable,
        )
        for i, d in enumerate(drafts)
    ]
    corpus = dict(
        title=fixture["title"],
        chunker_version=CHUNKER_VERSION,
        sections=[s.model_dump() for s in document.sections],
        chunks=rows,
    )
    data = {"corpus": {"rmsnorm": {"sha256": fixture["pdf_sha256"]}}}
    papers, store = await import_current(c, data, {"rmsnorm": corpus})
    scope = RetrievalScope(c.owner, paper_ids=(papers["rmsnorm"],))
    anchor = next(
        r
        for r in rows
        if any(s["block_id"] == fixture["anchor_block_id"] for s in r["source_spans"])
    )
    target = next(
        r
        for r in rows
        if any(s["block_id"] == fixture["target_block_id"] for s in r["source_spans"])
    )
    assert anchor["section_id"] != target["section_id"]
    with Session(c.engine) as db:
        seed = db.scalar(
            select(Chunk).where(
                Chunk.paper_id == papers["rmsnorm"], Chunk.ordinal == anchor["ordinal"]
            )
        )
        hit = store.document(seed, fixture["title"])
    expanded = await store.expand_context(scope, [hit])
    target_doc = next(d for d in expanded if d.metadata["ordinal"] == target["ordinal"])
    assert target_doc.metadata["retrieval_role"] == "cross_ref"
    assert target_doc.metadata["locations"] == [{"page_idx": 2, "bbox": None}]
    assert {d.metadata["user_id"] for d in expanded} == {str(c.owner)}
    inspected = await store.inspect(scope, "equation:3")
    assert {anchor["ordinal"], target["ordinal"]} <= {d.metadata["ordinal"] for d in inspected}


def result(content, citations, status="answered"):
    return dict(
        status="completed",
        result_status=status,
        retrieval_count=1,
        generation_count=1,
        message=dict(content=content, citations=citations),
    )


def test_smoke_rejects_wrong_scope_invalid_citations_and_missing_target_tool():
    module = script("pre_m6_live")
    case = next(c for c in module.CASES if c["id"] == "table")
    checks = module.assess(
        case,
        result("Claim [2].", [dict(number=1, text="Source", deleted=False, paper_id="foreign")]),
        {"maps": 1, "plans": [], "reports": []},
        [{"kind": "done"}],
        {"owner"},
        3,
    )
    assert not checks["citation_numbers_valid"]
    assert not checks["citation_scope_valid"]
    assert not checks["target_tool_used"]


def test_refusal_smoke_does_not_accept_arbitrary_answer_or_invent_faithfulness():
    module = script("pre_m6_live")
    case = next(c for c in module.CASES if c["id"] == "unsupported")
    wrong = module.assess(
        case,
        result(
            "GSM8K is 95%. [1]", [dict(number=1, text="BLEU 28.4", deleted=False, paper_id="owner")]
        ),
        {},
        [{"kind": "done"}],
        {"owner"},
        3,
    )
    assert not wrong["abstention_smoke"]
    refused = module.assess(
        case,
        result("当前证据未报告该值。", [], "insufficient"),
        {},
        [{"kind": "done"}],
        {"owner"},
        3,
    )
    assert refused["abstention_smoke"]


async def test_current_smoke_harness_runs_real_api_sql_graph_and_replay_with_provider_doubles(
    infrastructure, tmp_path, monkeypatch
):
    module = script("pre_m6_live")
    settings = Settings(
        **{k.lower(): v for k, v in infrastructure.items()},
        storage_root=tmp_path / "files",
        user_daily_model_tokens=10000000,
        global_daily_model_tokens=20000000,
    )
    monkeypatch.setattr("app.workers.mail.get_settings", lambda: settings)

    class Models:
        def __init__(self, *args):
            pass

        async def invoke(self, run, node, payload, config, *, schema=None, on_token=None):
            if node == "route":
                value = {"route": "chitchat" if payload["question"] == "你好" else "paper_qa"}
            elif node in {"rewrite_query", "transform_query"}:
                value = {"query": payload["question"]}
            elif node == "grade_documents":
                value = {"sufficient": True}
            elif node == "verify_grounding":
                value = {"grounded": True}
            elif node == "plan_paper_tools":
                if "Table 2" in payload["question"] or "Equation (3)" in payload["question"]:
                    value = {
                        "calls": [
                            {
                                "name": "inspect_element",
                                "args": {
                                    "label": "Table 2"
                                    if "Table 2" in payload["question"]
                                    else "Equation (3)"
                                },
                            }
                        ]
                    }
                else:
                    value = {
                        "calls": [
                            {
                                "name": "read_paper_section",
                                "args": {"section_std": "method", "mode": "full"},
                            }
                        ]
                    }
            elif node == "direct_answer":
                return "你好。"
            else:
                answer = "根据论文原文说明[1]。"
                if on_token:
                    await on_token(answer)
                return answer
            return schema.model_validate(value)

    monkeypatch.setattr("app.workers.chat.ChatModels", Models)
    actual_startup = module.startup

    async def start(ctx):
        await actual_startup(ctx)
        await ctx["checkpointer"].setup()

    monkeypatch.setattr(module, "startup", start)
    dataset_path = ROOT / "eval/datasets/m3-gold-v7.json"
    _, corpus = load_dataset(dataset_path)
    cache = dict(
        identity=dict(
            provider=settings.embedding_base_url.rstrip("/"),
            model=settings.embedding_model,
            dimension=settings.embedding_dim,
        ),
        vectors={
            r["content_hash"]: [0.1] * settings.embedding_dim
            for p in corpus.values()
            for r in p["chunks"]
            if r["indexable"]
        },
    )
    cache_path = tmp_path / "cache.json"
    cache_path.write_text(json.dumps(cache))
    output = tmp_path / "smoke.json"
    report = await module.smoke(
        settings,
        dataset_path,
        cache_path,
        output,
        ["agentic_section", "follow_up", "table", "formula_element", "empty_scope", "greeting"],
    )
    assert report["status"] == "passed", report["cases"]
    assert len(report["cases"]) == 6 and all(r["passed"] for r in report["cases"])
    assert json.loads(output.read_bytes())["status"] == "passed"
