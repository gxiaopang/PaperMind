"""Explicit paid retrieval evaluation; only disposable infrastructure is mutated."""

import argparse
import asyncio
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from uuid import NAMESPACE_URL, uuid4, uuid5

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from testcontainers.postgres import PostgresContainer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import Settings  # noqa: E402
from app.db.models import (  # noqa: E402
    Chunk,
    ChunkParent,
    Collection,
    Paper,
    PaperVersion,
    ParseJob,
    Section,
    UsageRecord,
    User,
    utcnow,
)
from app.eval.retrieval import detailed_metrics, load_dataset  # noqa: E402
from app.services.chunking.paper_map import bind_section_ids, section_std  # noqa: E402
from app.services.providers.embedding_batch import AccountedEmbeddings  # noqa: E402
from app.services.retrieval import RetrievalScope, create_retriever  # noqa: E402
from app.services.retrieval.accounting import RetrievalAccounting  # noqa: E402


async def usage_rows(sessions, operations=None):
    stmt = select(
        UsageRecord.node,
        func.sum(UsageRecord.amount),
        UsageRecord.estimated,
        func.count(UsageRecord.id),
    ).group_by(UsageRecord.node, UsageRecord.estimated)
    if operations is not None:
        stmt = stmt.where(UsageRecord.operation_id.in_(operations))
    async with sessions() as db:
        return [
            dict(node=node, tokens=int(amount), estimated=estimated, calls=calls)
            for node, amount, estimated, calls in (await db.execute(stmt)).all()
        ]


async def import_corpus(sessions, dataset, corpus, identity, vectors, owner, foreign):
    paper_ids, chunk_keys = {}, {}
    async with sessions() as db:
        for user_id in [owner, foreign]:
            collection = Collection(id=uuid4(), user_id=user_id, name="Frozen retrieval corpus")
            db.add(collection)
            await db.flush()
            for name, document in corpus.items():
                paper = Paper(
                    id=uuid4(),
                    user_id=user_id,
                    collection_id=collection.id,
                    title=document["title"],
                    sha256=dataset["corpus"][name]["sha256"],
                    file_path="not-a-file",
                    size_bytes=0,
                    page_count=max(
                        (
                            loc["page_idx"] + 1
                            for row in document["chunks"]
                            for loc in row["locations"]
                        ),
                        default=1,
                    ),
                    status="ready",
                )
                db.add(paper)
                await db.flush()
                job = ParseJob(
                    id=uuid4(),
                    user_id=user_id,
                    paper_id=paper.id,
                    result_path="not-a-file",
                    config={},
                )
                db.add(job)
                await db.flush()
                version = PaperVersion(
                    id=uuid4(),
                    user_id=user_id,
                    paper_id=paper.id,
                    parse_job_id=job.id,
                    operation="upload",
                    chunker_version=document["chunker_version"],
                    embedding_provider=identity["provider"],
                    embedding_model=identity["model"],
                    embedding_dim=identity["dimension"],
                    reference_entries=document.get("reference_entries", []),
                    status="ready",
                )
                db.add(version)
                await db.flush()
                paper.active_version_id = version.id
                if user_id == owner:
                    paper_ids[name] = paper.id
                section_ids = {
                    section["id"]: uuid5(version.id, f"section:{section['id']}")
                    for section in document.get("sections", [])
                }
                for section in document.get("sections", []):
                    db.add(
                        Section(
                            id=section_ids[section["id"]],
                            user_id=user_id,
                            version_id=version.id,
                            parent_id=section_ids.get(section.get("parent_id")),
                            title=section["title"],
                            level=section["level"],
                            path=section["path"],
                            section_std=section.get("section_std") or section_std(section["path"]),
                        )
                    )
                await db.flush()
                if document.get("paper_map"):
                    # Every tenant/version receives its own real SQL section IDs.
                    version.paper_map = bind_section_ids(
                        json.loads(json.dumps(document["paper_map"])), section_ids
                    )
                parent_ids = {}
                for parent in document.get("parents", []):
                    parent_id = uuid4()
                    parent_ids[parent["ordinal"]] = parent_id
                    db.add(
                        ChunkParent(
                            id=parent_id,
                            user_id=user_id,
                            paper_id=paper.id,
                            version_id=version.id,
                            ordinal=parent["ordinal"],
                            kind=parent["kind"],
                            text=parent["text"],
                            locations=parent["locations"],
                            child_count=parent["child_count"],
                            section_id=section_ids.get(parent.get("section_id")),
                        )
                    )
                await db.flush()
                for row in document["chunks"]:
                    locations = row["locations"]
                    pages = [loc["page_idx"] for loc in locations] or [0]
                    chunk_id = uuid5(
                        NAMESPACE_URL,
                        f"papermind:m3:{'owner' if user_id == owner else 'foreign'}:{name}:{row['ordinal']}:{row['content_hash']}",
                    )
                    if user_id == owner:
                        chunk_keys[str(chunk_id)] = (name, row["content_hash"])
                    db.add(
                        Chunk(
                            id=chunk_id,
                            user_id=user_id,
                            collection_id=collection.id,
                            paper_id=paper.id,
                            version_id=version.id,
                            parent_id=parent_ids.get(row.get("parent_ordinal")),
                            section_id=section_ids.get(row.get("section_id")),
                            ordinal=row["ordinal"],
                            kind=row["kind"],
                            text=row["text"],
                            embedding_input=row["embedding_input"],
                            section_path=row["section_path"],
                            locations=locations,
                            source_spans=row.get("source_spans", []),
                            chunk_metadata=row.get("chunk_metadata", {}),
                            page_start=min(pages),
                            page_end=max(pages),
                            token_count=row["token_count"],
                            content_hash=row["content_hash"],
                            indexable=row["indexable"],
                            unindexed_reason=row.get("unindexed_reason"),
                            embedding=vectors[row["content_hash"]] if row["indexable"] else None,
                        )
                    )
        await db.commit()
    return paper_ids, chunk_keys


def require_reviewed(dataset):
    if (
        dataset.get("status") == "draft"
        or dataset.get("human_review", {}).get("status") == "pending"
    ):
        raise ValueError(
            "Dataset is a pending human-review draft; review and freeze it before paid evaluation"
        )


def case_scope(owner, paper_ids, case):
    names = case.get("papers")
    if case["scope"] != "all":
        names = (
            [case["scope"]] if names is None else [name for name in names if name == case["scope"]]
        )
    return RetrievalScope(
        owner, paper_ids=tuple(paper_ids[name] for name in names) if names is not None else None
    )


async def evaluate(settings, dataset_path, cache_path, selected_variants=None):
    source_fingerprints = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [
            ROOT / "scripts/m3_eval.py",
            *sorted((ROOT / "backend/app/services/retrieval").glob("*.py")),
        ]
    }
    selected_variants = selected_variants or ["vector", "bm25", "hybrid", "hybrid_rerank"]
    if settings.rerank_top_n != 8:
        raise ValueError("This evaluation requires rerank_top_n=8 for comparable @8 metrics")
    dataset, corpus = load_dataset(dataset_path)
    require_reviewed(dataset)
    engine = create_async_engine(
        settings.database_url.get_secret_value(), pool_size=5, max_overflow=5
    )
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    accounting = RetrievalAccounting(sessions, engine, settings)
    owner, foreign = uuid4(), uuid4()
    started = time.monotonic()
    try:
        async with sessions() as db:
            for user_id, name in [(owner, "eval"), (foreign, "isolation")]:
                db.add(
                    User(
                        id=user_id,
                        email=f"{name}@example.org",
                        password_hash="no-login",
                        email_verified_at=utcnow(),
                    )
                )
            await db.commit()
        identity = {
            "provider": settings.embedding_base_url.rstrip("/"),
            "model": settings.embedding_model,
            "dimension": settings.embedding_dim,
        }
        cached = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        vectors = cached.get("vectors", {}) if cached.get("identity") == identity else {}
        indexed = [row for paper in corpus.values() for row in paper["chunks"] if row["indexable"]]
        missing = [row for row in indexed if row["content_hash"] not in vectors]
        embedder = AccountedEmbeddings(settings)
        try:
            for offset in range(0, len(missing), settings.embedding_batch_size):
                batch = missing[offset : offset + settings.embedding_batch_size]
                async with accounting.slot(owner):
                    values = await accounting.call(
                        owner,
                        uuid4(),
                        "query_embedding",
                        sum(r["token_count"] for r in batch),
                        lambda batch=batch: embedder.aembed_documents(
                            [r["embedding_input"] for r in batch]
                        ),
                        lambda: embedder.last_usage,
                    )
                if len(values) != len(batch) or any(
                    len(v) != settings.embedding_dim
                    or not any(v)
                    or any(not math.isfinite(x) for x in v)
                    for v in values
                ):
                    raise ValueError("Invalid corpus embeddings")
                for row, value in zip(batch, values, strict=True):
                    vectors[row["content_hash"]] = value
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps({"identity": identity, "vectors": vectors}))
                cache_path.chmod(0o600)
                print(
                    f"Corpus embeddings: {min(offset + len(batch), len(missing))}/{len(missing)}",
                    flush=True,
                )
        finally:
            await embedder.aclose()
        paper_ids, chunk_keys = await import_corpus(
            sessions, dataset, corpus, identity, vectors, owner, foreign
        )
        variants = {}
        for variant in selected_variants:
            rows, operations = [], []
            answerability_rows = []
            for case in dataset["cases"] + dataset.get("answerability_cases", []):
                scope = case_scope(owner, paper_ids, case)
                retriever = create_retriever(settings, sessions, engine, scope)
                query_started = time.perf_counter()
                result = await retriever.search(case["query"], variant=variant)
                latency_ms = (time.perf_counter() - query_started) * 1000
                operations.append(result.operation_id)
                if result.degraded:
                    raise RuntimeError(
                        f"Evaluation cannot accept degraded retrieval: {result.degraded}"
                    )
                expected = {(r["paper"], r["content_hash"]) for r in case["relevant"]}
                # Expanded evidence is citable in document order. Scope checks
                # cover every delivered source, including unindexed graph targets.
                for document in result.documents:
                    if (
                        document.metadata["user_id"] != str(owner)
                        or document.metadata["chunk_id"] not in chunk_keys
                    ):
                        raise AssertionError("Tenant leakage")
                    name, content_hash = chunk_keys[document.metadata["chunk_id"]]
                    if (case["scope"] != "all" and name != case["scope"]) or (
                        case.get("papers") and name not in case["papers"]
                    ):
                        raise AssertionError("Scope leakage")
                if case in dataset.get("answerability_cases", []):
                    answerability_rows.append(
                        {
                            "id": case["id"],
                            "expected_answerability": "insufficient",
                            "generation_evaluated": False,
                            "returned_sources": len(result.documents),
                            "latency_ms": round(latency_ms, 3),
                            "plan": result.plan,
                            "context_returned": [
                                {
                                    "paper": chunk_keys[d.metadata["chunk_id"]][0],
                                    "ordinal": d.metadata["ordinal"],
                                    "role": d.metadata.get("retrieval_role", "hit"),
                                    "rerank_score": d.metadata.get("rerank_score"),
                                }
                                for d in result.documents
                            ],
                        }
                    )
                    continue
                ranked = sorted(
                    (
                        d
                        for d in result.documents
                        if d.metadata.get("retrieval_role", "hit") == "hit"
                    ),
                    key=lambda d: d.metadata.get("hit_rank", 0),
                )[:8]
                ranks, returned = [], []
                for rank, document in enumerate(ranked, 1):
                    name, content_hash = chunk_keys[document.metadata["chunk_id"]]
                    if (name, content_hash) in expected:
                        ranks.append(rank)
                    returned.append(
                        {
                            "paper": name,
                            "ordinal": document.metadata["ordinal"],
                            "content_hash": content_hash,
                            "pages": [
                                document.metadata["page_start"],
                                document.metadata["page_end"],
                            ],
                            "evidence_group_rank": document.metadata.get("hit_rank"),
                        }
                    )
                context_keys = {chunk_keys[d.metadata["chunk_id"]] for d in result.documents}
                context_relevant_count = len(context_keys & expected)
                rows.append(
                    {
                        "id": case["id"],
                        "relevant_ranks": ranks,
                        "relevant_count": len(expected),
                        "returned": returned,
                        "context_sources": len(result.documents),
                        "context_hit": context_relevant_count > 0,
                        "context_relevant_count": context_relevant_count,
                        "core_bytes": sum(len(d.page_content.encode()) for d in ranked),
                        "context_bytes": sum(
                            len(d.page_content.encode()) for d in result.documents
                        ),
                        "latency_ms": round(latency_ms, 3),
                        "channel_counts": result.channel_counts,
                        "plan": result.plan,
                        "expected_retrieval": case.get("expected_retrieval", {}),
                        "context_returned": [
                            {
                                "paper": chunk_keys[d.metadata["chunk_id"]][0],
                                "ordinal": d.metadata["ordinal"],
                                "content_hash": chunk_keys[d.metadata["chunk_id"]][1],
                                "role": d.metadata.get("retrieval_role", "hit"),
                            }
                            for d in result.documents
                        ],
                    }
                )
            variants[variant] = {
                **detailed_metrics(rows),
                "cases": rows,
                "leaks": 0,
                "usage": await usage_rows(sessions, operations),
                "core_miss_ids": [r["id"] for r in rows if not r["relevant_ranks"]],
                "context_miss_ids": [r["id"] for r in rows if not r["context_hit"]],
                "answerability_cases": answerability_rows,
            }
            print(
                f"{variant}: Hit@8={variants[variant]['hit_at_8']:.4f}, MRR@8={variants[variant]['mrr_at_8']:.4f}",
                flush=True,
            )
        async with sessions() as db:
            versions = dict(
                (
                    await db.execute(
                        text(
                            "SELECT extname, extversion FROM pg_extension WHERE extname IN ('vector', 'pg_search')"
                        )
                    )
                ).all()
            )
        return {
            "status": ("passed" if variants["hybrid_rerank"]["hit_at_8"] >= 0.8 else "failed")
            if "hybrid_rerank" in variants
            else "completed",
            "acceptance": {
                "applied": "hybrid_rerank" in variants,
                "criterion": "hybrid_rerank hit_at_8 >= 0.80; baselines are descriptive",
            },
            "metric_definitions": {
                "ks": [1, 3, 5, 8],
                "averaging": "macro over questions; frozen binary child-chunk labels",
                "fusion": "Weighted RRF: sum(channel_weight / (rrf_k + one-based channel rank)); absent channels contribute zero. Weights are positive relative values, not raw score interpolation.",
                "core": "First 8 delivered seed child sources (role=hit), ordered by evidence-group rank. Top-8 applies to evidence bundles assembled before reranking; expanded sources do not become independent seed hits. Not directly comparable to historical pre-expansion child rankings.",
                "precision_at_k": "Relevant core hits in first K / K; missing slots count as nonrelevant.",
                "map_at_k": "Mean sum(precision at each relevant rank <= K) / min(K, total labeled relevant chunks).",
                "ndcg_at_k": "Binary gain with log2(rank+1) discount; ideal has min(K, total relevant) positives.",
                "full_evidence": "All labeled child chunks present; distinct from at least one hit.",
                "context": "Unordered union of seed, parent, neighbor and explicit cross-reference sources; includes authorized unindexed targets.",
                "answerability": "Unsupported questions are reported separately and excluded from positive ranking metric denominators. This retrieval-only script does not judge generation abstention or answer correctness; returning related background is not itself a failure.",
                "latency": "Milliseconds per sequential search including query embedding, SQL, rerank where enabled and expansion; excludes corpus import/embedding and is not a load test. Percentiles use linear interpolation.",
                "usage": "Variant usage is filtered by search operation IDs, excluding corpus embedding; overall usage includes any uncached corpus embedding. Calls count ledger records; not low-level transport retries.",
            },
            "dataset": dataset["version"],
            "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
            "execution_source_sha256": source_fingerprints,
            "annotation": dataset["annotation"],
            "cases": len(dataset["cases"]),
            "answerability_cases": len(dataset.get("answerability_cases", [])),
            "review_status": dataset.get("human_review", {}).get("status", "legacy_unspecified"),
            "chunker_version": dataset.get("chunker_version", "legacy_unspecified"),
            "retrieval_strategy": dataset.get("retrieval_strategy", "legacy_unspecified"),
            "evaluation_path": "retrieval.search only; model-selected paper tools and generation are not evaluated",
            "total_chunks_per_tenant": sum(len(paper["chunks"]) for paper in corpus.values()),
            "indexed_chunks_per_tenant": len(indexed),
            "tenants": 2,
            "embedding": identity,
            "rerank_model": settings.rerank_model,
            "configuration": {
                key: getattr(settings, key)
                for key in [
                    "retrieval_top_k",
                    "rerank_top_n",
                    "rrf_k",
                    "rrf_vector_weight",
                    "rrf_bm25_weight",
                    "rrf_metadata_weight",
                    "rerank_score_threshold",
                    "retrieval_max_subqueries",
                    "retrieval_merge_min_hits",
                    "retrieval_cross_ref_max_chunks",
                    "retrieval_background_weight",
                    "retrieval_table_weight",
                    "retrieval_experiment_weight",
                    "rerank_candidate_bytes",
                    "retrieval_context_bytes",
                    "retrieval_parent_bytes",
                    "retrieval_max_sources",
                    "retrieval_parent_max_chunks",
                    "retrieval_neighbor_window",
                ]
            },
            "extensions": versions,
            "variants": variants,
            "usage": await usage_rows(sessions),
            "corpus_embedding_cache": {
                "hits": len(indexed) - len(missing),
                "new": len(missing),
            },
            "elapsed_seconds": round(time.monotonic() - started, 2),
        }
    finally:
        await engine.dispose()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-paid", action="store_true", required=True)
    parser.add_argument("--dataset", type=Path, default=ROOT / "eval/datasets/m3-gold-v7.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/m3-local/retrieval.json")
    parser.add_argument("--cache", type=Path, default=ROOT / "reports/m3-local/embeddings.json")
    parser.add_argument("--vector-weight", type=float, help="Override RRF vector weight in (0, 1]")
    parser.add_argument("--bm25-weight", type=float, help="Override RRF BM25 weight in (0, 1]")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["vector", "bm25", "hybrid", "hybrid_rerank"],
        default=["vector", "bm25", "hybrid", "hybrid_rerank"],
        help="Run only the selected retrieval variants; one name runs an independent ablation",
    )
    args = parser.parse_args()
    if len(set(args.variants)) != len(args.variants):
        parser.error("--variants must not contain duplicates")
    weights = {}
    if (args.vector_weight is None) != (args.bm25_weight is None):
        parser.error("provide both --vector-weight and --bm25-weight")
    if args.vector_weight is not None:
        for name, weight in (
            ("rrf_vector_weight", args.vector_weight),
            ("rrf_bm25_weight", args.bm25_weight),
        ):
            if not math.isfinite(weight) or not 0 < weight <= 1:
                parser.error("RRF weights must be finite and in (0, 1]")
            weights[name] = weight
    # Verify frozen inputs before starting containers or making paid calls.
    dataset, _ = load_dataset(args.dataset)
    require_reviewed(dataset)
    with PostgresContainer(
        "paradedb/paradedb:0.26.3-pg16",
        username="m3_eval",
        password="test-only",
        dbname="m3_eval",
    ) as pg:
        database = pg.get_connection_url().replace(
            "postgresql+psycopg2://", "postgresql+asyncpg://"
        )
        values = {
            "DATABASE_URL": database,
            "APP_ENV": "test",
            "TRACING_ENABLED": "false",
        }
        subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            cwd=ROOT / "backend",
            env={**os.environ, **values},
            check=True,
        )
        settings = Settings(
            **{k.lower(): v for k, v in values.items()},
            user_daily_model_tokens=10000000,
            global_daily_model_tokens=20000000,
            **weights,
        )
        report = asyncio.run(evaluate(settings, args.dataset, args.cache, args.variants))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "variants"}, ensure_ascii=False))
    raise SystemExit(0 if report["status"] in {"passed", "completed"} else 1)


if __name__ == "__main__":
    main()
