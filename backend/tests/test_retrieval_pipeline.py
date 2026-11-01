"""Offline behavior tests using real scope SQL and deterministic provider doubles."""

import asyncio
from contextlib import asynccontextmanager

import pytest
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Chunk, Paper, utcnow
from app.services.chunking.chunker import chunk_document
from app.services.chunking.crossrefs import acronyms, labels
from app.services.retrieval.accounting import RetrievalAccounting
from app.services.retrieval.hybrid import HybridRetriever
from app.services.retrieval.planning import expand_acronyms, keywords, route_query
from app.services.retrieval.types import RetrievalScope
from tests.test_chunking import block, document
from tests.test_parent_retrieval_store import hit
from tests.test_parent_retrieval_store import parent_store as parent_store


@pytest.mark.parametrize(
    "query,intent",
    [
        ("这篇论文的核心创新点和局限性是什么？", "overview"),
        ("Summarize the main contributions", "overview"),
        ("在 GSM8K 数据集上 pass@1 是多少？", "detail"),
        ("λ_1 的超参是多少", "detail"),
        ("本文方法如何解决引言中的第二个痛点？", "comparison"),
        ("Compare methods in these papers", "comparison"),
        ("How is the objective defined?", "local"),
        ("What hidden dimensionality is used in the encoder?", "detail"),
        ("Which optimizer and beta values are used?", "detail"),
        ("How long does pretraining take on eight GPUs?", "detail"),
        ("How much memory is required for adaptation?", "detail"),
        ("适配权重的 checkpoint 有多大？", "detail"),
        ("固定参数预算下哪种投影适配效果最好？", "detail"),
        ("How small can the adaptation rank be?", "detail"),
        ("Explain the intuition behind low-rank updates", "local"),
        ("讨论 GPU 并行的基本思想", "local"),
    ],
)
def test_query_intent(query, intent):
    assert route_query(query).intent == intent


def test_decomposition_is_bounded_and_background_is_explicit():
    plan = route_query("本文方法如何解决引言中第二个痛点？")
    assert len(plan.queries) == 3 and "引言" in plan.queries[1]
    assert "方法" in plan.queries[2]
    assert len(route_query("compare A and B", max_queries=2).queries) == 2
    assert route_query("Explain related work").include_background
    assert not route_query("Explain the method").include_background
    assert "λ_1" in keywords("GSM8K pass@1 λ_1")
    assert "λ_1" in keywords("λ_1是多少")
    assert route_query("Compare GSM8K pass@1 in both papers").metric_focus


def test_acronyms_use_definitions_word_boundaries_and_byte_budget():
    text = "We use Group Relative Policy Optimization (GRPO). Transformer (TF) is unrelated."
    assert acronyms(text)["GRPO"] == "Group Relative Policy Optimization"
    dictionary = {"GRPO": [acronyms(text)["GRPO"]]}
    assert "Group Relative" in expand_acronyms("GRPO算法", dictionary, max_bytes=200)
    assert expand_acronyms("GRPOX", dictionary, max_bytes=200) == "GRPOX"
    assert expand_acronyms("GRPO", dictionary, max_bytes=4) == "GRPO"


def test_explicit_graph_targets_have_exact_labels_and_real_source_ordinals():
    blocks = [
        block(
            "Group Relative Policy Optimization (GRPO) uses Equation (4), see Table 2 and refer to Appendix B.1. Table 20 is absent.",
            page=0,
        ),
        block(r"$$x=y \tag{4}$$", "formula", page=1, section="eq", path=["3 Objective"]),
        block(
            "<table><tr><td>GSM8K</td><td>92.3</td></tr></table>",
            "table",
            page=2,
            section="exp",
            path=["4 Experiments"],
        ),
        block(
            "Table 2: results compared with Table 9.",
            "caption",
            page=3,
            section="exp",
            path=["4 Experiments"],
        ),
        block("Proof of convergence.", page=4, section="app", path=["Appendix B", "B.1 Proof"]),
    ]
    drafts = chunk_document(document(blocks), "Paper")
    edges = {
        edge["label"]: edge["target_ordinals"] for edge in drafts[0].chunk_metadata["cross_refs"]
    }
    assert (
        drafts[0].chunk_metadata["document_acronyms"]["GRPO"]
        == "Group Relative Policy Optimization"
    )
    assert drafts[edges["equation:4"][0]].kind == "formula"
    assert drafts[edges["table:2"][0]].kind == "table"
    assert "92.3" in drafts[edges["table:2"][0]].text
    assert drafts[edges["appendix:b.1"][0]].section_path[-1] == "B.1 Proof"
    assert edges["table:20"] == []
    assert not any("table:9" in d.chunk_metadata["defined_labels"] for d in drafts)
    assert labels("见表2和公式（4），参考附录B.1。") == ["table:2", "equation:4", "appendix:b.1"]


def test_section_alias_resolves_defined_appendix_without_inventing_other_targets():
    drafts = chunk_document(
        document(
            [
                block(
                    "See Section D.1 and Section 1, but Section D.10 is missing.",
                    path=["2 Results"],
                ),
                block(
                    "Fair comparison settings.",
                    page=1,
                    section="app",
                    path=["Appendix D", "D.1 Settings"],
                ),
                block("Introduction content.", page=2, section="intro", path=["1 Introduction"]),
            ]
        ),
        "Paper",
    )
    anchor = next(d for d in drafts if "See Section" in d.text and d.kind == "text")
    edges = {e["label"]: e["target_ordinals"] for e in anchor.chunk_metadata["cross_refs"]}
    assert edges["section:d.1"] and edges["section:1"]
    assert edges["section:d.10"] == []
    assert all(drafts[i].section_path[-1] == "D.1 Settings" for i in edges["section:d.1"])
    assert all(drafts[i].section_path[-1] == "1 Introduction" for i in edges["section:1"])


async def test_multiple_hits_merge_parent_before_rerank_and_single_hit_gets_window(parent_store):
    c = parent_store
    units = await c.store.expand_candidates(c.scope, [hit(c, 0), hit(c, 2)])
    assert len(units) == 1
    assert all(f"Original source {i}." in units[0].page_content for i in range(3))
    docs = await c.store.finalize_context(c.scope, units)
    assert [d.metadata["ordinal"] for d in docs] == [0, 1, 2]
    assert all(d.metadata["parent_complete"] for d in docs)
    assert [d.metadata["locations"][0]["page_idx"] for d in docs] == [0, 1, 2]
    units = await c.store.expand_candidates(c.scope, [hit(c, 0)])
    assert len(units[0].metadata["evidence"]) == 2
    assert not any(
        d.metadata["parent_complete"] for d in await c.store.finalize_context(c.scope, units)
    )


def install_graph(c, *, oversized=False):
    c.settings.retrieval_neighbor_window = 0
    with Session(c.engine) as db:
        anchor = db.get(Chunk, c.chunks[0][0])
        anchor.text = "Results are given in Table 2."
        anchor.chunk_metadata = {
            "document_acronyms": {"GRPO": "Group Relative Policy Optimization"},
            "cross_refs": [{"label": "table:2", "target_ordinals": [4]}],
        }
        target = db.get(Chunk, c.chunks[0][4])
        target.kind = "table"
        target.text = "GSM8K pass@1 = 92.3" + ("x" * 13000 if oversized else "")
        target.embedding_input = "GSM8K results"
        target.indexable = False
        target.embedding = None
        target.chunk_metadata = {"cross_refs": [{"label": "equation:4", "target_ordinals": [3]}]}
        db.commit()


async def test_one_hop_fetches_full_unindexed_target_and_stops_before_second_hop(parent_store):
    c = parent_store
    install_graph(c)
    units = await c.store.expand_candidates(c.scope, [hit(c, 0)])
    assert "92.3" in units[0].metadata["embedding_input"]
    docs = await c.store.finalize_context(c.scope, units)
    assert [d.metadata["ordinal"] for d in docs] == [0, 4]
    assert docs[1].metadata["retrieval_role"] == "cross_ref"
    assert docs[1].metadata["locations"][0]["page_idx"] == 4
    assert not any(d.metadata["ordinal"] == 3 for d in docs)
    assert await c.store.expand_candidates(c.scope, [hit(c, foreign=True)]) == []
    assert await c.store.expand_candidates(RetrievalScope(c.owner, paper_ids=()), [hit(c, 0)]) == []
    dictionary = await c.store.acronym_dictionary(c.scope)
    assert dictionary == {"GRPO": ["Group Relative Policy Optimization"]}


async def test_reference_bundles_are_atomic_under_context_and_source_budgets(parent_store):
    c = parent_store
    install_graph(c, oversized=True)
    units = await c.store.expand_candidates(c.scope, [hit(c, 0)])
    assert units == []  # Impossible groups must not consume paid reranking slots.
    assert await c.store.finalize_context(c.scope, units) == []
    c.settings.retrieval_context_bytes = 20000
    units = await c.store.expand_candidates(c.scope, [hit(c, 0)])
    assert len(await c.store.finalize_context(c.scope, units)) == 2


async def test_oversized_optional_expansion_keeps_independent_seeds_and_required_refs(parent_store):
    c = parent_store
    c.settings.retrieval_context_bytes = 1024
    with Session(c.engine) as db:
        # An optional sibling cites a target that cannot fit. Other seeds in
        # this parent remain independently useful and must not be suppressed.
        db.get(Chunk, c.chunks[0][1]).chunk_metadata = {
            "cross_refs": [{"label": "table:2", "target_ordinals": [4]}]
        }
        db.get(Chunk, c.chunks[0][4]).text = "x" * 2000
        db.commit()
    units = await c.store.expand_candidates(c.scope, [hit(c, 0), hit(c, 1), hit(c, 2)])
    assert [u.metadata["ordinal"] for u in units] == [0, 2]
    assert all("context_budget" in u.metadata["expansion_incomplete"] for u in units)
    docs = await c.store.finalize_context(c.scope, units)
    assert [d.metadata["ordinal"] for d in docs] == [0, 2]
    assert all(not d.metadata["parent_complete"] for d in docs)


async def test_deletion_or_version_switch_during_rerank_revokes_reference_bundle(parent_store):
    c = parent_store
    install_graph(c)
    units = await c.store.expand_candidates(c.scope, [hit(c, 0)])
    docs = await c.store.finalize_context(c.scope, units)
    with Session(c.engine) as db:
        db.delete(db.get(Chunk, c.chunks[0][4]))
        db.commit()
    assert await c.store.finalize_context(c.scope, units) == []
    assert await c.store.refresh(c.scope, docs) == []


async def test_shared_source_retains_later_seed_role_rank_and_reference_dependencies(parent_store):
    c = parent_store
    install_graph(c)
    units = await c.store.expand_candidates(c.scope, [hit(c, 0), hit(c, 4)])
    docs = await c.store.finalize_context(c.scope, units)
    seed = next(d for d in docs if d.metadata["ordinal"] == 4)
    assert seed.metadata["retrieval_role"] == "hit" and seed.metadata["hit_rank"] == 2
    assert str(c.chunks[0][3]) in seed.metadata["required_cross_ref_ids"]
    with Session(c.engine) as db:
        db.delete(db.get(Chunk, c.chunks[0][3]))
        db.commit()
    # Promotion must preserve the new seed's own dependencies, so a removed
    # target cannot leave either dependent group falsely valid.
    assert await c.store.refresh(c.scope, docs) == []


async def test_unresolved_ref_is_reported_without_fabricated_target(parent_store):
    c = parent_store
    with Session(c.engine) as db:
        db.get(Chunk, c.chunks[0][0]).chunk_metadata = {
            "cross_refs": [{"label": "table:999", "target_ordinals": []}]
        }
        db.commit()
    units = await c.store.expand_candidates(c.scope, [hit(c, 0)])
    assert units[0].metadata["unresolved_cross_refs"] == ["table:999"]


async def test_overview_has_section_boundaries_and_obeys_scope_without_embeddings(parent_store):
    c = parent_store
    with Session(c.engine) as db:
        db.get(Chunk, c.chunks[0][1]).kind = "text"
        db.get(Chunk, c.chunks[0][3]).embedding = None
        db.commit()
    docs = await c.store.overview(c.scope)
    assert [d.metadata["ordinal"] for d in docs] == [0, 3, 4]
    assert all(d.metadata["user_id"] == str(c.owner) for d in docs)
    assert await c.store.overview(RetrievalScope(c.owner, paper_ids=())) == []


async def test_metadata_prior_downweights_background_unless_asked_and_boosts_experiments(
    parent_store,
):
    c = parent_store
    with Session(c.engine) as db:
        db.get(Chunk, c.chunks[0][0]).section_path = ["Related Work"]
        db.get(Chunk, c.chunks[0][4]).kind = "table"
        db.commit()

        def weights(query):
            return dict(
                db.execute(
                    select(Chunk.ordinal, c.store.metadata_weight(query)).where(
                        Chunk.user_id == c.owner
                    )
                ).all()
            )

        assert weights("Explain the method")[0] < weights("Explain the method")[1]
        assert weights("Explain related work")[0] == weights("Explain related work")[1]
        assert weights("GSM8K pass@1是多少")[4] > weights("GSM8K pass@1是多少")[1]


class OfflineAccounting(RetrievalAccounting):
    @asynccontextmanager
    async def slot(self, user_id):
        yield

    async def call(self, user_id, operation_id, node, amount, callback, usage):
        return await callback()


def offline_retriever(c, monkeypatch, *, score=0.9):
    seen = {"queries": [], "rerank": [], "scopes": []}

    class Embedder:
        last_usage = 1

        async def aembed_documents(self, queries):
            seen["queries"] += queries
            return [[0.1] * 1024 for _ in queries]

        async def aclose(self):
            pass

    class Reranker:
        last_usage = 1

        async def acompress_documents(self, documents, query):
            seen["rerank"] += documents
            for doc in documents:
                doc.metadata["rerank_score"] = score
            return documents[: c.settings.rerank_top_n]

    async def recalled(scope, *args):
        await asyncio.sleep(0)
        seen["scopes"].append(scope)
        return [hit(c, 0)]

    monkeypatch.setattr(c.store, "vector", recalled)
    monkeypatch.setattr(c.store, "bm25", recalled)
    retriever = HybridRetriever(
        settings=c.settings,
        scope=c.scope,
        store=c.store,
        accounting=OfflineAccounting(c.store.sessions, None, c.settings),
        embedder_factory=Embedder,
        reranker_factory=Reranker,
    )
    return retriever, seen


async def test_production_flow_reranks_expanded_evidence_and_filters_scores(
    parent_store, monkeypatch
):
    c = parent_store
    install_graph(c)
    retriever, seen = offline_retriever(c, monkeypatch)
    result = await retriever.search("GRPO method")
    assert "Group Relative Policy Optimization" in seen["queries"][0]
    assert "92.3" in seen["rerank"][0].page_content
    assert [d.metadata["ordinal"] for d in result.documents] == [0, 4]
    assert result.plan["intent"] == "local" and not result.degraded
    retriever, seen = offline_retriever(c, monkeypatch, score=0.01)
    assert not (await retriever.search("GRPO method")).documents


async def test_overview_skips_providers_and_complex_queries_run_parallel_recall(
    parent_store, monkeypatch
):
    c = parent_store
    retriever, seen = offline_retriever(c, monkeypatch)
    result = await retriever.search("概述核心创新点和局限性")
    assert result.documents and not seen["queries"] and not seen["rerank"]
    result = await retriever.search("本文方法如何解决引言中第二个痛点？")
    assert len(seen["queries"]) == 3
    assert result.plan["intent"] == "comparison"
    assert result.channel_counts == {"vector": 3, "bm25": 3}


async def test_new_pipeline_rechecks_tenant_scope_and_revocation(parent_store, monkeypatch):
    c = parent_store
    retriever, seen = offline_retriever(c, monkeypatch)
    retriever.scope = RetrievalScope(c.owner, paper_ids=(c.papers[1],))
    with pytest.raises(HTTPException) as error:
        await retriever.search("GRPO")
    assert error.value.status_code == 404 and seen["queries"] == []
    retriever.scope = c.scope
    with Session(c.engine) as db:
        db.get(Paper, c.papers[0]).deleted_at = utcnow()
        db.commit()
    assert not (await retriever.search("GRPO")).documents


async def test_metadata_channel_preserves_exact_symbols_and_runs_for_metric_comparisons(
    parent_store, monkeypatch
):
    c = parent_store
    with Session(c.engine) as db:
        table = db.get(Chunk, c.chunks[0][4])
        table.kind = "table"
        table.embedding_input = "GSM8K pass@1 λ_1 = 0.5"
        other = db.get(Chunk, c.chunks[0][3])
        other.kind = "table"
        other.embedding_input = "λx1 = 0.7"
        db.commit()
    assert [d.metadata["ordinal"] for d in await c.store.metadata(c.scope, "λ_1是多少", 10)] == [4]
    retriever, _ = offline_retriever(c, monkeypatch)
    result = await retriever.search("Compare GSM8K pass@1 in both papers")
    assert result.plan["intent"] == "comparison"
    assert result.channel_counts["metadata"] > 0


async def test_one_failed_subquery_preserves_healthy_recall_and_quota_errors_propagate(
    parent_store, monkeypatch
):
    c = parent_store
    retriever, _ = offline_retriever(c, monkeypatch)

    async def partial(scope, query, limit):
        if "引言中提出" in query:
            raise TimeoutError()
        return [hit(c, 0)]

    monkeypatch.setattr(c.store, "bm25", partial)
    result = await retriever.search("本文方法如何解决引言中第二个痛点？", variant="bm25")
    assert result.documents and result.degraded == ["bm25_unavailable"]
    assert result.channel_counts["bm25"] == 2

    async def denied(scope, query, limit):
        raise HTTPException(429, "quota")

    monkeypatch.setattr(c.store, "bm25", denied)
    with pytest.raises(HTTPException) as error:
        await retriever.search("本文方法如何解决引言中第二个痛点？", variant="bm25")
    assert error.value.status_code == 429
